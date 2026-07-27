r"""Training with instrumented trajectory capture.

Two things are recorded at a fixed stride: the parameter vector :math:`\theta_t`, which
defines the plane, and the mini-batch gradient :math:`g_{B_t}` that produced the step,
which is what the stochastic-surface model in :mod:`stam.viz.animate` needs.

Capture is not free, and the overhead is reported rather than assumed.  Each snapshot is
one device-to-host copy of :math:`N` floats; at stride :math:`s` over :math:`E` epochs
of :math:`M` steps that is :math:`2NEM/s` floats crossing PCIe.  The loop below
overlaps those copies with compute on a dedicated stream and measures the residual cost
by running matched control epochs with capture disabled.

The reference loss on the fixed evaluation sets is also recorded at every snapshot.
That single number is what later makes the projection gap measurable: it is the true
loss *at the trajectory point*, against which the loss at the trajectory's *shadow on
the plane* can be compared.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import time
from typing import Any, Callable

import numpy as np
import torch

from .data import DataBundle, EvalSet
from .device import human_time
from .flat import FlatParams
from .models import Task


@dataclasses.dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 5e-4
    optimizer: str = "adamw"
    momentum: float = 0.9
    warmup_steps: int = 100
    schedule: str = "cosine"
    grad_clip: float = 1.0
    snapshot_stride: int = 20
    capture_gradients: bool = True
    # Gradient snapshots are only ever used projected onto a 2-plane, where half
    # precision costs ~1e-3 relative error -- far below the Monte-Carlo noise on any
    # quantity derived from them -- and halves a buffer that reaches several GiB.
    grad_dtype: str = "float16"
    seed: int = 0
    control_epochs: int = 2  # matched epochs run without capture, for the overhead figure
    eval_batch: int = 500


@dataclasses.dataclass
class Trajectory:
    """Everything recorded during a run."""

    params: torch.Tensor            # (T, N) float32, host memory
    grads: torch.Tensor | None      # (T, N) float32, host memory
    steps: np.ndarray               # global step index of each snapshot
    epochs: np.ndarray              # fractional epoch of each snapshot
    loss_train: np.ndarray          # exact L_train at each snapshot
    loss_val: np.ndarray            # exact L_val at each snapshot
    batch_loss: np.ndarray          # mini-batch loss that produced the step
    history: list[dict]
    timing: dict
    config: dict

    @property
    def T(self) -> int:
        return int(self.params.shape[0])

    @property
    def N(self) -> int:
        return int(self.params.shape[1])

    def save(self, path: str | pathlib.Path) -> None:
        path = pathlib.Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"params": self.params, "grads": self.grads},
            path / "trajectory.pt",
        )
        np.savez(
            path / "trajectory_meta.npz",
            steps=self.steps, epochs=self.epochs, loss_train=self.loss_train,
            loss_val=self.loss_val, batch_loss=self.batch_loss,
        )
        (path / "train_report.json").write_text(
            json.dumps({"history": self.history, "timing": self.timing, "config": self.config},
                       indent=2)
        )

    @staticmethod
    def load(path: str | pathlib.Path) -> "Trajectory":
        path = pathlib.Path(path)
        blob = torch.load(path / "trajectory.pt", map_location="cpu", weights_only=True)
        meta = np.load(path / "trajectory_meta.npz")
        report = json.loads((path / "train_report.json").read_text())
        return Trajectory(
            params=blob["params"], grads=blob["grads"], steps=meta["steps"],
            epochs=meta["epochs"], loss_train=meta["loss_train"], loss_val=meta["loss_val"],
            batch_loss=meta["batch_loss"], history=report["history"], timing=report["timing"],
            config=report["config"],
        )


@torch.no_grad()
def exact_loss(task: Task, eval_set: EvalSet, batch_size: int = 500) -> tuple[float, float]:
    """Exact mean loss and accuracy over a fixed evaluation set."""
    task.eval()
    total, correct_sum, n = 0.0, 0.0, 0
    for batch in eval_set.batches(batch_size):
        per = task.per_example_loss(batch)
        total += float(per.sum())
        correct_sum += task.accuracy(batch) * per.numel()
        n += per.numel()
    task.train()
    return total / n, correct_sum / n


def _make_optimizer(cfg: TrainConfig, params) -> torch.optim.Optimizer:
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(
            params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay, nesterov=True
        )
    raise ValueError(f"unknown optimizer {cfg.optimizer!r}")


def _lr_at(cfg: TrainConfig, step: int, total: int) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(cfg.warmup_steps, 1)
    if cfg.schedule == "cosine":
        p = (step - cfg.warmup_steps) / max(total - cfg.warmup_steps, 1)
        return cfg.lr * 0.5 * (1 + np.cos(np.pi * min(p, 1.0)))
    return cfg.lr


def train_with_capture(
    task: Task,
    flat: FlatParams,
    data: DataBundle,
    cfg: TrainConfig,
    log: Callable[[str], None] = print,
) -> Trajectory:
    """Run training, recording snapshots at ``cfg.snapshot_stride``."""
    device = flat.device
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    n_train = len(data.train)
    steps_per_epoch = n_train // cfg.batch_size
    total_steps = steps_per_epoch * cfg.epochs
    n_snaps = total_steps // cfg.snapshot_stride + 2

    # Pinned host buffers: pinned memory is what lets the copies overlap with compute.
    pin = device.type == "cuda"
    params_buf = torch.empty(n_snaps, flat.numel, dtype=torch.float32, pin_memory=pin)
    grad_dtype = getattr(torch, cfg.grad_dtype)
    grads_buf = (
        torch.empty(n_snaps, flat.numel, dtype=grad_dtype, pin_memory=pin)
        if cfg.capture_gradients
        else None
    )
    copy_stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None

    opt = _make_optimizer(cfg, flat.parameters())
    steps, epochs_f, l_tr, l_va, b_loss = [], [], [], [], []
    history: list[dict] = []
    snap = 0
    step = 0

    capture_seconds = 0.0
    compute_seconds = 0.0
    eval_seconds = 0.0

    def take_snapshot(batch_loss_value: float) -> None:
        nonlocal snap, capture_seconds, eval_seconds
        if snap >= n_snaps:
            return
        t0 = time.perf_counter()
        if copy_stream is not None:
            copy_stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(copy_stream):
                params_buf[snap].copy_(flat.vector, non_blocking=True)
                if grads_buf is not None:
                    grads_buf[snap].copy_(flat.grad_vector, non_blocking=True)
            torch.cuda.current_stream(device).wait_stream(copy_stream)
        else:
            params_buf[snap].copy_(flat.vector)
            if grads_buf is not None:
                grads_buf[snap].copy_(flat.grad_vector)
        capture_seconds += time.perf_counter() - t0

        t1 = time.perf_counter()
        lt, _ = exact_loss(task, data.eval_train, cfg.eval_batch)
        lv, _ = exact_loss(task, data.eval_val, cfg.eval_batch)
        eval_seconds += time.perf_counter() - t1

        steps.append(step)
        epochs_f.append(step / max(steps_per_epoch, 1))
        l_tr.append(lt)
        l_va.append(lv)
        b_loss.append(batch_loss_value)
        snap += 1

    task.train()
    take_snapshot(float("nan"))
    wall0 = time.perf_counter()

    for epoch in range(cfg.epochs):
        perm = torch.randperm(n_train, generator=gen).to(device)
        running = 0.0
        t_epoch = time.perf_counter()
        for i in range(steps_per_epoch):
            idx = perm[i * cfg.batch_size : (i + 1) * cfg.batch_size]
            batch = data.train.index(idx)

            t0 = time.perf_counter()
            for group in opt.param_groups:
                group["lr"] = _lr_at(cfg, step, total_steps)
            flat.zero_grad()
            loss = task(batch)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(flat.parameters(), cfg.grad_clip)
            opt.step()
            compute_seconds += time.perf_counter() - t0

            lv = float(loss.detach())
            running += lv
            step += 1
            if step % cfg.snapshot_stride == 0:
                take_snapshot(lv)

        if device.type == "cuda":
            torch.cuda.synchronize()
        lt, at = exact_loss(task, data.eval_train, cfg.eval_batch)
        lva, av = exact_loss(task, data.eval_val, cfg.eval_batch)
        history.append(
            {
                "epoch": epoch + 1,
                "batch_loss": running / steps_per_epoch,
                "train_loss": lt, "train_acc": at,
                "val_loss": lva, "val_acc": av,
                "seconds": time.perf_counter() - t_epoch,
                "lr": _lr_at(cfg, step, total_steps),
            }
        )
        log(
            f"  epoch {epoch + 1:3d}/{cfg.epochs}  train {lt:.4f} ({at * 100:5.2f}%)  "
            f"val {lva:.4f} ({av * 100:5.2f}%)  {human_time(history[-1]['seconds'])}"
        )

    wall = time.perf_counter() - wall0
    take_snapshot(float(loss.detach()))

    # Matched control: identical work with capture disabled, so the reported overhead is
    # a measured difference rather than a subtraction of instrumented timers.
    control_seconds = None
    if cfg.control_epochs > 0:
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(cfg.control_epochs):
            perm = torch.randperm(n_train, generator=gen).to(device)
            for i in range(steps_per_epoch):
                idx = perm[i * cfg.batch_size : (i + 1) * cfg.batch_size]
                batch = data.train.index(idx)
                flat.zero_grad()
                loss = task(batch)
                loss.backward()
                if cfg.grad_clip:
                    torch.nn.utils.clip_grad_norm_(flat.parameters(), cfg.grad_clip)
                opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        control_seconds = (time.perf_counter() - t0) / cfg.control_epochs

    captured_epoch = float(np.mean([h["seconds"] for h in history]))
    timing: dict[str, Any] = {
        "wall_seconds": wall,
        "compute_seconds": compute_seconds,
        "capture_seconds": capture_seconds,
        "snapshot_eval_seconds": eval_seconds,
        "snapshots": snap,
        "steps": step,
        "steps_per_epoch": steps_per_epoch,
        "seconds_per_epoch_with_capture": captured_epoch,
        "seconds_per_epoch_control": control_seconds,
        "snapshot_bytes": int(params_buf[:snap].nbytes + (grads_buf[:snap].nbytes if grads_buf is not None else 0)),
    }
    if control_seconds:
        # The honest overhead figure: end-to-end epoch time with capture, including the
        # exact reference evaluations, against the same epoch without any of it.
        timing["capture_overhead_fraction"] = captured_epoch / control_seconds - 1.0
        timing["capture_overhead_excl_eval"] = (
            captured_epoch - eval_seconds / max(cfg.epochs, 1)
        ) / control_seconds - 1.0

    return Trajectory(
        params=params_buf[:snap].clone(),
        grads=grads_buf[:snap].clone() if grads_buf is not None else None,
        steps=np.asarray(steps), epochs=np.asarray(epochs_f, dtype=np.float64),
        loss_train=np.asarray(l_tr), loss_val=np.asarray(l_va),
        batch_loss=np.asarray(b_loss), history=history, timing=timing,
        config=dataclasses.asdict(cfg),
    )


__all__ = ["TrainConfig", "Trajectory", "train_with_capture", "exact_loss"]
