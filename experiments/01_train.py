"""Stage 1: train the two landscape subjects and capture their trajectories.

Produces, per subject, a directory under ``runs/<name>/`` containing the parameter and
gradient snapshots, the exact loss at every snapshot, and a timing report from which the
capture overhead is read.  Everything downstream reads from here, so this stage fixes
the random seeds, the evaluation sets, and the optimiser once and for all.

Both subjects use AdamW with a cosine schedule and a short linear warm-up.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import torch

from stam import kernels
from stam.capture import TrainConfig, train_with_capture
from stam.data import cost_unit, load_data
from stam.device import describe_environment, human_time, set_matmul_precision
from stam.flat import FlatParams
from stam.models import build_task, parameter_report
from stam.parallel import autotune_micro_batch, memory_report

PRESETS = {
    "cnn": TrainConfig(
        epochs=30, batch_size=128, lr=1.5e-3, weight_decay=5e-4, optimizer="adamw",
        warmup_steps=200, snapshot_stride=39, grad_clip=1.0, seed=0, eval_batch=2000,
    ),
    "gpt": TrainConfig(
        epochs=40, batch_size=32, lr=3e-4, weight_decay=0.01, optimizer="adamw",
        warmup_steps=200, snapshot_stride=68, grad_clip=1.0, seed=0, eval_batch=256,
    ),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["cnn", "gpt"], required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    set_matmul_precision()
    kernels.build_extension()

    cfg = PRESETS[args.task]
    if args.epochs:
        cfg.epochs = args.epochs
    out = pathlib.Path(args.out or f"runs/{args.task}")
    out.mkdir(parents=True, exist_ok=True)

    print(f"=== training {args.task} on {device} ===")
    t0 = time.perf_counter()
    data = load_data(args.task, device)
    task = build_task(args.task, device)
    flat = FlatParams(task.model)
    print(f"  {flat}")
    print(f"  {json.dumps(parameter_report(task.model))}")
    print(f"  data: {json.dumps(data.meta)}")

    # Size the training micro-batch to the device rather than guessing.  The tuned batch
    # is used for the *evaluation* passes; the optimisation batch stays at the
    # configured value because it is part of the training dynamics being visualised.
    def eval_step(b: int) -> None:
        idx = torch.arange(min(b, len(data.eval_train)), device=device)
        with torch.no_grad():
            task.per_example_loss(data.eval_train.index(idx)).mean()

    tuning = autotune_micro_batch(eval_step, device, start=128,
                                  maximum=min(1 << 15, len(data.eval_train)))
    cfg.eval_batch = tuning.micro_batch
    print(f"  eval micro-batch tuned to {tuning.micro_batch} "
          f"({tuning.throughput:,.0f} ex/s, peak {tuning.peak_bytes / 2**30:.2f} GiB)")

    traj = train_with_capture(task, flat, data, cfg)
    traj.save(out)

    report = {
        "task": args.task,
        "environment": describe_environment(),
        "model": parameter_report(task.model),
        "data": data.meta,
        "cost_unit": cost_unit(args.task, data),
        "train_config": traj.config,
        "timing": traj.timing,
        "batch_tuning": tuning.to_dict(),
        "memory": memory_report([device.index or 0]),
        "final": traj.history[-1] if traj.history else None,
        "eval_train_fingerprint": data.eval_train.fingerprint(),
        "eval_val_fingerprint": data.eval_val.fingerprint(),
        "total_seconds": time.perf_counter() - t0,
    }
    (out / "stage1_report.json").write_text(json.dumps(report, indent=2, default=str))

    ov = traj.timing.get("capture_overhead_fraction")
    print(f"\n  snapshots: {traj.T}  parameters: {traj.N:,}")
    print(f"  trajectory buffer: {traj.timing['snapshot_bytes'] / 2**30:.2f} GiB")
    if ov is not None:
        print(f"  capture overhead: {ov * 100:.1f}% of epoch time "
              f"({traj.timing['capture_overhead_excl_eval'] * 100:.1f}% excluding "
              f"reference evaluations)")
    print(f"  total: {human_time(report['total_seconds'])}")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
