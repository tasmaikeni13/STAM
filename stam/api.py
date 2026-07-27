r"""The two-line interface: attach STAM to a training loop you already have.

::

    from stam import LandscapeRecorder

    recorder = LandscapeRecorder(model, loss_fn, eval_batches, every=50)

    for batch in loader:
        loss = loss_fn(model, batch).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=False)
        recorder.step()                       # <- 1

    report = recorder.render("stam_out")      # <- 2

``loss_fn(model, batch)`` returns either per-example losses (preferred -- it gives the
variance estimate the certificate needs for free) or a scalar. ``eval_batches`` is any
sequence of batches your ``loss_fn`` accepts; it defines the surface being drawn, so it
should be a fixed set, not a shuffling loader.

What ``render`` does: builds the plane from the recorded trajectory, spends a pilot
slice measuring the constants, allocates the rest, probes, reconstructs, certifies, and
writes a figure that states its own error. What it never does is consult anything the
caller has not paid for.

Two requirements the wrapper enforces rather than assumes:

* ``optimizer.zero_grad(set_to_none=False)``. STAM rebinds ``p.grad`` to views of one
  contiguous buffer so a gradient read is zero-copy; ``set_to_none=True`` drops those
  views. :meth:`LandscapeRecorder.zero_grad` does the right thing if you use it.
* The gradient must still be populated when :meth:`step` is called, i.e.\ call it after
  ``optimizer.step()`` and before zeroing. If it is not, the mini-batch gradients are
  recorded as zeros and the animation is disabled rather than silently wrong.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import time
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn as nn

from . import kernels
from .basis import build_plane, project_trajectory, residual_norms
from .data import BatchEvalSet, EvalSet, _infer_size
from .design import render_grid
from .device import describe_environment, human_time, set_matmul_precision
from .flat import FlatParams
from .models import Task
from .pipeline import certified_landscape
from .probe import PlaneProbe, calibrate_cost, tune_micro_batches
from .reconstruct import reconstruct

LossFn = Callable[[nn.Module, Any], torch.Tensor]


class CallableTask(Task):
    """Adapts an arbitrary ``loss_fn(model, batch)`` to the probe's interface."""

    def __init__(self, model: nn.Module, loss_fn: LossFn, name: str = "user"):
        super().__init__(model, name)
        self.loss_fn = loss_fn

    def per_example_loss(self, batch) -> torch.Tensor:
        out = self.loss_fn(self.model, batch)
        if not torch.is_tensor(out):
            raise TypeError("loss_fn must return a tensor")
        if out.dim() == 0:
            # A scalar loss still works; it just means the per-example spread is
            # unavailable, so the variance estimate falls back to the between-batch
            # spread across micro-batches.  Expanding to the batch size keeps every
            # downstream mean and weighting correct.
            n = _infer_size(batch)
            return out.expand(max(n, 1))
        return out.reshape(-1)

    @torch.no_grad()
    def accuracy(self, batch) -> float:
        return float("nan")


@dataclasses.dataclass
class RenderReport:
    """Everything :meth:`LandscapeRecorder.render` produced, and what it cost."""

    figure: str
    animation: str | None
    certificate: dict
    plane: dict
    allocation: dict
    fidelity: dict
    budget: float
    seconds: float
    arrays: str

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def summary(self) -> str:
        c = self.certificate
        return (
            f"{self.figure}: {self.allocation['n_anchors']} anchors x "
            f"{self.allocation['examples_per_anchor']} examples; "
            f"95% of the domain within {c['q95_upper']:.4g} "
            f"({100 * c['q95_upper'] / max(c.get('surface_range', 1), 1e-12):.1f}% of relief)"
        )


class LandscapeRecorder:
    """Records an optimiser's trajectory, then renders a certified landscape.

    Parameters
    ----------
    model:
        Any ``nn.Module``.  Its trainable parameters are rebound to a contiguous buffer.
    loss_fn:
        ``loss_fn(model, batch) -> Tensor``, per-example losses or a scalar.
    eval_batches:
        A fixed sequence of batches defining the surface.  Keep it small enough to
        evaluate repeatedly: a few thousand examples is plenty, and the cost of every
        probe scales with it.
    every:
        Snapshot stride, in calls to :meth:`step`.
    max_snapshots:
        Cap on the number of snapshots.  Once reached, the stride doubles and every
        second existing snapshot is dropped, so a run of unknown length stays bounded
        in memory while keeping coverage of the whole trajectory.
    store:
        ``"cpu"`` (default) keeps snapshots in host memory; ``"cuda"`` keeps them on the
        device, which is faster but needs ``2 T N`` floats of VRAM.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: LossFn,
        eval_batches: Sequence,
        *,
        every: int = 50,
        max_snapshots: int = 256,
        capture_gradients: bool = True,
        store: str = "cpu",
        eval_val_batches: Sequence | None = None,
        name: str = "model",
    ):
        self.task = CallableTask(model, loss_fn, name)
        self.flat = FlatParams(model)
        self.every = int(every)
        self.max_snapshots = int(max_snapshots)
        self.capture_gradients = bool(capture_gradients)
        self.store = torch.device(store)
        self.name = name

        self.eval_set = _as_eval_set(eval_batches, "eval")
        self.eval_val = _as_eval_set(eval_val_batches, "eval-val") if eval_val_batches else None

        self._params: list[torch.Tensor] = []
        self._grads: list[torch.Tensor] = []
        self._steps: list[int] = []
        self._n = 0
        self._grad_warned = False
        set_matmul_precision()

    # -- training-loop hooks ---------------------------------------------------

    def zero_grad(self) -> None:
        """Zero gradients without dropping the contiguous views STAM relies on."""
        self.flat.zero_grad()

    def step(self) -> None:
        """Call once per optimiser step, after ``optimizer.step()``."""
        self._n += 1
        if self._n % self.every == 0:
            self.snapshot()

    @torch.no_grad()
    def snapshot(self) -> None:
        """Record the current parameters (and gradient) unconditionally."""
        self._params.append(self.flat.vector.detach().to(self.store, copy=True))
        if self.capture_gradients:
            g = self.flat.grad_vector
            if not self._grad_warned and float(g.abs().max()) == 0.0 and self._n > 0:
                import warnings

                warnings.warn(
                    "LandscapeRecorder.step() saw an all-zero gradient. Call it after "
                    "backward()/optimizer.step() and before zero_grad(), or pass "
                    "capture_gradients=False. The mini-batch animation will be skipped.",
                    RuntimeWarning, stacklevel=2,
                )
                self._grad_warned = True
            self._grads.append(g.detach().to(self.store, copy=True, dtype=torch.float16))
        self._steps.append(self._n)
        if len(self._params) > self.max_snapshots:
            self._thin()

    def _thin(self) -> None:
        """Halve the stored trajectory and double the stride.

        Dropping every second snapshot preserves coverage of the whole path, which is
        what the plane is built from; keeping only the tail would bias the plane toward
        the converged region.
        """
        self._params = self._params[::2]
        self._grads = self._grads[::2]
        self._steps = self._steps[::2]
        self.every *= 2

    # -- rendering -------------------------------------------------------------

    @property
    def n_snapshots(self) -> int:
        return len(self._params)

    def trajectory(self) -> torch.Tensor:
        return torch.stack(self._params)

    def render(
        self,
        out_dir: str | pathlib.Path = "stam_out",
        *,
        budget: float | None = None,
        budget_seconds: float | None = 60.0,
        resolution: int = 65,
        margin: float = 0.25,
        anchoring: str = "centered",
        animate: bool = True,
        frames: int = 90,
        seed: int = 0,
        device: torch.device | str | None = None,
    ) -> RenderReport:
        """Build the plane, probe, certify, and write the figures.

        ``budget`` is in example-forward-equivalents.  If it is ``None`` the budget is
        derived from ``budget_seconds`` and the *measured* throughput of this model, so
        "give me a landscape in a minute" is a thing you can ask for.
        """
        from .viz import animate as A
        from .viz import render as R
        from .viz import style as S

        if self.n_snapshots < 8:
            raise RuntimeError(
                f"only {self.n_snapshots} snapshots recorded; need at least 8. "
                "Lower `every` or train for longer."
            )
        t_start = time.perf_counter()
        out = pathlib.Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        dev = torch.device(device) if device is not None else self.flat.device
        kernels.build_extension()
        S.use_paper_style()

        traj = self.trajectory()
        plane = build_plane(traj, dim=2, anchoring=anchoring, device=dev)
        coords = project_trajectory(plane, traj).cpu().numpy()
        resid = residual_norms(plane, traj).cpu().numpy()
        domain = _bbox_domain(coords, margin)

        probe = PlaneProbe(self.task, self.flat, plane, self.eval_set)
        probe.micro_batch_by_order = tune_micro_batches(probe)
        cost = calibrate_cost(probe)
        probe.cost_model = cost

        if budget is None:
            budget = float(budget_seconds or 60.0) / cost.seconds_per_example
        queries, shape = render_grid(domain, resolution)

        res = certified_landscape(probe, domain, budget, cost, queries,
                                  method="pu-taylor-2", seed=seed)
        cert = res.certificate

        Z = res.surface.values.reshape(shape)
        G = res.surface.grads.reshape(*shape, 2)
        X = queries[:, 0].reshape(shape)
        Y = queries[:, 1].reshape(shape)
        traj_z = reconstruct("pu-taylor-2", res.anchors, coords, device=dev,
                             radii=res.radii).values

        Zv, Gv, tzv = Z, G, traj_z
        res_val = None
        if self.eval_val is not None:
            pv = PlaneProbe(self.task, self.flat, plane, self.eval_val,
                            micro_batch=probe.micro_batch,
                            micro_batch_by_order=dict(probe.micro_batch_by_order),
                            cost_model=cost)
            res_val = certified_landscape(pv, domain, budget, cost, queries,
                                          method="pu-taylor-2", seed=seed + 1)
            Zv = res_val.surface.values.reshape(shape)
            Gv = res_val.surface.grads.reshape(*shape, 2)
            tzv = reconstruct("pu-taylor-2", res_val.anchors, coords, device=dev,
                              radii=res_val.radii).values

        err_v = res_val.certificate.rmse if res_val else cert.rmse
        fig, axes = R.landscape_figure(
            X, Y, Z, Zv, G, Gv, coords, traj_z, tzv,
            err_train=cert.rmse, err_val=err_v, anchors=res.anchors.coords,
            single=self.eval_val is None,
        )
        rel = cert.q95_upper / max(float(np.nanmax(Z) - np.nanmin(Z)), 1e-12)
        R.annotate_certificate(
            axes[1], loc="upper left",
            cert_text=f"certified: 95% of the domain within {cert.q95_upper:.3g}"
                      f" ({rel:.1%} of relief)\n"
                      f"{res.allocation.n_anchors} anchors x "
                      f"{res.allocation.examples_per_anchor} examples "
                      f"= {budget:,.0f} example-equivalents",
        )
        fig.tight_layout()
        fig_path = out / f"landscape_{self.name}.png"
        fig.savefig(fig_path, dpi=190)
        fig.savefig(out / f"landscape_{self.name}.pdf")

        gif_path = None
        if animate and self._grads and not self._grad_warned:
            stride = max(1, len(coords) // frames)
            idx = np.arange(0, len(coords), stride)
            gproj = np.stack([
                kernels.project(self._grads[t].to(dev).float(),
                                plane.basis).double().cpu().numpy()
                for t in idx
            ])
            at = reconstruct("pu-taylor-2", res.anchors, coords[idx], device=dev,
                             radii=res.radii)
            fr = A.build_frames(coords[idx], gproj, at.grads, np.zeros(len(idx)),
                                tolerance=max(cert.rmse, 1e-9),
                                radius_cap=0.45 * (domain[0][1] - domain[0][0]))
            gif_path = out / f"landscape_{self.name}.gif"
            A.animate_landscape(
                X, Y, Z, Zv, G, Gv, fr, traj_z[idx], tzv[idx], out_path=gif_path,
                err_train=cert.rmse, err_val=err_v,
                surface_fn=lambda q: reconstruct("pu-taylor-2", res.anchors, q,
                                                 device=dev, radii=res.radii).values,
                fps=12, dpi=88,
            )

        np.savez_compressed(
            out / f"landscape_{self.name}.npz", X=X, Y=Y, Z=Z, G=G, Z_val=Zv, G_val=Gv,
            traj=coords, traj_z=traj_z, residual=resid,
            anchors=res.anchors.coords, anchor_values=res.anchors.values,
        )
        cert_d = cert.to_dict()
        cert_d["surface_range"] = float(np.nanmax(Z) - np.nanmin(Z))
        report = RenderReport(
            figure=str(fig_path),
            animation=str(gif_path) if gif_path else None,
            certificate=cert_d,
            plane={
                **plane.spectrum_report(),
                "anchoring": anchoring,
                "residual_mean": float(resid.mean()),
                "residual_max": float(resid.max()),
                "snapshots": self.n_snapshots,
                "parameters": int(self.flat.numel),
            },
            allocation=res.allocation.to_dict(),
            fidelity={"domain": [list(domain[0]), list(domain[1])],
                      "cost_model": cost.to_dict()},
            budget=float(budget),
            seconds=time.perf_counter() - t_start,
            arrays=str(out / f"landscape_{self.name}.npz"),
        )
        (out / f"report_{self.name}.json").write_text(
            json.dumps(report.to_dict() | {"environment": describe_environment()},
                       indent=2, default=str)
        )
        print(f"[stam] {report.summary()}  ({human_time(report.seconds)})")
        return report


def _as_eval_set(batches: Sequence, name: str):
    """Use the fast tensor path when the batches are uniform tensor tuples."""
    if isinstance(batches, (EvalSet, BatchEvalSet)):
        return batches
    lst = list(batches)
    if not lst:
        raise ValueError(f"{name}: no batches given")
    first = lst[0]
    if (
        isinstance(first, (tuple, list))
        and all(torch.is_tensor(t) for t in first)
        and all(isinstance(b, (tuple, list)) and len(b) == len(first) for b in lst)
    ):
        try:
            cat = tuple(torch.cat([b[i] for b in lst]) for i in range(len(first)))
            return EvalSet(cat, name)
        except Exception:
            pass
    return BatchEvalSet(lst, name=name)


def _bbox_domain(coords: np.ndarray, margin: float
                 ) -> tuple[tuple[float, float], tuple[float, float]]:
    """Trajectory bounding box expanded by ``margin``, per axis.

    Per-axis rather than square: a trajectory is usually far longer than it is wide, and
    a square domain would reach into the region where the loss diverges along the short
    axis while adding nothing to the picture.
    """
    out = []
    for k in range(2):
        lo, hi = float(coords[:, k].min()), float(coords[:, k].max())
        pad = max((hi - lo) * margin, 1e-6)
        out.append((lo - pad, hi + pad))
    return (out[0], out[1])


__all__ = ["LandscapeRecorder", "CallableTask", "RenderReport"]
