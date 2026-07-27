r"""How much the plane misses, measured rather than asserted.

A 2-plane through a :math:`10^6`-dimensional parameter space cannot show everything, and
the honest question is not whether information is lost but *how much, in the units the
picture is read in*.

Two quantities are usually conflated and should not be.

**Trajectory capture** :math:`\rho_2 = (\sigma_1^2+\sigma_2^2)/\sum_i\sigma_i^2` says what
fraction of the *parameter-space* variance of the path lies in the plane.  It is a
statement about a curve in :math:`\mathbb R^N`, and a high value does not by itself
license any claim about the surface.

**The projection gap**

.. math::
   \gamma_t \;=\; L(\theta_t) \;-\; L(\Pi\theta_t),
   \qquad \Pi\theta = c + V^\top V(\theta-c),

is the quantity that matters: the difference between the loss the optimiser actually
had, and the loss at the point directly beneath it on the drawn surface.  It is the
error a reader makes when they read the trajectory's height off the picture.  It costs
nothing extra to measure -- :math:`L(\theta_t)` is recorded during training and
:math:`L(\Pi\theta_t)` is one probe -- and it is reported in loss units alongside the
range of the surface, so a reader can see immediately whether the picture supports the
conclusion being drawn from it.

The two are related but not interchangeable.  A first-order expansion gives

.. math::
   |\gamma_t| \le \|\nabla L(\Pi\theta_t)\|\, r_t + \tfrac12 M_2 r_t^2,
   \qquad r_t = \|(I-\Pi)(\theta_t-c)\| ,

so a small residual :math:`r_t` bounds the gap only when the *full* gradient norm is
also small.  Early in training it is not, and a plane with :math:`\rho_2 = 0.95` can
still misplace the trajectory's height by more than the total relief of the surface.
Reporting :math:`\rho_2` alone would hide that; reporting :math:`\gamma_t` cannot.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch

from .basis import Plane, project_trajectory, residual_norms
from .capture import Trajectory
from .probe import PlaneProbe


@dataclasses.dataclass
class FidelityReport:
    coords: np.ndarray            # (T, 2) trajectory in plane coordinates
    residual: np.ndarray          # (T,) out-of-plane parameter-space distance
    loss_true: np.ndarray         # (T,) L(theta_t), exact
    loss_on_plane: np.ndarray     # (T,) L(Pi theta_t)
    gap: np.ndarray               # (T,) projection gap
    captured_variance: float
    surface_range: float
    displacement: np.ndarray      # (T,) ||theta_t - c||

    def summary(self) -> dict:
        rel = np.abs(self.gap) / max(self.surface_range, 1e-12)
        finite = np.isfinite(self.gap)
        return {
            "captured_variance_rho2": round(self.captured_variance, 6),
            "gap_mean_abs": float(np.mean(np.abs(self.gap[finite]))),
            "gap_max_abs": float(np.max(np.abs(self.gap[finite]))),
            "gap_final_abs": float(abs(self.gap[finite][-1])) if finite.any() else float("nan"),
            "gap_relative_mean": float(np.mean(rel[finite])),
            "gap_relative_max": float(np.max(rel[finite])),
            "residual_mean": float(np.mean(self.residual)),
            "residual_max": float(np.max(self.residual)),
            "residual_fraction_mean": float(
                np.mean(self.residual / np.maximum(self.displacement, 1e-30))
            ),
            "surface_range": float(self.surface_range),
        }


def measure_fidelity(
    plane: Plane,
    trajectory: Trajectory,
    probe: PlaneProbe,
    loss_reference: np.ndarray,
    surface_range: float,
    subsample: int = 1,
    examples: int | None = None,
    seed: int = 7,
) -> FidelityReport:
    """Compare the true loss along the path with the loss on the plane beneath it.

    ``loss_reference`` is the exact loss recorded at each snapshot (``trajectory
    .loss_train`` or ``.loss_val``).  ``examples=None`` evaluates the plane point on the
    full evaluation set, making both sides of the comparison exact and the gap free of
    sampling error.
    """
    coords = project_trajectory(plane, trajectory.params).cpu().numpy()
    resid = residual_norms(plane, trajectory.params).cpu().numpy()
    disp = (
        (trajectory.params.to(plane.center.device) - plane.center[None, :].cpu()
         if plane.center.device.type == "cpu" else None)
    )
    # ||theta_t - c|| without materialising the difference for the whole trajectory.
    disp_norms = []
    for lo in range(0, trajectory.params.shape[0], 64):
        block = trajectory.params[lo : lo + 64].to(plane.center.device).float()
        disp_norms.append((block - plane.center[None, :]).norm(dim=1).cpu())
    disp = torch.cat(disp_norms).numpy()

    idx = np.arange(0, coords.shape[0], subsample)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    on_plane = np.full(coords.shape[0], np.nan)
    for i in idx:
        if examples is None:
            r = probe.exact(coords[i], order=0)
        else:
            r = probe.probe(coords[i], examples, order=0, generator=gen)
        on_plane[i] = r.value

    gap = loss_reference - on_plane
    return FidelityReport(
        coords=coords, residual=resid, loss_true=np.asarray(loss_reference),
        loss_on_plane=on_plane, gap=gap, captured_variance=plane.captured_variance,
        surface_range=surface_range, displacement=disp,
    )


def compare_anchorings(
    reports: dict[str, FidelityReport],
) -> dict[str, dict]:
    """Side-by-side summary of several plane constructions.

    The comparison the table is for: mean-centred planes capture more trajectory
    variance *by construction*, but the question is whether that translates into a
    smaller projection gap -- a loss-space quantity the optimality theorem says nothing
    about.
    """
    return {name: rep.summary() for name, rep in reports.items()}


__all__ = ["FidelityReport", "measure_fidelity", "compare_anchorings"]
