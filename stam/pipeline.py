r"""The method, end to end.

Given a plane, an evaluation set, and a compute budget :math:`C`, produce a rendered
surface together with a measured statement about its error.  Four steps:

1. **Pilot.**  Spend a small fraction of the budget on a handful of second-order probes.
   These give the constants the allocation needs -- the per-example loss standard
   deviation :math:`\sigma`, a curvature scale :math:`M_2`, and a third-derivative scale
   :math:`M_3` -- from the problem at hand rather than from an assumption.  The pilot is
   charged to the budget and its measurements are *not* reused as anchors: every method
   in the comparison pays the same 4%, including the value-only baselines that do not
   benefit from second-order pilot data, so the comparison stays like-for-like.
2. **Allocate.**  Solve for the anchor count and per-anchor sample size that minimise the
   predicted error under the remaining budget (:mod:`stam.design`).
3. **Probe and reconstruct.**  Measure at the allocated design and blend the Taylor
   patches (:mod:`stam.reconstruct`).
4. **Certify.**  Spend a final slice on independent hold-out probes and report the
   variance-corrected error of the surface actually produced (:mod:`stam.certify`).

Everything the pipeline consumes is metered in the same unit, and the returned budget
accounting adds up: pilot + anchors + certification equals the budget spent.  No step
consults the ground-truth reference; that is used only to score the result afterwards.
"""

from __future__ import annotations

import dataclasses
import math
import time
from typing import Callable

import numpy as np
import torch

from .certify import Certificate, certify_surface, estimate_m3, estimate_sigma
from .design import Allocation, allocate, design_points, predicted_error
from .probe import CostModel, PlaneProbe, ProbeResult
from .reconstruct import AnchorSet, Surface, reconstruct, support_radii


@dataclasses.dataclass
class LandscapeResult:
    surface: Surface
    anchors: AnchorSet
    allocation: Allocation
    certificate: Certificate | None
    sigma: dict
    m3: dict
    m2: float
    budget: float
    spent: dict[str, float]
    seconds: dict[str, float]
    method: str
    radii: np.ndarray
    domain: tuple[tuple[float, float], tuple[float, float]]

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "budget": self.budget,
            "spent": self.spent,
            "spent_total": sum(self.spent.values()),
            "seconds": self.seconds,
            "allocation": self.allocation.to_dict(),
            "sigma": self.sigma,
            "m3": self.m3,
            "m2": self.m2,
            "certificate": self.certificate.to_dict() if self.certificate else None,
            "n_anchors": int(self.anchors.coords.shape[0]),
            "examples_per_anchor": self.allocation.examples_per_anchor,
            "mean_radius": float(np.mean(self.radii)),
            "coverage": float(np.mean(np.isfinite(self.surface.values))),
        }


def _probe_design(
    probe: PlaneProbe,
    points: np.ndarray,
    n_examples: int,
    order: int,
    generator: torch.Generator,
) -> list[ProbeResult]:
    return [probe.probe(p, n_examples, order=order, generator=generator) for p in points]


def certified_landscape(
    probe: PlaneProbe,
    domain: tuple[tuple[float, float], tuple[float, float]],
    budget: float,
    cost: CostModel,
    queries: np.ndarray,
    method: str = "pu-taylor",
    order: int | None = None,
    pilot_fraction: float = 0.04,
    certify_fraction: float = 0.06,
    pilot_points: int = 5,
    design: str = "halton",
    seed: int = 0,
    overlap: float = 1.6,
    n_min: int = 4,
    fixed_anchors: int | None = None,
    grid_shape: tuple[int, int] | None = None,
) -> LandscapeResult:
    """Run the full pipeline at a given budget and return the rendered surface.

    ``fixed_anchors`` bypasses the allocator, which is how the fixed-resolution grid
    baselines are run: their anchor count is chosen in advance and the budget only
    changes how many examples each point sees.
    """
    if order is None:
        order = {"pu-taylor": 2, "pu-taylor-2": 2, "pu-taylor-1": 1, "pu-taylor-0": 0}.get(
            method, 0
        )
    gen = torch.Generator(device="cpu").manual_seed(seed)
    rng_radius = 0.5 * (domain[0][1] - domain[0][0])
    spent: dict[str, float] = {}
    seconds: dict[str, float] = {}

    # ---- 1. pilot -----------------------------------------------------------
    t0 = time.perf_counter()
    pilot_budget = budget * pilot_fraction
    pilot_pts = design_points(pilot_points, domain, kind="halton", seed=seed + 101, inset=0.25)
    per_pilot = max(
        16, int((pilot_budget / max(len(pilot_pts), 1) - cost.tau[2]) / cost.kappa[2])
    )
    per_pilot = min(per_pilot, len(probe.eval_set))
    pilot = _probe_design(probe, pilot_pts, per_pilot, 2, gen)
    pilot_set = AnchorSet.from_probes(pilot)
    sigma_est = estimate_sigma(pilot_set)
    m3_est = estimate_m3(pilot_set)
    m2 = float(np.nanmax(np.abs(pilot_set.hess))) if pilot_set.hess.size else 1.0
    if not np.isfinite(m2) or m2 <= 0:
        m2 = 1.0
    spent["pilot"] = pilot_set.total_cost
    seconds["pilot"] = time.perf_counter() - t0

    sigma = sigma_est["sigma"]
    m3 = m3_est.get("m3_denoised") or m3_est["m3"]
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    if not np.isfinite(m3) or m3 <= 0:
        m3 = 1.0

    # ---- 2. allocate --------------------------------------------------------
    main_budget = budget * (1 - pilot_fraction - certify_fraction)
    if fixed_anchors is not None:
        B = max(
            1,
            min(
                int((main_budget / fixed_anchors - cost.tau[order]) / cost.kappa[order]),
                len(probe.eval_set),
            ),
        )
        total, bias, noise = predicted_error(fixed_anchors, B, order, sigma, m3,
                                             rng_radius, m2=m2)
        alloc = Allocation(
            n_anchors=fixed_anchors, examples_per_anchor=B, order=order, budget=main_budget,
            used=fixed_anchors * (cost.tau[order] + cost.kappa[order] * B),
            predicted_error=total, predicted_bias=bias, predicted_noise=noise,
            spacing=rng_radius / math.sqrt(fixed_anchors),
            detail={"fixed_anchors": True, "sigma": sigma, "m3": m3},
        )
    else:
        alloc = allocate(
            main_budget, cost, sigma, m3, rng_radius, order=order, n_min=n_min,
            examples_available=len(probe.eval_set), m2=m2,
        )

    # ---- 3. probe and reconstruct ------------------------------------------
    t1 = time.perf_counter()
    kind = "grid" if design == "grid" else design
    pts = design_points(alloc.n_anchors, domain, kind=kind, seed=seed, inset=0.0)
    probes = _probe_design(probe, pts, alloc.examples_per_anchor, order, gen)
    anchors = AnchorSet.from_probes(probes)
    spent["anchors"] = anchors.total_cost
    seconds["probe"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    device = probe.plane.basis.device
    kw: dict = {}
    is_pu = method.startswith("pu-taylor")
    radii = support_radii(anchors.coords, overlap=overlap) if is_pu else np.zeros(1)
    if is_pu:
        kw = {"radii": radii, "overlap": overlap}
    surf = reconstruct(method, anchors, queries, device=device, grid_shape=grid_shape, **kw)
    seconds["reconstruct"] = time.perf_counter() - t2

    # ---- 4. certify ---------------------------------------------------------
    cert = None
    if certify_fraction > 0:
        t3 = time.perf_counter()
        cert_budget = budget * certify_fraction
        # A near-square count so the stratification has equal cells.
        # The quantile statement's precision is set by the probe count, so the slice is
        # spent on many shallow probes rather than few deep ones -- but the count must be
        # feasible: with a per-anchor overhead tau, each probe costs at least
        # tau + kappa*per_min however shallow it is, and asking for more probes than the
        # slice can pay for is how a budget silently overruns.
        per_min = 8
        affordable = cert_budget / max(cost.tau[0] + cost.kappa[0] * per_min, 1e-9)
        side = int(math.sqrt(max(affordable, 1.0)))
        side = max(4, min(side, 18))
        n_cert = side * side
        per_cert = max(
            per_min,
            min(
                int((cert_budget / n_cert - cost.tau[0]) / max(cost.kappa[0], 1e-9)),
                len(probe.eval_set),
            ),
        )

        def rebuild(q: np.ndarray) -> Surface:
            if method.startswith("pu-taylor"):
                return reconstruct(method, anchors, q, device=device, radii=radii,
                                   overlap=overlap)
            return reconstruct(method, anchors, q, device=device, grid_shape=grid_shape)

        cert = certify_surface(
            probe, rebuild, domain, n_probes=n_cert, examples_per_probe=per_cert,
            seed=seed + 991,
        )
        spent["certify"] = cert.probe_cost
        seconds["certify"] = time.perf_counter() - t3

    return LandscapeResult(
        surface=surf, anchors=anchors, allocation=alloc, certificate=cert,
        sigma=sigma_est, m3=m3_est, m2=m2, budget=budget, spent=spent, seconds=seconds,
        method=method, radii=radii, domain=domain,
    )


def surface_evaluator(
    result: LandscapeResult, method: str | None = None
) -> Callable[[np.ndarray], Surface]:
    """A closure that re-evaluates the *fitted* reconstruction at new points."""
    method = method or result.method
    device = None
    kw: dict = {}
    if method.startswith("pu-taylor"):
        kw = {"radii": result.radii}

    def f(q: np.ndarray) -> Surface:
        return reconstruct(method, result.anchors, q, device=device, **kw)

    return f


__all__ = ["LandscapeResult", "certified_landscape", "surface_evaluator"]
