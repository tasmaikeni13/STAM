"""End-to-end smoke test of the estimation pipeline.

Runs every reconstruction method at a small budget on the real CNN plane and checks the
invariants that must hold regardless of accuracy: the budget accounting closes, the
certificate is finite and positive, every method covers the render domain, and the
quadratic-reproduction property survives the full pipeline.  Cheap enough to run before
committing an hour to a sweep.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch

from stam import kernels
from stam.basis import Plane
from stam.data import load_data
from stam.design import allocate, asymptotic_optimum, design_points, render_grid
from stam.device import set_matmul_precision
from stam.flat import FlatParams
from stam.metrics import curvature_error, hessian_by_differencing, surface_error
from stam.models import build_task
from stam.pipeline import certified_landscape
from stam.probe import CostModel, PlaneProbe, calibrate_cost
from stam.reconstruct import AnchorSet, reconstruct, support_radii

FAIL: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def test_allocator() -> None:
    print("allocator")
    cost = CostModel(kappa={0: 1.0, 1: 4.0, 2: 21.0}, tau={0: 70.0, 1: 82.0, 2: 550.0},
                     seconds_per_example=8e-6, device="cpu")
    ns = []
    # Constants in the range the pilot actually measures: sigma ~ O(1) per-example loss
    # spread, M3 small because the restricted loss is smooth at the scale of the domain.
    for C in (1e5, 1e6, 1e7, 1e8):
        a = allocate(C, cost, sigma=1.0, m3=1e-4, radius=40.0, order=2)
        ns.append(a.n_anchors)
        check(f"C={C:.0e} feasible", a.used <= C * 1.001 and a.examples_per_anchor >= 1,
              f"n={a.n_anchors} B={a.examples_per_anchor} used={a.used:.3g}")
    check("n* increases with budget", all(b >= a for a, b in zip(ns, ns[1:])), str(ns))
    # n* should grow like C^{1/4}: a 1000x budget gives about 5.6x the anchors.
    growth = ns[-1] / max(ns[0], 1)
    check("n* growth follows C^1/4 (1000x budget -> ~5.6x)", 3.0 <= growth <= 10.0,
          f"{growth:.2f}x over 1000x budget")
    asy = asymptotic_optimum(1e7, 21.0, 1.0, 1e-4, 40.0)
    check("closed form agrees with the scan within 2x",
          0.5 <= asy["n_star"] / ns[2] <= 2.0,
          f"closed form {asy['n_star']:.1f} vs scan {ns[2]}")


def test_reconstruction_exactness() -> None:
    print("reconstruction")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    domain = ((-2.0, 2.0), (-2.0, 2.0))
    pts = design_points(48, domain, kind="halton", seed=0)
    c = np.array([0.4, -1.1, 0.7, 0.9, -0.5, 0.3])

    def q(p):
        x, y = p[:, 0], p[:, 1]
        return c[0] + c[1] * x + c[2] * y + c[3] * x * x + c[4] * x * y + c[5] * y * y

    def dq(p):
        x, y = p[:, 0], p[:, 1]
        return np.stack([c[1] + 2 * c[3] * x + c[4] * y,
                         c[2] + c[4] * x + 2 * c[5] * y], -1)

    n = pts.shape[0]
    anchors = AnchorSet(
        coords=pts, values=q(pts), value_var=np.zeros(n), grads=dq(pts),
        grad_var=np.zeros((n, 2)),
        hess=np.tile([2 * c[3], c[4], 2 * c[5]], (n, 1)), hess_var=np.zeros((n, 3)),
        order=2, total_cost=0.0, n_examples=0,
    )
    qq, shape = render_grid(domain, 41)
    s = reconstruct("pu-taylor-2", anchors, qq, device=dev,
                    radii=support_radii(pts, overlap=1.6))
    ok = np.isfinite(s.values)
    check("covers the domain", ok.mean() > 0.999, f"{ok.mean():.4f}")
    check("reproduces a quadratic through the pipeline",
          np.abs(s.values[ok] - q(qq)[ok]).max() < 1e-3,
          f"{np.abs(s.values[ok] - q(qq)[ok]).max():.2e}")

    # Curvature recovered by differencing the rendered surface must match the truth.
    H = hessian_by_differencing(
        lambda p: reconstruct("pu-taylor-2", anchors, p, device=dev,
                              radii=support_radii(pts, overlap=1.6)).grads,
        qq[ok][:400], delta=0.02,
    )
    ref = np.tile([2 * c[3], c[4], 2 * c[5]], (H.shape[0], 1))
    ce = curvature_error(H, ref)
    check("curvature by differencing the render", ce["rmse_relative"] < 5e-2,
          f"rel {ce['rmse_relative']:.2e}")

    for method in ("rbf", "lstsq2", "bilinear", "dense"):
        gs = (7, 7) if method in ("bilinear", "dense") else None
        a2 = anchors
        if method in ("bilinear", "dense"):
            gp = design_points(49, domain, kind="grid", seed=0)
            a2 = AnchorSet(coords=gp, values=q(gp), value_var=np.zeros(49),
                           grads=dq(gp), grad_var=np.zeros((49, 2)),
                           hess=np.zeros((49, 3)), hess_var=np.zeros((49, 3)),
                           order=0, total_cost=0.0, n_examples=0)
        s2 = reconstruct(method, a2, qq, device=dev, grid_shape=gs)
        e = surface_error(s2.values, q(qq))
        check(f"{method} runs and is finite", np.isfinite(e.rmse), f"rmse {e.rmse:.3g}")


def test_full_pipeline() -> None:
    print("full pipeline on the real plane")
    run = pathlib.Path("runs/cnn")
    if not (run / "plane_centered.pt").exists():
        check("plane available", False, "run stage 1+2 first")
        return
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_matmul_precision()
    kernels.build_extension()
    data = load_data("cnn", dev)
    task = build_task("cnn", dev)
    flat = FlatParams(task.model)
    b = torch.load(run / "plane_centered.pt", map_location="cpu", weights_only=True)
    plane = Plane(center=b["center"].to(dev), basis=b["basis"].to(dev),
                  singular_values=b["singular_values"], anchoring=b["anchoring"],
                  captured_variance=float(b["captured_variance"]),
                  orthonormality_error=float(b["orthonormality_error"]),
                  build_seconds=0.0, gram_dtype=str(b["gram_dtype"]))

    probe = PlaneProbe(task, flat, plane, data.eval_train, micro_batch=1024)
    cost = calibrate_cost(probe, repeats=2, warmup=1)
    probe.cost_model = cost
    print(f"    kappa={ {k: round(v, 2) for k, v in cost.kappa.items()} } "
          f"tau={ {k: round(v, 1) for k, v in cost.tau.items()} }")
    check("kappa is ordered and > 1", 1.0 == cost.kappa[0] < cost.kappa[1] < cost.kappa[2])

    # Symmetry of the restricted Hessian: the two routes to H_12 must agree.
    r = probe.probe(np.zeros(2), 512, order=2)
    check("restricted Hessian symmetric",
          abs(r.hess[0, 1] - r.hess[1, 0]) < 1e-6 * max(abs(r.hess).max(), 1e-12))
    check("variance estimates positive and finite",
          np.isfinite(r.value_var) and r.value_var > 0
          and np.all(np.isfinite(r.grad_var)) and np.all(r.grad_var >= 0),
          f"var(l)={r.value_var:.3g} var(g)={r.grad_var}")

    # An exact evaluation must lie within a few sigma of a sampled one.
    ex = probe.exact(np.zeros(2), order=0)
    z = abs(ex.value - r.value) / max(np.sqrt(r.value_var), 1e-12)
    check("sampled probe agrees with exact within 4 sigma", z < 4.0, f"z={z:.2f}")

    domain = ((-40.0, 40.0), (-40.0, 40.0))
    qq, shape = render_grid(domain, 33)
    budget = 40 * (cost.tau[2] + cost.kappa[2] * 256)
    for method in ("pu-taylor-2", "pu-taylor-1", "lstsq2", "rbf"):
        res = certified_landscape(probe, domain, budget, cost, qq, method=method, seed=3)
        spent = sum(res.spent.values())
        check(f"{method}: budget accounting closes", spent <= budget * 1.15,
              f"spent {spent:,.0f} of {budget:,.0f}")
        check(f"{method}: certificate finite and positive",
              res.certificate is not None and np.isfinite(res.certificate.rmse)
              and res.certificate.rmse >= 0
              and res.certificate.rmse_upper >= res.certificate.rmse,
              f"rmse {res.certificate.rmse:.4g} <= {res.certificate.rmse_upper:.4g}")
        cov = np.mean(np.isfinite(res.surface.values))
        check(f"{method}: covers the render grid", cov > 0.995, f"{cov:.4f}")

    # Grid baselines take a fixed anchor count.
    res = certified_landscape(probe, domain, budget, cost, qq, method="bilinear",
                              order=0, design="grid", fixed_anchors=49,
                              grid_shape=(7, 7), seed=3)
    check("bilinear baseline runs", np.isfinite(res.certificate.rmse),
          f"rmse {res.certificate.rmse:.4g}")


def main() -> int:
    test_allocator()
    test_reconstruction_exactness()
    test_full_pipeline()
    print()
    if FAIL:
        print(f"{len(FAIL)} FAILURES: {FAIL}")
        return 1
    print("pipeline smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
