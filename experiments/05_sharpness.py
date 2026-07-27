r"""Stage 5: how much of the sharpness in a landscape plot is Monte-Carlo noise.

"Sharpness" -- the curvature of the loss surface at a minimum -- is the quantity most
often *read off* these pictures and used to draw conclusions.  It is also the quantity a
noisy grid estimates worst, because reading curvature from values means second
differencing, and second differencing amplifies noise by :math:`h^{-2}`.

For a grid of spacing :math:`h` whose values carry independent noise of standard
deviation :math:`\sigma/\sqrt B`, the second-difference estimate of a curvature has a
noise component of size

.. math::
   \lambda_{\text{noise}} \;\sim\; \frac{c\,\sigma}{h^2\sqrt B},

independent of the surface.  Since the estimator is a squared quantity in expectation,
the *apparent* sharpness behaves like
:math:`\sqrt{\lambda_{\text{true}}^2 + \lambda_{\text{noise}}^2}`: it never falls below
the noise floor, and at the sample sizes typical of a landscape figure the floor can
exceed the signal outright.

This experiment measures it.  A grid of fixed resolution is evaluated at a range of
per-point sample sizes; at each, the curvature is read off the grid exactly as a reader
would, and compared with the exact curvature from the reference.  The predicted floor is
overlaid -- with :math:`\sigma` measured, not fitted.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import torch

from stam import kernels
from stam.basis import Plane
from stam.data import load_data
from stam.design import render_grid
from stam.device import describe_environment, human_time, set_matmul_precision
from stam.flat import FlatParams
from stam.metrics import sharpness
from stam.models import build_task
from stam.parallel import gpu_devices, run_jobs
from stam.probe import CostModel, PlaneProbe, micro_table


def grid_hessian(Z: np.ndarray, h: float) -> np.ndarray:
    r"""Curvature read off a regular grid by second differences.

    The interior stencils are the standard five- and four-point forms; this is what a
    reader estimating curvature from a plotted grid is doing, whether or not they say so.
    Returns ``(m, 3)`` packed ``(xx, xy, yy)`` over interior points.
    """
    ny, nx = Z.shape
    zxx = (Z[1:-1, 2:] - 2 * Z[1:-1, 1:-1] + Z[1:-1, :-2]) / h**2
    zyy = (Z[2:, 1:-1] - 2 * Z[1:-1, 1:-1] + Z[:-2, 1:-1]) / h**2
    zxy = (Z[2:, 2:] - Z[2:, :-2] - Z[:-2, 2:] + Z[:-2, :-2]) / (4 * h**2)
    return np.stack([zxx.ravel(), zxy.ravel(), zyy.ravel()], axis=1)


def stencil_constant(resolution: int, trials: int = 400, seed: int = 0) -> float:
    r"""RMS largest eigenvalue produced by the estimator from *pure* unit-variance noise.

    Relating the node-noise variance to the RMS of :math:`\lambda_{\max}` analytically
    means tracking the correlations between overlapping second differences and then the
    distribution of an eigenvalue of a correlated symmetric matrix.  Applying the exact
    estimator to synthetic noise gives the same constant with no algebra and no
    approximation, and it is a constant -- independent of the surface, of :math:`h`, and
    of :math:`B`.  Scaling: the estimate is linear in the node values and
    :math:`\propto h^{-2}`, so
    :math:`\lambda_{\mathrm{noise}} = k \sqrt{v}/h^2` with this :math:`k`.
    """
    rng = np.random.default_rng(seed)
    acc = []
    for _ in range(trials):
        Z = rng.standard_normal((resolution, resolution))
        acc.append(sharpness(grid_hessian(Z, 1.0)) ** 2)
    return float(np.sqrt(np.mean(np.concatenate(acc))))


def setup(spec: dict, device: torch.device):
    torch.cuda.set_device(device)
    set_matmul_precision()
    kernels.build_extension()
    data = load_data(spec["task"], device)
    task = build_task(spec["task"], device)
    flat = FlatParams(task.model)
    b = torch.load(spec["plane_path"], map_location="cpu", weights_only=True)
    plane = Plane(center=b["center"].to(device), basis=b["basis"].to(device),
                  singular_values=b["singular_values"], anchoring=b["anchoring"],
                  captured_variance=float(b["captured_variance"]),
                  orthonormality_error=float(b["orthonormality_error"]),
                  build_seconds=0.0, gram_dtype=str(b["gram_dtype"]))
    cm = spec["cost_model"]
    cost = CostModel(
        kappa={int(k): float(v) for k, v in cm["kappa"].items()},
        tau={int(k): float(v) for k, v in cm["tau_example_equivalents"].items()},
        seconds_per_example=float(cm["seconds_per_example"]), device=str(device),
    )
    eval_set = data.eval_train if spec["split"] == "train" else data.eval_val
    table = micro_table(spec["micro_batch"])
    probe = PlaneProbe(task, flat, plane, eval_set, micro_batch=table[0],
                       cost_model=cost, micro_batch_by_order=table)
    return {"probe": probe, "spec": spec}


def work(ctx, job: dict) -> dict:
    probe: PlaneProbe = ctx["probe"]
    spec = ctx["spec"]
    domain = tuple(map(tuple, spec["domain"]))
    res = int(spec["resolution"])
    pts, shape = render_grid(domain, res)
    h = (domain[0][1] - domain[0][0]) / (res - 1)

    gen = torch.Generator(device="cpu").manual_seed(int(job["seed"]))
    B = int(job["examples"])
    vals = np.empty(pts.shape[0])
    var = np.empty(pts.shape[0])
    t0 = time.perf_counter()
    for i, p in enumerate(pts):
        r = probe.probe(p, B, order=0, generator=gen)
        vals[i] = r.value
        var[i] = r.value_var
    Z = vals.reshape(shape)
    H = grid_hessian(Z, h)
    lam = sharpness(H)
    return {
        "examples": B, "seed": int(job["seed"]),
        "apparent_sharpness_rms": float(np.sqrt(np.mean(lam**2))),
        "apparent_sharpness_p90": float(np.percentile(np.abs(lam), 90)),
        "value_var_mean": float(np.mean(var)),
        "spacing": h, "seconds": time.perf_counter() - t0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["cnn", "gpt"], required=True)
    ap.add_argument("--run", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--resolution", type=int, default=21)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--points", type=int, default=7)
    args = ap.parse_args()

    run = pathlib.Path(args.run or f"runs/{args.task}")
    stage2 = json.loads((run / "stage2_report.json").read_text())
    ref = np.load(run / "reference.npz")
    devices = gpu_devices()

    n_eval = int(stage2["reference_exact_examples"])
    sizes = np.unique(np.geomspace(8, n_eval, args.points).astype(int))
    jobs = [{"examples": int(b), "seed": s} for b in sizes for s in range(args.seeds)]

    spec = {
        "task": args.task, "split": args.split,
        "plane_path": str(run / f"plane_{stage2['anchoring']}.pt"),
        "domain": stage2.get("analysis_domain", stage2["domain"]),
        "cost_model": stage2["cost_model"],
        "micro_batch": stage2["micro_batch"][args.split],
        "resolution": args.resolution,
    }

    # Exact curvature from the reference derivative field, restricted to the same domain.
    from stam.design import inside

    dmask = inside(ref["deriv_points"], tuple(map(tuple, spec["domain"])))
    ref_hess = ref[f"hess_{args.split}"][dmask]
    ok = np.isfinite(ref_hess).all(1)
    true_rms = float(np.sqrt(np.mean(sharpness(ref_hess[ok]) ** 2)))
    true_p90 = float(np.percentile(np.abs(sharpness(ref_hess[ok])), 90))

    print(f"=== sharpness vs sample size: {args.task}/{args.split} ===")
    print(f"  grid {args.resolution}x{args.resolution}, sample sizes "
          f"{sizes[0]}..{sizes[-1]}, {args.seeds} seeds")
    print(f"  exact sharpness (from second-order probes): RMS {true_rms:.4g}, "
          f"p90 {true_p90:.4g}")

    t0 = time.perf_counter()
    results = run_jobs(jobs, setup, work, spec, devices=devices)
    out = run / f"sharpness_{args.split}.json"
    out.write_text(json.dumps({
        "task": args.task, "split": args.split, "resolution": args.resolution,
        "sizes": sizes.tolist(), "seeds": args.seeds, "results": results,
        "true_sharpness_rms": true_rms, "true_sharpness_p90": true_p90,
        "stencil_constant": stencil_constant(args.resolution),
        "domain": spec["domain"], "environment": describe_environment(),
        "seconds": time.perf_counter() - t0,
    }, indent=2, default=str))

    for b in sizes:
        rs = [r for r in results if r["examples"] == b]
        if rs:
            m = float(np.mean([r["apparent_sharpness_rms"] for r in rs]))
            print(f"    B={b:>6}  apparent RMS sharpness {m:10.4g}  "
                  f"({m / max(true_rms, 1e-30):6.1f}x the true value)")
    print(f"\n  {human_time(time.perf_counter() - t0)}, wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
