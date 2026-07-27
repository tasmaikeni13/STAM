"""Stage 3: the budget-error study.

The central experiment.  Every method is given the *same* compute budget, measured in
example-forward-equivalents, and scored against the exact reference surface from stage 2.
Sweeping the budget over two and a half decades gives the convergence exponent, which is
what the theory predicts and what distinguishes a method that converges from one that
plateaus.

Six methods, spanning what is actually done and what the analysis proposes:

``grid-interp``
    A fixed-resolution grid of noisy values, drawn by bilinear interpolation.  This is
    standard practice.  Growing the budget only buys more examples per grid point.
``grid-refine``
    A grid whose resolution grows with the budget at fixed per-point sample size -- the
    other way practitioners spend a larger budget.
``rbf``
    Multiquadric radial-basis interpolation of values on an allocated design.
``lstsq2``
    Locally weighted quadratic regression on values.  The strongest value-only method,
    and the one classical theory says is rate-optimal for a :math:`C^3` function.
``pu-taylor-1``
    Partition-of-unity blending of first-order Taylor patches: value and exact gradient.
``pu-taylor-2``
    Second-order patches: value, exact gradient, exact restricted Hessian.

Three quantities are scored for each: the surface, the gradient field, and the curvature
field.  Curvature is always obtained by differencing the *rendered* surface, so no method
gets credit for derivative information it did not put into the picture.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import time

import numpy as np
import torch

from stam import kernels
from stam.basis import Plane
from stam.data import load_data
from stam.device import describe_environment, human_time, set_matmul_precision
from stam.flat import FlatParams
from stam.metrics import (curvature_error, hessian_by_differencing, surface_error,
                          vector_field_error)
from stam.models import build_task
from stam.parallel import gpu_devices, memory_report, run_jobs
from stam.pipeline import certified_landscape
from stam.probe import CostModel, PlaneProbe, micro_table

METHODS = {
    "grid-interp":  {"method": "bilinear",    "order": 0, "design": "grid",   "fixed_res": 21},
    "grid-refine":  {"method": "bilinear",    "order": 0, "design": "grid",   "fixed_B": 128},
    "rbf":          {"method": "rbf",         "order": 0, "design": "halton"},
    "lstsq2":       {"method": "lstsq2",      "order": 0, "design": "halton"},
    "pu-taylor-1":  {"method": "pu-taylor-1", "order": 1, "design": "halton"},
    "pu-taylor-2":  {"method": "pu-taylor-2", "order": 2, "design": "halton"},
}


def load_plane(path: str, device: torch.device) -> Plane:
    blob = torch.load(path, map_location="cpu", weights_only=True)
    return Plane(
        center=blob["center"].to(device), basis=blob["basis"].to(device),
        singular_values=blob["singular_values"], anchoring=blob["anchoring"],
        captured_variance=float(blob["captured_variance"]),
        orthonormality_error=float(blob["orthonormality_error"]),
        build_seconds=0.0, gram_dtype=str(blob["gram_dtype"]),
    )


def setup(spec: dict, device: torch.device):
    torch.cuda.set_device(device)
    set_matmul_precision()
    kernels.build_extension()
    data = load_data(spec["task"], device)
    task = build_task(spec["task"], device)
    flat = FlatParams(task.model)
    plane = load_plane(spec["plane_path"], device)
    eval_set = data.eval_train if spec["split"] == "train" else data.eval_val
    cm = spec["cost_model"]
    cost = CostModel(
        kappa={int(k): float(v) for k, v in cm["kappa"].items()},
        tau={int(k): float(v) for k, v in cm["tau_example_equivalents"].items()},
        seconds_per_example=float(cm["seconds_per_example"]), device=str(device),
    )
    table = micro_table(spec["micro_batch"])
    probe = PlaneProbe(task, flat, plane, eval_set, micro_batch=table[0],
                       cost_model=cost, micro_batch_by_order=table)
    ref = np.load(spec["reference_path"])
    return {"probe": probe, "cost": cost, "ref": ref, "device": device, "spec": spec}


def work(ctx, job: dict) -> dict:
    spec = ctx["spec"]
    probe: PlaneProbe = ctx["probe"]
    cost: CostModel = ctx["cost"]
    ref = ctx["ref"]
    split = spec["split"]
    domain = tuple(map(tuple, spec["domain"]))

    cfg = METHODS[job["name"]]
    # Score only where the surface is worth drawing (stage 2b): the reference is exact
    # everywhere, but a box scaled to the whole trajectory reaches into a region where
    # the loss diverges, and an error relative to that range would be meaningless.
    from stam.design import inside

    vmask = inside(ref["value_points"], domain)
    dmask = inside(ref["deriv_points"], domain)
    value_pts = ref["value_points"][vmask]
    deriv_pts = ref["deriv_points"][dmask]
    ref_value = ref[f"value_{split}"].ravel()[vmask]
    ref_grad = ref[f"grad_{split}"][dmask]
    ref_hess = ref[f"hess_{split}"][dmask]

    budget = float(job["budget"])
    kwargs: dict = {
        "method": cfg["method"], "order": cfg["order"], "design": cfg["design"],
        "seed": int(job["seed"]),
    }

    if "fixed_res" in cfg:
        res = int(cfg["fixed_res"])
        kwargs["fixed_anchors"] = res * res
        kwargs["grid_shape"] = (res, res)
    elif "fixed_B" in cfg:
        # Resolution grows with the budget at fixed sample size per point.
        per_point = cost.tau[0] + cost.kappa[0] * cfg["fixed_B"]
        usable = budget * (1 - 0.04 - 0.06)
        res = max(3, int(math.floor(math.sqrt(usable / per_point))))
        kwargs["fixed_anchors"] = res * res
        kwargs["grid_shape"] = (res, res)

    t0 = time.perf_counter()
    res_obj = certified_landscape(
        probe, domain, budget, cost, value_pts, **kwargs
    )
    # Re-evaluate the fitted surface on the derivative grid for the field metrics.
    from stam.reconstruct import reconstruct

    rk: dict = {}
    if cfg["method"].startswith("pu-taylor"):
        rk = {"radii": res_obj.radii}
    gshape = kwargs.get("grid_shape")

    def at(q: np.ndarray):
        return reconstruct(cfg["method"], res_obj.anchors, q,
                           device=ctx["device"], grid_shape=gshape, **rk)

    dsurf = at(deriv_pts)
    delta = max(float(res_obj.allocation.spacing) / 4.0, 1e-4)
    pred_hess = hessian_by_differencing(lambda q: at(q).grads, deriv_pts, delta)

    se = surface_error(res_obj.surface.values, ref_value)
    ge = vector_field_error(dsurf.grads, ref_grad)
    ce = curvature_error(pred_hess, ref_hess)

    out = {
        "name": job["name"], "budget": budget, "seed": int(job["seed"]), "split": split,
        "surface": se.to_dict(), "gradient": ge, "curvature": ce,
        "pipeline": res_obj.to_dict(),
        "wall_seconds": time.perf_counter() - t0,
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["cnn", "gpt"], required=True)
    ap.add_argument("--run", default=None)
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--budgets", type=int, default=8)
    ap.add_argument("--min-budget", type=float, default=None)
    ap.add_argument("--max-budget", type=float, default=None)
    ap.add_argument("--methods", default=",".join(METHODS))
    args = ap.parse_args()

    run = pathlib.Path(args.run or f"runs/{args.task}")
    stage2 = json.loads((run / "stage2_report.json").read_text())
    devices = gpu_devices()

    cm = stage2["cost_model"]
    kappa = {int(k): float(v) for k, v in cm["kappa"].items()}
    tau = {int(k): float(v) for k, v in cm["tau_example_equivalents"].items()}

    # Budget floor: enough for the smallest sensible order-2 design.  Ceiling: the point
    # at which every method could evaluate the full evaluation set at every anchor, past
    # which the comparison stops being about sampling.
    n_eval = int(stage2["reference_exact_examples"])
    lo = args.min_budget or 8 * (tau[2] + kappa[2] * 64)
    hi = args.max_budget or 64 * (tau[2] + kappa[2] * n_eval) / 4
    budgets = np.geomspace(lo, hi, args.budgets)

    names = [n for n in args.methods.split(",") if n in METHODS]
    jobs = [
        {"name": n, "budget": float(b), "seed": s}
        for n in names for b in budgets for s in range(args.seeds)
    ]

    domain = stage2.get("analysis_domain", stage2["domain"])
    spec = {
        "task": args.task,
        "split": args.split,
        "plane_path": str(run / f"plane_{stage2['anchoring']}.pt"),
        "reference_path": str(run / "reference.npz"),
        "domain": domain,
        "cost_model": cm,
        "micro_batch": stage2["micro_batch"][args.split],
    }

    print(f"=== budget sweep: {args.task}/{args.split} ===")
    print(f"  methods: {names}")
    print(f"  budgets: {budgets[0]:.3g} .. {budgets[-1]:.3g} example-equivalents "
          f"({args.budgets} points, {args.seeds} seeds)")
    print(f"  {len(jobs)} jobs across devices {devices}")
    est = float(np.sum([b for _ in names for b in budgets for _ in range(args.seeds)])) \
        * float(cm["seconds_per_example"]) / max(len(devices), 1)
    print(f"  estimated probe time {human_time(est)}")

    t0 = time.perf_counter()

    def progress(i: int, n: int, r: dict) -> None:
        if i % max(1, n // 25) == 0 or i == n:
            el = time.perf_counter() - t0
            print(f"    {i}/{n}  {human_time(el)} elapsed, "
                  f"{human_time(el / i * (n - i))} remaining", flush=True)

    results = run_jobs(jobs, setup, work, spec, devices=devices, progress=progress)

    out = run / f"sweep_{args.split}.json"
    out.write_text(json.dumps({
        "task": args.task, "split": args.split, "budgets": budgets.tolist(),
        "methods": names, "seeds": args.seeds, "results": results,
        "cost_model": cm, "domain": domain,
        "surface_range": stage2.get("analysis_surface_range",
                                    stage2["surface_range"])[args.split],
        "environment": describe_environment(),
        "memory": memory_report(devices),
        "seconds": time.perf_counter() - t0,
    }, indent=2, default=str))
    print(f"\n  {len(results)} results in {human_time(time.perf_counter() - t0)}")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
