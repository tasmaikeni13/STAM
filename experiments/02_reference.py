"""Stage 2: build the visualisation planes and compute exact reference surfaces.

Every accuracy claim later in this work is measured against a surface computed here, so
this stage does no estimation at all: at each grid point the loss is evaluated over the
*entire* evaluation set, making the reference exact by construction (:mod:`stam.data`
defines the target as a fixed finite average precisely so that this is possible).

Three things are produced per subject:

* the planes -- mean-centred, :math:`\\theta_0`-anchored, and endpoint-anchored -- with
  their spectra, so the affine-optimality question is settled by measurement;
* a dense grid of exact loss values on the chosen plane, for both the training and
  validation evaluation sets;
* a coarser grid of exact restricted gradients and Hessians, which is the ground truth
  for the derived-field accuracy comparison.

Grid points are sharded across all visible GPUs.
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
from stam.basis import build_plane, project_trajectory, residual_norms
from stam.capture import Trajectory
from stam.data import load_data
from stam.design import render_grid
from stam.device import describe_environment, human_time, set_matmul_precision
from stam.flat import FlatParams
from stam.models import build_task
from stam.parallel import gpu_devices, memory_report, run_jobs
from stam.probe import PlaneProbe, calibrate_cost, tune_micro_batches


def square_domain(coords: np.ndarray, margin: float = 0.30
                  ) -> tuple[tuple[float, float], tuple[float, float]]:
    """Smallest square containing the projected trajectory, expanded by ``margin``.

    A square domain keeps the geometry isotropic, which the error analysis assumes when
    it speaks of a single domain radius :math:`R` and a single spacing :math:`h`.
    """
    cx = 0.5 * (coords[:, 0].max() + coords[:, 0].min())
    cy = 0.5 * (coords[:, 1].max() + coords[:, 1].min())
    half = 0.5 * max(np.ptp(coords[:, 0]), np.ptp(coords[:, 1])) * (1 + 2 * margin)
    half = max(half, 1e-6)
    return ((cx - half, cx + half), (cy - half, cy + half))


# ---------------------------------------------------------------------------
# Worker: rebuilds the model and evaluation set on its own device
# ---------------------------------------------------------------------------


def setup(spec: dict, device: torch.device):
    torch.cuda.set_device(device)
    set_matmul_precision()
    kernels.build_extension()
    data = load_data(spec["task"], device)
    task = build_task(spec["task"], device)
    flat = FlatParams(task.model)

    blob = torch.load(spec["plane_path"], map_location="cpu", weights_only=True)
    from stam.basis import Plane

    plane = Plane(
        center=blob["center"].to(device), basis=blob["basis"].to(device),
        singular_values=blob["singular_values"], anchoring=blob["anchoring"],
        captured_variance=float(blob["captured_variance"]),
        orthonormality_error=float(blob["orthonormality_error"]),
        build_seconds=0.0, gram_dtype=str(blob["gram_dtype"]),
    )

    probes = {}
    for split in ("train", "val"):
        eval_set = data.eval_train if split == "train" else data.eval_val
        table = {int(k): int(v) for k, v in spec["micro_batch"][split].items()}
        probes[split] = PlaneProbe(task, flat, plane, eval_set,
                                   micro_batch=table.get(0, 256),
                                   micro_batch_by_order=table)
    return {"probes": probes, "device": device}


def work(ctx, job: dict) -> dict:
    probe = ctx["probes"][job["split"]]
    pts = np.asarray(job["points"], dtype=np.float64)
    order = int(job["order"])
    vals, grads, hess = [], [], []
    for p in pts:
        r = probe.exact(p, order=order)
        vals.append(r.value)
        if order >= 1:
            grads.append(r.grad)
        if order >= 2:
            hess.append(r.hess_packed())
    if order >= 2 and torch.cuda.is_available():
        # Release the double-backward working set between jobs so fragmentation does not
        # accumulate across a run of several thousand second-order probes.
        torch.cuda.empty_cache()
    out = {"index": job["index"], "split": job["split"], "order": order, "values": vals}
    if order >= 1:
        out["grads"] = [g.tolist() for g in grads]
    if order >= 2:
        out["hess"] = [h.tolist() for h in hess]
    return out


def shard(points: np.ndarray, split: str, order: int, block: int) -> list[dict]:
    jobs = []
    for lo in range(0, points.shape[0], block):
        jobs.append({
            "index": list(range(lo, min(lo + block, points.shape[0]))),
            "points": points[lo : lo + block].tolist(),
            "split": split, "order": order,
        })
    return jobs


def gather(results: list[dict], n: int, key: str, width: int | None = None) -> np.ndarray:
    out = np.full((n,) if width is None else (n, width), np.nan)
    for r in results:
        if key not in r:
            continue
        out[np.asarray(r["index"])] = np.asarray(r[key])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["cnn", "gpt"], required=True)
    ap.add_argument("--run", default=None)
    ap.add_argument("--value-resolution", type=int, default=None)
    ap.add_argument("--deriv-resolution", type=int, default=None)
    ap.add_argument("--margin", type=float, default=0.30)
    ap.add_argument("--anchoring", default="centered",
                    choices=["centered", "origin", "endpoint"])
    args = ap.parse_args()

    run = pathlib.Path(args.run or f"runs/{args.task}")
    devices = gpu_devices()
    dev0 = torch.device(f"cuda:{devices[0]}" if devices else "cpu")
    torch.cuda.set_device(dev0)
    set_matmul_precision()
    kernels.build_extension()

    print(f"=== reference surfaces for {args.task} on devices {devices} ===")
    t_start = time.perf_counter()
    traj = Trajectory.load(run)
    print(f"  trajectory: {traj.T} snapshots x {traj.N:,} parameters")

    data = load_data(args.task, dev0)
    task = build_task(args.task, dev0)
    flat = FlatParams(task.model)

    # ---- planes -------------------------------------------------------------
    planes = {}
    for anchoring in ("centered", "origin", "endpoint"):
        p = build_plane(traj.params, dim=2, anchoring=anchoring, device=dev0)
        planes[anchoring] = p
        coords = project_trajectory(p, traj.params).cpu().numpy()
        resid = residual_norms(p, traj.params).cpu().numpy()
        print(f"  plane[{anchoring:>8}] rho2={p.captured_variance:.4f} "
              f"orth_err={p.orthonormality_error:.2e} "
              f"residual mean={resid.mean():.4g} max={resid.max():.4g} "
              f"({p.build_seconds:.1f}s, gram in {p.gram_dtype})")

    plane = planes[args.anchoring]
    coords = project_trajectory(plane, traj.params).cpu().numpy()
    domain = square_domain(coords, args.margin)
    radius = 0.5 * (domain[0][1] - domain[0][0])
    print(f"  domain: x{domain[0]} y{domain[1]}  radius {radius:.4g}")

    plane_path = run / f"plane_{args.anchoring}.pt"
    torch.save(
        {
            "center": plane.center.cpu(), "basis": plane.basis.cpu(),
            "singular_values": plane.singular_values, "anchoring": plane.anchoring,
            "captured_variance": plane.captured_variance,
            "orthonormality_error": plane.orthonormality_error,
            "gram_dtype": plane.gram_dtype,
        },
        plane_path,
    )

    # ---- micro-batch tuning and cost calibration ----------------------------
    micro = {}
    for split in ("train", "val"):
        eval_set = data.eval_train if split == "train" else data.eval_val
        pr = PlaneProbe(task, flat, plane, eval_set, micro_batch=64)
        table = tune_micro_batches(pr)
        micro[split] = {str(k): int(v) for k, v in table.items()}
        print(f"  micro-batch[{split}] by order: {table}")

    probe0 = PlaneProbe(task, flat, plane, data.eval_train,
                        micro_batch=int(micro["train"]["0"]),
                        micro_batch_by_order={int(k): int(v)
                                              for k, v in micro["train"].items()})
    cost = calibrate_cost(probe0)
    print(f"  cost model: kappa={ {k: round(v, 2) for k, v in cost.kappa.items()} } "
          f"tau={ {k: round(v, 1) for k, v in cost.tau.items()} } example-equivalents, "
          f"{cost.seconds_per_example * 1e6:.2f} us/example")

    # ---- resolution: sized from measured throughput -------------------------
    ex_per_sec = 1.0 / cost.seconds_per_example
    budget_seconds = 900.0 * max(len(devices), 1)
    n_eval = len(data.eval_train)
    if args.value_resolution:
        vres = args.value_resolution
    else:
        affordable = budget_seconds * ex_per_sec / (2 * n_eval)  # two splits
        vres = int(max(33, min(97, math.floor(math.sqrt(affordable)) | 1)))
    dres = args.deriv_resolution or max(17, (vres // 2) | 1)
    est = (2 * vres**2 + dres**2 * cost.kappa[2]) * n_eval / ex_per_sec / max(len(devices), 1)
    print(f"  value grid {vres}x{vres}, derivative grid {dres}x{dres}; "
          f"estimated {human_time(est)}")

    value_pts, value_shape = render_grid(domain, vres)
    deriv_pts, deriv_shape = render_grid(domain, dres)

    spec = {"task": args.task, "plane_path": str(plane_path), "micro_batch": micro}
    block = max(1, min(64, value_pts.shape[0] // (4 * max(len(devices), 1))))

    jobs = []
    for split in ("train", "val"):
        jobs += shard(value_pts, split, 0, block)
    jobs += shard(deriv_pts, "train", 2, max(1, block // 4))
    jobs += shard(deriv_pts, "val", 2, max(1, block // 4))
    print(f"  {len(jobs)} jobs across {max(len(devices), 1)} device(s)")

    done = [0]
    t_jobs = time.perf_counter()

    def progress(i: int, n: int, _r: dict) -> None:
        done[0] = i
        if i % max(1, n // 20) == 0 or i == n:
            el = time.perf_counter() - t_jobs
            print(f"    {i}/{n} jobs  {human_time(el)} elapsed, "
                  f"{human_time(el / i * (n - i))} remaining", flush=True)

    results = run_jobs(jobs, setup, work, spec, devices=devices, progress=progress)

    ref: dict[str, np.ndarray] = {}
    for split in ("train", "val"):
        rs = [r for r in results if r["split"] == split and r["order"] == 0]
        ref[f"value_{split}"] = gather(rs, value_pts.shape[0], "values").reshape(value_shape)
        rd = [r for r in results if r["split"] == split and r["order"] == 2]
        ref[f"dvalue_{split}"] = gather(rd, deriv_pts.shape[0], "values")
        ref[f"grad_{split}"] = gather(rd, deriv_pts.shape[0], "grads", 2)
        ref[f"hess_{split}"] = gather(rd, deriv_pts.shape[0], "hess", 3)

    np.savez_compressed(
        run / "reference.npz",
        value_points=value_pts, value_shape=np.asarray(value_shape),
        deriv_points=deriv_pts, deriv_shape=np.asarray(deriv_shape),
        traj_coords=coords, traj_loss_train=traj.loss_train, traj_loss_val=traj.loss_val,
        domain=np.asarray(domain), **ref,
    )

    report = {
        "task": args.task,
        "anchoring": args.anchoring,
        "domain": [list(domain[0]), list(domain[1])],
        "radius": radius,
        "value_resolution": vres,
        "deriv_resolution": dres,
        "planes": {k: {**v.spectrum_report(),
                       "orthonormality_error": v.orthonormality_error,
                       "build_seconds": v.build_seconds,
                       "gram_dtype": v.gram_dtype,
                       "residual_mean": float(residual_norms(v, traj.params).mean()),
                       "residual_max": float(residual_norms(v, traj.params).max())}
                   for k, v in planes.items()},
        "cost_model": cost.to_dict(),
        "micro_batch": micro,
        "surface_range": {
            s: float(np.nanmax(ref[f"value_{s}"]) - np.nanmin(ref[f"value_{s}"]))
            for s in ("train", "val")
        },
        "reference_exact_examples": n_eval,
        "devices": devices,
        "memory": memory_report(devices),
        "environment": describe_environment(),
        "seconds": time.perf_counter() - t_start,
    }
    (run / "stage2_report.json").write_text(json.dumps(report, indent=2, default=str))

    print(f"\n  surface range (train): {report['surface_range']['train']:.4f}")
    print(f"  total: {human_time(report['seconds'])}")
    print(f"  wrote {run / 'reference.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
