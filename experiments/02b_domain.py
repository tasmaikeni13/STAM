r"""Stage 2b: choose the region worth drawing.

A plane through a trained network's parameter space is not uniformly interesting. Far
enough from the trajectory the weights are effectively random at a large scale and the
cross-entropy diverges: on the CNN reference the loss reaches 479 at the corners of a box
scaled to the whole trajectory, against a maximum of 4.2 anywhere the optimiser actually
went. Drawing that box is not informative -- the entire trained region is compressed into
the bottom percent of the colour scale -- and, worse for this work, an error reported
relative to a 479-wide range is meaningless.

So the render domain is chosen by a stated criterion rather than by a margin parameter:

    the largest box, centred on the trajectory, on which the loss stays within
    ``cap_multiple`` times the largest loss the optimiser itself experienced.

At ``cap_multiple = 3`` this keeps everything the optimiser saw plus a surrounding band,
and excludes only the divergent outskirts. The criterion is applied to the *exact*
reference computed in stage 2, so it costs nothing extra, and it is applied identically
for every method compared later.

The box is found by growing a centred sub-box of the reference grid until the cap is
exceeded, separately in each axis, so an anisotropic trajectory gets an anisotropic
domain rather than a square that reaches into the divergent region along its short axis.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np


def largest_box_under_cap(
    X: np.ndarray, Y: np.ndarray, Z: np.ndarray, centre: tuple[float, float], cap: float,
    seed_box: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> tuple[tuple[float, float], tuple[float, float], dict]:
    """Grow an axis-aligned sub-box of the grid outward from ``centre`` while
    ``max Z <= cap``.

    Each of the four sides is grown independently, so the box adapts to an anisotropic
    trajectory and is not forced to stop on one side because the grid boundary was
    reached on the other -- which would otherwise clip the end of the optimiser's path.
    """
    ny, nx = Z.shape
    xs, ys = X[0, :], Y[:, 0]
    ci = int(np.argmin(np.abs(xs - centre[0])))
    cj = int(np.argmin(np.abs(ys - centre[1])))

    i0 = i1 = ci
    j0 = j1 = cj
    if seed_box is not None:
        # The domain must contain the optimiser's path: a picture that crops the
        # trajectory is not a picture of that trajectory.  The cap then governs only how
        # far *beyond* the path the surface is drawn.
        (sx0, sx1), (sy0, sy1) = seed_box
        i0 = int(np.clip(np.searchsorted(xs, sx0, side="right") - 1, 0, nx - 1))
        i1 = int(np.clip(np.searchsorted(xs, sx1, side="left"), 0, nx - 1))
        j0 = int(np.clip(np.searchsorted(ys, sy0, side="right") - 1, 0, ny - 1))
        j1 = int(np.clip(np.searchsorted(ys, sy1, side="left"), 0, ny - 1))

    def ok(a0, a1, b0, b1) -> bool:
        return float(np.nanmax(Z[b0 : b1 + 1, a0 : a1 + 1])) <= cap

    seed_max = float(np.nanmax(Z[j0 : j1 + 1, i0 : i1 + 1]))
    if not ok(i0, i1, j0, j1):
        # The trajectory's own neighbourhood already exceeds the cap; keep it anyway and
        # report the fact rather than silently cropping the path.
        pass

    grew = True
    while grew:
        grew = False
        for side in ("left", "right", "down", "up"):
            trial = [i0, i1, j0, j1]
            if side == "left" and i0 > 0:
                trial[0] -= 1
            elif side == "right" and i1 < nx - 1:
                trial[1] += 1
            elif side == "down" and j0 > 0:
                trial[2] -= 1
            elif side == "up" and j1 < ny - 1:
                trial[3] += 1
            else:
                continue
            if ok(*trial):
                i0, i1, j0, j1 = trial
                grew = True

    dom_x = (float(xs[i0]), float(xs[i1]))
    dom_y = (float(ys[j0]), float(ys[j1]))
    sub = Z[j0 : j1 + 1, i0 : i1 + 1]
    info = {
        "grid_index_x": [i0, i1], "grid_index_y": [j0, j1],
        "seed_box_max_loss": seed_max,
        "points": int(sub.size),
        "loss_min": float(np.nanmin(sub)), "loss_max": float(np.nanmax(sub)),
        "loss_median": float(np.nanmedian(sub)),
    }
    return dom_x, dom_y, info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["cnn", "gpt"], required=True)
    ap.add_argument("--run", default=None)
    ap.add_argument("--cap-multiple", type=float, default=3.0)
    args = ap.parse_args()

    run = pathlib.Path(args.run or f"runs/{args.task}")
    ref = np.load(run / "reference.npz")
    report = json.loads((run / "stage2_report.json").read_text())

    shape = tuple(ref["value_shape"])
    X = ref["value_points"][:, 0].reshape(shape)
    Y = ref["value_points"][:, 1].reshape(shape)
    coords = ref["traj_coords"]
    centre = (float(np.median(coords[:, 0])), float(np.median(coords[:, 1])))

    traj_max = float(np.nanmax(ref["traj_loss_train"]))
    cap = args.cap_multiple * traj_max

    print(f"=== analysis domain for {args.task} ===")
    print(f"  full domain: x{tuple(report['domain'][0])} y{tuple(report['domain'][1])}")
    for split in ("train", "val"):
        Z = ref[f"value_{split}"].reshape(shape)
        print(f"  {split} reference: median {np.nanmedian(Z):.3f}  p95 {np.nanpercentile(Z, 95):.3f}  "
              f"max {np.nanmax(Z):.1f}")
    print(f"  optimiser's own maximum loss: {traj_max:.3f}  ->  cap "
          f"{args.cap_multiple:g}x = {cap:.3f}")

    # Both splits must satisfy the cap, so the two surfaces are drawn on the same domain.
    Zt = ref["value_train"].reshape(shape)
    Zv = ref["value_val"].reshape(shape)
    Zmax = np.fmax(Zt, Zv)
    pad_x = 0.03 * float(np.ptp(coords[:, 0]))
    pad_y = 0.03 * float(np.ptp(coords[:, 1]))
    seed = ((coords[:, 0].min() - pad_x, coords[:, 0].max() + pad_x),
            (coords[:, 1].min() - pad_y, coords[:, 1].max() + pad_y))
    dom_x, dom_y, info = largest_box_under_cap(X, Y, Zmax, centre, cap, seed_box=seed)

    inside_traj = (
        (coords[:, 0] >= dom_x[0]) & (coords[:, 0] <= dom_x[1])
        & (coords[:, 1] >= dom_y[0]) & (coords[:, 1] <= dom_y[1])
    )
    report["analysis_domain"] = [list(dom_x), list(dom_y)]
    report["analysis_domain_info"] = {
        **info,
        "cap_multiple": args.cap_multiple,
        "cap": cap,
        "trajectory_max_loss": traj_max,
        "trajectory_fraction_inside": float(inside_traj.mean()),
        "radius": 0.5 * float(np.sqrt((dom_x[1] - dom_x[0]) * (dom_y[1] - dom_y[0]))),
    }
    report["analysis_surface_range"] = {}
    for split in ("train", "val"):
        Z = ref[f"value_{split}"].reshape(shape)
        i0, i1 = info["grid_index_x"]
        j0, j1 = info["grid_index_y"]
        sub = Z[j0 : j1 + 1, i0 : i1 + 1]
        report["analysis_surface_range"][split] = float(np.nanmax(sub) - np.nanmin(sub))

    (run / "stage2_report.json").write_text(json.dumps(report, indent=2, default=str))

    print(f"  analysis domain: x{dom_x} y{dom_y}")
    print(f"    {info['points']} reference points, loss {info['loss_min']:.3f}"
          f"..{info['loss_max']:.3f} (median {info['loss_median']:.3f})")
    print(f"    contains {report['analysis_domain_info']['trajectory_fraction_inside']:.1%} "
          f"of the trajectory; effective radius "
          f"{report['analysis_domain_info']['radius']:.3g}")
    print(f"    surface range: train {report['analysis_surface_range']['train']:.3f}, "
          f"val {report['analysis_surface_range']['val']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
