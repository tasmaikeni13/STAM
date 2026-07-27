"""Stage 4: the certified landscape, the fidelity audit, and the animation.

Runs the full pipeline once at a realistic budget and produces the artefacts a
practitioner would actually look at, each carrying its measured accuracy:

* the four-panel figure, with contour intervals no finer than the certified error;
* the signed error map against the exact reference, which the certificate is checked
  against (the certificate never sees the reference);
* the projection-gap audit -- the true loss along the optimiser's path against the loss
  on the plane beneath it -- for all three plane constructions;
* the animated mini-batch surface with its trust region.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import torch

from stam import kernels
from stam.basis import Plane, build_plane, project_trajectory
from stam.capture import Trajectory
from stam.data import load_data
from stam.device import describe_environment, human_time, set_matmul_precision
from stam.fidelity import measure_fidelity
from stam.flat import FlatParams
from stam.metrics import surface_error
from stam.models import build_task
from stam.pipeline import certified_landscape
from stam.probe import CostModel, PlaneProbe, micro_table
from stam.reconstruct import reconstruct
from stam.viz import animate as A
from stam.viz import render as R
from stam.viz import style as S


def load_plane(path, device) -> Plane:
    b = torch.load(path, map_location="cpu", weights_only=True)
    return Plane(center=b["center"].to(device), basis=b["basis"].to(device),
                 singular_values=b["singular_values"], anchoring=b["anchoring"],
                 captured_variance=float(b["captured_variance"]),
                 orthonormality_error=float(b["orthonormality_error"]),
                 build_seconds=0.0, gram_dtype=str(b["gram_dtype"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["cnn", "gpt"], required=True)
    ap.add_argument("--run", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--frames", type=int, default=140)
    ap.add_argument("--no-animation", action="store_true")
    args = ap.parse_args()

    run = pathlib.Path(args.run or f"runs/{args.task}")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    set_matmul_precision()
    kernels.build_extension()
    S.use_paper_style()

    stage2 = json.loads((run / "stage2_report.json").read_text())
    ref = np.load(run / "reference.npz")
    domain = tuple(map(tuple, stage2.get("analysis_domain", stage2["domain"])))
    cm = stage2["cost_model"]
    cost = CostModel(
        kappa={int(k): float(v) for k, v in cm["kappa"].items()},
        tau={int(k): float(v) for k, v in cm["tau_example_equivalents"].items()},
        seconds_per_example=float(cm["seconds_per_example"]), device=str(device),
    )

    traj = Trajectory.load(run)
    data = load_data(args.task, device)
    task = build_task(args.task, device)
    flat = FlatParams(task.model)
    plane = load_plane(run / f"plane_{stage2['anchoring']}.pt", device)

    n_eval = int(stage2["reference_exact_examples"])
    budget = args.budget or 24 * (cost.tau[2] + cost.kappa[2] * min(1024, n_eval))
    # Render on the sub-grid of the exact reference that lies in the analysis domain, so
    # every rendered pixel has an exact value to be scored against.
    full_shape = tuple(ref["value_shape"])
    info = stage2.get("analysis_domain_info")
    if info:
        i0, i1 = info["grid_index_x"]
        j0, j1 = info["grid_index_y"]
    else:
        i0, j0 = 0, 0
        i1, j1 = full_shape[1] - 1, full_shape[0] - 1
    sl = (slice(j0, j1 + 1), slice(i0, i1 + 1))
    shape = (j1 - j0 + 1, i1 - i0 + 1)
    X = ref["value_points"][:, 0].reshape(full_shape)[sl]
    Y = ref["value_points"][:, 1].reshape(full_shape)[sl]
    value_pts = np.stack([X.ravel(), Y.ravel()], -1)
    ref_sub = {s_: ref[f"value_{s_}"].reshape(full_shape)[sl] for s_ in ("train", "val")}

    print(f"=== certified landscape: {args.task} ===")
    print(f"  budget {budget:,.0f} example-forward-equivalents "
          f"(~{budget * cost.seconds_per_example:.1f} s of probing per split)")

    out: dict = {"task": args.task, "budget": budget, "domain": stage2["domain"]}
    surfaces, certs, results = {}, {}, {}
    for split in ("train", "val"):
        eval_set = data.eval_train if split == "train" else data.eval_val
        table = micro_table(stage2["micro_batch"][split])
        probe = PlaneProbe(task, flat, plane, eval_set, micro_batch=table[0],
                           cost_model=cost, micro_batch_by_order=table)
        t0 = time.perf_counter()
        res = certified_landscape(probe, domain, budget, cost, value_pts,
                                  method="pu-taylor-2", seed=17)
        results[split] = res
        surfaces[split] = res.surface
        certs[split] = res.certificate
        err = surface_error(res.surface.values, ref_sub[split].ravel())
        out[split] = {
            "pipeline": res.to_dict(),
            "true_error": err.to_dict(),
            "seconds": time.perf_counter() - t0,
        }
        c = res.certificate
        print(f"  [{split}] {res.allocation.n_anchors} anchors x "
              f"{res.allocation.examples_per_anchor} examples")
        print(f"        certified RMSE {c.rmse:.4g} (<= {c.rmse_upper:.4g} @95%), "
              f"true RMSE {err.rmse:.4g}, surface range {err.reference_range:.3g}")
        print(f"        relative: certified {c.rmse / err.reference_range:.3%}, "
              f"true {err.rmse_relative:.3%}")

    # ---- figures ------------------------------------------------------------
    figdir = pathlib.Path("figures")
    figdir.mkdir(exist_ok=True)

    Zt = surfaces["train"].values.reshape(shape)
    Zv = surfaces["val"].values.reshape(shape)
    Gt = surfaces["train"].grads.reshape(*shape, 2)
    Gv = surfaces["val"].grads.reshape(*shape, 2)

    coords = project_trajectory(plane, traj.params).cpu().numpy()
    st = reconstruct("pu-taylor-2", results["train"].anchors, coords,
                     device=device, radii=results["train"].radii)
    sv = reconstruct("pu-taylor-2", results["val"].anchors, coords,
                     device=device, radii=results["val"].radii)

    fig, axes = R.landscape_figure(
        X, Y, Zt, Zv, Gt, Gv, coords, st.values, sv.values,
        err_train=certs["train"].rmse, err_val=certs["val"].rmse,
        anchors=results["train"].anchors.coords,
    )
    R.annotate_certificate(
        axes[1], loc="upper left", cert_text=
        f"certified: 95% of the domain within "
        f"{certs['train'].q95_upper:.3g} ({certs['train'].q95_upper / out['train']['true_error']['reference_range']:.1%} of relief)\n"
        f"RMSE {certs['train'].rmse:.3g} "
        f"\n{results['train'].allocation.n_anchors} anchors x "
        f"{results['train'].allocation.examples_per_anchor} examples "
        f"= {budget:,.0f} example-equivalents",
    )
    fig.savefig(figdir / f"landscape_{args.task}.pdf")
    fig.savefig(figdir / f"landscape_{args.task}.png", dpi=200)
    print(f"  wrote {figdir / f'landscape_{args.task}.pdf'}")

    import matplotlib.pyplot as plt

    fig2, axs = plt.subplots(1, 2, figsize=(8.6, 3.5))
    for ax, split, Z in ((axs[0], "train", Zt), (axs[1], "val", Zv)):
        E = Z - ref_sub[split]
        R.error_panel(ax, X, Y, E, title=f"{split}: reconstruction $-$ exact",
                      cbar_label="loss units")
    fig2.tight_layout()
    fig2.savefig(figdir / f"error_map_{args.task}.pdf")
    fig2.savefig(figdir / f"error_map_{args.task}.png", dpi=190)
    plt.close(fig2)

    # ---- projection fidelity ------------------------------------------------
    print("  measuring projection gap ...")
    fid_report = {}
    tbl_tr = micro_table(stage2["micro_batch"]["train"])
    sub = max(1, traj.T // 60)
    for anchoring in ("centered", "origin", "endpoint"):
        p = build_plane(traj.params, dim=2, anchoring=anchoring, device=device)
        pr = PlaneProbe(task, flat, p, data.eval_train, micro_batch=tbl_tr[0],
                        cost_model=cost, micro_batch_by_order=tbl_tr)
        rep = measure_fidelity(p, traj, pr, traj.loss_train,
                               surface_range=out["train"]["true_error"]["reference_range"],
                               subsample=sub, examples=None)
        fid_report[anchoring] = rep.summary()
        np.savez(run / f"fidelity_{anchoring}.npz", coords=rep.coords, residual=rep.residual,
                 loss_true=rep.loss_true, loss_on_plane=rep.loss_on_plane, gap=rep.gap,
                 epochs=traj.epochs, displacement=rep.displacement)
        print(f"    [{anchoring:>8}] rho2={rep.captured_variance:.4f}  "
              f"mean|gap|={fid_report[anchoring]['gap_mean_abs']:.4g}  "
              f"max|gap|={fid_report[anchoring]['gap_max_abs']:.4g}  "
              f"({fid_report[anchoring]['gap_relative_mean']:.1%} of relief)")
    out["fidelity"] = fid_report

    # ---- animation ----------------------------------------------------------
    if not args.no_animation and traj.grads is not None:
        print("  building the mini-batch animation ...")
        t0 = time.perf_counter()
        stride = max(1, traj.T // args.frames)
        idx = np.arange(0, traj.T, stride)
        gproj = np.zeros((len(idx), 2))
        for i, t in enumerate(idx):
            g = traj.grads[t].to(device).float()
            gproj[i] = kernels.project(g, plane.basis).double().cpu().numpy()
        at_traj = reconstruct("pu-taylor-2", results["train"].anchors, coords[idx],
                              device=device, radii=results["train"].radii)
        frames = A.build_frames(
            coords[idx], gproj, at_traj.grads, traj.batch_loss[idx],
            tolerance=max(certs["train"].rmse, 1e-6),
            radius_cap=0.45 * (domain[0][1] - domain[0][0]),
        )
        gif = figdir / f"landscape_{args.task}.gif"

        def surface_fn(q: np.ndarray) -> np.ndarray:
            """Re-evaluate the fitted reconstruction, for the zoom panel."""
            return reconstruct("pu-taylor-2", results["train"].anchors, q,
                               device=device, radii=results["train"].radii).values

        A.animate_landscape(
            X, Y, Zt, Zv, Gt, Gv, frames, st.values[idx], sv.values[idx],
            out_path=gif, err_train=certs["train"].rmse, err_val=certs["val"].rmse,
            surface_fn=surface_fn, fps=12, dpi=100,
        )
        R_dom = 0.5 * (domain[0][1] - domain[0][0])
        out["animation"] = {
            "path": str(gif), "frames": int(len(idx)),
            "tolerance": float(frames.tolerance),
            "trust_radius_mean": float(np.mean(frames.radius)),
            "trust_radius_min": float(np.min(frames.radius)),
            # The headline number: how much of the plotted domain the first-order
            # mini-batch model is actually certified over.
            "trust_radius_fraction_mean": float(np.mean(frames.radius) / R_dom),
            "trust_radius_fraction_max": float(np.max(frames.radius) / R_dom),
            "domain_radius": float(R_dom),
            "noise_norm_mean": float(np.mean(np.linalg.norm(frames.noise, axis=1))),
            "seconds": time.perf_counter() - t0,
        }
        print(f"    trust region covers {out['animation']['trust_radius_fraction_mean']:.2%} "
              f"of the domain radius on average "
              f"(tolerance {frames.tolerance:.3g})")
        np.savez(run / "stochastic_frames.npz", coords=frames.coords, noise=frames.noise,
                 radius=frames.radius, batch_loss=frames.batch_loss,
                 epochs=traj.epochs[idx])
        print(f"    wrote {gif} ({len(idx)} frames, "
              f"{human_time(out['animation']['seconds'])})")

    out["environment"] = describe_environment()
    (run / "stage4_report.json").write_text(json.dumps(out, indent=2, default=str))
    np.savez_compressed(
        run / "landscape.npz", X=X, Y=Y, Z_train=Zt, Z_val=Zv, G_train=Gt, G_val=Gv,
        traj=coords, traj_z_train=st.values, traj_z_val=sv.values,
        anchors_train=results["train"].anchors.coords,
        anchors_val=results["val"].anchors.coords,
    )
    print(f"  wrote {run / 'stage4_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
