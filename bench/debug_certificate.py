"""Check the certificate against the exact reference, point by point.

The certificate estimates the reconstruction's error from independent probes; the
reference gives the same error exactly.  If they disagree, one of three things is wrong:
the probes are not measuring what the reference measures, the reconstruction is being
evaluated differently in the two paths, or the debiasing is over-subtracting.  This
script separates those cases.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import torch

from stam import kernels
from stam.basis import Plane
from stam.certify import certify_surface
from stam.data import load_data
from stam.design import inside
from stam.device import set_matmul_precision
from stam.flat import FlatParams
from stam.metrics import surface_error
from stam.models import build_task
from stam.pipeline import certified_landscape
from stam.probe import CostModel, PlaneProbe, micro_table
from stam.reconstruct import reconstruct


def main() -> int:
    run = pathlib.Path("runs/cnn")
    stage2 = json.loads((run / "stage2_report.json").read_text())
    ref = np.load(run / "reference.npz")
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    set_matmul_precision()
    kernels.build_extension()

    b = torch.load(run / "plane_centered.pt", map_location="cpu", weights_only=True)
    plane = Plane(center=b["center"].to(dev), basis=b["basis"].to(dev),
                  singular_values=b["singular_values"], anchoring=b["anchoring"],
                  captured_variance=float(b["captured_variance"]),
                  orthonormality_error=float(b["orthonormality_error"]),
                  build_seconds=0.0, gram_dtype=str(b["gram_dtype"]))
    data = load_data("cnn", dev)
    task = build_task("cnn", dev)
    flat = FlatParams(task.model)
    cm = stage2["cost_model"]
    cost = CostModel(kappa={int(k): float(v) for k, v in cm["kappa"].items()},
                     tau={int(k): float(v) for k, v in cm["tau_example_equivalents"].items()},
                     seconds_per_example=float(cm["seconds_per_example"]), device=str(dev))
    tbl = micro_table(stage2["micro_batch"]["train"])
    probe = PlaneProbe(task, flat, plane, data.eval_train, micro_batch=tbl[0],
                       cost_model=cost, micro_batch_by_order=tbl)

    domain = tuple(map(tuple, stage2["analysis_domain"]))
    vmask = inside(ref["value_points"], domain)
    grid = ref["value_points"][vmask]
    ref_val = ref["value_train"].ravel()[vmask]

    budget = 7.1e5
    res = certified_landscape(probe, domain, budget, cost, grid,
                              method="pu-taylor-2", seed=0)
    err = surface_error(res.surface.values, ref_val)
    c = res.certificate
    print(f"budget {budget:,.0f}: n={res.allocation.n_anchors} "
          f"B={res.allocation.examples_per_anchor}")
    print(f"  true RMSE on the reference grid : {err.rmse:.4f}")
    print(f"  certified RMSE                  : {c.rmse:.4f} "
          f"(raw {c.rmse_raw:.4f}, noise floor {c.noise_floor:.4f})")
    print(f"  certification: {c.n_probes} probes x {c.examples_per_probe} examples")

    # 1. Is the reconstruction the same object in both paths?
    at_grid = reconstruct("pu-taylor-2", res.anchors, grid, device=dev, radii=res.radii)
    print(f"  reconstruction reproducible: "
          f"{np.nanmax(np.abs(at_grid.values - res.surface.values)):.2e}")

    # 2. Evaluate the TRUE error at the certificate's own probe locations, exactly.
    rng = np.random.default_rng(991)
    (x0, x1), (y0, y1) = domain
    m = 48
    q = np.stack([rng.uniform(x0, x1, m), rng.uniform(y0, y1, m)], 1)
    exact = np.array([probe.exact(p, order=0).value for p in q])
    recon = reconstruct("pu-taylor-2", res.anchors, q, device=dev, radii=res.radii).values
    ok = np.isfinite(recon)
    true_at_probes = float(np.sqrt(np.mean((exact[ok] - recon[ok]) ** 2)))
    print(f"  true RMSE at random probe points: {true_at_probes:.4f}")

    # 3. Run the certificate machinery against those same exact values.
    cert2 = certify_surface(
        probe,
        lambda qq: reconstruct("pu-taylor-2", res.anchors, qq, device=dev, radii=res.radii),
        domain, n_probes=m, examples_per_probe=2000, seed=991,
    )
    print(f"  certificate re-run              : {cert2.rmse:.4f} "
          f"(upper {cert2.rmse_upper:.4f})")

    # 4. Where does the error live?  If it concentrates on a small set of grid points,
    #    a uniform sample of 48 will usually miss it and the two numbers legitimately
    #    differ -- a property of the estimand, not a bug.
    d = np.abs(at_grid.values - ref_val)
    d = d[np.isfinite(d)]
    print(f"  |error| on the grid: mean {d.mean():.4f} median {np.median(d):.4f} "
          f"p90 {np.percentile(d, 90):.4f} p99 {np.percentile(d, 99):.4f} "
          f"max {d.max():.4f}")
    print(f"  share of MSE from the worst 1% of points: "
          f"{(np.sort(d)[-len(d) // 100:] ** 2).sum() / (d ** 2).sum():.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
