"""Kernel benchmark: hand-written CUDA against the PyTorch baseline, operation by operation.

Decides which implementation ``auto`` dispatch uses on *this* machine, and produces the
numbers reported in the systems section: achieved bandwidth against the device's
theoretical peak, speedup over the PyTorch baseline, and accuracy against an fp64
reference computed independently of all three.

Selection rule, applied per operation: among implementations whose maximum relative
error against the fp64 reference is within ``ACC_TOL``, take the fastest at the largest
problem size; break near-ties (within 5%) in favour of the more accurate one.  Speed
alone is the wrong criterion here -- these kernels sit inside an error analysis, so an
implementation that is 10% faster and 10x less accurate is a loss, not a win.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any, Callable

import numpy as np
import torch

from stam import kernels as K
from stam.device import describe_environment, get_caps

ACC_TOL = 1e-3
TIE = 0.05


def cuda_time(fn: Callable[[], Any], warmup: int = 5, iters: int = 30) -> float:
    """Median wall time per call, measured with CUDA events.

    Events time the device rather than the host, so launch-queue effects do not leak
    into the measurement; the median over repeats suppresses clock-boost transients,
    which on a workstation card are large.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) / 1e3 for s, e in zip(starts, ends)]
    return float(np.median(times))


def rel_err(a: torch.Tensor, ref: torch.Tensor) -> float:
    a = a.double()
    ref = ref.double()
    m = torch.isfinite(a) & torch.isfinite(ref)
    if m.sum() == 0:
        return float("inf")
    denom = ref[m].abs().max().clamp(min=1e-30)
    return float((a[m] - ref[m]).abs().max() / denom)


# ---------------------------------------------------------------------------
# Per-operation benchmark definitions
# ---------------------------------------------------------------------------


def bench_plane_point(device: str, sizes: list[int], k: int = 2) -> list[dict]:
    rows = []
    for n in sizes:
        theta0 = torch.randn(n, device=device)
        basis = torch.randn(k, n, device=device)
        coords = torch.randn(k, device=device)
        ref = theta0.double() + (coords.double()[:, None] * basis.double()).sum(0)
        # traffic: read theta0 + k basis vectors, write out
        traffic = (2 + k) * n * 4
        for backend in ("torch", "cuda"):
            try:
                out = K.plane_point(theta0, basis, coords, backend=backend)
                t = cuda_time(lambda: K.plane_point(theta0, basis, coords, backend=backend))
            except Exception as exc:
                rows.append({"op": "plane_point", "n": n, "backend": backend,
                             "error": str(exc)[:120]})
                continue
            rows.append({
                "op": "plane_point", "n": n, "backend": backend, "seconds": t,
                "gbps": traffic / t / 1e9, "rel_err": rel_err(out, ref),
            })
    return rows


def bench_project(device: str, sizes: list[int], k: int = 2) -> list[dict]:
    rows = []
    for n in sizes:
        v = torch.randn(n, device=device)
        basis = torch.randn(k, n, device=device)
        ref = (basis.double() * v.double()).sum(1)
        traffic = (1 + k) * n * 4
        for backend in ("torch", "cuda"):
            try:
                out = K.project(v, basis, backend=backend)
                t = cuda_time(lambda: K.project(v, basis, backend=backend))
            except Exception as exc:
                rows.append({"op": "project", "n": n, "backend": backend, "error": str(exc)[:120]})
                continue
            rows.append({
                "op": "project", "n": n, "backend": backend, "seconds": t,
                "gbps": traffic / t / 1e9, "rel_err": rel_err(out, ref),
            })
    return rows


def bench_gram(device: str, sizes: list[int], T: int = 64) -> list[dict]:
    rows = []
    for n in sizes:
        D = torch.randn(T, n, device=device)
        ref = D.double() @ D.double().t()
        traffic = T * n * 4
        flops = 2.0 * T * T * n / 2  # lower triangle only
        for backend in ("torch", "cuda"):
            try:
                G = K.symmetrise(K.gram_chunk(D, backend=backend))
                t = cuda_time(lambda: K.gram_chunk(D, torch.zeros(T, T, dtype=torch.float64,
                                                                 device=device),
                                                   backend=backend), warmup=3, iters=10)
            except Exception as exc:
                rows.append({"op": "gram_chunk", "n": n, "backend": backend,
                             "error": str(exc)[:120]})
                continue
            rows.append({
                "op": "gram_chunk", "n": n, "T": T, "backend": backend, "seconds": t,
                "gbps": traffic / t / 1e9, "gflops": flops / t / 1e9, "rel_err": rel_err(G, ref),
            })
    return rows


def bench_pu_taylor(device: str, sizes: list[int], n_anchors: int = 96) -> list[dict]:
    rows = []
    torch.manual_seed(0)
    anchors = torch.rand(n_anchors, 2, device=device) * 4 - 2
    radii = torch.full((n_anchors,), 0.9, device=device)
    values = torch.randn(n_anchors, device=device)
    grads = torch.randn(n_anchors, 2, device=device)
    hess = torch.randn(n_anchors, 3, device=device)
    for m in sizes:
        q = torch.rand(m, 2, device=device) * 3 - 1.5
        ref, _, _ = K.pu_taylor(anchors, radii, values, grads, hess, q, order=2,
                                backend="torch")
        flops = m * n_anchors * 40.0
        for backend in ("torch", "cuda"):
            try:
                out, _, _ = K.pu_taylor(anchors, radii, values, grads, hess, q, order=2,
                                        backend=backend)
                t = cuda_time(
                    lambda: K.pu_taylor(anchors, radii, values, grads, hess, q, order=2,
                                        backend=backend),
                    warmup=3, iters=15,
                )
            except Exception as exc:
                rows.append({"op": "pu_taylor", "m": m, "backend": backend,
                             "error": str(exc)[:120]})
                continue
            rows.append({
                "op": "pu_taylor", "m": m, "n_anchors": n_anchors, "backend": backend,
                "seconds": t, "gflops": flops / t / 1e9, "rel_err": rel_err(out, ref),
            })
    return rows


# ---------------------------------------------------------------------------


def decide(rows: list[dict]) -> tuple[dict[str, str], dict[str, Any]]:
    """Pick a winner per operation at the largest problem size."""
    choice: dict[str, str] = {}
    detail: dict[str, Any] = {}
    ops = sorted({r["op"] for r in rows})
    for op in ops:
        sel = [r for r in rows if r["op"] == op and "seconds" in r]
        if not sel:
            continue
        size_key = "m" if op == "pu_taylor" else "n"
        biggest = max(r[size_key] for r in sel)
        cands = [r for r in sel if r[size_key] == biggest]
        accurate = [r for r in cands if r["rel_err"] < ACC_TOL] or cands
        best = min(accurate, key=lambda r: r["seconds"])
        near = [r for r in accurate if r["seconds"] <= best["seconds"] * (1 + TIE)]
        best = min(near, key=lambda r: r["rel_err"])
        choice[op] = best["backend"]
        base = next((r for r in cands if r["backend"] == "torch"), None)
        detail[op] = {
            "size": biggest,
            "winner": best["backend"],
            "seconds": best["seconds"],
            "rel_err": best["rel_err"],
            "speedup_vs_torch": (base["seconds"] / best["seconds"]) if base else None,
            "candidates": {
                r["backend"]: {"seconds": r["seconds"], "rel_err": r["rel_err"]} for r in cands
            },
        }
    return choice, detail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/bakeoff.json")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device; bake-off is a GPU comparison")
        return 1

    device = "cuda"
    caps = get_caps()
    print(caps.summary())
    print(f"extension: {K.extension_status()['available']}")
    print()

    stream_sizes = [1 << 20, 1 << 23] if args.quick else [1 << 20, 1 << 22, 1 << 24, 1 << 26]
    gram_sizes = [1 << 18, 1 << 20] if args.quick else [1 << 18, 1 << 20, 1 << 22]
    pu_sizes = [1 << 14, 1 << 18] if args.quick else [1 << 12, 1 << 16, 1 << 20]

    rows: list[dict] = []
    for name, fn, sizes in [
        ("plane_point", bench_plane_point, stream_sizes),
        ("project", bench_project, stream_sizes),
        ("gram_chunk", bench_gram, gram_sizes),
        ("pu_taylor", bench_pu_taylor, pu_sizes),
    ]:
        print(f"--- {name} ---")
        r = fn(device, sizes)
        rows += r
        for row in r:
            if "error" in row:
                print(f"  {row['backend']:>7}  FAILED  {row['error']}")
                continue
            size = row.get("n", row.get("m"))
            extra = ""
            if "gbps" in row:
                pct = 100 * row["gbps"] / max(caps.peak_bandwidth_gbps, 1e-9)
                extra = f"{row['gbps']:8.1f} GB/s ({pct:4.1f}% peak)"
            if "gflops" in row:
                extra += f"  {row['gflops']:8.1f} GFLOP/s"
            print(f"  {row['backend']:>7}  n={size:>10,}  {row['seconds'] * 1e6:9.1f} us  "
                  f"{extra}  relerr {row['rel_err']:.2e}")
        print()

    choice, detail = decide(rows)
    print("=== decision ===")
    for op, backend in choice.items():
        d = detail[op]
        sp = d["speedup_vs_torch"]
        print(f"  {op:<12} -> {backend:<7} "
              f"({d['seconds'] * 1e6:.1f} us, {sp:.2f}x vs torch)" if sp else
              f"  {op:<12} -> {backend}")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "environment": describe_environment(),
        "rows": rows, "choice": choice, "detail": detail,
        "peak_bandwidth_gbps": caps.peak_bandwidth_gbps,
        "acc_tol": ACC_TOL,
    }, indent=2, default=str))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
