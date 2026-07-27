"""Correctness tests for the fused kernels.

Each kernel is checked against (a) its pure-PyTorch twin and (b) an fp64 reference
computed independently.  The fp64 reference matters: two implementations that agree
with each other can still both be wrong, and the accumulation strategy is precisely
what these kernels change.
"""

from __future__ import annotations

import sys

import torch

from stam import kernels as K
from stam.device import get_caps, get_policy

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FAIL: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.double()
    b = b.double()
    denom = b.abs().max().clamp(min=1e-30)
    return float((a - b).abs().max() / denom)


# ---------------------------------------------------------------------------


def test_plane_point() -> None:
    print("plane_point")
    g = torch.Generator(device=DEV).manual_seed(0)
    for n, k, dtype in [(1 << 20, 2, torch.float32), (1 << 20 + 3, 3, torch.float32),
                        (1 << 18, 2, torch.float16), (1 << 18, 4, torch.bfloat16)]:
        theta0 = torch.randn(n, device=DEV, generator=g)
        basis = torch.randn(k, n, device=DEV, generator=g).to(dtype)
        coords = torch.randn(k, device=DEV, generator=g)

        ref = theta0.double() + (coords.double()[:, None] * basis.double()).sum(0)
        cuda = K.plane_point(theta0, basis, coords, backend="cuda" if DEV == "cuda" else "torch")
        torchp = K.plane_point(theta0, basis, coords, backend="torch")
        tol = 3e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-5
        check(
            f"n={n} k={k} basis={dtype}",
            rel_err(cuda, ref) < tol and rel_err(torchp, ref) < tol,
            f"cuda={rel_err(cuda, ref):.2e} torch={rel_err(torchp, ref):.2e}",
        )

    # Non-multiple-of-4 length forces the scalar path; both must still agree.
    n = 1000003
    theta0 = torch.randn(n, device=DEV)
    basis = torch.randn(2, n, device=DEV)
    coords = torch.tensor([0.3, -1.7], device=DEV)
    ref = theta0.double() + 0.3 * basis[0].double() - 1.7 * basis[1].double()
    out = K.plane_point(theta0, basis, coords)
    check("odd length (scalar path)", rel_err(out, ref) < 1e-5, f"{rel_err(out, ref):.2e}")

    # In-place output buffer.
    buf = torch.empty(n, device=DEV)
    out2 = K.plane_point(theta0, basis, coords, out=buf)
    check("in-place out", out2.data_ptr() == buf.data_ptr() and rel_err(buf, ref) < 1e-5)


def test_project() -> None:
    print("project")
    g = torch.Generator(device=DEV).manual_seed(1)
    for n, k, dtype in [(1 << 22, 2, torch.float32), (1 << 22, 3, torch.float16),
                        (1 << 20, 8, torch.float32)]:
        v = torch.randn(n, device=DEV, generator=g)
        basis = torch.randn(k, n, device=DEV, generator=g).to(dtype)
        ref = (basis.double() * v.double()).sum(1)
        cuda = K.project(v, basis, backend="cuda" if DEV == "cuda" else "torch")
        torchp = K.project(v, basis, backend="torch")
        tol = 5e-3 if dtype in (torch.float16, torch.bfloat16) else 2e-5
        check(
            f"n={n} k={k} basis={dtype}",
            rel_err(cuda, ref) < tol and rel_err(torchp, ref) < tol,
            f"cuda={rel_err(cuda, ref):.2e} torch={rel_err(torchp, ref):.2e}",
        )

    if DEV == "cuda":
        # Compensated summation must beat plain fp32 on an adversarial input: many
        # small same-sign terms plus one large one, which is exactly the regime a
        # gradient projection over 10^7 coordinates lands in.
        n = 1 << 24
        v = torch.full((n,), 1e-4, device=DEV)
        v[0] = 1e5
        basis = torch.ones(1, n, device=DEV)
        ref = float(v.double().sum())
        comp = float(K.project(v, basis, compensated=True)[0])
        plain = float(K.project(v, basis, compensated=False)[0])
        e_comp = abs(comp - ref) / abs(ref)
        e_plain = abs(plain - ref) / abs(ref)
        check(
            "compensated beats plain fp32",
            e_comp <= e_plain,
            f"comp={e_comp:.2e} plain={e_plain:.2e} ({e_plain / max(e_comp, 1e-16):.0f}x)",
        )

        # Determinism: fixed block count, no float atomics.
        a = K.project(v, basis)
        b = K.project(v, basis)
        check("bitwise deterministic", bool(torch.equal(a, b)))


def test_gram() -> None:
    print("gram_chunk")
    g = torch.Generator(device=DEV).manual_seed(2)
    T, n = 37, 1 << 18
    D = torch.randn(T, n, device=DEV, generator=g)
    ref = (D.double() @ D.double().t())

    G = K.gram_chunk(D, backend="cuda" if DEV == "cuda" else "torch")
    G = K.symmetrise(G)
    check("fp32 storage", rel_err(G, ref) < 1e-5, f"{rel_err(G, ref):.2e}")

    Gt = K.symmetrise(K.gram_chunk(D, backend="torch"))
    check("torch path", rel_err(Gt, ref) < 1e-10, f"{rel_err(Gt, ref):.2e}")

    # Half storage: the accuracy question that decides whether a 400-snapshot
    # trajectory of a 5M-parameter model fits in 24 GB.
    Dh = D.to(torch.float16)
    ref_h = Dh.double() @ Dh.double().t()
    Gh = K.symmetrise(K.gram_chunk(Dh))
    check("fp16 storage vs its own fp64 Gram", rel_err(Gh, ref_h) < 1e-4,
          f"{rel_err(Gh, ref_h):.2e}")

    # Streaming: chunking the columns must give the same answer as one shot.
    Gs = torch.zeros(T, T, dtype=torch.float64, device=DEV)
    for lo in range(0, n, 40000):
        K.gram_chunk(D[:, lo : lo + 40000].contiguous(), Gs)
    check("streaming == one-shot", rel_err(K.symmetrise(Gs), ref) < 1e-5,
          f"{rel_err(K.symmetrise(Gs), ref):.2e}")


def test_pu_taylor() -> None:
    print("pu_taylor")
    torch.manual_seed(3)
    n, m = 61, 4096
    anchors = torch.rand(n, 2, device=DEV) * 4 - 2
    radii = torch.full((n,), 1.2, device=DEV)
    values = torch.randn(n, device=DEV)
    grads = torch.randn(n, 2, device=DEV)
    hess = torch.randn(n, 3, device=DEV)
    queries = torch.rand(m, 2, device=DEV) * 3 - 1.5

    v_c, g_c, w_c = K.pu_taylor(anchors, radii, values, grads, hess, queries, order=2)
    v_t, g_t, w_t = K.pu_taylor(anchors, radii, values, grads, hess, queries, order=2,
                                backend="torch")
    good = torch.isfinite(v_c) & torch.isfinite(v_t)
    check("value matches reference", rel_err(v_c[good], v_t[good]) < 1e-4,
          f"{rel_err(v_c[good], v_t[good]):.2e}")
    check("gradient matches reference", rel_err(g_c[good], g_t[good]) < 1e-3,
          f"{rel_err(g_c[good], g_t[good]):.2e}")
    check("weight sum matches", rel_err(w_c, w_t) < 1e-5)

    # Polynomial reproduction: a partition-of-unity blend of exact Taylor patches of a
    # quadratic must reproduce that quadratic *exactly*, at every query point, for any
    # anchor placement.  This is the property the whole error analysis rests on.
    c = torch.tensor([0.7, -1.3, 0.45, 0.9, -0.6, 0.25], device=DEV)  # 1,x,y,xx,xy,yy

    def quad(p: torch.Tensor) -> torch.Tensor:
        x, y = p[:, 0], p[:, 1]
        return c[0] + c[1] * x + c[2] * y + c[3] * x * x + c[4] * x * y + c[5] * y * y

    def dquad(p: torch.Tensor) -> torch.Tensor:
        x, y = p[:, 0], p[:, 1]
        return torch.stack([c[1] + 2 * c[3] * x + c[4] * y, c[2] + c[4] * x + 2 * c[5] * y], -1)

    fv = quad(anchors)
    gv = dquad(anchors)
    hv = torch.stack([2 * c[3], c[4], 2 * c[5]]).expand(n, 3).contiguous()
    v_q, g_q, w_q = K.pu_taylor(anchors, radii, fv, gv, hv, queries, order=2)
    good = torch.isfinite(v_q)
    ev = float((v_q[good] - quad(queries)[good]).abs().max())
    eg = float((g_q[good] - dquad(queries)[good]).abs().max())
    check("reproduces quadratics exactly", ev < 1e-4 and eg < 1e-3, f"val={ev:.2e} grad={eg:.2e}")

    # The rendered gradient must be the true gradient of the rendered surface: check by
    # central differences on a non-polynomial field, where the two are not trivially equal.
    h = 1e-3
    qs = queries[:512]
    for axis in (0, 1):
        off = torch.zeros(1, 2, device=DEV)
        off[0, axis] = h
        vp, _, _ = K.pu_taylor(anchors, radii, values, grads, hess, qs + off, order=2)
        vm, _, _ = K.pu_taylor(anchors, radii, values, grads, hess, qs - off, order=2)
        ga, _, _ = K.pu_taylor(anchors, radii, values, grads, hess, qs, order=2)
        _, gan, _ = K.pu_taylor(anchors, radii, values, grads, hess, qs, order=2)
        fd = (vp - vm) / (2 * h)
        ok = torch.isfinite(fd) & torch.isfinite(gan[:, axis])
        err = float((fd[ok] - gan[ok, axis]).abs().max() / gan[ok, axis].abs().max())
        check(f"grad == d/d{'xy'[axis]} of surface", err < 5e-3, f"{err:.2e}")

    # Uncovered queries are flagged, never extrapolated.
    far = torch.tensor([[50.0, 50.0]], device=DEV)
    vf, gf, wf = K.pu_taylor(anchors, radii, values, grads, hess, far, order=2)
    check("uncovered -> NaN, weight 0", bool(torch.isnan(vf).all()) and float(wf[0]) == 0.0)

    # Order truncation must actually change the answer in the right direction.
    v0, _, _ = K.pu_taylor(anchors, radii, fv, gv, hv, queries, order=0)
    v1, _, _ = K.pu_taylor(anchors, radii, fv, gv, hv, queries, order=1)
    target = quad(queries)
    e0 = float((v0[good] - target[good]).abs().mean())
    e1 = float((v1[good] - target[good]).abs().mean())
    check("order 0 > order 1 > order 2 error", e0 > e1 > ev, f"{e0:.3f} > {e1:.3f} > {ev:.1e}")


def main() -> int:
    print(f"device: {get_caps().summary()}")
    if torch.cuda.is_available():
        print(f"policy: {get_policy().describe()}")
    print(f"extension: {K.extension_status()['available']}")
    print()
    test_plane_point()
    test_project()
    test_gram()
    test_pu_taylor()
    print()
    if FAIL:
        print(f"{len(FAIL)} FAILURES: {FAIL}")
        return 1
    print("all kernel tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
