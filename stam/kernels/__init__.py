"""Fused kernels with transparent fallbacks.

Every operation has two implementations, and they must agree numerically to within
their stated tolerance:

``cuda``
    Hand-written CUDA in ``stam_cuda.cu``, JIT-compiled for exactly the architectures
    present on the machine plus embedded PTX, so the same build runs on newer hardware.
``torch``
    A pure-PyTorch path.  Correct everywhere, including CPU, and the reference against
    which the CUDA kernels are checked.
``auto``
    ``cuda`` where the extension built and the tensors are on a GPU, else ``torch``.

A Triton implementation of all four operations was written and benchmarked against
these; ``bench/bakeoff.py`` records the comparison.  Hand-written CUDA won every
operation -- decisively on the trajectory Gram (14.7x) and the reconstruction sweep
(1.3x), and on a tie-break for the two pure streaming kernels, where the two toolchains
reach the same fraction of peak bandwidth but the compensated CUDA reduction is an order
of magnitude more accurate.  The Triton path was therefore dropped rather than carried.

Nothing in STAM *requires* a compiler: every result is reproducible on the PyTorch path,
only more slowly -- and since the per-anchor overhead enters the optimal experiment
design, more slowly also means less accurately at a fixed budget.
"""

from __future__ import annotations

import functools
import os
import pathlib
import warnings
from typing import Any, Literal

import torch

Backend = Literal["auto", "cuda", "torch"]

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parent.parent

_BUILD_ERROR: str | None = None
_EXTENSION: Any = None
_TRIED = False


# ---------------------------------------------------------------------------
# Toolchain discovery
# ---------------------------------------------------------------------------


def _nvcc_version(nvcc: pathlib.Path) -> tuple[int, int] | None:
    import re
    import subprocess

    try:
        out = subprocess.run([str(nvcc), "--version"], capture_output=True, text=True, timeout=30)
    except Exception:
        return None
    m = re.search(r"release (\d+)\.(\d+)", out.stdout)
    return (int(m.group(1)), int(m.group(2))) if m else None


def find_cuda_home() -> str | None:
    """Locate a CUDA toolkit whose major version matches the one torch was built with.

    Distribution toolkits are frequently older than the host compiler supports (CUDA
    11.5 cannot parse GCC 11.3 headers, for instance), so a repo-local toolkit under
    ``.toolchain/cuda-<version>`` takes precedence when present.
    """
    torch_cuda = torch.version.cuda
    if torch_cuda is None:
        return None
    want_major = int(torch_cuda.split(".")[0])

    candidates: list[pathlib.Path] = []
    local = _REPO / ".toolchain"
    if local.is_dir():
        candidates += sorted(local.glob("cuda-*"), reverse=True)
    for env in ("CUDA_HOME", "CUDA_PATH"):
        if os.environ.get(env):
            candidates.append(pathlib.Path(os.environ[env]))
    candidates += [pathlib.Path("/usr/local/cuda"), pathlib.Path("/usr")]

    for c in candidates:
        nvcc = c / "bin" / "nvcc"
        if not nvcc.is_file():
            continue
        ver = _nvcc_version(nvcc)
        if ver is None or ver[0] != want_major:
            continue
        return str(c)
    return None


def _arch_list() -> str:
    """Compile for the architectures actually present, and embed PTX for the newest so
    the same build keeps working on hardware that did not exist at build time."""
    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return os.environ["TORCH_CUDA_ARCH_LIST"]
    archs = set()
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        archs.add((p.major, p.minor))
    if not archs:
        return "7.5+PTX"
    ordered = sorted(archs)
    parts = [f"{a}.{b}" for a, b in ordered[:-1]]
    parts.append(f"{ordered[-1][0]}.{ordered[-1][1]}+PTX")
    return ";".join(parts)


def build_extension(verbose: bool = False, force: bool = False) -> Any:
    """Compile (or load from cache) the CUDA extension.  Returns ``None`` on failure."""
    global _EXTENSION, _BUILD_ERROR, _TRIED
    if _EXTENSION is not None and not force:
        return _EXTENSION
    if _TRIED and not force:
        return None
    _TRIED = True

    if not torch.cuda.is_available():
        _BUILD_ERROR = "no CUDA device"
        return None
    if os.environ.get("STAM_DISABLE_CUDA_EXT"):
        _BUILD_ERROR = "disabled by STAM_DISABLE_CUDA_EXT"
        return None

    cuda_home = find_cuda_home()
    if cuda_home is None:
        _BUILD_ERROR = f"no CUDA toolkit matching torch's CUDA {torch.version.cuda}"
        return None

    prev_home = os.environ.get("CUDA_HOME")
    prev_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["CUDA_HOME"] = cuda_home
    os.environ["TORCH_CUDA_ARCH_LIST"] = _arch_list()
    try:
        from torch.utils.cpp_extension import load

        _EXTENSION = load(
            name="stam_kernels",
            sources=[str(_HERE / "stam_cuda.cu")],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-lineinfo",
                "--expt-relaxed-constexpr",
            ],
            extra_cflags=["-O3"],
            verbose=verbose,
        )
        _BUILD_ERROR = None
    except Exception as exc:  # pragma: no cover - environment dependent
        _EXTENSION = None
        _BUILD_ERROR = f"{type(exc).__name__}: {exc}"
    finally:
        if prev_home is None:
            os.environ.pop("CUDA_HOME", None)
        else:
            os.environ["CUDA_HOME"] = prev_home
        if prev_arch is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = prev_arch
    return _EXTENSION


def get_extension() -> Any:
    return build_extension(verbose=False)


def extension_status() -> dict[str, Any]:
    ext = get_extension()
    status: dict[str, Any] = {
        "available": ext is not None,
        "error": _BUILD_ERROR,
        "cuda_home": find_cuda_home(),
        "arch_list": _arch_list() if torch.cuda.is_available() else None,
    }
    if ext is not None:
        status["build"] = dict(ext.build_info())
    return status



def resolve(op: str, backend: Backend, *tensors: torch.Tensor) -> str:
    """Decide which implementation to run for ``op``.

    An explicit backend is honoured and raises rather than silently degrading, so
    benchmarks measure what they claim to.
    """
    on_gpu = all(t.is_cuda for t in tensors)
    if backend == "torch":
        return "torch"
    if backend == "cuda":
        if not on_gpu:
            raise RuntimeError("backend='cuda' requires CUDA tensors")
        if get_extension() is None:
            raise RuntimeError(f"CUDA extension unavailable: {_BUILD_ERROR}")
        return "cuda"
    if not on_gpu:
        return "torch"
    return "cuda" if get_extension() is not None else "torch"


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


def plane_point(
    theta0: torch.Tensor,
    basis: torch.Tensor,
    coords: torch.Tensor,
    out: torch.Tensor | None = None,
    backend: Backend = "auto",
) -> torch.Tensor:
    r"""Evaluate :math:`\theta_0 + \sum_i c_i v_i` into a contiguous buffer.

    ``basis`` is ``(K, N)``; ``coords`` is ``(K,)``.  The fused form reads the basis
    once and writes the output once, replacing a per-parameter-tensor Python loop.
    """
    impl = resolve("plane_point", backend, theta0, basis)
    if impl == "cuda":
        return get_extension().plane_point(theta0, basis, coords, out)
    coords = coords.to(device=theta0.device, dtype=torch.float32).reshape(-1)
    acc = torch.addmv(
        theta0.to(torch.float32),
        basis.to(torch.float32).t(),
        coords,
    )
    acc = acc.to(theta0.dtype)
    if out is None:
        return acc
    out.copy_(acc)
    return out


def project(
    g: torch.Tensor,
    basis: torch.Tensor,
    compensated: bool | None = None,
    backend: Backend = "auto",
) -> torch.Tensor:
    r"""Compute :math:`(\langle g, v_1\rangle, \dots, \langle g, v_K\rangle)` in one pass.

    ``compensated`` selects Neumaier accumulation inside each thread's serial loop.
    The default follows the device policy: compensation is enabled wherever fp64 is
    rate-limited, which is exactly where an fp32 accumulator would otherwise be the
    accuracy bottleneck.
    """
    if compensated is None:
        from ..device import get_policy

        compensated = not get_policy().caps.fp64_is_cheap if g.is_cuda else True
    impl = resolve("project", backend, g, basis)
    if impl == "cuda":
        return get_extension().project(g, basis, bool(compensated))
    return torch.mv(basis.to(torch.float32), g.to(torch.float32).reshape(-1))


def gram_chunk(
    D: torch.Tensor,
    G: torch.Tensor | None = None,
    backend: Backend = "auto",
) -> torch.Tensor:
    r"""Accumulate :math:`G \mathrel{+}= D D^\top` for a column chunk of ``D``.

    ``D`` is ``(T, n_chunk)``, ``G`` is ``(T, T)`` float64.  Only the lower triangle is
    written; :func:`symmetrise` completes it.
    """
    T = D.shape[0]
    if G is None:
        G = torch.zeros(T, T, dtype=torch.float64, device=D.device)
    impl = resolve("gram_chunk", backend, D, G)
    if impl == "cuda":
        return get_extension().gram_chunk(D, G)
    Df = D.to(torch.float64)
    G += torch.tril(Df @ Df.t())
    return G


def symmetrise(G: torch.Tensor) -> torch.Tensor:
    """Mirror a lower-triangular Gram accumulation into a full symmetric matrix."""
    lower = torch.tril(G)
    return lower + torch.tril(lower, -1).t()


def pu_taylor(
    anchors: torch.Tensor,
    radii: torch.Tensor,
    values: torch.Tensor,
    grads: torch.Tensor,
    hess: torch.Tensor,
    queries: torch.Tensor,
    order: int = 2,
    backend: Backend = "auto",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Partition-of-unity Taylor reconstruction and its exact gradient.

    Returns ``(values, gradients, weight_sum)``.  Query points not covered by any
    anchor's support yield ``NaN`` with weight sum zero; callers must treat those as
    outside the certified region rather than extrapolating into them.
    """
    impl = resolve("pu_taylor", backend, anchors, queries)
    if impl == "cuda":
        return get_extension().pu_taylor(
            anchors, radii, values, grads, hess, queries, int(order)
        )
    return _pu_taylor_torch(anchors, radii, values, grads, hess, queries, order)


def _pu_taylor_torch(
    anchors: torch.Tensor,
    radii: torch.Tensor,
    values: torch.Tensor,
    grads: torch.Tensor,
    hess: torch.Tensor,
    queries: torch.Tensor,
    order: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference implementation.  Same formulae, same weight function, no tiling."""
    dtype = torch.float32
    a = anchors.to(dtype)
    rho = radii.to(dtype)
    f = values.to(dtype)
    g = grads.to(dtype)
    H = hess.to(dtype)
    q = queries.to(dtype)

    d = q[:, None, :] - a[None, :, :]            # (m, n, 2)
    dist = d.norm(dim=-1)                        # (m, n)
    r = dist / rho[None, :]

    om = (1 - r).clamp(min=0)
    w = om.pow(4) * (4 * r + 1)
    dphi = -20 * r * om.pow(3)

    Q = f[None, :].expand_as(w).clone()
    G = torch.zeros_like(d)
    if order >= 1:
        Q = Q + (d * g[None, :, :]).sum(-1)
        G = G + g[None, :, :].expand_as(d).clone()
    if order >= 2:
        hxx, hxy, hyy = H[:, 0], H[:, 1], H[:, 2]
        dx, dy = d[..., 0], d[..., 1]
        Q = Q + 0.5 * (hxx * dx * dx + 2 * hxy * dx * dy + hyy * dy * dy)
        G = G + torch.stack([hxx * dx + hxy * dy, hxy * dx + hyy * dy], dim=-1)

    safe = dist.clamp(min=1e-12)
    scale = torch.where(dist > 1e-12, dphi / (rho[None, :] * safe), torch.zeros_like(dist))
    gw = scale[..., None] * d                    # (m, n, 2)

    W = w.sum(1)
    covered = W > 1e-12
    invW = torch.where(covered, 1.0 / W.clamp(min=1e-12), torch.zeros_like(W))
    R = (w * Q).sum(1) * invW
    term1 = (w[..., None] * G).sum(1)
    term2 = ((Q - R[:, None])[..., None] * gw).sum(1)
    grad = invW[:, None] * (term1 + term2)

    nan = torch.full_like(R, float("nan"))
    return (
        torch.where(covered, R, nan),
        torch.where(covered[:, None], grad, nan[:, None].expand_as(grad)),
        W,
    )


@functools.lru_cache(maxsize=1)
def warn_if_unavailable() -> None:
    if get_extension() is None and torch.cuda.is_available():
        warnings.warn(
            f"STAM CUDA kernels unavailable ({_BUILD_ERROR}); using the PyTorch path. "
            "Results are unchanged; per-anchor overhead will be higher.",
            RuntimeWarning,
            stacklevel=2,
        )


__all__ = [
    "build_extension", "get_extension", "extension_status", "find_cuda_home",
    "plane_point", "project", "gram_chunk", "symmetrise", "pu_taylor",
    "warn_if_unavailable",
]
