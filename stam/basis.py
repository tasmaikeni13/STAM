r"""Choosing the plane, and reporting honestly how much it misses.

Given snapshots :math:`\theta_0,\dots,\theta_T\in\mathbb R^N` of an optimiser's path, we
need a 2-dimensional affine subspace :math:`\mathcal S = c + \mathrm{span}\{v_1,v_2\}` to
draw the loss on.

**The optimal affine plane is centred at the mean, not at the initial point.**  Writing
:math:`P` for orthogonal projection onto :math:`\mathrm{span}\{v_1,v_2\}`, the residual to
minimise is

.. math::
   \sum_t \bigl\| (\theta_t - c) - P(\theta_t - c) \bigr\|^2 ,

jointly over :math:`c` and the subspace.  For any fixed subspace the inner minimisation
over :math:`c` is a least-squares problem solved by :math:`c=\bar\theta`, after which
Eckart--Young gives the top two right singular vectors of the **mean-centred**
displacement matrix.  Anchoring at :math:`\theta_0` and taking an uncentred SVD -- the
common choice -- solves a different problem and is generally suboptimal; the gap is
:math:`T\|\bar\theta-\theta_0\|^2` minus its in-plane part, which is not small when the
trajectory drifts, as trajectories do.  Both are implemented so the difference can be
measured rather than asserted.

**Computing the basis without an N-dimensional SVD.**  With :math:`T\sim10^2` snapshots
and :math:`N\sim10^6` parameters, form the Gram matrix :math:`G = DD^\top\in\mathbb
R^{T\times T}`, eigendecompose it, and recover :math:`v_i = D^\top u_i/\sigma_i`.  The
Gram is accumulated in column chunks streamed from host memory, so the trajectory never
has to fit in device memory.

**What the plane cannot show.**  The captured-variance ratio :math:`\rho_2` describes the
*trajectory*, not the surface.  The surface drawn on :math:`\mathcal S` is a
**restriction** of :math:`L`, and structure orthogonal to :math:`\mathcal S` is invisible
at any :math:`\rho_2`.  :mod:`stam.fidelity` measures that separately.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Literal

import torch

from . import kernels
from .device import get_policy

Anchoring = Literal["centered", "origin", "endpoint"]


@dataclasses.dataclass
class Plane:
    """An affine 2-plane with an orthonormal basis."""

    center: torch.Tensor          # (N,) the affine origin c
    basis: torch.Tensor           # (K, N) orthonormal rows
    singular_values: torch.Tensor  # (T,) of the centred displacement matrix
    anchoring: Anchoring
    captured_variance: float       # rho_K
    orthonormality_error: float    # max |V V^T - I|
    build_seconds: float
    gram_dtype: str

    @property
    def dim(self) -> int:
        return int(self.basis.shape[0])

    @property
    def numel(self) -> int:
        return int(self.basis.shape[1])

    def to(self, device: torch.device | str) -> "Plane":
        return dataclasses.replace(
            self, center=self.center.to(device), basis=self.basis.to(device),
            singular_values=self.singular_values.to(device),
        )

    def point(self, coords: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        r"""Map plane coordinates to a parameter vector, :math:`c + \sum_i \alpha_i v_i`."""
        return kernels.plane_point(self.center, self.basis, coords, out=out)

    def project(self, theta: torch.Tensor) -> torch.Tensor:
        r"""Map a parameter vector to its plane coordinates, :math:`V(\theta - c)`."""
        return kernels.project(theta - self.center, self.basis)

    def project_direction(self, v: torch.Tensor) -> torch.Tensor:
        """Project a direction (gradient, update) with no centring."""
        return kernels.project(v, self.basis)

    def spectrum_report(self) -> dict:
        s2 = (self.singular_values.double() ** 2)
        total = float(s2.sum())
        cum = (s2.cumsum(0) / max(total, 1e-30)).tolist()
        return {
            "singular_values": self.singular_values.tolist(),
            "captured_variance_by_rank": [round(c, 6) for c in cum[: min(10, len(cum))]],
            "rho_2": round(cum[1], 6) if len(cum) > 1 else 1.0,
            "effective_rank": round(float(torch.exp(_entropy(s2))), 3),
        }


def _entropy(s2: torch.Tensor) -> torch.Tensor:
    p = s2 / s2.sum().clamp(min=1e-30)
    p = p[p > 0]
    return -(p * p.log()).sum()


# ---------------------------------------------------------------------------
# Gram accumulation
# ---------------------------------------------------------------------------


def auto_chunk(T: int, N: int, device: torch.device | str, budget_bytes: int | None = None,
               min_chunk: int = 1 << 14) -> int:
    """Columns per streamed block, sized so ``T x chunk`` fp32 fits in a memory budget.

    The block is ``T`` rows deep, so a chunk width chosen without reference to the
    trajectory length silently scales the working set with the number of snapshots --
    the difference between 130 MB and 13 GB for the same nominal setting.
    """
    dev = torch.device(device)
    if budget_bytes is None:
        if dev.type == "cuda":
            free, _ = torch.cuda.mem_get_info(dev.index or 0)
            budget_bytes = int(free * 0.25)
        else:
            budget_bytes = 1 << 30
    # Two fp32 blocks live at once (the staged copy and the centred result).
    per_col = T * 4 * 2
    return int(max(min_chunk, min(N, budget_bytes // max(per_col, 1))))


def streaming_gram(
    trajectory: torch.Tensor,
    center: torch.Tensor,
    device: torch.device | str,
    chunk_elems: int | None = None,
    store_dtype: torch.dtype | None = None,
    backend: kernels.Backend = "auto",
) -> tuple[torch.Tensor, float, float]:
    r"""Accumulate :math:`G = DD^\top` for :math:`D = \Theta - \mathbf 1 c^\top`.

    ``trajectory`` is ``(T, N)`` and may live in host memory.  Columns are streamed to
    the device in chunks, centred, rescaled to unit RMS, and cast to ``store_dtype``.

    The rescaling is what makes half-precision storage safe: displacements have a wide
    dynamic range across layers, and fp16's 5-bit exponent would otherwise flush the
    small ones to zero.  After rescaling the whole chunk sits near 1, where fp16 has
    :math:`2^{-11}` relative resolution -- far below the Monte-Carlo noise on anything
    computed from it.  The scale factor is removed from ``G`` at the end.

    Returns ``(G, scale, seconds)``.
    """
    T, N = trajectory.shape
    policy = get_policy()
    store_dtype = store_dtype or policy.store_dtype
    dev = torch.device(device)
    if chunk_elems is None:
        chunk_elems = auto_chunk(T, N, dev)

    t0 = time.perf_counter()
    # Two passes: one for the scale (a single fp64 reduction), one for the Gram.
    # `sum(dtype=...)` accumulates in fp64 without materialising an fp64 copy.
    sq = torch.zeros((), dtype=torch.float64, device=dev)
    for lo in range(0, N, chunk_elems):
        hi = min(lo + chunk_elems, N)
        block = trajectory[:, lo:hi].to(dev, non_blocking=True).float()
        block -= center[lo:hi].to(dev).float()
        sq += block.pow_(2).sum(dtype=torch.float64)
        del block
    rms = float((sq / (T * N)).sqrt().clamp(min=1e-30))
    scale = 1.0 / rms

    G = torch.zeros(T, T, dtype=torch.float64, device=dev)
    for lo in range(0, N, chunk_elems):
        hi = min(lo + chunk_elems, N)
        block = trajectory[:, lo:hi].to(dev, non_blocking=True).float()
        block -= center[lo:hi].to(dev).float()
        block *= scale
        kernels.gram_chunk(block.to(store_dtype).contiguous(), G, backend=backend)
        del block
    G = kernels.symmetrise(G) / (scale * scale)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return G, scale, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Plane construction
# ---------------------------------------------------------------------------


def build_plane(
    trajectory: torch.Tensor,
    dim: int = 2,
    anchoring: Anchoring = "centered",
    device: torch.device | str = "cuda",
    chunk_elems: int | None = None,
    store_dtype: torch.dtype | None = None,
) -> Plane:
    r"""Construct the visualisation plane from trajectory snapshots.

    ``anchoring``
        ``"centered"``
            :math:`c=\bar\theta` with the top eigenvectors of the mean-centred Gram.
            Frobenius-optimal over all affine 2-planes (Eckart--Young after the
            least-squares reduction).
        ``"origin"``
            :math:`c=\theta_0` with the top eigenvectors of the uncentred Gram --
            the construction used in prior trajectory-PCA visualisations, kept as a
            baseline so the suboptimality can be measured.
        ``"endpoint"``
            :math:`v_1 \propto \theta_T-\theta_0`, :math:`v_2` the leading centred
            direction orthogonal to it.  Guarantees the endpoint is on-plane, which the
            optimal plane does not.
    """
    T, N = trajectory.shape
    if T < dim + 1:
        raise ValueError(f"need at least {dim + 1} snapshots, got {T}")
    dev = torch.device(device)
    policy = get_policy()
    store_dtype = store_dtype or (policy.store_dtype if dev.type == "cuda" else torch.float32)
    if chunk_elems is None:
        chunk_elems = auto_chunk(T, N, dev)

    t_start = time.perf_counter()
    if anchoring == "origin":
        center_cpu = trajectory[0].clone()
    else:
        center_cpu = trajectory.mean(0)

    G, _, gram_seconds = streaming_gram(
        trajectory, center_cpu, dev, chunk_elems=chunk_elems, store_dtype=store_dtype
    )

    # G is symmetric PSD; eigh is the numerically sound route to the singular vectors.
    evals, evecs = torch.linalg.eigh(G)
    order = torch.argsort(evals, descending=True)
    evals = evals[order].clamp(min=0)
    evecs = evecs[:, order]
    svals = evals.sqrt()

    keep = dim
    if anchoring == "endpoint":
        keep = dim - 1

    # v_i = D^T u_i / sigma_i, accumulated in column chunks so D is never materialised.
    basis = torch.empty(dim, N, dtype=torch.float32, device=dev)
    if keep > 0:
        u = evecs[:, :keep].to(dev).float()                      # (T, keep)
        inv_s = torch.where(svals[:keep] > 1e-12, 1.0 / svals[:keep], torch.zeros_like(svals[:keep]))
        coef = (u * inv_s.to(dev).float()[None, :]).t().contiguous()  # (keep, T)
        for lo in range(0, N, chunk_elems):
            hi = min(lo + chunk_elems, N)
            block = trajectory[:, lo:hi].to(dev, non_blocking=True).float()
            block -= center_cpu[lo:hi].to(dev).float()
            basis[dim - keep :, lo:hi] = coef @ block
            del block

    if anchoring == "endpoint":
        d = (trajectory[-1] - trajectory[0]).to(dev).float()
        d = d / d.norm().clamp(min=1e-30)
        basis[0] = d
        # Gram-Schmidt the remaining directions against the endpoint direction.
        for i in range(1, dim):
            basis[i] -= (basis[i] @ basis[0]) * basis[0]

    # Re-orthonormalise: the Gram route loses a few digits at small singular values, and
    # the certificate's geometry assumes an exactly orthonormal frame.
    q, _ = torch.linalg.qr(basis.t().double())
    basis = q.t().float().contiguous()
    # QR fixes the frame up to sign; align each vector with the pre-QR direction.
    orth_err = float((basis @ basis.t() - torch.eye(dim, device=dev)).abs().max())

    s2 = evals.double()
    captured = float(s2[:dim].sum() / s2.sum().clamp(min=1e-30))

    center = center_cpu.to(dev).float().contiguous()
    return Plane(
        center=center,
        basis=basis,
        singular_values=svals.float().cpu(),
        anchoring=anchoring,
        captured_variance=captured,
        orthonormality_error=orth_err,
        build_seconds=time.perf_counter() - t_start,
        gram_dtype=str(store_dtype).replace("torch.", ""),
    )


def project_trajectory(plane: Plane, trajectory: torch.Tensor, chunk_rows: int = 64) -> torch.Tensor:
    """Plane coordinates of every snapshot, ``(T, dim)``."""
    coords = []
    for lo in range(0, trajectory.shape[0], chunk_rows):
        block = trajectory[lo : lo + chunk_rows].to(plane.basis.device).float()
        block = block - plane.center[None, :]
        coords.append(block @ plane.basis.t())
    return torch.cat(coords, 0)


def residual_norms(plane: Plane, trajectory: torch.Tensor, chunk_rows: int = 64) -> torch.Tensor:
    r"""Out-of-plane residual :math:`\|(I-P)(\theta_t-c)\|` for every snapshot.

    This is the *parameter-space* miss.  It bounds, but does not equal, the loss-space
    miss that actually determines whether the picture is trustworthy.
    """
    out = []
    for lo in range(0, trajectory.shape[0], chunk_rows):
        block = trajectory[lo : lo + chunk_rows].to(plane.basis.device).float()
        block = block - plane.center[None, :]
        c = block @ plane.basis.t()
        recon = c @ plane.basis
        out.append((block - recon).norm(dim=1))
    return torch.cat(out, 0)


__all__ = ["Plane", "build_plane", "streaming_gram", "auto_chunk",
           "project_trajectory", "residual_norms"]
