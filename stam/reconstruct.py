r"""Reconstructing the surface from probes, and the baselines it is measured against.

Given probes :math:`\{(\alpha_i, \hat\ell_i, \hat g_i, \hat H_i)\}_{i=1}^n` on a plane,
build an estimate :math:`\mathcal R` of the restricted loss :math:`\ell` over a render
domain, together with its gradient field.

**The method: partition-of-unity Taylor patches.**  Attach to each anchor its
second-order Taylor model

.. math::
   Q_i(\alpha) = \hat\ell_i + \hat g_i^\top(\alpha-\alpha_i)
                 + \tfrac12 (\alpha-\alpha_i)^\top \hat H_i (\alpha-\alpha_i),

and blend them with compactly supported Wendland :math:`C^2` weights normalised to a
partition of unity, :math:`\mathcal R = \sum_i w_i Q_i / \sum_i w_i`.  Two properties
matter:

*Local accuracy transfers to global accuracy.*  If each patch satisfies
:math:`|\ell-Q_i|\le\varepsilon` on its support then, because the weights are
non-negative and sum to one,

.. math::
   |\ell - \mathcal R| = \Bigl|\sum_i \hat w_i (\ell - Q_i)\Bigr| \le \max_i \varepsilon_i ,

with no constant lost in the blend.  Taylor's theorem supplies
:math:`\varepsilon_i \le \tfrac16 M_3 \rho_i^3` for a third-derivative bound
:math:`M_3`, giving the cubic order that drives the rate analysis.

*Quadratics are reproduced exactly.*  If :math:`\ell` is a quadratic then every
:math:`Q_i=\ell`, hence :math:`\mathcal R=\ell` identically -- for any anchor placement
and any radii.  Scattered-data interpolants of *values* have no such guarantee at
:math:`n=6`, and none at all when the values are noisy.

**Baselines.**  Four, spanning what the literature actually does:
``dense`` (plot the noisy grid -- current practice), ``rbf`` (multiquadric
interpolation of values), ``lstsq2`` (local quadratic regression on values -- the
strongest value-only competitor, and the one the minimax theory says is rate-optimal),
and ``pu-taylor`` at orders 0/1/2.
"""

from __future__ import annotations

import dataclasses
from typing import Literal, Sequence

import numpy as np
import torch

from . import kernels
from .probe import ProbeResult

Method = Literal["pu-taylor", "pu-taylor-1", "pu-taylor-0", "rbf", "lstsq2", "dense", "nearest"]


@dataclasses.dataclass
class Surface:
    """A reconstructed surface sampled on a query set."""

    queries: np.ndarray        # (m, 2)
    values: np.ndarray         # (m,)
    grads: np.ndarray | None   # (m, 2)
    coverage: np.ndarray       # (m,) partition-of-unity weight sum; 0 = outside support
    method: str
    n_anchors: int
    detail: dict

    @property
    def covered(self) -> np.ndarray:
        return np.isfinite(self.values)

    def as_grid(self, shape: tuple[int, int]) -> np.ndarray:
        return self.values.reshape(shape)

    def grad_as_grid(self, shape: tuple[int, int]) -> np.ndarray:
        assert self.grads is not None
        return self.grads.reshape(*shape, 2)


@dataclasses.dataclass
class AnchorSet:
    """Probe results in the array form the reconstructors consume."""

    coords: np.ndarray         # (n, 2)
    values: np.ndarray         # (n,)
    value_var: np.ndarray      # (n,)
    grads: np.ndarray          # (n, 2)   zeros if unmeasured
    grad_var: np.ndarray       # (n, 2)
    hess: np.ndarray           # (n, 3)   packed xx, xy, yy; zeros if unmeasured
    hess_var: np.ndarray       # (n, 3)
    order: int
    total_cost: float
    n_examples: int

    @staticmethod
    def from_probes(probes: Sequence[ProbeResult]) -> "AnchorSet":
        n = len(probes)
        K = probes[0].coords.shape[0]
        coords = np.stack([p.coords for p in probes])
        values = np.array([p.value for p in probes])
        value_var = np.array([p.value_var for p in probes])
        grads = np.zeros((n, K))
        grad_var = np.zeros((n, K))
        hess = np.zeros((n, 3))
        hess_var = np.zeros((n, 3))
        for i, p in enumerate(probes):
            if p.grad is not None:
                grads[i] = p.grad
                grad_var[i] = np.nan_to_num(p.grad_var, nan=0.0)
            if p.hess is not None:
                hess[i] = p.hess_packed()
                hv = np.nan_to_num(p.hess_var, nan=0.0)
                hess_var[i] = [hv[0, 0], hv[0, 1], hv[1, 1]]
        return AnchorSet(
            coords=coords, values=values, value_var=value_var, grads=grads, grad_var=grad_var,
            hess=hess, hess_var=hess_var, order=min(p.order for p in probes),
            total_cost=float(sum(p.cost_units for p in probes)),
            n_examples=int(sum(p.n_examples for p in probes)),
        )


# ---------------------------------------------------------------------------
# Support radii
# ---------------------------------------------------------------------------


def support_radii(coords: np.ndarray, overlap: float = 1.6, k: int = 4) -> np.ndarray:
    r"""Per-anchor support radius.

    Set :math:`\rho_i` to ``overlap`` times the mean distance to the ``k`` nearest other
    anchors.  Two competing requirements fix the constant: the supports must cover the
    domain (else the reconstruction has holes), and they must stay small (the Taylor
    error grows as :math:`\rho^3`).  ``overlap`` slightly above the nearest-neighbour
    distance is the smallest choice that guarantees coverage for a quasi-uniform design;
    the default is calibrated for the Halton designs used here and checked at run time by
    the coverage field returned with every surface.
    """
    n = coords.shape[0]
    if n == 1:
        return np.array([float("inf")])
    kk = min(k, n - 1)
    if n > 2048:
        # A dense distance matrix is O(n^2) memory; a KD-tree gives the same k nearest
        # neighbours in O(n log n) and matters once the baselines reach n ~ 10^4.
        from scipy.spatial import cKDTree

        d, _ = cKDTree(coords).query(coords, k=kk + 1)
        near = d[:, 1:].mean(axis=1)
    else:
        d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        near = np.sort(d, axis=1)[:, :kk].mean(axis=1)
    return overlap * near


# ---------------------------------------------------------------------------
# Reconstructors
# ---------------------------------------------------------------------------


def _to_device(a: np.ndarray, device: torch.device | str) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=device)


def pu_taylor_surface(
    anchors: AnchorSet,
    queries: np.ndarray,
    order: int = 2,
    overlap: float = 1.6,
    device: torch.device | str = "cuda",
    radii: np.ndarray | None = None,
) -> Surface:
    """Partition-of-unity Taylor reconstruction of the requested order."""
    if radii is None:
        radii = support_radii(anchors.coords, overlap=overlap)
    v, g, w = kernels.pu_taylor(
        _to_device(anchors.coords, device),
        _to_device(radii, device),
        _to_device(anchors.values, device),
        _to_device(anchors.grads, device),
        _to_device(anchors.hess, device),
        _to_device(queries, device),
        order=order,
    )
    return Surface(
        queries=queries, values=v.double().cpu().numpy(), grads=g.double().cpu().numpy(),
        coverage=w.double().cpu().numpy(), method=f"pu-taylor-{order}",
        n_anchors=anchors.coords.shape[0],
        detail={"overlap": overlap, "mean_radius": float(np.mean(radii))},
    )


def rbf_surface(
    anchors: AnchorSet,
    queries: np.ndarray,
    kernel: str = "multiquadric",
    smoothing: float = 0.0,
    epsilon: float | None = None,
) -> Surface:
    """Radial-basis interpolation of the *values only*.

    Included because it is what prior loss-landscape tooling uses to upsample a sparse
    grid.  With ``smoothing=0`` it interpolates the noisy anchor values exactly, which
    is precisely the property that gives it an error floor.
    """
    from scipy.interpolate import RBFInterpolator

    n = anchors.coords.shape[0]
    if epsilon is None:
        d = np.linalg.norm(anchors.coords[:, None] - anchors.coords[None], axis=-1)
        np.fill_diagonal(d, np.inf)
        epsilon = 1.0 / max(float(np.median(d.min(axis=1))), 1e-9)
    interp = RBFInterpolator(
        anchors.coords, anchors.values, kernel=kernel, epsilon=epsilon,
        smoothing=smoothing, degree=1,
    )
    values = interp(queries)
    # Gradient by central differences on the interpolant: RBF gives no gradient field,
    # which is itself part of why value-only pipelines draw quiver plots that are not
    # the gradient of the surface they are drawn on.
    h = 1e-3 * max(np.ptp(anchors.coords), 1e-9)
    gx = (interp(queries + [h, 0]) - interp(queries - [h, 0])) / (2 * h)
    gy = (interp(queries + [0, h]) - interp(queries - [0, h])) / (2 * h)
    return Surface(
        queries=queries, values=values, grads=np.stack([gx, gy], -1),
        coverage=np.ones(queries.shape[0]), method=f"rbf-{kernel}", n_anchors=n,
        detail={"epsilon": epsilon, "smoothing": smoothing},
    )


def local_quadratic_surface(
    anchors: AnchorSet,
    queries: np.ndarray,
    bandwidth: float | None = None,
    ridge: float = 1e-8,
    min_points: int = 6,
) -> Surface:
    """Locally weighted quadratic regression on values only.

    This is the strongest value-only competitor and the one classical nonparametric
    theory identifies as rate-optimal for a :math:`C^3` function observed with noise:
    the smoothing it performs removes the interpolation floor that ``dense`` and ``rbf``
    suffer from.  It needs at least six anchors inside every window to determine the six
    quadratic coefficients, which forces a wider window -- and hence a larger bias
    constant -- than the Taylor patches need.
    """
    n = anchors.coords.shape[0]
    if bandwidth is None:
        r = support_radii(anchors.coords, overlap=1.0, k=min(min_points, max(n - 1, 1)))
        bandwidth = float(np.mean(r) * 2.2)

    A = anchors.coords
    z = anchors.values
    out = np.empty(queries.shape[0])
    grad = np.empty((queries.shape[0], 2))
    for j, q in enumerate(queries):
        d = np.linalg.norm(A - q, axis=1)
        h = bandwidth
        # Grow the window until the design is determined; this is the structural cost of
        # not having derivative observations.
        for _ in range(24):
            m = d < h
            if m.sum() >= min_points:
                break
            h *= 1.3
        idx = np.where(m)[0]
        if idx.size < 3:
            idx = np.argsort(d)[:min_points]
            h = max(d[idx].max(), 1e-12)
        dx = A[idx, 0] - q[0]
        dy = A[idx, 1] - q[1]
        u = np.clip(d[idx] / h, 0, 1)
        w = (1 - u) ** 4 * (4 * u + 1)
        X = np.stack([np.ones_like(dx), dx, dy, 0.5 * dx * dx, dx * dy, 0.5 * dy * dy], 1)
        if idx.size < 6:
            X = X[:, : idx.size]
        W = w[:, None]
        XtW = (X * W).T
        M = XtW @ X + ridge * np.eye(X.shape[1])
        try:
            beta = np.linalg.solve(M, XtW @ z[idx])
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(M, XtW @ z[idx], rcond=None)[0]
        out[j] = beta[0]
        grad[j] = [beta[1] if len(beta) > 1 else 0.0, beta[2] if len(beta) > 2 else 0.0]
    return Surface(
        queries=queries, values=out, grads=grad, coverage=np.ones(queries.shape[0]),
        method="lstsq2", n_anchors=n, detail={"bandwidth": bandwidth},
    )


def dense_surface(anchors: AnchorSet, queries: np.ndarray, grid_shape: tuple[int, int] | None
                  = None) -> Surface:
    """Nearest-anchor lookup: the raw measured grid, drawn as-is.

    This models what a dense-grid landscape plot actually shows.  It reproduces every
    anchor value exactly, noise included, which is the defining property behind the
    error floor.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(anchors.coords)
    _, idx = tree.query(queries)
    return Surface(
        queries=queries, values=anchors.values[idx], grads=anchors.grads[idx],
        coverage=np.ones(queries.shape[0]), method="dense", n_anchors=anchors.coords.shape[0],
        detail={"grid_shape": grid_shape},
    )


def bilinear_surface(anchors: AnchorSet, queries: np.ndarray, grid_shape: tuple[int, int]
                     ) -> Surface:
    """Bilinear interpolation of a regular grid of noisy values -- what ``contourf``
    draws.  Interpolating, hence subject to the same floor as ``dense``."""
    from scipy.interpolate import RegularGridInterpolator

    ny, nx = grid_shape
    xs = np.unique(anchors.coords[:, 0])
    ys = np.unique(anchors.coords[:, 1])
    order = np.lexsort((anchors.coords[:, 0], anchors.coords[:, 1]))
    z = anchors.values[order].reshape(ny, nx)
    interp = RegularGridInterpolator(
        (ys, xs), z, method="linear", bounds_error=False, fill_value=None
    )
    values = interp(np.stack([queries[:, 1], queries[:, 0]], -1))
    gy, gx = np.gradient(z, ys, xs)
    gi_x = RegularGridInterpolator((ys, xs), gx, bounds_error=False, fill_value=None)
    gi_y = RegularGridInterpolator((ys, xs), gy, bounds_error=False, fill_value=None)
    qq = np.stack([queries[:, 1], queries[:, 0]], -1)
    return Surface(
        queries=queries, values=values, grads=np.stack([gi_x(qq), gi_y(qq)], -1),
        coverage=np.ones(queries.shape[0]), method="bilinear",
        n_anchors=anchors.coords.shape[0], detail={"grid_shape": grid_shape},
    )


def reconstruct(
    method: Method | str,
    anchors: AnchorSet,
    queries: np.ndarray,
    device: torch.device | str = "cuda",
    grid_shape: tuple[int, int] | None = None,
    **kw,
) -> Surface:
    """Dispatch to a named reconstruction method."""
    if method in ("pu-taylor", "pu-taylor-2"):
        return pu_taylor_surface(anchors, queries, order=2, device=device, **kw)
    if method == "pu-taylor-1":
        return pu_taylor_surface(anchors, queries, order=1, device=device, **kw)
    if method == "pu-taylor-0":
        return pu_taylor_surface(anchors, queries, order=0, device=device, **kw)
    if method.startswith("rbf"):
        return rbf_surface(anchors, queries, **kw)
    if method == "lstsq2":
        return local_quadratic_surface(anchors, queries, **kw)
    if method == "dense":
        return dense_surface(anchors, queries, grid_shape)
    if method == "bilinear":
        if grid_shape is None:
            raise ValueError("bilinear requires grid_shape")
        return bilinear_surface(anchors, queries, grid_shape)
    raise ValueError(f"unknown reconstruction method {method!r}")


__all__ = [
    "AnchorSet", "Surface", "support_radii", "pu_taylor_surface", "rbf_surface",
    "local_quadratic_surface", "dense_surface", "bilinear_surface", "reconstruct",
]
