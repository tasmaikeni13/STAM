r"""Error metrics against an exact reference.

Three families, because a landscape picture is read for three different things.

*The surface itself* -- normalised RMS and sup error against the exact grid.  Errors are
reported relative to the **relief** of the true surface, :math:`\max\ell-\min\ell` over
the domain, because that is the scale a reader judges a contour plot on.  An absolute
error of 0.05 nats is negligible on a surface spanning 4 nats and meaningless on one
spanning 0.05.

*The gradient field* -- the quiver plot, and the thing that purports to explain why the
optimiser moved as it did.

*The curvature* -- "sharpness", the quantity most often used to draw conclusions from
these pictures.  It is estimated here the way a reader would have to estimate it: by
differencing the *rendered* surface, uniformly across methods, so no method is credited
with derivative information it did not put into the picture.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np


@dataclasses.dataclass
class SurfaceError:
    rmse: float
    rmse_relative: float
    max_abs: float
    max_relative: float
    q90: float
    q95: float
    bias: float
    coverage: float
    reference_range: float

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def surface_error(pred: np.ndarray, ref: np.ndarray) -> SurfaceError:
    pred = np.asarray(pred, dtype=np.float64).ravel()
    ref = np.asarray(ref, dtype=np.float64).ravel()
    ok = np.isfinite(pred) & np.isfinite(ref)
    rng = float(np.nanmax(ref) - np.nanmin(ref))
    rng = max(rng, 1e-12)
    if ok.sum() == 0:
        return SurfaceError(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.0, rng)
    d = pred[ok] - ref[ok]
    ad = np.abs(d)
    rmse = float(np.sqrt(np.mean(d * d)))
    mx = float(ad.max())
    return SurfaceError(
        rmse=rmse, rmse_relative=rmse / rng, max_abs=mx, max_relative=mx / rng,
        q90=float(np.percentile(ad, 90)), q95=float(np.percentile(ad, 95)),
        bias=float(np.mean(d)), coverage=float(ok.mean()), reference_range=rng,
    )


def vector_field_error(pred: np.ndarray, ref: np.ndarray) -> dict:
    """RMS error of a 2-D field, plus the angular error of its direction.

    Direction is separated from magnitude because a quiver plot is read mostly for
    direction: a field with the right arrows and the wrong lengths still tells the
    correct story about where the optimiser is pushed.
    """
    pred = np.asarray(pred, dtype=np.float64).reshape(-1, 2)
    ref = np.asarray(ref, dtype=np.float64).reshape(-1, 2)
    ok = np.isfinite(pred).all(1) & np.isfinite(ref).all(1)
    if ok.sum() == 0:
        return {"rmse": np.nan, "rmse_relative": np.nan, "angle_deg": np.nan, "coverage": 0.0}
    p, r = pred[ok], ref[ok]
    scale = float(np.sqrt(np.mean((r * r).sum(1))))
    rmse = float(np.sqrt(np.mean(((p - r) ** 2).sum(1))))
    pn = np.linalg.norm(p, axis=1)
    rn = np.linalg.norm(r, axis=1)
    m = (pn > 1e-12) & (rn > 1e-12)
    cos = np.clip((p[m] * r[m]).sum(1) / (pn[m] * rn[m]), -1, 1)
    return {
        "rmse": rmse,
        "rmse_relative": rmse / max(scale, 1e-12),
        "angle_deg": float(np.degrees(np.arccos(cos)).mean()) if m.any() else np.nan,
        "reference_scale": scale,
        "coverage": float(ok.mean()),
    }


def hessian_by_differencing(
    grad_at: Callable[[np.ndarray], np.ndarray],
    points: np.ndarray,
    delta: float,
) -> np.ndarray:
    r"""Symmetric Hessian field from central differences of a rendered gradient field.

    Applied identically to every method, so the comparison measures what a reader could
    extract from the picture rather than what the method happened to compute internally.
    Returns ``(m, 3)`` packed as ``(xx, xy, yy)``.
    """
    ex = np.array([delta, 0.0])
    ey = np.array([0.0, delta])
    gxp = grad_at(points + ex)
    gxm = grad_at(points - ex)
    gyp = grad_at(points + ey)
    gym = grad_at(points - ey)
    hxx = (gxp[:, 0] - gxm[:, 0]) / (2 * delta)
    hyy = (gyp[:, 1] - gym[:, 1]) / (2 * delta)
    # Average the two routes to the mixed partial; they differ only through error.
    hxy = 0.5 * ((gxp[:, 1] - gxm[:, 1]) / (2 * delta) + (gyp[:, 0] - gym[:, 0]) / (2 * delta))
    return np.stack([hxx, hxy, hyy], axis=1)


def sharpness(hess_packed: np.ndarray) -> np.ndarray:
    r"""Largest eigenvalue of each packed :math:`2\times2` Hessian.

    For :math:`\begin{pmatrix}a&b\\b&c\end{pmatrix}` the eigenvalues are
    :math:`\tfrac{a+c}2 \pm \sqrt{\bigl(\tfrac{a-c}2\bigr)^2 + b^2}`.
    """
    h = np.asarray(hess_packed, dtype=np.float64).reshape(-1, 3)
    a, b, c = h[:, 0], h[:, 1], h[:, 2]
    mean = 0.5 * (a + c)
    disc = np.sqrt(np.maximum(((a - c) / 2) ** 2 + b * b, 0.0))
    return mean + disc


def curvature_error(pred_hess: np.ndarray, ref_hess: np.ndarray) -> dict:
    """Frobenius error of the Hessian field and the error of the sharpness read off it."""
    p = np.asarray(pred_hess, dtype=np.float64).reshape(-1, 3)
    r = np.asarray(ref_hess, dtype=np.float64).reshape(-1, 3)
    ok = np.isfinite(p).all(1) & np.isfinite(r).all(1)
    if ok.sum() == 0:
        return {"rmse": np.nan, "sharpness_rmse": np.nan, "coverage": 0.0}
    # Frobenius norm of a symmetric 2x2 from packed form weights the off-diagonal twice.
    w = np.array([1.0, 2.0, 1.0])
    d2 = (((p[ok] - r[ok]) ** 2) * w).sum(1)
    ref2 = ((r[ok] ** 2) * w).sum(1)
    sp, sr = sharpness(p[ok]), sharpness(r[ok])
    return {
        "rmse": float(np.sqrt(np.mean(d2))),
        "rmse_relative": float(np.sqrt(np.mean(d2) / max(np.mean(ref2), 1e-30))),
        "sharpness_rmse": float(np.sqrt(np.mean((sp - sr) ** 2))),
        "sharpness_relative": float(
            np.sqrt(np.mean((sp - sr) ** 2)) / max(np.sqrt(np.mean(sr**2)), 1e-30)
        ),
        "sharpness_bias": float(np.mean(sp - sr)),
        "reference_sharpness_mean": float(np.mean(sr)),
        "coverage": float(ok.mean()),
    }


def fit_rate(budgets: np.ndarray, errors: np.ndarray) -> dict:
    r"""Fit :math:`E \propto C^{-p}` by least squares in log-log, returning ``p``.

    The exponent, not the constant, is what the theory predicts, and it is what
    distinguishes a method that converges from one that plateaus.
    """
    b = np.asarray(budgets, dtype=np.float64)
    e = np.asarray(errors, dtype=np.float64)
    ok = np.isfinite(b) & np.isfinite(e) & (b > 0) & (e > 0)
    if ok.sum() < 3:
        return {"exponent": np.nan, "intercept": np.nan, "r2": np.nan, "n": int(ok.sum())}
    x = np.log(b[ok])
    y = np.log(e[ok])
    A = np.stack([np.ones_like(x), x], 1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "exponent": float(-coef[1]),
        "intercept": float(coef[0]),
        "r2": float(1 - (resid**2).sum() / max(ss_tot, 1e-30)),
        "n": int(ok.sum()),
    }


__all__ = [
    "SurfaceError", "surface_error", "vector_field_error", "hessian_by_differencing",
    "sharpness", "curvature_error", "fit_rate",
]
