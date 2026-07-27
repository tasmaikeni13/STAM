r"""Error certificates: turning an assumption into a measurement.

Every bound on a Taylor-based reconstruction has the form
:math:`\varepsilon \le \tfrac16 M_3 \rho^3`, and :math:`M_3` -- a bound on the third
derivative of the restricted loss -- is not knowable a priori.  For a ReLU network it is
not even finite in the classical sense: the loss is piecewise smooth, and a plane
generically cuts the kink set in curves across which the second derivative jumps.

Rather than assert a constant, STAM *measures the error itself*.

**The estimator.**  Draw :math:`m` locations :math:`q_j` uniformly from the render
domain, independently of the anchor design, and probe the true loss there:
:math:`\hat\ell_j = \ell(q_j) + \xi_j` with :math:`\mathbb E\xi_j = 0` and
:math:`\mathrm{Var}(\xi_j)=v_j` estimated from the same draw.  With the reconstruction
:math:`\mathcal R` held fixed, the residual :math:`d_j = \hat\ell_j - \mathcal R(q_j)`
satisfies

.. math::
   \mathbb E\bigl[d_j^2 - \hat v_j \,\big|\, \mathcal R\bigr] = \bigl(\ell(q_j)-\mathcal R(q_j)\bigr)^2 ,

so the variance-corrected mean :math:`\widehat{\mathrm{MSE}} = \frac1m\sum_j (d_j^2-\hat
v_j)` is unbiased for the mean squared reconstruction error over the domain.  Without the
correction the certificate would charge the reconstruction for the *probe's* noise and
could never report an error below the noise floor.

**The confidence bound** is empirical Bernstein (Maurer--Pontil): distribution-free,
finite-sample, and using the observed variance rather than an assumed range.

**The sup-norm statement** is an order statistic: with :math:`m` uniform draws, at least
a :math:`1-p` fraction of the domain has error below the observed maximum, with
confidence :math:`1-\delta`, whenever :math:`(1-p)^m\le\delta`.  Claiming a true supremum
from finitely many samples would not be honest, and this is the strongest thing the data
supports.

Nothing here assumes smoothness, a noise distribution, or a correct model.  A
reconstruction that is badly wrong produces a large certified error; that is the point.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Callable

import numpy as np
import torch

from .probe import PlaneProbe, ProbeResult
from .reconstruct import AnchorSet, Surface


@dataclasses.dataclass
class Certificate:
    """A measured statement about the error of a rendered surface."""

    rmse: float                   # variance-corrected RMS error
    rmse_upper: float             # (1-delta) upper confidence bound on the RMS error
    rmse_raw: float               # uncorrected, for comparison
    noise_floor: float            # RMS probe noise; the correction that was subtracted
    max_abs_residual: float
    q90: float                    # empirical 90th percentile of |residual|
    q95: float                    # empirical 95th percentile of |residual|
    q95_upper: float              # distribution-free upper confidence bound on the 95th pct
    quantile_level: float         # the 1-p in "1-p of the domain is below max_abs_residual"
    confidence: float             # 1 - delta
    n_probes: int
    examples_per_probe: int
    probe_cost: float             # example-forward-equivalents spent on certification
    coverage: float               # fraction of probe points inside the reconstruction support
    relative_rmse: float          # rmse / range of the reconstructed surface
    detail: dict

    def to_dict(self) -> dict:
        return {
            "rmse": self.rmse, "rmse_upper": self.rmse_upper, "rmse_raw": self.rmse_raw,
            "noise_floor": self.noise_floor, "max_abs_residual": self.max_abs_residual,
            "q90": self.q90, "q95": self.q95, "q95_upper": self.q95_upper,
            "quantile_level": self.quantile_level, "confidence": self.confidence,
            "n_probes": self.n_probes, "examples_per_probe": self.examples_per_probe,
            "probe_cost": self.probe_cost, "coverage": self.coverage,
            "relative_rmse": self.relative_rmse, **self.detail,
        }

    def summary(self) -> str:
        return (
            f"95% of the domain within {self.q95_upper:.4g} (w.p. {self.confidence:.0%}); "
            f"RMSE {self.rmse:.4g}; noise floor {self.noise_floor:.4g}; "
            f"{self.n_probes} probes x {self.examples_per_probe} examples"
        )


def empirical_bernstein_upper(x: np.ndarray, delta: float = 0.05) -> float:
    r"""Distribution-free upper confidence bound on :math:`\mathbb E[x]`.

    Maurer--Pontil: with probability at least :math:`1-\delta`,

    .. math::
       \mathbb E[x] \le \bar x + \sqrt{\frac{2 V_m \ln(2/\delta)}{m}}
                          + \frac{7R\ln(2/\delta)}{3(m-1)} ,

    where :math:`V_m` is the sample variance and :math:`R` the observed range.  Using the
    observed variance rather than a worst-case range makes the bound usable at the
    :math:`m\sim10^2` sample sizes a certification budget affords.
    """
    m = x.size
    if m < 2:
        return float("inf")
    mean = float(np.mean(x))
    var = float(np.var(x, ddof=1))
    rng = float(np.ptp(x)) if m > 1 else 0.0
    ln = math.log(2.0 / delta)
    return mean + math.sqrt(2.0 * var * ln / m) + 7.0 * rng * ln / (3.0 * (m - 1))


def order_statistic_bound(x: np.ndarray, level: float = 0.95, delta: float = 0.05) -> float:
    r"""Distribution-free upper confidence bound on the ``level`` quantile of ``|x|``.

    With :math:`m` i.i.d.\ draws, the :math:`k`-th order statistic exceeds the true
    ``level`` quantile with probability :math:`\Pr[\mathrm{Bin}(m,\text{level}) < k]`.
    Choosing the smallest :math:`k` for which that probability is at least
    :math:`1-\delta` gives a bound that holds whatever the distribution -- which is what
    a heavy-tailed error field requires, since a mean-based bound built on the observed
    range can be defeated by a tail the sample missed.
    """
    from scipy.stats import binom

    a = np.sort(np.abs(np.asarray(x, dtype=float)))
    m = a.size
    if m == 0:
        return float("inf")
    for k in range(1, m + 1):
        if binom.cdf(k - 1, m, level) >= 1 - delta:
            return float(a[k - 1])
    return float(a[-1])


def quantile_confidence(m: int, delta: float = 0.05) -> float:
    r"""Largest :math:`1-p` such that :math:`(1-p)^m \le \delta`.

    With :math:`m` uniform draws all below a threshold, the probability that a
    :math:`p`-fraction of the domain exceeds it is at most :math:`(1-p)^m`.
    """
    if m <= 0:
        return 0.0
    return float(delta ** (1.0 / m))


def stratified_points(
    n: int, domain: tuple[tuple[float, float], tuple[float, float]], rng: np.random.Generator
) -> np.ndarray:
    r"""One uniform draw inside each cell of a near-square partition of the domain.

    Reconstruction error on a loss surface is strongly spatially structured and heavily
    tailed -- on the CNN reference, ~90% of the mean squared error comes from ~1% of the
    domain, in the steep outer region. A plain uniform sample of the few dozen points a
    certification budget affords will usually miss that region entirely and report an
    optimistic mean.

    Stratification fixes the marginal count per cell while leaving the draw within each
    cell uniform, so the estimator stays unbiased for the domain mean and its variance
    falls to the within-cell component. It costs nothing.
    """
    (x0, x1), (y0, y1) = domain
    side = max(int(round(math.sqrt(n))), 1)
    ex = np.linspace(x0, x1, side + 1)
    ey = np.linspace(y0, y1, side + 1)
    pts = []
    for i in range(side):
        for j in range(side):
            pts.append([rng.uniform(ex[i], ex[i + 1]), rng.uniform(ey[j], ey[j + 1])])
    return np.asarray(pts)


def certify_surface(
    probe: PlaneProbe,
    reconstruct_at: Callable[[np.ndarray], Surface],
    domain: tuple[tuple[float, float], tuple[float, float]],
    n_probes: int = 49,
    examples_per_probe: int = 512,
    seed: int = 991,
    delta: float = 0.05,
    surface_range: float | None = None,
    stratified: bool = True,
) -> Certificate:
    r"""Certify a reconstruction against fresh, independent probes.

    ``reconstruct_at(queries)`` must evaluate the *already-built* reconstruction; it must
    not be refitted using the certification probes, or the estimate becomes an in-sample
    residual and loses its meaning.
    """
    rng = np.random.default_rng(seed)
    (x0, x1), (y0, y1) = domain
    if stratified:
        q = stratified_points(n_probes, domain, rng)
        n_probes = q.shape[0]
    else:
        q = np.stack(
            [rng.uniform(x0, x1, n_probes), rng.uniform(y0, y1, n_probes)], axis=1
        )

    gen = torch.Generator(device="cpu").manual_seed(seed)
    truths, variances, cost = [], [], 0.0
    for point in q:
        r: ProbeResult = probe.probe(point, examples_per_probe, order=0, generator=gen)
        truths.append(r.value)
        variances.append(r.value_var)
        cost += r.cost_units
    truths = np.asarray(truths)
    variances = np.nan_to_num(np.asarray(variances), nan=0.0)

    surf = reconstruct_at(q)
    ok = np.isfinite(surf.values)
    coverage = float(ok.mean())
    if ok.sum() < 2:
        raise RuntimeError("reconstruction covers too few certification points")

    d = truths[ok] - surf.values[ok]
    v = variances[ok]
    corrected = d * d - v
    mse = float(np.mean(corrected))
    mse_upper = empirical_bernstein_upper(corrected, delta)

    rng_surface = surface_range
    if rng_surface is None:
        finite = surf.values[ok]
        rng_surface = float(np.ptp(finite)) if finite.size else 1.0
    rng_surface = max(rng_surface, 1e-12)

    rmse = math.sqrt(max(mse, 0.0))
    ad = np.abs(d)
    return Certificate(
        rmse=rmse,
        rmse_upper=math.sqrt(max(mse_upper, 0.0)),
        rmse_raw=float(np.sqrt(np.mean(d * d))),
        noise_floor=float(np.sqrt(np.mean(v))),
        max_abs_residual=float(np.max(ad)),
        q90=float(np.percentile(ad, 90)),
        q95=float(np.percentile(ad, 95)),
        q95_upper=order_statistic_bound(ad, level=0.95, delta=delta),
        quantile_level=quantile_confidence(int(ok.sum()), delta),
        confidence=1.0 - delta,
        n_probes=n_probes,
        examples_per_probe=examples_per_probe,
        probe_cost=cost,
        coverage=coverage,
        relative_rmse=rmse / rng_surface,
        detail={
            "mse_point_estimate": mse,
            "mse_upper": mse_upper,
            "residual_mean": float(np.mean(d)),
            "surface_range": rng_surface,
            "stratified": bool(stratified),
            # Share of the estimated MSE contributed by the single worst probe.  On a
            # heavy-tailed error field this is large, and it is the signal that the
            # point estimate should be read together with the upper bound rather than
            # on its own.
            "worst_probe_share": float(
                np.max(corrected) / max(np.sum(np.clip(corrected, 0, None)), 1e-30)
            ),
        },
    )


# ---------------------------------------------------------------------------
# A-priori constants, measured from the anchors themselves
# ---------------------------------------------------------------------------


def estimate_m3(anchors: AnchorSet, k: int = 6) -> dict[str, float]:
    r"""Estimate a third-derivative scale from measured Hessians.

    For :math:`\ell\in C^3`, :math:`\|\nabla^2\ell(a)-\nabla^2\ell(b)\| \le M_3\|a-b\|`.
    Every anchor pair therefore gives a lower bound on :math:`M_3`, and the largest over
    nearby pairs is a usable estimate.  Only *nearby* pairs are used: distant pairs
    under-report because the bound is loose, and the quantity wanted is the local
    modulus that governs the patch error.

    The measured Hessians are noisy, so a pair separated by :math:`h` yields a spurious
    contribution :math:`\sim\sigma_H/h`.  The reported ``m3_denoised`` subtracts that in
    quadrature using the per-anchor Hessian variances.
    """
    A = anchors.coords
    n = A.shape[0]
    if n < 2:
        return {"m3": float("nan"), "m3_denoised": float("nan"), "pairs": 0}

    H = np.stack(
        [
            np.array([[h[0], h[1]], [h[1], h[2]]]) for h in anchors.hess
        ]
    )
    Hvar = anchors.hess_var
    d = np.linalg.norm(A[:, None] - A[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    kk = min(k, n - 1)
    nbr = np.argsort(d, axis=1)[:, :kk]

    ratios, denoised = [], []
    for i in range(n):
        for j in nbr[i]:
            h = d[i, j]
            if not np.isfinite(h) or h <= 0:
                continue
            diff = np.linalg.norm(H[i] - H[j], ord=2)
            noise = math.sqrt(float(Hvar[i].sum() + Hvar[j].sum()))
            ratios.append(diff / h)
            denoised.append(max(diff * diff - noise * noise, 0.0) ** 0.5 / h)
    if not ratios:
        return {"m3": float("nan"), "m3_denoised": float("nan"), "pairs": 0}
    return {
        "m3": float(np.percentile(ratios, 90)),
        "m3_denoised": float(np.percentile(denoised, 90)),
        "m3_median": float(np.median(ratios)),
        "pairs": len(ratios),
    }


def estimate_sigma(anchors: AnchorSet) -> dict[str, float]:
    r"""Per-example loss standard deviation, from the anchors' own variance estimates.

    ``anchors.value_var`` is the variance *of the mean* at the sample size used, so
    :math:`\sigma^2 = B \cdot \mathrm{Var}(\hat\ell)` recovers the per-example scale that
    the allocation theory needs.
    """
    B = max(anchors.n_examples // max(anchors.coords.shape[0], 1), 1)
    v = np.asarray(anchors.value_var, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"sigma": float("nan"), "sigma_grad": float("nan")}
    sigma = float(np.sqrt(np.mean(v) * B))
    gv = np.asarray(anchors.grad_var, dtype=float)
    gv = gv[np.isfinite(gv)]
    sigma_g = float(np.sqrt(np.mean(gv) * B)) if gv.size else float("nan")
    hv = np.asarray(anchors.hess_var, dtype=float)
    hv = hv[np.isfinite(hv)]
    sigma_h = float(np.sqrt(np.mean(hv) * B)) if hv.size else float("nan")
    return {"sigma": sigma, "sigma_grad": sigma_g, "sigma_hess": sigma_h, "B": B}


def taylor_bound(m3: float, radius: float) -> float:
    r"""A-priori patch error :math:`\tfrac16 M_3 \rho^3` from Taylor's theorem with
    Lagrange remainder, which the partition of unity carries to the whole domain."""
    return m3 * radius**3 / 6.0


__all__ = [
    "Certificate", "certify_surface", "empirical_bernstein_upper", "quantile_confidence",
    "order_statistic_bound", "stratified_points",
    "estimate_m3", "estimate_sigma", "taylor_bound",
]
