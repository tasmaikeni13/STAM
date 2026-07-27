r"""Where to probe, and how deeply: the budget-allocation problem.

Fix a compute budget :math:`C`, in example-forward-equivalents.  Spending it on
:math:`n` anchors of :math:`B` examples each costs :math:`n(\tau + \kappa B)=C`, and the
error of the resulting surface has two terms that pull in opposite directions:

.. math::
   E(n, B) \;\approx\;
   \underbrace{c_1 M_3 h^3}_{\text{approximation}} \;+\;
   \underbrace{c_2 \sigma B^{-1/2}}_{\text{Monte Carlo}},
   \qquad h \simeq R n^{-1/2},

with :math:`R` the domain radius.  More anchors shrink the first term and starve the
second.  Substituting :math:`B=(C/n-\tau)/\kappa` and minimising gives, in the regime
:math:`\tau \ll C/n`,

.. math::
   n^\star \simeq \Bigl(\tfrac{3c_1M_3R^3}{c_2\sigma}\Bigr)^{1/2}(C/\kappa)^{1/4},
   \qquad
   E^\star \simeq c\,(M_3R^3)^{1/4}\Bigl(\tfrac{\kappa\sigma^2}{C}\Bigr)^{3/8}.

Three consequences worth stating plainly:

* :math:`n^\star` grows only as :math:`C^{1/4}`.  Quadrupling the budget should buy
  :math:`\sqrt2` times as many anchors, not four times as many -- the rest goes into
  making each one less noisy.  Dense-grid practice does the opposite.
* :math:`E^\star \propto \kappa^{3/8}`, so paying :math:`\kappa\approx4` for a full
  second-order probe costs :math:`4^{3/8}\approx1.7\times` in error *for the value
  surface* -- and buys a strictly better rate for the gradient and curvature fields,
  which is where it pays for itself.
* :math:`E^\star \propto \tau^{3/8}` through the effective budget once :math:`\tau` is
  non-negligible: fixed per-anchor overhead is not a constant factor on the wall clock,
  it degrades the achievable accuracy.  This is why the fused kernels are part of the
  method.

The optimum is computed numerically here, from *measured* :math:`\kappa,\tau,\sigma`
and an estimated :math:`M_3`, rather than from the asymptotic formula -- the formula is
the explanation, the numerics are the recommendation.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Literal, Sequence  # noqa: F401

import numpy as np

from .probe import CostModel

DesignKind = Literal["grid", "halton", "sobol", "jittered", "random"]


# ---------------------------------------------------------------------------
# Point sets
# ---------------------------------------------------------------------------


def _van_der_corput(n: int, base: int) -> np.ndarray:
    out = np.zeros(n)
    for i in range(n):
        f, r, k = 1.0 / base, 0.0, i + 1
        while k > 0:
            r += f * (k % base)
            k //= base
            f /= base
        out[i] = r
    return out


def halton(n: int, bases: tuple[int, int] = (2, 3)) -> np.ndarray:
    """Halton sequence on :math:`[0,1)^2`.

    Low-discrepancy designs give a smaller *and less variable* covering radius than
    random placement at the same :math:`n`, which is what the approximation term
    depends on; unlike a lattice they degrade gracefully at any :math:`n`, so the
    allocator is free to choose non-square anchor counts.
    """
    return np.stack([_van_der_corput(n, bases[0]), _van_der_corput(n, bases[1])], axis=1)


def design_points(
    n: int,
    domain: tuple[tuple[float, float], tuple[float, float]],
    kind: DesignKind = "halton",
    seed: int = 0,
    inset: float = 0.0,
) -> np.ndarray:
    """``n`` probe locations in ``domain``.

    ``inset`` shrinks the design towards the centre, leaving a margin the supports can
    reach over.  Anchors exactly on the boundary make the partition of unity one-sided
    there, which inflates the error in a band the width of one support radius.
    """
    (x0, x1), (y0, y1) = domain
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    hx, hy = 0.5 * (x1 - x0) * (1 - inset), 0.5 * (y1 - y0) * (1 - inset)
    lo = np.array([cx - hx, cy - hy])
    hi = np.array([cx + hx, cy + hy])

    if kind == "grid":
        side = int(round(math.sqrt(n)))
        side = max(side, 2)
        gx = np.linspace(0, 1, side)
        gy = np.linspace(0, 1, side)
        u = np.stack(np.meshgrid(gx, gy, indexing="xy"), -1).reshape(-1, 2)
    elif kind == "halton":
        u = halton(n)
    elif kind == "sobol":
        from scipy.stats import qmc

        u = qmc.Sobol(d=2, scramble=True, seed=seed).random(n)
    elif kind == "jittered":
        side = int(math.ceil(math.sqrt(n)))
        rng = np.random.default_rng(seed)
        gx, gy = np.meshgrid((np.arange(side) + 0.5) / side, (np.arange(side) + 0.5) / side)
        u = np.stack([gx.ravel(), gy.ravel()], -1)
        u = u + (rng.random(u.shape) - 0.5) / side
        u = np.clip(u[:n], 0, 1)
    elif kind == "random":
        u = np.random.default_rng(seed).random((n, 2))
    else:
        raise ValueError(f"unknown design {kind!r}")
    return lo + u * (hi - lo)


def inside(points: np.ndarray,
           domain: tuple[tuple[float, float], tuple[float, float]]) -> np.ndarray:
    """Boolean mask of the points lying in a closed axis-aligned box."""
    (x0, x1), (y0, y1) = domain
    p = np.asarray(points)
    return ((p[:, 0] >= x0) & (p[:, 0] <= x1) & (p[:, 1] >= y0) & (p[:, 1] <= y1))


def render_grid(
    domain: tuple[tuple[float, float], tuple[float, float]], resolution: int
) -> tuple[np.ndarray, tuple[int, int]]:
    """Query points for rendering, row-major with ``y`` varying slowest."""
    (x0, x1), (y0, y1) = domain
    xs = np.linspace(x0, x1, resolution)
    ys = np.linspace(y0, y1, resolution)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    return np.stack([X.ravel(), Y.ravel()], -1), (resolution, resolution)


# ---------------------------------------------------------------------------
# Allocation
# ---------------------------------------------------------------------------

# Geometric constants of the reconstruction, fixed by the scheme rather than the problem.
# c1: patch error is bounded by M3 rho^3 / 6, and the covering radius of a quasi-uniform
#     design of n points in a disc of radius R is about 1.6 R n^{-1/2} (the 1.6 is the
#     support overlap factor used by `support_radii`).
# c2: the partition of unity averages roughly `neff` overlapping patches, so anchor noise
#     is attenuated by sqrt(neff); measured at 1 to stay conservative.
_C1 = 1.6**3 / 6.0
_C2 = 1.0


@dataclasses.dataclass
class Allocation:
    n_anchors: int
    examples_per_anchor: int
    order: int
    budget: float
    used: float
    predicted_error: float
    predicted_bias: float
    predicted_noise: float
    spacing: float
    detail: dict

    def to_dict(self) -> dict:
        return {
            "n_anchors": self.n_anchors, "examples_per_anchor": self.examples_per_anchor,
            "order": self.order, "budget": self.budget, "used": self.used,
            "predicted_error": self.predicted_error, "predicted_bias": self.predicted_bias,
            "predicted_noise": self.predicted_noise, "spacing": self.spacing, **self.detail,
        }


# A value-only local quadratic fit needs at least six anchors in its window, forcing a
# window roughly 2.2x wider than a Taylor patch of the same order -- the structural cost
# of not observing derivatives.  The constant enters cubed.
_WIDEN = 2.2


def predicted_error(
    n: int, B: int, order: int, sigma: float, m3: float, radius: float,
    m2: float | None = None, n_eff: float = 1.0,
    sigma_grad: float | None = None, sigma_hess: float | None = None,
) -> tuple[float, float, float]:
    r"""Predicted (total, bias, noise) error of the reconstruction.

    The bias is the Taylor remainder of the polynomial degree each patch reproduces:

    * ``order=2`` -- quadratic patches, :math:`\tfrac16 M_3 h^3`;
    * ``order=1`` -- linear patches, :math:`\tfrac12 M_2 h^2`, with :math:`M_2`
      measured directly from the pilot's Hessians (it is exactly what they bound);
    * ``order=0`` -- value-only local quadratic regression, which does reach
      :math:`O(h^3)` but only over a window wide enough to determine six coefficients,
      so the same :math:`M_3` enters with a constant larger by ``_WIDEN**3``.

    This is where the derivative probes pay: not in the exponent for the surface, but in
    a bias constant roughly :math:`2.2^3 \approx 11` times smaller at equal spacing.

    **The noise term is not only the value's.**  A Taylor patch evaluated a distance
    :math:`\rho` from its anchor also carries that anchor's *derivative* noise, scaled by
    the distance:

    .. math::
       \sigma_{\mathrm{eff}}^2(\rho) = \frac1B\Bigl(\sigma_\ell^2 + \rho^2\sigma_g^2
         + \tfrac14\rho^4\sigma_H^2\Bigr).

    These terms vanish asymptotically, since :math:`\rho\simeq h\to0`, but they are what
    stops the optimum from collapsing to a single enormous patch, and at small :math:`n`
    they bind.  All three variances are measured by the pilot, so including them is free.
    """
    h = radius / max(math.sqrt(n), 1e-9)
    if order >= 2:
        bias = _C1 * m3 * h**3
    elif order == 1:
        m2_eff = m2 if (m2 is not None and math.isfinite(m2) and m2 > 0) else m3 * radius
        bias = (1.6**2 / 2.0) * m2_eff * h**2
    else:
        bias = _C1 * (_WIDEN**3) * m3 * h**3

    var = sigma * sigma
    rho = 1.6 * h  # the support radius the reconstruction actually uses
    if order >= 1 and sigma_grad is not None and math.isfinite(sigma_grad):
        var += (rho * sigma_grad) ** 2
    if order >= 2 and sigma_hess is not None and math.isfinite(sigma_hess):
        var += (0.5 * rho * rho * sigma_hess) ** 2
    noise = _C2 * math.sqrt(var) / math.sqrt(max(B, 1) * max(n_eff, 1.0))
    return bias + noise, bias, noise


def allocate(
    budget: float,
    cost: CostModel,
    sigma: float,
    m3: float,
    radius: float,
    order: int = 2,
    n_min: int = 4,
    n_max: int = 65536,
    examples_available: int | None = None,
    m2: float | None = None,
    examples_min: int = 8,
    sigma_grad: float | None = None,
    sigma_hess: float | None = None,
) -> Allocation:
    """Choose ``(n_anchors, examples_per_anchor)`` minimising the predicted error.

    Solved by scanning :math:`n` and reading :math:`B` off the budget constraint.  A
    scan rather than the closed form because the fixed overhead :math:`\\tau` makes the
    objective non-monotone near the feasibility boundary, and because :math:`B` is
    capped by the size of the evaluation set -- both effects the asymptotics drop.
    """
    kappa, tau = cost.kappa[order], cost.tau[order]
    best: Allocation | None = None
    for n in range(n_min, n_max + 1):
        per_anchor = budget / n
        B = (per_anchor - tau) / kappa
        if B < 1:
            break
        B_int = int(B)
        if examples_available is not None:
            B_int = min(B_int, examples_available)
        # Below a handful of examples the per-anchor variance estimate -- which the
        # certificate needs -- has no degrees of freedom, so the design is rejected
        # rather than merely penalised.
        if B_int < examples_min:
            break
        total, bias, noise = predicted_error(
            n, B_int, order, sigma, m3, radius, m2=m2,
            sigma_grad=sigma_grad, sigma_hess=sigma_hess,
        )
        used = n * (tau + kappa * B_int)
        cand = Allocation(
            n_anchors=n, examples_per_anchor=B_int, order=order, budget=budget, used=used,
            predicted_error=total, predicted_bias=bias, predicted_noise=noise,
            spacing=radius / math.sqrt(n),
            detail={"kappa": kappa, "tau": tau, "sigma": sigma, "m3": m3, "m2": m2,
                    "radius": radius},
        )
        if best is None or cand.predicted_error < best.predicted_error:
            best = cand
    if best is None:
        raise ValueError(
            f"budget {budget:g} too small for order {order} "
            f"(tau={tau:.1f}, needs at least {n_min * (tau + kappa):.1f})"
        )
    return best


def asymptotic_optimum(
    budget: float, kappa: float, sigma: float, m3: float, radius: float
) -> dict[str, float]:
    r"""Closed-form optimum with :math:`\tau=0`, for comparison with the numeric scan.

    .. math::
       n^\star = \Bigl(\frac{3c_1M_3R^3}{c_2\sigma}\Bigr)^{1/2}
                 \Bigl(\frac{C}{\kappa}\Bigr)^{1/4},
       \qquad
       E^\star = 4\Bigl(\frac{c_1M_3R^3}{3}\Bigr)^{1/4}
                 \Bigl(\frac{3c_2\sigma}{4}\Bigr)^{3/4}
                 \Bigl(\frac{\kappa}{C}\Bigr)^{3/8}\!\Big/\,\ldots
    """
    a = _C1 * m3 * radius**3       # bias = a n^{-3/2}
    b = _C2 * sigma                # noise = b (kappa n / C)^{1/2}
    if a <= 0 or b <= 0 or budget <= 0:
        return {"n_star": float("nan"), "error_star": float("nan")}
    # minimise a n^{-3/2} + b (kappa/C)^{1/2} n^{1/2}
    n_star = (3 * a / b * math.sqrt(budget / kappa)) ** 0.5
    err = a * n_star**-1.5 + b * math.sqrt(kappa * n_star / budget)
    return {
        "n_star": n_star,
        "examples_star": budget / (kappa * max(n_star, 1e-9)),
        "error_star": err,
        "exponent": -3.0 / 8.0,
    }


def dense_grid_allocation(budget: float, cost: CostModel, resolution: int, order: int = 0
                          ) -> Allocation:
    """The comparison point: a fixed :math:`n\\times n` grid, whatever is left per point.

    This is what current practice does.  Because the resolution is fixed in advance,
    growing the budget only grows :math:`B`, and the surface is drawn by interpolating
    the resulting noisy values -- so the error cannot fall below the per-point noise.
    """
    n = resolution * resolution
    kappa, tau = cost.kappa[order], cost.tau[order]
    B = max(int((budget / n - tau) / kappa), 1)
    return Allocation(
        n_anchors=n, examples_per_anchor=B, order=order, budget=budget,
        used=n * (tau + kappa * B), predicted_error=float("nan"),
        predicted_bias=float("nan"), predicted_noise=float("nan"),
        spacing=float("nan"), detail={"resolution": resolution, "fixed_grid": True},
    )


__all__ = [
    "halton", "design_points", "render_grid", "inside", "Allocation", "allocate", "predicted_error",
    "asymptotic_optimum", "dense_grid_allocation",
]
