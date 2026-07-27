r"""The stochastic surface, drawn only where the model for it holds.

An optimiser never descends the surface that gets plotted.  At step :math:`t` it sees a
mini-batch loss :math:`\ell_{B_t}`, a different function on every step, and it is that
sequence of functions -- not their mean -- that produces the observed dynamics.  Showing
it is worth doing, and it can be done without re-evaluating anything.

**The model.**  During training the mini-batch gradient :math:`g_{B_t}` is recorded, and
its restriction to the plane is :math:`\hat g_t = V g_{B_t}\in\mathbb R^2`.  The rendered
surface supplies the population gradient :math:`\nabla\mathcal R` at the same point
analytically.  Their difference

.. math::
   \eta_t = \hat g_t - \nabla\mathcal R(\alpha_t)

is the realised stochastic gradient noise, projected into the plane, and the first-order
model of the mini-batch surface is

.. math::
   \ell_{B_t}(\alpha) \approx \mathcal R(\alpha) + \eta_t^\top(\alpha - \alpha_t).

**Where it is drawn.**  A linear model extrapolates without limit, so it needs a stated
domain of validity, not a decorative envelope.  Fix a tolerance :math:`\varepsilon` in
loss units -- the certified error of the surface is the natural choice, since a
perturbation smaller than that is not resolvable anyway -- and draw the tilt only inside

.. math::
   \rho_t = \varepsilon / \|\eta_t\|,

the radius at which the tilt itself reaches :math:`\varepsilon`.  Outside, the deviation
being modelled is larger than the model's own accuracy and nothing is drawn.  The cutoff
is a Wendland :math:`C^2` window, so the rendered surface stays twice differentiable.

The trust radius is drawn on the contour panel.  It shrinks when the gradient noise is
large -- which is exactly when the mini-batch surface departs most from the mean -- so
the picture becomes visibly less confident precisely where it should.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from . import render as R
from . import style as S


def wendland(r: np.ndarray) -> np.ndarray:
    """Wendland C^2 window on [0, 1]: ``(1-r)^4 (4r+1)``."""
    om = np.clip(1.0 - r, 0.0, None)
    return om**4 * (4.0 * r + 1.0)


@dataclasses.dataclass
class StochasticFrames:
    """Per-step tilt vectors and their certified trust radii."""

    coords: np.ndarray      # (T, 2) optimiser position in the plane
    noise: np.ndarray       # (T, 2) eta_t
    radius: np.ndarray      # (T,)   rho_t
    batch_loss: np.ndarray  # (T,)   the mini-batch loss actually observed
    tolerance: float

    @property
    def T(self) -> int:
        return int(self.coords.shape[0])


def build_frames(
    coords: np.ndarray,
    batch_grad_proj: np.ndarray,
    surface_grad_at: np.ndarray,
    batch_loss: np.ndarray,
    tolerance: float,
    radius_cap: float,
) -> StochasticFrames:
    r"""Assemble the per-step stochastic model.

    ``surface_grad_at`` is :math:`\nabla\mathcal R(\alpha_t)`, read off the rendered
    surface analytically at the optimiser's own position -- not interpolated from a
    separate gradient field, which would introduce an inconsistency exactly where the
    difference is being taken.
    """
    eta = np.asarray(batch_grad_proj, dtype=np.float64) - np.asarray(
        surface_grad_at, dtype=np.float64
    )
    mag = np.linalg.norm(eta, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.where(mag > 1e-12, tolerance / np.maximum(mag, 1e-12), radius_cap)
    rho = np.clip(rho, 0.0, radius_cap)
    return StochasticFrames(
        coords=np.asarray(coords, dtype=np.float64), noise=eta, radius=rho,
        batch_loss=np.asarray(batch_loss, dtype=np.float64), tolerance=tolerance,
    )


def warp(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, frames: StochasticFrames, t: int
         ) -> np.ndarray:
    r"""The mini-batch surface at step ``t``: :math:`\mathcal R + \eta_t^\top d` inside
    the trust radius, tapered to :math:`\mathcal R` at its edge."""
    ax, ay = frames.coords[t]
    dx, dy = X - ax, Y - ay
    rho = max(float(frames.radius[t]), 1e-9)
    r = np.hypot(dx, dy) / rho
    tilt = frames.noise[t, 0] * dx + frames.noise[t, 1] * dy
    return Z + tilt * wendland(r)


def animate_landscape(
    X, Y, Z_train, Z_val, G_train, G_val, frames: StochasticFrames,
    traj_z_train, traj_z_val, *, out_path, err_train=0.0, err_val=0.0,
    surface_fn=None, title="", fps=12, stride=1, dpi=110, figsize=(12.6, 7.4),
    zoom_res: int = 41, zoom_factor: float = 1.15,
):
    r"""Write the animated view.

    Six panels, and the layout is a consequence of the analysis rather than a choice.
    On a domain whose radius is set by the *whole* trajectory, the certified validity
    radius :math:`\rho_t=\varepsilon/\|\eta_t\|` of the first-order mini-batch model is
    typically a few percent of it -- so a warp drawn honestly on the global surface is
    invisible, and a warp large enough to see on the global surface is not supported by
    the data. Prior "breathing landscape" animations resolve this by choosing an
    envelope width freely; we resolve it by showing the local structure at the scale
    where it exists.

    * global training surface and contour map, with the trust disc drawn to scale;
    * a **zoom** to the trust region, where the mini-batch tilt is rendered at a scale
      that makes it legible -- the same model, a different magnification;
    * the validation surface and contour map;
    * the realised gradient-noise magnitude and trust radius over training.

    ``surface_fn(points) -> values`` re-evaluates the fitted reconstruction; without it
    the zoom panel is omitted.
    """
    idx = np.arange(0, frames.T, stride)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, hspace=0.20, wspace=0.16)
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2], projection="3d")
    ax4 = fig.add_subplot(gs[1, 0], projection="3d")
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    lv_t = R.contour_levels(Z_train, err_train)
    lv_v = R.contour_levels(Z_val, err_val)
    zlim_t = (float(np.nanmin(Z_train)), float(np.nanmax(Z_train)))
    zlim_v = (float(np.nanmin(Z_val)), float(np.nanmax(Z_val)))
    pad_t = 0.12 * (zlim_t[1] - zlim_t[0] + 1e-12)
    pad_v = 0.12 * (zlim_v[1] - zlim_v[0] + 1e-12)
    domain_radius = 0.5 * float(X.max() - X.min())
    noise_mag = np.linalg.norm(frames.noise, axis=1)
    steps = np.arange(frames.T)

    def zoom_grid(t: int):
        ax_, ay_ = frames.coords[t]
        h = max(float(frames.radius[t]) * zoom_factor, 1e-9)
        u = np.linspace(-h, h, zoom_res)
        ZX, ZY = np.meshgrid(ax_ + u, ay_ + u)
        return ZX, ZY

    def draw(k: int) -> None:
        t = int(idx[k])
        for ax in (ax1, ax2, ax3, ax4, ax5, ax6):
            ax.clear()

        R.surface_panel(ax1, X, Y, Z_train, traj=frames.coords[: t + 1],
                        traj_z=traj_z_train[: t + 1],
                        title="training loss surface (global)")
        ax1.set_zlim(zlim_t[0] - pad_t, zlim_t[1] + pad_t)

        R.contour_panel(ax2, X, Y, Z_train, G_train[..., 0], G_train[..., 1],
                        traj=frames.coords[: t + 1], levels=lv_t,
                        title="training: contours, $-\\nabla\\ell$, trust region",
                        label_contours=False)
        ax2.add_patch(plt.Circle(tuple(frames.coords[t]), frames.radius[t], fill=False,
                                 color=S.SERIES[1], lw=1.3, zorder=8))
        ax2.set_xlim(X.min(), X.max())
        ax2.set_ylim(Y.min(), Y.max())

        # The zoom: the mini-batch surface where the model for it is certified.
        if surface_fn is not None:
            ZX, ZY = zoom_grid(t)
            base = surface_fn(np.stack([ZX.ravel(), ZY.ravel()], -1)).reshape(ZX.shape)
            tilted = warp(ZX, ZY, base, frames, t)
            R.surface_panel(ax3, ZX, ZY, tilted, title="mini-batch surface, zoomed to "
                                                       "the trust region", stride=1)
            lo = float(np.nanmin([np.nanmin(base), np.nanmin(tilted)]))
            hi = float(np.nanmax([np.nanmax(base), np.nanmax(tilted)]))
            ax3.set_zlim(lo - 0.08 * (hi - lo + 1e-12), hi + 0.08 * (hi - lo + 1e-12))
        else:
            ax3.axis("off")

        R.surface_panel(ax4, X, Y, Z_val, traj=frames.coords[: t + 1],
                        traj_z=traj_z_val[: t + 1],
                        title="validation loss surface (global)")
        ax4.set_zlim(zlim_v[0] - pad_v, zlim_v[1] + pad_v)

        R.contour_panel(ax5, X, Y, Z_val, G_val[..., 0], G_val[..., 1],
                        traj=frames.coords[: t + 1], levels=lv_v,
                        title="validation: contours and $-\\nabla\\ell$",
                        label_contours=False)
        ax5.set_xlim(X.min(), X.max())
        ax5.set_ylim(Y.min(), Y.max())

        ax6.plot(steps, noise_mag, color=S.SERIES[1], lw=1.2,
                 label=r"$\|\eta_t\|$  (projected gradient noise)")
        ax6.plot(steps, frames.radius / domain_radius, color=S.SERIES[0], lw=1.2,
                 label=r"$\rho_t/R$  (certified trust radius)")
        ax6.axvline(t, color=S.INK_MUTED, lw=0.9)
        ax6.set_yscale("log")
        ax6.set_xlabel("snapshot")
        ax6.set_title("what the animation is entitled to claim", loc="left")
        ax6.legend(fontsize=6.2, loc="upper right")
        S.despine(ax6)

        R.annotate_certificate(
            ax2,
            f"step {t + 1}/{frames.T}\n"
            f"$\\|\\eta_t\\|$ = {noise_mag[t]:.3g}\n"
            f"$\\rho_t$ = {frames.radius[t]:.3g} "
            f"({frames.radius[t] / domain_radius:.1%} of $R$)\n"
            f"$\\varepsilon$ = {frames.tolerance:.3g}",
        )
        if title:
            fig.suptitle(title, y=0.975, fontsize=plt.rcParams["font.size"] + 1.5)

    anim = FuncAnimation(fig, draw, frames=len(idx), interval=1000 / fps, blit=False)
    anim.save(str(out_path), writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return out_path


__all__ = ["StochasticFrames", "build_frames", "warp", "animate_landscape", "wendland"]
