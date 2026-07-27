"""Rendering a certified landscape.

The panels are the conventional ones -- a 3-D surface and a contour map with a gradient
field -- with three differences that follow from the analysis rather than from taste:

* the quiver field is the **exact gradient of the surface being drawn**, because the
  reconstruction differentiates analytically; a field interpolated separately from the
  values need not be consistent with them, and the inconsistency is invisible;
* the certified error is printed on the panel, so the picture states its own accuracy;
* contour levels are spaced by a multiple of that certified error, so no contour is
  drawn that the measurement cannot support.

The last point is the one that changes what a reader may conclude.  A contour interval
finer than the error of the surface manufactures topography.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

from . import style as S


def contour_levels(Z: np.ndarray, error: float, target: int = 14,
                   min_multiple: float = 2.0) -> np.ndarray:
    """Contour levels no finer than ``min_multiple`` times the certified error.

    Refusing to draw a contour the measurement cannot resolve is the visual counterpart
    of not quoting a digit beyond the uncertainty.
    """
    lo, hi = float(np.nanmin(Z)), float(np.nanmax(Z))
    span = max(hi - lo, 1e-12)
    step = max(span / target, min_multiple * max(error, 0.0))
    n = max(int(np.ceil(span / step)), 2)
    return np.linspace(lo, lo + n * step, n + 1)


def surface_panel(ax, X, Y, Z, *, traj=None, traj_z=None, title="", zlabel=r"$\ell$",
                  cmap=None, elev=32.0, azim=-125.0, stride=1) -> None:
    """3-D surface with the optimiser's path drawn on it."""
    cmap = cmap or S.SEQ_R
    Zm = np.ma.masked_invalid(Z)
    ax.plot_surface(
        X, Y, Zm, cmap=cmap, rstride=stride, cstride=stride, linewidth=0,
        antialiased=True, alpha=0.94, rasterized=True,
    )
    if traj is not None and traj_z is not None:
        ok = np.isfinite(traj_z)
        ax.plot(traj[ok, 0], traj[ok, 1], traj_z[ok], color=S.INK, lw=1.5, zorder=10)
        ax.scatter(traj[ok, 0][-1:], traj[ok, 1][-1:], traj_z[ok][-1:], color=S.INK,
                   s=14, depthshade=False, zorder=11)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel(r"$\alpha_1$", labelpad=-6)
    ax.set_ylabel(r"$\alpha_2$", labelpad=-6)
    ax.set_zlabel(zlabel, labelpad=-6)
    ax.tick_params(labelsize=6, pad=-2)
    ax.set_title(title, pad=2)
    ax.grid(False)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0.03)
        pane.pane.set_edgecolor(S.GRID)


def contour_panel(ax, X, Y, Z, GX=None, GY=None, *, traj=None, title="", error=0.0,
                  quiver_stride=6, cmap=None, levels=None, label_contours=True,
                  colorbar=False, cbar_label=r"$\ell$") -> None:
    """Filled contour map with the negated gradient field and the optimiser's path.

    The ramp runs dark (low loss) to light (high loss), so the basin the optimiser is in
    reads as the dark region.  Contour lines and arrows are drawn in the warm accent
    rather than white: white is invisible against the light end of the ramp, and a mark
    that disappears over part of its own plot is worse than one that is merely quiet.
    """
    cmap = cmap or S.SEQ_R
    levels = contour_levels(Z, error) if levels is None else levels
    cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, extend="both")
    for c in cf.collections if hasattr(cf, "collections") else []:
        c.set_rasterized(True)
    cs = ax.contour(X, Y, Z, levels=levels, colors=["#c9d6e8"], linewidths=0.4, alpha=0.75)
    if label_contours:
        ax.clabel(cs, cs.levels[::4], inline=True, fontsize=5.5, fmt="%.2f",
                  colors="#e8eef7")

    if GX is not None and GY is not None:
        s = quiver_stride
        u, v = -GX[::s, ::s], -GY[::s, ::s]
        mag = np.hypot(u, v)
        scale = np.nanpercentile(mag[np.isfinite(mag)], 90) if np.isfinite(mag).any() else 1.0
        ax.quiver(
            X[::s, ::s], Y[::s, ::s], u, v, color=S.SERIES[1], alpha=0.95,
            width=0.004, headwidth=3.4, headlength=3.8,
            scale=max(scale, 1e-9) * 22, scale_units="width",
        )
    if colorbar:
        cb = ax.figure.colorbar(cf, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(cbar_label, fontsize=7)
        cb.ax.tick_params(labelsize=6)
        cb.outline.set_visible(False)
    if traj is not None:
        ax.plot(traj[:, 0], traj[:, 1], color="#ffffff", lw=2.6, zorder=6, alpha=0.85)
        ax.plot(traj[:, 0], traj[:, 1], color=S.INK, lw=1.4, zorder=7)
        ax.scatter(traj[:1, 0], traj[:1, 1], s=26, facecolor="#ffffff",
                   edgecolor=S.INK, lw=1.0, zorder=8)
        ax.scatter(traj[-1:, 0], traj[-1:, 1], s=54, facecolor=S.INK,
                   edgecolor="#ffffff", lw=0.7, zorder=8, marker="*")
    ax.set_xlabel(r"$\alpha_1$")
    ax.set_ylabel(r"$\alpha_2$")
    ax.set_title(title, pad=3)
    ax.set_aspect("equal")
    ax.grid(False)
    return cf


def error_panel(ax, X, Y, E, *, title="", vmax=None, cbar_label="signed error") -> None:
    """Signed error map on a diverging ramp with a neutral midpoint at zero."""
    lo, hi = S.symmetric_limits(E) if vmax is None else (-vmax, vmax)
    im = ax.pcolormesh(X, Y, E, cmap=S.DIV, vmin=lo, vmax=hi, shading="auto",
                       rasterized=True)
    ax.set_title(title, pad=3)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\alpha_1$")
    ax.set_ylabel(r"$\alpha_2$")
    ax.grid(False)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(cbar_label, fontsize=7)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_visible(False)
    return im


# Above this many anchors the markers stop conveying the design and start obscuring the
# surface they sit on; the count is printed in the caption either way.
_MAX_ANCHOR_MARKERS = 220


def landscape_figure(
    X, Y, Z_train, Z_val, G_train, G_val, traj, traj_z_train, traj_z_val,
    *, err_train=0.0, err_val=0.0, title="", anchors=None, figsize=None,
    labels=("training", "validation"), single=False,
):
    """Surface and gradient field, for one or two evaluation sets.

    ``single=True`` draws one row: with only one evaluation set the second row would
    repeat the first, and a duplicated panel invites the reader to look for a difference
    that is not there.
    """
    rows = 1 if single else 2
    figsize = figsize or ((9.4, 4.3) if single else (9.4, 8.2))
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(rows, 2, hspace=0.16, wspace=0.14)
    name = labels[0]
    pre = "" if single else f"{name} "

    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    surface_panel(ax1, X, Y, Z_train, traj=traj, traj_z=traj_z_train,
                  title=f"{pre}loss surface".strip())

    ax2 = fig.add_subplot(gs[0, 1])
    contour_panel(ax2, X, Y, Z_train, G_train[..., 0], G_train[..., 1], traj=traj,
                  title=f"{pre}contours and $-\\nabla\\ell$".strip().capitalize()
                  if single else f"{name}: contours and $-\\nabla\\ell$",
                  error=err_train, colorbar=True)
    if anchors is not None and len(anchors) <= _MAX_ANCHOR_MARKERS:
        ax2.scatter(anchors[:, 0], anchors[:, 1], s=7, facecolor="none",
                    edgecolor=S.SERIES[3], lw=0.8, zorder=5)

    axes = [ax1, ax2]
    if not single:
        ax3 = fig.add_subplot(gs[1, 0], projection="3d")
        surface_panel(ax3, X, Y, Z_val, traj=traj, traj_z=traj_z_val,
                      title=f"{labels[1]} loss surface")
        ax4 = fig.add_subplot(gs[1, 1])
        contour_panel(ax4, X, Y, Z_val, G_val[..., 0], G_val[..., 1], traj=traj,
                      title=f"{labels[1]}: contours and $-\\nabla\\ell$",
                      error=err_val, colorbar=True)
        axes += [ax3, ax4]

    if title:
        fig.suptitle(title, y=0.99 if single else 0.955,
                     fontsize=plt.rcParams["font.size"] + 1.5)
    return fig, tuple(axes)


def annotate_certificate(ax, cert_text: str, loc: str = "lower left") -> None:
    """Print the surface's own certified accuracy on the panel."""
    xy = {"lower left": (0.02, 0.02), "lower right": (0.98, 0.02),
          "upper left": (0.02, 0.98), "upper right": (0.98, 0.98)}[loc]
    ha = "right" if "right" in loc else "left"
    va = "top" if "upper" in loc else "bottom"
    ax.text(xy[0], xy[1], cert_text, transform=ax.transAxes, fontsize=6.5,
            color="#ffffff", ha=ha, va=va,
            bbox=dict(boxstyle="round,pad=0.28", facecolor=S.INK, alpha=0.55,
                      edgecolor="none"))


__all__ = [
    "contour_levels", "surface_panel", "contour_panel", "error_panel",
    "landscape_figure", "annotate_certificate",
]
