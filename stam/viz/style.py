"""Figure style: one palette, three colour jobs, no decoration.

Colour is assigned by the job it does, never by rank or by what looks nice:

*Categorical* (which method) -- a fixed six-hue order, validated for colour-vision
deficiency: every adjacent pair separates by ΔE ≥ 9.1 under protanopia and ≥ 19.6 under
normal vision (OKLab ×100).  The assignment from method to hue is fixed once here, so
adding or dropping a method from a panel never repaints the others.

*Sequential* (how much loss) -- a single hue, light to dark.  A rainbow ramp invents
boundaries where the data has none, which on a loss surface reads as topography that
is not there.

*Diverging* (signed error) -- two hues with a **neutral grey** midpoint, so zero error
is visually null and the sign of a deviation is immediately legible.

Three of the six categorical hues fall below 3:1 contrast on white; every series that
uses them carries a direct label at the end of its line, so identity never rests on
colour alone.
"""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb

# -- Categorical: fixed order, validated as a set -----------------------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]

# Fixed method -> hue assignment.  Stable across every figure in the paper.
METHOD_COLOR = {
    "pu-taylor-2": SERIES[0],
    "pu-taylor-1": SERIES[1],
    "lstsq2":      SERIES[2],
    "rbf":         SERIES[3],
    "grid-refine": SERIES[4],
    "grid-interp": SERIES[5],
}

METHOD_LABEL = {
    "pu-taylor-2": "STAM (value+grad+Hess)",
    "pu-taylor-1": "STAM (value+grad)",
    "lstsq2":      "local quadratic",
    "rbf":         "RBF interpolation",
    "grid-refine": "grid, refine",
    "grid-interp": "grid, fixed",
}

METHOD_SHORT = {
    "pu-taylor-2": "PU-Taylor-2",
    "pu-taylor-1": "PU-Taylor-1",
    "lstsq2":      "local quad.",
    "rbf":         "RBF",
    "grid-refine": "grid-refine",
    "grid-interp": "grid-fixed",
}

# Marker per method: a second, redundant channel so the panels survive greyscale
# printing and the three low-contrast hues.
METHOD_MARKER = {
    "pu-taylor-2": "o", "pu-taylor-1": "s", "lstsq2": "^",
    "rbf": "D", "grid-refine": "v", "grid-interp": "P",
}

INK = "#1a1a19"
INK_SECONDARY = "#55544d"
INK_MUTED = "#8a8880"
GRID = "#e4e3dd"
SURFACE = "#ffffff"


def _ramp(name: str, stops: list[str]) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(name, [to_rgb(c) for c in stops], N=256)


# Sequential: one hue, light to dark, monotone in lightness.
SEQ = _ramp("stam_seq", ["#f2f6fc", "#cddff5", "#8fbaea", "#4a90d9", "#2a78d6", "#154a87",
                         "#0b2b4f"])
SEQ_R = _ramp("stam_seq_r", list(reversed(
    ["#f2f6fc", "#cddff5", "#8fbaea", "#4a90d9", "#2a78d6", "#154a87", "#0b2b4f"])))

# Diverging: warm and cool poles about a neutral grey midpoint.
DIV = _ramp("stam_div", ["#8a3410", "#c85520", "#eb6834", "#f4a986", "#eeeeea",
                         "#a8c8ec", "#4a90d9", "#2a78d6", "#154a87"])


def use_paper_style(font_size: float = 9.0) -> None:
    """Publication defaults: serif to match the document, recessive furniture.

    Gridlines and axis rules are solid and pale rather than dashed -- dashing adds
    texture that competes with the data for attention.
    """
    mpl.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "mathtext.fontset": "dejavuserif",
        "font.size": font_size,
        "axes.titlesize": font_size + 0.5,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1,
        "axes.edgecolor": INK_MUTED,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "text.color": INK,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.0,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.columnspacing": 1.2,
        "legend.labelspacing": 0.35,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
    })


def despine(ax, left: bool = True, bottom: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)


def direct_label(ax, x, y, text: str, color: str, dx: float = 0.02, **kw) -> None:
    """Label a series at its right-hand end.

    Series identity should never rest on colour alone; the low-contrast hues in the
    palette make that a hard requirement rather than a preference.
    """
    ax.annotate(
        text, xy=(x, y), xytext=(dx, 0), textcoords="offset fontsize",
        color=color, va="center", ha="left", fontsize=mpl.rcParams["font.size"] - 1.5,
        annotation_clip=False, **kw,
    )


def symmetric_limits(a: np.ndarray) -> tuple[float, float]:
    """Limits centred on zero, so a diverging ramp's neutral midpoint means zero."""
    m = float(np.nanmax(np.abs(a)))
    m = max(m, 1e-12)
    return -m, m


def reference_slope(ax, exponent: float, x: np.ndarray, y_at: float, label: str,
                    color: str = INK_MUTED) -> None:
    """Draw a guide line of slope ``-exponent`` on log-log axes."""
    x = np.asarray(x, dtype=float)
    y = y_at * (x / x[0]) ** (-exponent)
    ax.plot(x, y, color=color, lw=0.9, zorder=1)
    ax.annotate(label, xy=(x[-1], y[-1]), xytext=(0.3, -0.3),
                textcoords="offset fontsize", color=color, fontsize=7.5,
                annotation_clip=False)


__all__ = [
    "SERIES", "METHOD_COLOR", "METHOD_LABEL", "METHOD_SHORT", "METHOD_MARKER",
    "INK", "INK_SECONDARY", "INK_MUTED", "GRID", "SEQ", "SEQ_R", "DIV",
    "use_paper_style", "despine", "direct_label", "symmetric_limits", "reference_slope",
]
