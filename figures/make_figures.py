"""Build every figure in the paper from the recorded run artefacts.

Each figure reads only JSON/NPZ written by the experiment stages, so the plots cannot
drift from the numbers they are supposed to show, and re-running this script after a
re-run of the experiments updates the paper without touching either.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from stam.metrics import fit_rate
from stam.viz import style as S

FIG = pathlib.Path("figures")
RUNS = pathlib.Path("runs")
ORDER = ["pu-taylor-2", "pu-taylor-1", "lstsq2", "rbf", "grid-refine", "grid-interp"]


def save(fig, name: str) -> None:
    """Write both a vector PDF (for the paper) and a raster PNG (for the README)."""
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=190)


def _load(path: pathlib.Path):
    return json.loads(path.read_text()) if path.exists() else None


def aggregate(sweep: dict, field: str, sub: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Median over seeds of ``result[field][sub]`` for each method, keyed by budget."""
    acc: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in sweep["results"]:
        v = r.get(field, {}).get(sub)
        if v is None or not np.isfinite(v):
            continue
        acc[r["name"]][r["budget"]].append(float(v))
    out = {}
    for name, by_b in acc.items():
        bs = np.array(sorted(by_b))
        ys = np.array([np.median(by_b[b]) for b in bs])
        out[name] = (bs, ys)
    return out


def _line(ax, x, y, name, label_it=False, xmax=None):
    c = S.METHOD_COLOR[name]
    ax.plot(x, y, color=c, marker=S.METHOD_MARKER[name], ms=3.4, mew=0,
            label=S.METHOD_LABEL[name], zorder=4)
    if label_it and len(x):
        S.direct_label(ax, x[-1], y[-1], S.METHOD_SHORT[name], c)


# ---------------------------------------------------------------------------


def fig_budget_error(task: str) -> None:
    sweep = _load(RUNS / task / "sweep_train.json")
    if sweep is None:
        return
    data = aggregate(sweep, "surface", "rmse_relative")
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    for name in ORDER:
        if name in data:
            x, y = data[name]
            _line(ax, x, y, name, label_it=name in ("pu-taylor-2", "lstsq2", "grid-interp"))

    if "pu-taylor-2" in data:
        x, y = data["pu-taylor-2"]
        S.reference_slope(ax, 3 / 8, x, y[0] * 1.9, r"$C^{-3/8}$")

    # Mark where the fixed grid's per-point sample reaches half the evaluation set.  Past
    # it the probes are nearly exhaustive, the sampling noise the analysis is about has
    # essentially vanished, and the comparison is one of approximation only -- a regime a
    # real dataset would never reach, and one this small evaluation set makes reachable.
    s2 = _load(RUNS / task / "stage2_report.json")
    if s2 and "grid-interp" in data:
        n_eval = int(s2["reference_exact_examples"])
        kappa0 = float(s2["cost_model"]["kappa"]["0"])
        tau0 = float(s2["cost_model"]["tau_example_equivalents"]["0"])
        c_sat = 441 * (tau0 + kappa0 * 0.5 * n_eval) / 0.9
        lo, hi = ax.get_xlim()
        if lo < c_sat < hi:
            ax.axvline(c_sat, color=S.INK_MUTED, lw=0.8, zorder=1)
            ax.annotate("fixed grid reaches\n$B = |\\mathcal{D}|/2$", xy=(c_sat, y.min()),
                        xytext=(-4.6, -0.2), textcoords="offset fontsize", fontsize=6.2,
                        color=S.INK_MUTED)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("budget $C$  (example-forward-equivalents)")
    ax.set_ylabel("relative RMS error of the surface")
    ax.set_title(f"{task}: surface accuracy at equal compute", loc="left")
    ax.legend(loc="lower left", ncol=2, fontsize=6.6)
    S.despine(ax)
    fig.tight_layout()
    save(fig, f"budget_error_{task}")
    plt.close(fig)

    # Fitted exponents, printed into the paper's table.
    rows = {}
    for name in ORDER:
        if name in data:
            x, y = data[name]
            rows[name] = fit_rate(x, y)
    (FIG / f"rates_{task}.json").write_text(json.dumps(rows, indent=2))


def fig_rate_separation(task: str) -> None:
    sweep = _load(RUNS / task / "sweep_train.json")
    if sweep is None:
        return
    panels = [
        ("surface", "rmse_relative", r"loss  $\ell$", 3 / 8),
        ("gradient", "rmse_relative", r"gradient  $\nabla\ell$", 1 / 3),
        ("curvature", "sharpness_relative", r"curvature  $\nabla^2\ell$", 1 / 4),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.2), sharex=True)
    rates: dict[str, dict] = {}
    for ax, (field, sub, title, slope) in zip(axs, panels):
        data = aggregate(sweep, field, sub)
        rates[field] = {}
        for name in ORDER:
            if name in data:
                x, y = data[name]
                _line(ax, x, y, name, label_it=(name == "pu-taylor-2" and ax is axs[0]))
                rates[field][name] = fit_rate(x, y)
        if "pu-taylor-2" in data:
            x, y = data["pu-taylor-2"]
            S.reference_slope(ax, slope, x, y[0] * 2.1,
                              rf"$C^{{-{slope:.3f}}}$".replace("0.375", "3/8")
                              .replace("0.333", "1/3").replace("0.250", "1/4"))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title, loc="left")
        ax.set_xlabel("budget $C$")
        S.despine(ax)
    axs[0].set_ylabel("relative RMS error")
    axs[0].legend(loc="lower left", fontsize=6.2, ncol=1)
    fig.suptitle(f"{task}: what a fixed budget buys, by quantity", y=1.0, x=0.005,
                 ha="left", fontsize=plt.rcParams["font.size"] + 1)
    fig.tight_layout()
    save(fig, f"rate_separation_{task}")
    plt.close(fig)
    (FIG / f"rates_all_{task}.json").write_text(json.dumps(rates, indent=2))


def fig_certificate(task: str) -> None:
    sweep = _load(RUNS / task / "sweep_train.json")
    if sweep is None:
        return
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.4))

    ax = axs[0]
    # The certified quantity is the 95th percentile of |error|: on a heavy-tailed error
    # field a mean-based statement needs far more probes than a certification slice
    # affords, while an order statistic is distribution-free at this sample size.
    pts = defaultdict(lambda: ([], [], []))
    for r in sweep["results"]:
        cert = (r.get("pipeline") or {}).get("certificate")
        if not cert or not np.isfinite(cert.get("q95_upper", np.nan)):
            continue
        true = r["surface"].get("q95")
        if true is None or not np.isfinite(true):
            continue
        a, b, c = pts[r["name"]]
        a.append(true)
        b.append(cert["q95"])
        c.append(cert["q95_upper"])
    lims = [np.inf, -np.inf]
    for name in ORDER:
        if name not in pts:
            continue
        t, e, u = (np.array(v) for v in pts[name])
        ax.scatter(t, e, s=11, color=S.METHOD_COLOR[name], alpha=0.85,
                   label=S.METHOD_LABEL[name], marker=S.METHOD_MARKER[name],
                   linewidths=0)
        lims[0] = min(lims[0], t.min(), e.min())
        lims[1] = max(lims[1], t.max(), e.max())
    lo, hi = max(lims[0] * 0.6, 1e-6), lims[1] * 1.6
    ax.plot([lo, hi], [lo, hi], color=S.INK_MUTED, lw=0.9, zorder=1)
    ax.annotate("exact", xy=(hi, hi), xytext=(-2.2, 0.3), textcoords="offset fontsize",
                fontsize=6.5, color=S.INK_MUTED)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"true 95th pct of $|$error$|$ (vs exact reference)")
    ax.set_ylabel(r"certified 95th pct of $|$error$|$")
    ax.set_title("the certificate never sees the reference", loc="left")
    ax.legend(loc="upper left", fontsize=6.2)
    S.despine(ax)

    ax = axs[1]
    names, cov, tight = [], [], []
    for name in ORDER:
        rs = [r for r in sweep["results"] if r["name"] == name
              and (r.get("pipeline") or {}).get("certificate")]
        if not rs:
            continue
        ok, ratios = [], []
        for r in rs:
            c = r["pipeline"]["certificate"]
            t = r["surface"].get("q95")
            if t is None or not np.isfinite(t) or not np.isfinite(c.get("q95_upper", np.nan)):
                continue
            ok.append(t <= c["q95_upper"])
            ratios.append(c["q95_upper"] / max(t, 1e-12))
        if ok:
            names.append(S.METHOD_SHORT[name])
            cov.append(100 * np.mean(ok))
            tight.append(np.median(ratios))
    y = np.arange(len(names))
    ax.barh(y, cov, height=0.62, color=[S.METHOD_COLOR[n] for n in ORDER[: len(names)]])
    ax.axvline(95, color=S.INK_MUTED, lw=0.9)
    ax.annotate("95% nominal", xy=(95, len(names) - 0.35), xytext=(0.3, 0),
                textcoords="offset fontsize", fontsize=6.5, color=S.INK_MUTED)
    ax.set_yticks(y, names, fontsize=7)
    ax.set_xlim(0, 104)
    ax.set_xlabel("coverage of the distribution-free bound  (%)")
    ax.set_title("does the bound hold?", loc="left")
    for yi, (c, t) in enumerate(zip(cov, tight)):
        ax.text(min(c + 1.5, 101), yi, f"{c:.0f}%  ({t:.2f}× loose)", va="center",
                fontsize=6.2, color=S.INK_SECONDARY)
    ax.grid(axis="y", visible=False)
    S.despine(ax, left=False)
    fig.tight_layout()
    save(fig, f"certificate_{task}")
    plt.close(fig)


def fig_fidelity(task: str) -> None:
    stage4 = _load(RUNS / task / "stage4_report.json")
    if stage4 is None or "fidelity" not in stage4:
        return
    fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.2))

    ax = axs[0]
    for i, anchoring in enumerate(("centered", "origin", "endpoint")):
        f = RUNS / task / f"fidelity_{anchoring}.npz"
        if not f.exists():
            continue
        d = np.load(f)
        ok = np.isfinite(d["gap"])
        ax.plot(d["epochs"][ok], np.abs(d["gap"][ok]), color=S.SERIES[i],
                label=f"{anchoring}", marker=["o", "s", "^"][i], ms=2.6, mew=0)
    rng = stage4["train"]["true_error"]["reference_range"]
    ax.axhline(rng, color=S.INK_MUTED, lw=0.9)
    ax.annotate("relief of the drawn surface", xy=(ax.get_xlim()[1], rng),
                xytext=(-8.5, 0.35), textcoords="offset fontsize", fontsize=6.5,
                color=S.INK_MUTED)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$|\gamma_t| = |L(\theta_t) - L(\Pi\theta_t)|$")
    ax.set_title("projection gap along the path", loc="left")
    ax.legend(fontsize=6.6, title="plane anchoring", title_fontsize=6.6)
    S.despine(ax)

    ax = axs[1]
    keys = ["centered", "origin", "endpoint"]
    rho = [stage4["fidelity"][k]["captured_variance_rho2"] for k in keys]
    res = [stage4["fidelity"][k]["residual_mean"] for k in keys]
    x = np.arange(len(keys))
    ax2 = ax
    w = 0.36
    ax2.bar(x - w / 2, rho, width=w, color=S.SERIES[0], label=r"captured variance $\rho_2$")
    ax2.bar(x + w / 2, np.array(res) / max(res), width=w, color=S.SERIES[1],
            label="mean residual (normalised)")
    for xi, (r_, s_) in enumerate(zip(rho, res)):
        ax2.text(xi - w / 2, r_ + 0.02, f"{r_:.3f}", ha="center", fontsize=6.2,
                 color=S.INK_SECONDARY)
        ax2.text(xi + w / 2, s_ / max(res) + 0.02, f"{s_:.2f}", ha="center", fontsize=6.2,
                 color=S.INK_SECONDARY)
    ax2.set_xticks(x, keys, fontsize=7)
    ax2.set_ylim(0, 1.18)
    ax2.set_title(r"a higher $\rho_2$ can mean a worse plane", loc="left")
    ax2.legend(fontsize=6.4, loc="lower right")
    ax2.grid(axis="x", visible=False)
    S.despine(ax2)
    fig.tight_layout()
    save(fig, f"fidelity_{task}")
    plt.close(fig)


def fig_sharpness(task: str) -> None:
    d = _load(RUNS / task / "sharpness_train.json")
    if d is None:
        return
    by_b: dict[int, list[float]] = defaultdict(list)
    var_b: dict[int, list[float]] = defaultdict(list)
    for r in d["results"]:
        by_b[r["examples"]].append(r["apparent_sharpness_rms"])
        var_b[r["examples"]].append(r["value_var_mean"])
    bs = np.array(sorted(by_b))
    ys = np.array([np.median(by_b[b]) for b in bs])
    true = d["true_sharpness_rms"]
    h = d["results"][0]["spacing"]

    # Predicted noise floor.  The stencil constant relating node-noise variance to the
    # RMS largest eigenvalue is obtained by applying the *same* estimator to pure noise
    # of unit variance -- exact by construction, and free of the algebra that relating
    # correlated second differences to an eigenvalue would otherwise require.  The only
    # measured input is v, the per-node variance the probes reported.
    v = np.array([np.median(var_b[b]) for b in bs])
    floor = float(d["stencil_constant"]) * np.sqrt(v) / h**2

    # At the largest sample size the probe uses the whole evaluation set, so its noise is
    # exactly zero (finite-population correction) and whatever remains is the grid's own
    # discretisation bias: a second difference over spacing h measures curvature at scale
    # h, and on a surface with structure below h that overstates it.  Two distinct
    # mechanisms inflate the apparent sharpness, and separating them is the point.
    disc = float(ys[-1])
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.plot(bs, ys, color=S.SERIES[0], marker="o", ms=3.6, mew=0, zorder=5,
            label="apparent sharpness, read off the grid")
    ax.plot(bs, floor, color=S.SERIES[1], lw=1.3,
            label=r"noise term  $k\sqrt{v}/h^2$ ($k$ from the stencil, $v$ measured)")
    ax.plot(bs, np.sqrt(disc**2 + floor**2), color=S.SERIES[2], lw=1.3, ls=(0, (4, 2)),
            label=r"$\sqrt{\lambda_{\rm disc}^2+\lambda_{\rm noise}^2}$  (both terms)")
    ax.axhline(disc, color=S.SERIES[4], lw=1.1)
    ax.axhline(true, color=S.INK, lw=1.2)
    ax.annotate("true sharpness", xy=(bs[0], true), xytext=(0.3, 0.35),
                textcoords="offset fontsize", fontsize=7, color=S.INK)
    ax.annotate(f"discretisation floor  ({disc / true:.1f}$\\times$ the truth,\n"
                f"noise-free)", xy=(bs[0], disc), xytext=(0.3, -1.9),
                textcoords="offset fontsize", fontsize=7, color=S.SERIES[4])
    cross = bs[int(np.argmin(np.abs(floor - disc)))]
    ax.axvline(cross, color=S.INK_MUTED, lw=0.8, zorder=1)
    ax.annotate(f"noise = discretisation\nat $B\\approx{cross:,}$", xy=(cross, true * 0.62),
                xytext=(0.35, 0.0), textcoords="offset fontsize", fontsize=6.5,
                color=S.INK_MUTED, va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(true * 0.45, ys[0] * 2.6)
    ax.set_xlabel("examples per grid point, $B$")
    ax.set_ylabel(r"RMS $\lambda_{\max}$ over the grid")
    ax.set_title(f"{task}: sharpness read off a {d['resolution']}$\\times${d['resolution']} "
                 f"grid overstates the truth by "
                 f"{ys[0] / true:.0f}$\\times$ to {disc / true:.0f}$\\times$", loc="left")
    ax.legend(fontsize=6.4, loc="upper right", ncol=1)
    S.despine(ax)
    fig.tight_layout()
    save(fig, f"sharpness_{task}")
    plt.close(fig)


def fig_allocation(task: str) -> None:
    stage2 = _load(RUNS / task / "stage2_report.json")
    sweep = _load(RUNS / task / "sweep_train.json")
    if stage2 is None:
        return
    from stam.design import allocate, asymptotic_optimum, predicted_error
    from stam.probe import CostModel

    cm = stage2["cost_model"]
    cost = CostModel(
        kappa={int(k): float(v) for k, v in cm["kappa"].items()},
        tau={int(k): float(v) for k, v in cm["tau_example_equivalents"].items()},
        seconds_per_example=float(cm["seconds_per_example"]), device="cpu",
    )
    radius = float(stage2["radius"])
    sigma, m3 = 1.0, 1.0
    if sweep:
        for r in sweep["results"]:
            p = r.get("pipeline") or {}
            if p.get("sigma", {}).get("sigma"):
                sigma = float(p["sigma"]["sigma"])
                m3 = float(p["m3"].get("m3_denoised") or p["m3"]["m3"])
                break

    fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.2))

    C = 4e6
    ns = np.arange(4, 400)
    ax = axs[0]
    for order, color in ((2, S.SERIES[0]), (1, S.SERIES[1]), (0, S.SERIES[2])):
        tot, bias, noise = [], [], []
        for n in ns:
            B = (C / n - cost.tau[order]) / cost.kappa[order]
            if B < 1:
                tot.append(np.nan); bias.append(np.nan); noise.append(np.nan); continue
            t, b_, nz = predicted_error(int(n), int(B), order, sigma, m3, radius)
            tot.append(t); bias.append(b_); noise.append(nz)
        ax.plot(ns, tot, color=color, label=f"order {order}")
        k = int(np.nanargmin(tot))
        ax.scatter([ns[k]], [tot[k]], s=20, color=color, zorder=5, marker="*")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of anchors $n$")
    ax.set_ylabel("predicted error")
    ax.set_title(f"where to spend $C={C:,.0f}$", loc="left")
    ax.legend(fontsize=6.6)
    S.despine(ax)

    ax = axs[1]
    budgets = np.geomspace(1e4, 1e8, 40)
    for order, color in ((2, S.SERIES[0]), (1, S.SERIES[1]), (0, S.SERIES[2])):
        n_star = []
        for b in budgets:
            try:
                a = allocate(b, cost, sigma, m3, radius, order=order)
                n_star.append(a.n_anchors)
            except ValueError:
                n_star.append(np.nan)
        ax.plot(budgets, n_star, color=color, label=f"order {order}")
    ref = asymptotic_optimum(budgets[0], cost.kappa[2], sigma, m3, radius)["n_star"]
    S.reference_slope(ax, -0.25, budgets, ref, r"$C^{1/4}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("budget $C$")
    ax.set_ylabel(r"optimal anchor count $n^\star$")
    ax.set_title("more budget buys depth, not resolution", loc="left")
    ax.legend(fontsize=6.6)
    S.despine(ax)
    fig.tight_layout()
    save(fig, f"allocation_{task}")
    plt.close(fig)


def fig_kernels() -> None:
    d = _load(RUNS / "bakeoff.json")
    tri = _load(RUNS / "bakeoff_cuda_vs_triton.json")
    if d is None:
        return
    peak = d.get("peak_bandwidth_gbps", 0) or 1.0
    fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.0))

    ax = axs[0]
    ops = ["plane_point", "project"]
    for i, op in enumerate(ops):
        rows = [r for r in d["rows"] if r["op"] == op and "gbps" in r]
        for j, backend in enumerate(("torch", "cuda")):
            rs = sorted([r for r in rows if r["backend"] == backend], key=lambda r: r["n"])
            if rs:
                ax.plot([r["n"] for r in rs], [100 * r["gbps"] / peak for r in rs],
                        color=S.SERIES[j], marker=["o", "s"][i], ms=3.4, mew=0,
                        ls=["-", (0, (4, 2))][i],
                        label=f"{backend} · {op}")
    ax.axhline(100, color=S.INK_MUTED, lw=0.9)
    ax.annotate("theoretical peak", xy=(ax.get_xlim()[1], 100), xytext=(-6.5, 0.35),
                textcoords="offset fontsize", fontsize=6.5, color=S.INK_MUTED)
    ax.set_xscale("log")
    ax.set_xlabel("vector length $N$")
    ax.set_ylabel("% of peak DRAM bandwidth")
    ax.set_ylim(0, 112)
    ax.set_title("streaming kernels", loc="left")
    ax.legend(fontsize=6.0, ncol=2)
    S.despine(ax)

    ax = axs[1]
    names, vals, colors = [], [], []
    for op, v in d["detail"].items():
        sp = v.get("speedup_vs_torch")
        if sp:
            names.append(op)
            vals.append(sp)
            colors.append(S.SERIES[0])
    if tri:
        for op, v in tri["detail"].items():
            c = v["candidates"]
            if "triton" in c and "cuda" in c:
                names.append(f"{op}\n(vs Triton)")
                vals.append(c["triton"]["seconds"] / c["cuda"]["seconds"])
                colors.append(S.SERIES[1])
    y = np.arange(len(names))
    ax.barh(y, vals, height=0.6, color=colors)
    ax.axvline(1, color=S.INK_MUTED, lw=0.9)
    ax.set_yticks(y, names, fontsize=6.2)
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(color=S.SERIES[0], label="vs PyTorch"),
                       Patch(color=S.SERIES[1], label="vs Triton")],
              fontsize=6.4, loc="lower right")
    ax.set_xscale("log")
    ax.set_xlabel("speedup of the CUDA kernel (×)")
    ax.set_title("against PyTorch and against Triton", loc="left")
    for yi, v in enumerate(vals):
        ax.text(v * 1.06, yi, f"{v:.1f}×", va="center", fontsize=6.2,
                color=S.INK_SECONDARY)
    ax.grid(axis="y", visible=False)
    S.despine(ax, left=False)
    fig.tight_layout()
    save(fig, "kernels")
    plt.close(fig)


def fig_spectrum() -> None:
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.0))
    drawn = 0
    for ax, task in zip(axs, ("cnn", "gpt")):
        s2 = _load(RUNS / task / "stage2_report.json")
        if s2 is None:
            continue
        drawn += 1
        for i, (name, p) in enumerate(s2["planes"].items()):
            sv = np.array(p["singular_values"], dtype=float)
            sv = sv[sv > 0]
            ax.plot(np.arange(1, len(sv) + 1), sv / sv[0], color=S.SERIES[i], label=name,
                    lw=1.4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("index $i$")
        ax.set_title(f"{task}: trajectory spectrum", loc="left")
        S.despine(ax)
    if drawn == 0:
        # An empty axes frame is worse than no figure: it would go into the paper
        # silently.  Raise so the caller reports the gap.
        plt.close(fig)
        raise FileNotFoundError("no stage2 reports yet")
    axs[0].set_ylabel(r"$\sigma_i/\sigma_1$")
    axs[0].legend(fontsize=6.6, title="anchoring", title_fontsize=6.6)
    fig.tight_layout()
    save(fig, "spectrum")
    plt.close(fig)


def fig_training() -> None:
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.0))
    drawn = 0
    for ax, task in zip(axs, ("cnn", "gpt")):
        rep = _load(RUNS / task / "train_report.json")
        if rep is None:
            continue
        drawn += 1
        h = rep["history"]
        ep = [x["epoch"] for x in h]
        ax.plot(ep, [x["train_loss"] for x in h], color=S.SERIES[0], label="train")
        ax.plot(ep, [x["val_loss"] for x in h], color=S.SERIES[1], label="validation")
        S.direct_label(ax, ep[-1], h[-1]["train_loss"], "train", S.SERIES[0])
        S.direct_label(ax, ep[-1], h[-1]["val_loss"], "val", S.SERIES[1])
        ax.set_xlabel("epoch")
        ax.set_title(f"{task}", loc="left")
        S.despine(ax)
    if drawn == 0:
        plt.close(fig)
        raise FileNotFoundError("no training reports yet")
    axs[0].set_ylabel("loss on the fixed evaluation set")
    axs[0].legend(fontsize=6.6)
    fig.tight_layout()
    save(fig, "training")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="cnn,gpt")
    args = ap.parse_args()
    S.use_paper_style()
    FIG.mkdir(exist_ok=True)

    made = []
    for task in args.tasks.split(","):
        for fn in (fig_budget_error, fig_rate_separation, fig_certificate, fig_fidelity,
                   fig_sharpness, fig_allocation):
            try:
                fn(task)
                made.append(f"{fn.__name__}({task})")
            except Exception as exc:
                print(f"  ! {fn.__name__}({task}): {type(exc).__name__}: {exc}")
    for fn in (fig_kernels, fig_spectrum, fig_training):
        try:
            fn()
            made.append(fn.__name__)
        except Exception as exc:
            print(f"  ! {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"built {len(made)} figures:")
    for m in made:
        print(f"  {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
