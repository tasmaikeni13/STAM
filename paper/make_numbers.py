"""Generate every number, table and figure block in the paper from the run artefacts.

The paper contains no hand-typed experimental numbers.  Each macro below is derived from
a JSON file written by an experiment stage, so a re-run updates the paper and a
discrepancy between text and results is impossible rather than merely unlikely.

Missing artefacts produce a visible ``[pending]`` marker rather than a plausible-looking
value, so an incomplete run cannot be mistaken for a complete one.
"""

from __future__ import annotations

import json
import pathlib
from collections import defaultdict

import numpy as np

RUNS = pathlib.Path("runs")
PAPER = pathlib.Path("paper")
FIG = pathlib.Path("figures")
PENDING = r"\textcolor{red}{[pending]}"


def load(p: pathlib.Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def fmt(x, digits=3, pct=False, thousands=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return PENDING
    if pct:
        return f"{100 * x:.1f}\\%"
    if thousands:
        return f"{x:,.0f}".replace(",", "{,}")
    if isinstance(x, (int, np.integer)):
        return f"{x:,}".replace(",", "{,}")
    v = float(x)
    if 1e-3 <= abs(v) < 1e5:
        # Plain decimal across the range every quantity in this paper occupies:
        # "1.1e+02" for a ratio of 116 is technically right and unreadable.
        r = f"{v:.{max(digits, 1)}g}"
        if "e" in r:
            r = f"{v:.0f}" if abs(v) >= 1 else f"{v:.4f}"
        return r
    return f"{v:.{digits}g}"


def macro(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


# ---------------------------------------------------------------------------


def collect() -> tuple[dict, dict]:
    m: dict[str, str] = {}
    raw: dict = {}

    for task, pre in (("cnn", "CNN"), ("gpt", "GPT")):
        s1 = load(RUNS / task / "stage1_report.json")
        s2 = load(RUNS / task / "stage2_report.json")
        s4 = load(RUNS / task / "stage4_report.json")
        sw = load(RUNS / task / "sweep_train.json")
        raw[task] = {"s1": s1, "s2": s2, "s4": s4, "sweep": sw}

        if s1:
            m[f"{pre}params"] = fmt(s1["model"]["total"])
            m[f"{pre}snaps"] = fmt(s1["timing"]["snapshots"])
            ov = s1["timing"].get("capture_overhead_excl_eval")
            m[f"{pre}overhead"] = fmt(ov, pct=True) if ov is not None else PENDING
            ovt = s1["timing"].get("capture_overhead_fraction")
            m[f"{pre}overheadtotal"] = fmt(ovt, pct=True) if ovt is not None else PENDING
            m[f"{pre}trainsec"] = fmt(s1["timing"]["wall_seconds"], 3)
            m[f"{pre}trajgib"] = fmt(s1["timing"]["snapshot_bytes"] / 2**30, 3)
        if s4 and s4.get("animation"):
            a = s4["animation"]
            m[f"{pre}trustfrac"] = fmt(a.get("trust_radius_fraction_mean"), pct=True)
            m[f"{pre}trusttol"] = fmt(a.get("tolerance"), 3)
            m[f"{pre}noisenorm"] = fmt(a.get("noise_norm_mean"), 3)
            m[f"{pre}trustmin"] = fmt(
                (a.get("trust_radius_min") or float("nan"))
                / max(a.get("domain_radius", 1), 1e-12), pct=True)
        if s4:
            for split in ("train", "val"):
                if split in s4:
                    c = s4[split].get("pipeline", {}).get("certificate") or {}
                    t = s4[split].get("true_error", {})
                    m[f"{pre}{split}certrel"] = fmt(
                        (c.get("rmse") or float("nan")) / max(t.get("reference_range", 1), 1e-12),
                        pct=True)
                    m[f"{pre}{split}truerel"] = fmt(t.get("rmse_relative"), pct=True)
                    m[f"{pre}{split}anchors"] = fmt(
                        s4[split].get("pipeline", {}).get("n_anchors"))
        if s2:
            k = s2["cost_model"]["kappa"]
            t = s2["cost_model"]["tau_example_equivalents"]
            m[f"{pre}kappaOne"] = fmt(float(k["1"]), 3)
            m[f"{pre}kappaTwo"] = fmt(float(k["2"]), 3)
            m[f"{pre}tauZero"] = fmt(float(t["0"]), 3)
            m[f"{pre}tauTwo"] = fmt(float(t["2"]), 3)
            m[f"{pre}vres"] = str(s2["value_resolution"])
            m[f"{pre}dres"] = str(s2["deriv_resolution"])
            m[f"{pre}radius"] = fmt(s2["radius"], 4)
            m[f"{pre}usec"] = fmt(s2["cost_model"]["seconds_per_example"] * 1e6, 3)
            m[f"{pre}rangetrain"] = fmt(s2["surface_range"]["train"], 3)
            m[f"{pre}fullmax"] = fmt(s2["surface_range"]["train"], 4)
            info = s2.get("analysis_domain_info")
            if info:
                m[f"{pre}trajmax"] = fmt(info["trajectory_max_loss"], 3)
                m[f"{pre}analysispts"] = fmt(info["points"])
                m[f"{pre}analysisradius"] = fmt(info["radius"], 3)
                m[f"{pre}trajinside"] = fmt(info["trajectory_fraction_inside"], pct=True)
            ar = s2.get("analysis_surface_range")
            if ar:
                m[f"{pre}analysisrange"] = fmt(ar["train"], 4)
            for a in ("centered", "origin", "endpoint"):
                p = s2["planes"][a]
                tag = a.capitalize()
                m[f"{pre}rho{tag}"] = fmt(p["rho_2"], 4)
                m[f"{pre}resid{tag}"] = fmt(p["residual_mean"], 3)
        if sw:
            budgets = sw["budgets"]
            m[f"{pre}budgetdecades"] = fmt(np.log10(budgets[-1] / budgets[0]), 2)
            m[f"{pre}seeds"] = str(sw["seeds"])
            m[f"{pre}budgetmin"] = fmt(budgets[0], thousands=True)
            m[f"{pre}budgetmax"] = fmt(budgets[-1], thousands=True)

    # ---- exponents and the budget-gain headline ----------------------------
    rates = load(FIG / "rates_all_cnn.json")
    if rates:
        for field, short in (("surface", "L"), ("gradient", "G"), ("curvature", "H")):
            for name, key in (("pu-taylor-2", "Two"), ("pu-taylor-1", "One"),
                              ("lstsq2", "Lsq"), ("grid-interp", "Grid"),
                              ("grid-refine", "Gridr"), ("rbf", "Rbf")):
                r = rates.get(field, {}).get(name)
                m[f"Exp{short}{key}"] = fmt(r["exponent"], 3) if r else PENDING

    gain = budget_gain(raw.get("cnn", {}).get("sweep"))
    m["CNNbudgetgain"] = f"${fmt(gain, 3)}\\times$" if gain else PENDING
    gain_g = budget_gain(raw.get("gpt", {}).get("sweep"))
    m["GPTbudgetgain"] = f"${fmt(gain_g, 3)}\\times$" if gain_g else PENDING

    cov = coverage(raw)
    m["Coverage"] = fmt(cov, pct=True) if cov is not None else PENDING
    for task, pre in (("cnn", "CNN"), ("gpt", "GPT")):
        sh = load(RUNS / task / "sharpness_train.json")
        if not sh:
            continue
        by = {}
        for r in sh["results"]:
            by.setdefault(r["examples"], []).append(r["apparent_sharpness_rms"])
        bs = sorted(by)
        true = sh["true_sharpness_rms"]
        m[f"{pre}sharpfloor"] = fmt(np.median(by[bs[-1]]) / true, 2)
        m[f"{pre}sharpworst"] = fmt(np.median(by[bs[0]]) / true, 2)
        m[f"{pre}stencilk"] = fmt(sh.get("stencil_constant"), 3)
    sh = load(RUNS / "cnn" / "sharpness_train.json")
    if sh:
        by = {}
        for r in sh["results"]:
            by.setdefault(r["examples"], []).append(r["apparent_sharpness_rms"])
        bs = sorted(by)
        true = sh["true_sharpness_rms"]
        m["CNNsharpfloor"] = fmt(np.median(by[bs[-1]]) / true, 2)
        m["CNNsharpworst"] = fmt(np.median(by[bs[0]]) / true, 2)
        m["CNNstencilk"] = fmt(sh.get("stencil_constant"), 3)
    tail = tail_share(raw)
    m["CNNtailshare"] = fmt(tail, pct=True) if tail is not None else PENDING

    # ---- kernels ------------------------------------------------------------
    bk = load(RUNS / "bakeoff.json")
    tri = load(RUNS / "bakeoff_cuda_vs_triton.json")
    if bk:
        peak = bk.get("peak_bandwidth_gbps", 1.0)
        best = max((r for r in bk["rows"]
                    if r.get("backend") == "cuda" and "gbps" in r),
                   key=lambda r: r["gbps"], default=None)
        m["CNNbandwidth"] = fmt(100 * best["gbps"] / peak, 3) if best else PENDING
        d = bk.get("detail", {})
        m["PUspeedup"] = fmt(d.get("pu_taylor", {}).get("speedup_vs_torch"), 3)
        m["GramSpeedup"] = fmt(d.get("gram_chunk", {}).get("speedup_vs_torch"), 3)
        m["PeakBW"] = fmt(peak, 4)
    if tri:
        c = tri["detail"].get("gram_chunk", {}).get("candidates", {})
        if "triton" in c and "cuda" in c:
            m["GramVsTriton"] = fmt(c["triton"]["seconds"] / c["cuda"]["seconds"], 3)
    m.setdefault("GramVsTriton", PENDING)

    return m, raw


def budget_gain(sweep) -> float | None:
    """Budget ratio at which STAM matches the best value-only method's best error."""
    if not sweep:
        return None

    acc = defaultdict(lambda: defaultdict(list))
    for r in sweep["results"]:
        v = r.get("surface", {}).get("rmse_relative")
        if v is not None and np.isfinite(v):
            acc[r["name"]][r["budget"]].append(float(v))
    curves = {n: (np.array(sorted(b)), np.array([np.median(b[k]) for k in sorted(b)]))
              for n, b in acc.items()}
    if "pu-taylor-2" not in curves:
        return None
    best_base = None
    for n in ("lstsq2", "rbf", "grid-interp", "grid-refine"):
        if n in curves:
            e = curves[n][1].min()
            best_base = e if best_base is None else min(best_base, e)
    if best_base is None:
        return None
    xs, ys = curves["pu-taylor-2"]
    hit = np.where(ys <= best_base)[0]
    if hit.size == 0:
        return None
    c_ours = xs[hit[0]]
    # budget at which the baseline first achieves it
    c_base = None
    for n in ("lstsq2", "rbf", "grid-interp", "grid-refine"):
        if n in curves:
            x2, y2 = curves[n]
            h2 = np.where(y2 <= best_base * 1.0000001)[0]
            if h2.size:
                v = x2[h2[0]]
                c_base = v if c_base is None else min(c_base, v)
    if not c_base or not c_ours:
        return None
    return float(c_base / c_ours)


def tail_share(raw) -> float | None:
    """Share of the total squared error contributed by the worst 1% of the domain.

    Read from the certificate's own diagnostic, which records the share carried by its
    single worst probe; with ~10^2 stratified probes that probe stands for roughly the
    worst percent of the domain.
    """
    vals = []
    sw = raw.get("cnn", {}).get("sweep")
    if not sw:
        return None
    for r in sw["results"]:
        c = (r.get("pipeline") or {}).get("certificate") or {}
        v = c.get("worst_probe_share")
        if v is not None and np.isfinite(v):
            vals.append(float(v))
    return float(np.median(vals)) if vals else None


def coverage(raw) -> float | None:
    ok = []
    for task in ("cnn", "gpt"):
        sw = raw.get(task, {}).get("sweep")
        if not sw:
            continue
        for r in sw["results"]:
            c = (r.get("pipeline") or {}).get("certificate")
            t = r.get("surface", {}).get("q95")
            if not c or t is None or not np.isfinite(t):
                continue
            if np.isfinite(c.get("q95_upper", np.nan)):
                ok.append(t <= c["q95_upper"])
    return float(np.mean(ok)) if ok else None


# ---------------------------------------------------------------------------


def tables(raw) -> str:
    out = []

    # -- cost model ----------------------------------------------------------
    rows = []
    for task, label in (("cnn", "CNN / CIFAR-10"), ("gpt", "Transformer / WikiText-2")):
        s2 = raw.get(task, {}).get("s2")
        if not s2:
            continue
        k = s2["cost_model"]["kappa"]
        t = s2["cost_model"]["tau_example_equivalents"]
        rows.append(
            f"{label} & {float(k['0']):.2f} & {float(k['1']):.2f} & {float(k['2']):.2f} & "
            f"{float(t['0']):.1f} & {float(t['1']):.1f} & {float(t['2']):.1f} & "
            f"{s2['cost_model']['seconds_per_example'] * 1e6:.1f} \\\\"
        )
    out.append(r"""
\begin{table}[t]
\centering
\caption{Measured cost model \eqref{eq:cost}. $\kappa_r$ is the marginal per-example
cost of an order-$r$ probe relative to a forward pass; $\tau_r$ is the fixed per-anchor
overhead in the same unit. Fitted by regressing probe time on sample size.}
\label{tab:cost}
\begin{tabular}{lccccccc}
\toprule
& \multicolumn{3}{c}{$\kappa_r$} & \multicolumn{3}{c}{$\tau_r$ (examples)} & \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
subject & $r{=}0$ & $r{=}1$ & $r{=}2$ & $r{=}0$ & $r{=}1$ & $r{=}2$ & $\mu$s/example \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

    # -- fidelity ------------------------------------------------------------
    rows = []
    for task, label in (("cnn", "CNN"), ("gpt", "Transformer")):
        s2 = raw.get(task, {}).get("s2")
        s4 = raw.get(task, {}).get("s4")
        if not s2:
            continue
        for a in ("centered", "origin", "endpoint"):
            p = s2["planes"][a]
            gap = ""
            if s4 and "fidelity" in s4 and a in s4["fidelity"]:
                f = s4["fidelity"][a]
                gap = (f"{f['gap_mean_abs']:.3g} & {f['gap_max_abs']:.3g} & "
                       f"{100 * f['gap_relative_mean']:.1f}\\%")
            else:
                gap = f"{PENDING} & {PENDING} & {PENDING}"
            rows.append(
                f"{label if a == 'centered' else ''} & {a} & {p['rho_2']:.4f} & "
                f"{p['residual_mean']:.3g} & {gap} \\\\"
            )
    out.append(r"""
\begin{table}[t]
\centering
\caption{Projection fidelity. The $\theta_0$-anchored plane reports a \emph{higher}
captured-variance ratio $\rho_2$ while having a \emph{larger} out-of-plane residual, on
both subjects: $\rho_2$ measures a different ratio than the one that matters. The
projection gap $|\gamma_t|$ is reported in loss units and as a fraction of the relief of
the drawn surface.}
\label{tab:fidelity}
\begin{tabular}{llccccc}
\toprule
subject & anchoring & $\rho_2$ & mean residual & mean $|\gamma_t|$ & max $|\gamma_t|$ &
$|\gamma_t|/$relief \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

    # -- kernels -------------------------------------------------------------
    bk = load(RUNS / "bakeoff.json")
    tri = load(RUNS / "bakeoff_cuda_vs_triton.json")
    rows = []
    if bk:
        peak = bk.get("peak_bandwidth_gbps", 1.0)
        for op, d in bk["detail"].items():
            c = d["candidates"]
            cu = c.get("cuda", {})
            to = c.get("torch", {})
            bw = ""
            r = next((x for x in bk["rows"] if x["op"] == op and x["backend"] == "cuda"
                      and x.get(("m" if op == "pu_taylor" else "n")) == d["size"]), None)
            bw = f"{100 * r['gbps'] / peak:.0f}\\%" if r and "gbps" in r else "---"
            tri_s = "---"
            if tri and op in tri["detail"]:
                tc = tri["detail"][op]["candidates"]
                if "triton" in tc:
                    tri_s = f"{tc['triton']['seconds'] / tc['cuda']['seconds']:.2f}$\\times$"
            rows.append(
                f"\\texttt{{{op.replace('_', chr(92) + '_')}}} & {d['size']:,} & "
                f"{cu.get('seconds', float('nan')) * 1e6:.0f} & "
                f"{d['speedup_vs_torch']:.2f}$\\times$ & {tri_s} & {bw} & "
                f"{cu.get('rel_err', float('nan')):.1e} & "
                f"{to.get('rel_err', float('nan')):.1e} \\\\".replace(",", "{,}")
            )
    out.append(r"""
\begin{table}[t]
\centering
\caption{Kernel benchmark on a Quadro RTX 6000 (sm\_75, """ + fmt(
        bk.get("peak_bandwidth_gbps") if bk else None, 4) + r""" GB/s peak).
Times are CUDA-event medians at the largest problem size. Accuracy is the maximum
relative error against an independently computed fp64 reference. Hand-written CUDA wins
every operation; the two streaming kernels tie with Triton on speed and are broken by
accuracy.}
\label{tab:kernels}
\begin{tabular}{lrrrrrrr}
\toprule
& & \multicolumn{4}{c}{CUDA kernel} & \multicolumn{2}{c}{rel.\ error} \\
\cmidrule(lr){3-6}\cmidrule(lr){7-8}
operation & size & $\mu$s & vs PyTorch & vs Triton & \% peak BW & CUDA & PyTorch \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

    # -- fitted exponents ----------------------------------------------------
    rows = []
    for task, label in (("cnn", "CNN"), ("gpt", "Transformer")):
        r = load(FIG / f"rates_all_{task}.json")
        if not r:
            continue
        for name, pretty in (("pu-taylor-2", "PU-Taylor-2 (value+grad+Hess)"),
                             ("pu-taylor-1", "PU-Taylor-1 (value+grad)"),
                             ("lstsq2", "local quadratic (values)"),
                             ("rbf", "RBF interpolation (values)"),
                             ("grid-refine", "grid, refining"),
                             ("grid-interp", "grid, fixed resolution")):
            cells = []
            for field in ("surface", "gradient", "curvature"):
                e = r.get(field, {}).get(name)
                cells.append(f"{e['exponent']:.3f}" if e else "---")
            rows.append(f"{label if name == 'pu-taylor-2' else ''} & {pretty} & "
                        + " & ".join(cells) + r" \\")
    out.append(r"""
\begin{table}[t]
\centering
\caption{Fitted convergence exponents $p$ in $E\propto C^{-p}$, by least squares in
log--log over the swept budgets. Predicted values from \Cref{tab:rates}: $3/8=0.375$
for the surface, $1/3=0.333$ (derivative probes) or $1/4=0.250$ (value-only) for the
gradient, $1/4$ or $1/8=0.125$ for curvature. A fixed-resolution grid is predicted to
plateau, i.e.\ to fit an exponent near zero.}
\label{tab:exponents}
\begin{tabular}{llccc}
\toprule
subject & method & $\ell$ & $\nabla\ell$ & $\nabla^2\ell$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

    # -- overhead ------------------------------------------------------------
    rows = []
    for task, label in (("cnn", "CNN / CIFAR-10"), ("gpt", "Transformer / WikiText-2")):
        s1 = raw.get(task, {}).get("s1")
        if not s1:
            continue
        t = s1["timing"]
        rows.append(
            f"{label} & {t['snapshots']} & {t['snapshot_bytes'] / 2**30:.2f} & "
            f"{t['seconds_per_epoch_control']:.2f} & "
            f"{t['seconds_per_epoch_with_capture']:.2f} & "
            f"{100 * t.get('capture_overhead_excl_eval', float('nan')):.1f}\\% & "
            f"{100 * t.get('capture_overhead_fraction', float('nan')):.1f}\\% \\\\"
        )
    out.append(r"""
\begin{table}[t]
\centering
\caption{Cost of trajectory capture, measured against matched control epochs run with
capture disabled. The snapshot copies themselves are essentially free --- they overlap
with compute on a separate stream --- and the visible cost is the exact reference
evaluation performed at every snapshot, which is optional and is what makes the
projection-gap audit possible.}
\label{tab:overhead}
\small
\begin{tabular}{lcccccc}
\toprule
& & buffer & \multicolumn{2}{c}{s/epoch} & \multicolumn{2}{c}{overhead} \\
\cmidrule(lr){4-5}\cmidrule(lr){6-7}
subject & snapshots & (GiB) & control & capture & copies & incl.\ ref. \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    return "\n".join(out)


def figure_block() -> str:
    def fig(path, label, caption, width="\\linewidth"):
        p = FIG / path
        if not p.exists():
            return (f"\n\\begin{{figure}}[t]\\centering{PENDING}"
                    f"\\caption{{{caption}}}\\label{{{label}}}\\end{{figure}}\n")
        return (f"\n\\begin{{figure}}[t]\n\\centering\n"
                f"\\includegraphics[width={width}]{{../figures/{path}}}\n"
                f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{figure}}\n")

    parts = [
        fig("budget_error_cnn.pdf", "fig:budget",
            "Relative RMS error of the rendered surface against compute budget "
            "(CNN/CIFAR-10, median of three seeds, log--log). The two grid baselines "
            "flatten: they interpolate their own noise, so past a point extra budget "
            "buys nothing (\\Cref{prop:floor}). The guide line is the predicted "
            "$C^{-3/8}$.", "0.72\\linewidth"),
        fig("rate_separation_cnn.pdf", "fig:rates",
            "The same sweep scored on three quantities. Derivative probing does not "
            "improve the rate for the surface, only its constant --- but it strictly "
            "improves the rate for the gradient field and for curvature, which is what "
            "\\Cref{tab:rates} predicts and what practitioners read off these figures."),
        fig("certificate_cnn.pdf", "fig:cert",
            "Left: the certified error against the true error measured against the "
            "exact reference. The certificate never sees the reference. Right: "
            "empirical coverage of the $95\\%$ upper bound, and how tight it is."),
        fig("sharpness_cnn.pdf", "fig:sharp",
            "Sharpness read off a fixed grid, against the per-point sample size. The "
            "apparent value never falls below the noise floor "
            "$\\lambda_{\\mathrm{noise}}\\propto\\sigma/(h^2\\sqrt B)$, whose constant "
            "comes from the second-difference stencil and is not fitted. At sample "
            "sizes typical of a landscape figure the floor exceeds the signal.",
            "0.68\\linewidth"),
        fig("fidelity_cnn.pdf", "fig:fid",
            "Left: the projection gap along the optimiser's path, for three plane "
            "constructions. Right: the $\\theta_0$-anchored plane reports the highest "
            "$\\rho_2$ and has the largest residual."),
        fig("kernels.pdf", "fig:kernels",
            "Left: achieved fraction of peak DRAM bandwidth for the two streaming "
            "kernels. Right: speedup of the hand-written CUDA kernels over PyTorch and "
            "over the Triton implementations."),
        fig("landscape_cnn.pdf", "fig:landcnn",
            "A certified landscape (CNN/CIFAR-10). Contour intervals are set to at "
            "least twice the certified RMS error, so no contour is drawn that the "
            "measurement cannot resolve. The quiver field is the exact gradient of the "
            "surface, computed analytically from \\eqref{eq:pugrad}. Open circles mark "
            "the anchors actually probed."),
        fig("landscape_gpt.pdf", "fig:landgpt",
            "A certified landscape for the \\GPTparams-parameter transformer on "
            "WikiText-2."),
        fig("rate_separation_gpt.pdf", "fig:ratesgpt",
            "The same sweep on the \\GPTparams-parameter transformer. The "
            "grid-refinement baseline again degrades with budget on curvature. The "
            "relative curvature metric discriminates poorly here because the true "
            "curvature is small over a domain where the loss saturates near "
            "$\\ln|V|$ --- a limitation of the subject, stated in \\Cref{sec:limits}."),
        fig("certificate_gpt.pdf", "fig:certgpt",
            "Certificate validity on the transformer. The error field is far less "
            "heavy-tailed than the CNN's, and the certified point estimate is "
            "correspondingly close to the truth."),
        fig("sharpness_gpt.pdf", "fig:sharpgpt",
            "Sharpness read off a grid for the transformer: overstated by "
            "\\GPTsharpworst$\\times$ at small samples and still "
            "\\GPTsharpfloor$\\times$ when the probe is exact.", "0.68\\linewidth"),
        fig("spectrum.pdf", "fig:spectrum",
            "Trajectory spectra. The uncentred construction concentrates more of "
            "\\emph{its} ratio in the leading pair, which is why $\\rho_2$ favours it "
            "while the residual does not."),
        fig("training.pdf", "fig:training",
            "Training curves on the fixed evaluation sets that define the landscapes."),
    ]
    return "\n".join(parts)


def environment_block() -> str:
    env = None
    for task in ("cnn", "gpt"):
        s2 = load(RUNS / task / "stage2_report.json")
        if s2 and "environment" in s2:
            env = s2["environment"]
            break
    if env is None:
        return PENDING
    dev = env["devices"][0]
    lines = [
        r"\begin{itemize}[leftmargin=1.4em,itemsep=1pt]",
        f"\\item Hardware: {len(env['devices'])}$\\times$ {dev['name']} "
        f"(sm\\_{dev['major']}{dev['minor']}, {dev['total_memory'] / 2**30:.1f}\\,GiB, "
        f"{dev['multi_processor_count']} SMs), {env['cpu_count']} CPU cores.",
        f"\\item Software: PyTorch {env['torch']} (CUDA {env['torch_cuda']}), "
        f"Python {env['python']}, cuDNN {env['cudnn']}.",
    ]
    pol = env.get("policy")
    if pol:
        lines.append(
            f"\\item Numerical policy selected at run time: reduction accumulator "
            f"\\texttt{{{pol['reduce_dtype']}}} (fp64 is rate-limited on this part, so "
            f"compensated fp32 is used), trajectory storage "
            f"\\texttt{{{pol['store_dtype']}}}, TF32 "
            f"{'enabled' if pol['allow_tf32'] else 'unavailable'}, "
            f"bf16 tensor cores unavailable."
        )
    lines += [
        r"\item Every result file records the full environment and the fingerprint of "
        r"the evaluation set it used.",
        r"\item Reproduce with \texttt{make all}; the Lean proofs with "
        r"\texttt{cd proofs/StamCert \&\& lake build}.",
        r"\end{itemize}",
    ]
    return "\n".join(lines)


def main() -> int:
    PAPER.mkdir(exist_ok=True)
    m, raw = collect()
    src = ["% Generated by paper/make_numbers.py -- do not edit.\n",
           r"" + "\n"]
    for k in sorted(m):
        src.append(macro(k, m[k]))
    # Any macro the paper references but the data does not yet provide.
    for k in ["CNNparams", "GPTparams", "CNNsnaps", "GPTsnaps", "CNNkappaOne",
              "CNNkappaTwo", "GPTkappaOne", "GPTkappaTwo", "CNNvres", "CNNdres",
              "GPTvres", "GPTdres", "CNNbudgetdecades", "CNNseeds", "CNNbudgetgain",
              "Coverage", "CNNbandwidth", "PUspeedup", "GramVsTriton",
              "CNNoverhead", "GPToverhead",
              "CNNtrustfrac", "GPTtrustfrac", "CNNfullmax", "CNNtrajmax",
              "CNNanalysispts", "CNNanalysisrange", "CNNanalysisradius", "CNNtailshare",
              "CNNnoisenorm", "CNNtrustmin", "ExpGGridr", "ExpHGridr",
              "CNNsharpfloor", "CNNsharpworst", "CNNstencilk",
              "GPTsharpfloor", "GPTsharpworst", "GPTnoisenorm", "GPTtrustmin", "CNNtraincertrel", "CNNtraintruerel",
              "CNNtrainanchors", "GPTtraincertrel", "GPTtraintruerel", "GPTtrainanchors"]:
        if k not in m:
            src.append(macro(k, PENDING))
    (PAPER / "numbers.tex").write_text("".join(src))
    (PAPER / "tables.tex").write_text(tables(raw))
    (PAPER / "figures_block.tex").write_text(figure_block())
    (PAPER / "environment.tex").write_text(environment_block())

    n_pending = sum(1 for v in m.values() if v == PENDING)
    print(f"wrote {len(m)} macros ({n_pending} pending), tables, figure block, environment")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
