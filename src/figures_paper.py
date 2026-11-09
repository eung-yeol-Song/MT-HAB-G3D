"""
figures_paper.py
================
Publication-ready comparison figures for the extended Table 3 (13 models),
written to **PDF** (vector, Type-42 fonts, embeddable in LaTeX).

    python -m src.figures_paper                      # everything it can build
    python -m src.figures_paper --out outputs/figures
    python -m src.figures_paper --only f01,f05,f13

Outputs
-------
    <out>/figures_all.pdf        every figure, one per page (for quick review)
    <out>/fig01_regression.pdf   … one standalone PDF per figure (for \\includegraphics)
    <out>/figure_index.csv       filename -> caption, for the manuscript

Figure list
-----------
    f01  Regression accuracy       RMSE / Peak-RMSE / R2 / TA+-1, ranked bars
    f02  Red-tide detection        F1 / AUPRC / MCC / TSS / Accuracy, ranked bars
    f03  Normalised metric heatmap model x 9 metrics, single-hue ramp
    f04  Skill score vs baseline   diverging bars against day-of-year climatology
    f05  Accuracy vs complexity    RMSE against parameter count
    f06  Rank profile              parallel-coordinate ranks across the 9 metrics
    f07  Horizon degradation       D+1..D+7, regression and classification panels
    f08  Taylor diagram            correlation / normalised sd / centred RMSE
    f09  Observed vs predicted     scatter grid for the key models
    f10  Precision-recall curves   event-adjacent subset
    f11  Reliability diagrams      calibration of the red-tide probability
    f12  Error distributions       signed-error boxplots per model
    f13  Test-year time series     truth vs baselines vs MT-HAB-G3D
    f14  Proposed vs baselines     margin of MT-HAB-G3D over the best baseline

Figures f08-f14 need per-run predictions (outputs/runs/<tag>/preds_test.npz).
They are skipped with a printed notice when those files are absent, so the
script always produces whatever the available results allow.

Design notes
------------
* Colour encodes the **model family**, never the individual model and never the
  rank, so adding or removing a model never repaints the survivors.  Five slots
  are used; the reference/naive family gets a reserved neutral grey.
* The categorical palette was checked for colour-vision deficiency: worst
  adjacent pair dE 6.3 (deuteranopia), which is inside the band that requires
  secondary encoding — hatching and direct labels are therefore always present,
  never colour alone.
* No figure uses two y-axes.  Quantities on different scales get their own panel.
* Sequential ramp = one hue light->dark.  Diverging ramp = two hues with a
  neutral grey midpoint.  No rainbow anywhere.
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# --------------------------------------------------------------------------- #
# 1.  Design tokens
# --------------------------------------------------------------------------- #
INK = "#1b1b1a"
INK_2 = "#54544f"
INK_MUTED = "#84837c"
GRID = "#dedcd4"
SURFACE = "#ffffff"

# validated categorical palette (fixed order, never cycled)
FAMILY_COLOR = {
    "reference": "#6E7781",   # reserved neutral - naive / climatological references
    "sequence":  "#0072B2",
    "classical": "#B26B00",
    "graph":     "#00875A",
    "proposed":  "#BA3A8B",
}
FAMILY_LABEL = {
    "reference": "Reference (no learning)",
    "classical": "Classical ML",
    "sequence":  "Sequence networks",
    "graph":     "Graph networks",
    "proposed":  "Proposed (MT-HAB-G3D)",
}
FAMILY_ORDER = ["reference", "classical", "sequence", "graph", "proposed"]

# single-hue sequential ramp (light -> dark), for the normalised heatmap
SEQ_CMAP = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f2f7fb", "#cfe3f1", "#9dc7e4", "#5ba3d0", "#1c7ab5", "#0a4c78"])
# diverging ramp: two hues + neutral grey midpoint
DIV_CMAP = LinearSegmentedColormap.from_list(
    "div_amber_blue", ["#8a4f00", "#c08a3d", "#e3d9c8", "#eeeeea",
                       "#cadced", "#5f9dc9", "#02537f"])

MODELS = [
    # key,            pretty label,                 family,      hatch
    ("climatology",   "Day-of-year climatology",    "reference", ""),
    ("histgb",        "Gradient-boosted trees",     "classical", ""),
    ("mlp",           "MLP",                        "classical", "xx"),
    ("lstm",          "LSTM",                       "sequence",  ""),
    ("gru",           "GRU",                        "sequence",  ".."),
    ("cnn_lstm",      "CNN-LSTM",                   "sequence",  "//"),
    ("tcn",           "Dilated TCN",                "sequence",  "\\\\"),
    ("transformer",   "Transformer",                "sequence",  "++"),
    ("stgcn",         "ST-GCN (no G3D)",            "graph",     ""),
    ("mtgnn",         "MTGNN",                      "graph",     ".."),
    ("am_mtgnn",      "AM-MTGNN",                   "graph",     "//"),
    ("mtgnn_nograph", "MTGNN (no graph)",           "graph",     "xx"),
    ("mthab_mse",     "MT-HAB-G3D (proposed)",      "proposed",  ""),
]
KEYS = [m[0] for m in MODELS]
LABEL = {m[0]: m[1] for m in MODELS}
FAMILY = {m[0]: m[2] for m in MODELS}
HATCH = {m[0]: m[3] for m in MODELS}
COLOR = {k: FAMILY_COLOR[FAMILY[k]] for k in KEYS}

# The proposed model. It is the only emphasised series in every figure: full
# opacity, a dark outline, a larger mark and a ▶ marker in the tick label,
# while every baseline is drawn recessively. Nothing else is singled out —
# CNN-LSTM and the rest are ordinary members of the comparison set.
HIGHLIGHT = "mthab_mse"
BASELINE = "climatology"      # reference used for the skill score
MARK = "  ▶"                  # appended to the proposed model's labels

# metric label, results-column TEMPLATE, +1 = higher is better, number format
METRIC_SPECS = [
    ("Chl-$a$ RMSE (µg/L)", "full_year|D+{h}|RMSE",       -1, "{:.2f}"),
    ("Peak-RMSE (µg/L)",    "full_year|D+{h}|PeakRMSE",   -1, "{:.1f}"),
    ("$R^2$",               "full_year|D+{h}|R2",         +1, "{:.2f}"),
    ("TA$_{\\pm1}$",        "full_year|D+{h}|TA1",        +1, "{:.3f}"),
    ("F1 (event-adj.)",     "event_adjacent|D+{h}|F1",    +1, "{:.2f}"),
    ("AUPRC (event-adj.)",  "event_adjacent|D+{h}|AUPRC", +1, "{:.2f}"),
    ("MCC (event-adj.)",    "event_adjacent|D+{h}|MCC",   +1, "{:.2f}"),
    ("TSS (event-adj.)",    "event_adjacent|D+{h}|TSS",   +1, "{:.2f}"),
    ("Accuracy",            "full_year|D+{h}|accuracy",   +1, "{:.3f}"),
]

HORIZONS = list(range(1, 8))          # D+1 .. D+7, all trained jointly


def metrics_for(h: int):
    return [(lab, col.replace("{h}", str(h)), better, fmt)
            for lab, col, better, fmt in METRIC_SPECS]


# The horizon the D+h figures are rendered at. main() rebinds both when
# --horizon / --all_horizons is used; every figure reads the globals.
HORIZON = 1
HZ = 0                                # zero-based index into the prediction arrays
METRICS = metrics_for(HORIZON)


def set_horizon(h: int):
    global HORIZON, HZ, METRICS
    HORIZON, HZ = h, h - 1
    METRICS = metrics_for(h)


CAPTIONS = {
    "fig01_regression": "Next-day (D+1) Chl-a regression accuracy on the 2020 "
        "full-year timeline. Bars are the seed mean, whiskers the s.d.",
    "fig02_detection": "Next-day red-tide detection on the event-adjacent "
        "subset. Bars are the seed mean, whiskers the s.d.",
    "fig03_heatmap": "All nine reported metrics, min-max normalised per metric "
        "so that 1 is the best model and 0 the worst.",
    "fig04_skill": "Skill score against the day-of-year climatology reference. "
        "Positive values indicate an improvement over the reference forecast.",
    "fig05_complexity": "Forecast error against model size. The dashed line is "
        "the Pareto front; models above it are dominated.",
    "fig06_rank": "Rank of every model on every metric (1 = best). A flat, low "
        "profile indicates a model that is good on all criteria at once.",
    "fig07_horizons": "Degradation of skill with forecast lead time, D+1 to D+7.",
    "fig08_taylor": "Taylor diagram of the D+1 Chl-a forecasts: correlation, "
        "normalised standard deviation and centred RMS difference in one plot.",
    "fig09_scatter": "Observed against predicted Chl-a on the 2020 test year.",
    "fig10_pr": "Precision-recall curves for the red-tide task on the "
        "event-adjacent subset. The dashed line is the no-skill rate.",
    "fig11_reliability": "Reliability diagrams. A calibrated model follows the "
        "diagonal; points below it indicate over-confident probabilities.",
    "fig12_errors": "Distribution of the signed prediction error (predicted "
        "minus observed Chl-a). A box centred off zero indicates bias.",
    "fig13_timeseries": "Observed and predicted Chl-a through the 2020 test "
        "year at the station with the most red-tide days.",
    "fig15_horizon_matrix": "Every model at every lead time. Cell colour is "
        "the rank within that lead time, so each column is read independently.",
    "fig14_proposed": "The proposed MT-HAB-G3D against the full baseline "
        "set: margin over the strongest baseline on each metric, the pairwise "
        "RMSE gap to every baseline, and the error distribution.",
}


def set_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
        "text.color": INK, "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "grid.alpha": 0.7, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.6, "lines.markersize": 5,
        "legend.frameon": False, "figure.dpi": 120,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def _family_handles(extra=None):
    h = [Patch(facecolor=FAMILY_COLOR[f], edgecolor="none", label=FAMILY_LABEL[f])
         for f in FAMILY_ORDER]
    return h + list(extra or [])


def family_legend(ax, loc="lower right", extra=None):
    ax.legend(handles=_family_handles(extra), loc=loc, fontsize=7.5, ncol=1)


def family_legend_fig(fig, extra=None, ncol=5, y=0.005):
    """Figure-level legend below the panels — never collides with the marks."""
    fig.legend(handles=_family_handles(extra), loc="lower center",
               bbox_to_anchor=(0.5, y), ncol=ncol, fontsize=8, frameon=False)


# --------------------------------------------------------------------------- #
# 2.  Data access
# --------------------------------------------------------------------------- #
def load_results(paths) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in paths if os.path.exists(p)]
    if not frames:
        raise SystemExit("no results CSV found - run `python -m src.run_table3` first")
    return pd.concat(frames, ignore_index=True)


def agg_table(res: pd.DataFrame) -> pd.DataFrame:
    """One row per model: mean, sd and seed count for every reported metric."""
    rows = []
    for k in KEYS:
        d = res[res["name"] == k]
        if not len(d):
            continue
        row = {"key": k, "label": LABEL[k], "family": FAMILY[k], "seeds": len(d)}
        for lab, col, _better, _fmt in METRICS:
            v = pd.to_numeric(d[col], errors="coerce").dropna() if col in d else pd.Series(dtype=float)
            row[col] = v.mean() if len(v) else np.nan
            row[col + "|sd"] = v.std(ddof=1) if len(v) > 1 else 0.0
        npar = pd.to_numeric(d.get("n_params", pd.Series(dtype=float)), errors="coerce").dropna()
        row["n_params"] = npar.mean() if len(npar) else np.nan
        rt = pd.to_numeric(d.get("runtime_s", pd.Series(dtype=float)), errors="coerce").dropna()
        row["runtime_s"] = rt.mean() if len(rt) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("key")


def find_preds(runs_dir: str, key: str):
    """All preds_test.npz belonging to a model key, sorted by seed."""
    hits = sorted(glob.glob(os.path.join(runs_dir, f"*_{key}_s*", "preds_test.npz")))
    # '*_lstm_s*' would also match '*_cnn_lstm_s*'; filter on the exact key
    out = []
    for h in hits:
        tag = os.path.basename(os.path.dirname(h))
        core = tag.split("_s")[0]
        core = core.split("_", 1)[1] if "_" in core else core
        if core == key:
            out.append(h)
    return out


def load_pred(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


# --------------------------------------------------------------------------- #
# 3.  Figures built from the results table alone
# --------------------------------------------------------------------------- #
def _ranked_bars(ax, t, col, sd_col, better, fmt, title):
    d = t[[col, sd_col]].dropna(subset=[col])
    d = d.sort_values(col, ascending=(better > 0))     # best ends up at the top
    y = np.arange(len(d))
    for i, (k, r) in enumerate(d.iterrows()):
        ax.barh(y[i], r[col], xerr=(r[sd_col] if r[sd_col] > 0 else None),
                color=COLOR[k], hatch=HATCH[k], height=0.72,
                edgecolor=SURFACE if k != HIGHLIGHT else INK,
                linewidth=0.8 if k != HIGHLIGHT else 1.4,
                error_kw=dict(ecolor=INK_MUTED, elinewidth=0.9, capsize=2.5))
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL[k] + (MARK if k == HIGHLIGHT else "")
                        for k in d.index],
                       fontsize=7.6,
                       color=INK)
    ax.set_title(title, loc="left", color=INK)
    ax.grid(axis="y", visible=False)
    span = float(np.nanmax(np.abs(d[col]))) or 1.0
    for i, (k, r) in enumerate(d.iterrows()):
        off = 0.015 * span * (1 if r[col] >= 0 else -1)
        ax.text(r[col] + off + (r[sd_col] if r[col] >= 0 else -r[sd_col]), y[i],
                fmt.format(r[col]), va="center",
                ha="left" if r[col] >= 0 else "right",
                fontsize=7, color=INK_2)
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, hi + 0.14 * (hi - lo))


def fig01_regression(t, **_):
    sub = METRICS[:4]
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.6))
    for ax, (lab, col, better, fmt) in zip(axes.ravel(), sub):
        arrow = "lower is better" if better < 0 else "higher is better"
        _ranked_bars(ax, t, col, col + "|sd", better, fmt, f"{lab}   ({arrow})")
    fig.suptitle(f"Chl-$a$ regression accuracy at D+{HORIZON} (2020 test year)",
                 x=0.012, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0.055, 1, 0.965])
    family_legend_fig(fig)
    return fig


def fig02_detection(t, **_):
    sub = METRICS[4:]
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.0))
    for ax, (lab, col, better, fmt) in zip(axes.ravel(), sub):
        _ranked_bars(ax, t, col, col + "|sd", better, fmt, f"{lab}   (higher is better)")
    axes.ravel()[-1].axis("off")
    family_legend(axes.ravel()[-1], loc="center")
    fig.suptitle(f"Red-tide detection at D+{HORIZON} (event-adjacent subset)",
                 x=0.012, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    return fig


def fig03_heatmap(t, **_):
    cols = [c for _l, c, _b, _f in METRICS]
    M = t[cols].astype(float)
    Z = np.zeros_like(M.values)
    for j, (_lab, col, better, _fmt) in enumerate(METRICS):
        v = M[col].values.astype(float)
        lo, hi = np.nanmin(v), np.nanmax(v)
        n = (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)
        Z[:, j] = n if better > 0 else 1 - n

    fig, ax = plt.subplots(figsize=(10.6, 6.4))
    im = ax.imshow(Z, cmap=SEQ_CMAP, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(METRICS)))
    ax.set_xticklabels([m[0] for m in METRICS], rotation=32, ha="right", fontsize=8)
    ax.set_yticks(range(len(M)))
    ax.set_yticklabels([t.loc[k, "label"] + (MARK if k == HIGHLIGHT else "")
                        for k in M.index], fontsize=8)
    for i, k in enumerate(M.index):
        for j, (_lab, col, _b, fmt) in enumerate(METRICS):
            val = M.iloc[i, j]
            if np.isnan(val):
                continue
            ax.text(j, i, fmt.format(val), ha="center", va="center", fontsize=6.8,
                    color="#ffffff" if Z[i, j] > 0.62 else INK_2)
    for i, k in enumerate(M.index):
        if k == HIGHLIGHT:
            ax.axhline(i - 0.5, color=INK, lw=1.2)
            ax.axhline(i + 0.5, color=INK, lw=1.2)
    ax.grid(False)
    ax.set_title("Normalised performance (1 = best model on that metric, 0 = worst)",
                 loc="left", color=INK)
    fig.colorbar(im, ax=ax, shrink=0.75, label="normalised score")
    fig.tight_layout()
    return fig


def fig04_skill(t, **_):
    """Skill score against the climatology reference: 1 - err/err_ref for error
    metrics, and the plain difference for score metrics."""
    if BASELINE not in t.index:
        return None
    ref = t.loc[BASELINE]
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 5.6))
    panels = [("Chl-$a$ RMSE", "full_year|D+1|RMSE", "skill"),
              ("$R^2$", "full_year|D+1|R2", "delta"),
              ("F1 (event-adj.)", "event_adjacent|D+1|F1", "delta")]
    for ax, (lab, col, mode) in zip(axes, panels):
        vals = {}
        for k in t.index:
            if k == BASELINE or np.isnan(t.loc[k, col]):
                continue
            if mode == "skill":
                vals[k] = 1.0 - t.loc[k, col] / ref[col]
            else:
                vals[k] = t.loc[k, col] - ref[col]
        d = pd.Series(vals).sort_values()
        y = np.arange(len(d))
        vmax = max(abs(d.min()), abs(d.max())) or 1.0
        for i, (k, v) in enumerate(d.items()):
            ax.barh(y[i], v, color=DIV_CMAP(0.5 + 0.5 * v / vmax),
                    edgecolor=INK if k == HIGHLIGHT else SURFACE,
                    linewidth=1.4 if k == HIGHLIGHT else 0.8,
                    hatch=HATCH[k], height=0.72)
        ax.axvline(0, color=INK_2, lw=1.0)
        ax.set_yticks(y)
        ax.set_yticklabels([LABEL[k] + (MARK if k == HIGHLIGHT else "")
                            for k in d.index], fontsize=7.4, color=INK)
        ax.set_xlabel("skill score" if mode == "skill" else "difference")
        ax.set_title(lab, loc="left", color=INK)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Improvement over the day-of-year climatology reference forecast "
                 "(> 0 means better than the reference)",
                 x=0.012, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def fig05_complexity(t, **_):
    d = t.dropna(subset=[f"full_year|D+{HORIZON}|RMSE"]).copy()
    d["n_params"] = pd.to_numeric(d["n_params"], errors="coerce").fillna(0.0)
    # models without trainable parameters cannot sit on a log parameter axis;
    # they are reported in a footnote instead of being drawn at a fake x = 1
    nonparam = d[d["n_params"] <= 0]
    d = d[d["n_params"] > 0]
    if d.empty:
        return None

    fig, ax = plt.subplots(figsize=(10.4, 6.6))
    # alternate the label side so the dense cluster stays readable
    order = d.sort_values("n_params").index.tolist()
    offsets = [(10, 5), (10, -11), (-10, 7), (-10, -13)]
    for i, k in enumerate(order):
        r = d.loc[k]
        x, yv = r["n_params"], r[f"full_year|D+{HORIZON}|RMSE"]
        ax.scatter(x, yv, s=180 if k == HIGHLIGHT else 85,
                   color=COLOR[k], edgecolor=INK if k == HIGHLIGHT else SURFACE,
                   linewidth=1.6 if k == HIGHLIGHT else 1.0, zorder=3)
        dx, dy = offsets[i % len(offsets)]
        ax.annotate(LABEL[k] + (MARK if k == HIGHLIGHT else ""), (x, yv),
                    textcoords="offset points", xytext=(dx, dy), fontsize=6.9,
                    ha="left" if dx > 0 else "right",
                    color=INK if k == HIGHLIGHT else INK_2, zorder=4)
    # Pareto front: minimal RMSE at or below each parameter count
    front_x, front_y, best = [], [], np.inf
    for k in order:
        r = d.loc[k]
        if r[f"full_year|D+{HORIZON}|RMSE"] < best:
            best = r[f"full_year|D+{HORIZON}|RMSE"]
            front_x.append(r["n_params"])
            front_y.append(best)
    ax.step(front_x, front_y, where="post", ls="--", lw=1.2, color=INK_MUTED, zorder=2)
    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10, subs=tuple(np.arange(2, 10) * 0.1)))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _p: f"{v/1000:.0f}k" if v >= 1000 else f"{v:.0f}"))
    ax.tick_params(axis="x", which="minor", labelsize=6.5)
    ax.set_xlabel("trainable parameters (log scale)")
    ax.set_ylabel(f"Chl-$a$ RMSE at D+{HORIZON} (µg/L)   — lower is better")
    ax.set_title("Accuracy against model complexity", loc="left", color=INK)
    ax.margins(x=0.18, y=0.14)
    if len(nonparam):
        note = "; ".join(f"{LABEL[k]} {nonparam.loc[k, f'full_year|D+{HORIZON}|RMSE']:.2f}"
                         for k in nonparam.index)
        fig.text(0.012, 0.012, f"Not shown (no trainable parameters) — RMSE: {note}",
                 fontsize=7.2, color=INK_MUTED)
    fig.tight_layout(rect=[0, 0.075, 1, 1])
    family_legend_fig(fig, extra=[Line2D([0], [0], ls="--", color=INK_MUTED,
                                         label="Pareto front")], ncol=6, y=0.038)
    return fig


def fig06_rank(t, **_):
    ranks = pd.DataFrame(index=t.index)
    for lab, col, better, _fmt in METRICS:
        v = t[col].astype(float)
        ranks[lab] = v.rank(ascending=(better < 0), method="min")
    fig, ax = plt.subplots(figsize=(11.6, 6.2))
    x = np.arange(len(ranks.columns))
    for k in ranks.index:
        ax.plot(x, ranks.loc[k].values, marker="o",
                color=COLOR[k], alpha=1.0 if k == HIGHLIGHT else 0.55,
                lw=2.6 if k == HIGHLIGHT else 1.3,
                markersize=7 if k == HIGHLIGHT else 4.5, zorder=3 if k == HIGHLIGHT else 2)
    # direct labels for the four models that matter most, never all of them
    for k in [HIGHLIGHT, "lstm", "gru", BASELINE]:
        if k in ranks.index:
            ax.annotate(LABEL[k], (x[-1], ranks.loc[k].values[-1]),
                        textcoords="offset points", xytext=(8, 0), fontsize=7.6,
                        color=COLOR[k], va="center", fontweight="bold" if k == HIGHLIGHT else "normal")
    ax.set_xticks(x)
    ax.set_xticklabels(ranks.columns, rotation=28, ha="right", fontsize=8)
    ax.set_ylabel("rank  (1 = best)")
    ax.invert_yaxis()
    ax.set_xlim(-0.3, len(x) - 0.3 + 2.2)
    ax.set_title("Rank profile across all nine reported metrics", loc="left", color=INK)
    family_legend(ax, loc="lower left")
    fig.tight_layout()
    return fig


def fig07_horizons(t, res=None, **_):
    """Lead-time sweep. All seven heads come from ONE trained model (direct
    multi-horizon), so this is a property of the same run that produced D+1."""
    # GRU is included on purpose: it is the strongest D+1 baseline, so leaving it
    # out of the lead-time comparison would overstate the proposed model.
    keys = [k for k in [BASELINE, "gru", "lstm", "cnn_lstm", HIGHLIGHT] if k in t.index]
    panels = [("Chl-$a$ RMSE (µg/L)", "full_year|D+{h}|RMSE", "lower is better"),
              ("$R^2$", "full_year|D+{h}|R2", "higher is better"),
              ("F1, event-adjacent", "event_adjacent|D+{h}|F1", "higher is better"),
              ("AUPRC, event-adjacent", "event_adjacent|D+{h}|AUPRC", "higher is better")]

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.4))
    any_data = False
    for ax, (ylab, tmpl, note) in zip(axes.ravel(), panels):
        for k in keys:
            d = res[res["name"] == k]
            vals, errs = [], []
            for h in HORIZONS:
                c = tmpl.replace("{h}", str(h))
                v = pd.to_numeric(d[c], errors="coerce").dropna() if c in d else pd.Series(dtype=float)
                vals.append(v.mean() if len(v) else np.nan)
                errs.append(v.std(ddof=1) if len(v) > 1 else 0.0)
            if np.all(np.isnan(vals)):
                continue
            any_data = True
            ax.errorbar(HORIZONS, vals, yerr=errs, marker="o", capsize=2.5,
                        color=COLOR[k], elinewidth=0.9,
                        lw=2.8 if k == HIGHLIGHT else 1.5,
                        markersize=7.5 if k == HIGHLIGHT else 5,
                        zorder=4 if k == HIGHLIGHT else 2,
                        label=LABEL[k] + (MARK if k == HIGHLIGHT else ""))
        ax.set_xlabel("forecast lead time (days)")
        ax.set_ylabel(f"{ylab}   — {note}")
        ax.set_xticks(HORIZONS)
        ax.set_xticklabels([f"D+{h}" for h in HORIZONS])
        ax.set_title(ylab, loc="left", color=INK)
    if not any_data:
        plt.close(fig)
        return None
    axes[0, 0].legend(fontsize=7.6, loc="best")
    fig.suptitle("Skill against forecast lead time — one jointly trained model, "
                 "seven direct heads (D+1 … D+7)",
                 x=0.012, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    return fig


def fig15_horizon_matrix(t, res=None, **_):
    """Model x lead-time matrix: where each model stops being useful."""
    specs = [("Chl-$a$ RMSE (µg/L)", "full_year|D+{h}|RMSE", -1, "{:.2f}"),
             ("F1, event-adjacent", "event_adjacent|D+{h}|F1", +1, "{:.2f}")]
    keys = [k for k in KEYS if k in t.index]
    if not keys:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.6))
    for ax, (title, tmpl, better, fmt) in zip(axes, specs):
        M = np.full((len(keys), len(HORIZONS)), np.nan)
        for i, k in enumerate(keys):
            d = res[res["name"] == k]
            for j, h in enumerate(HORIZONS):
                c = tmpl.replace("{h}", str(h))
                v = pd.to_numeric(d[c], errors="coerce").dropna() if c in d else pd.Series(dtype=float)
                if len(v):
                    M[i, j] = v.mean()
        # normalise per COLUMN so each lead time is ranked on its own scale
        Z = np.full_like(M, np.nan)
        for j in range(M.shape[1]):
            col = M[:, j]
            lo, hi = np.nanmin(col), np.nanmax(col)
            n = (col - lo) / (hi - lo) if hi > lo else np.zeros_like(col)
            Z[:, j] = n if better > 0 else 1 - n
        im = ax.imshow(Z, cmap=SEQ_CMAP, vmin=0, vmax=1, aspect="auto")
        for i in range(len(keys)):
            for j in range(len(HORIZONS)):
                if np.isnan(M[i, j]):
                    continue
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center",
                        fontsize=7, color="#ffffff" if Z[i, j] > 0.62 else INK_2)
            if keys[i] == HIGHLIGHT:
                ax.axhline(i - 0.5, color=INK, lw=1.3)
                ax.axhline(i + 0.5, color=INK, lw=1.3)
        ax.set_xticks(range(len(HORIZONS)))
        ax.set_xticklabels([f"D+{h}" for h in HORIZONS], fontsize=8)
        ax.set_yticks(range(len(keys)))
        ax.set_yticklabels([LABEL[k] + (MARK if k == HIGHLIGHT else "") for k in keys],
                           fontsize=8)
        ax.grid(False)
        ax.set_title(title + "   (colour = rank within that lead time)",
                     loc="left", color=INK, fontsize=9.5)
        fig.colorbar(im, ax=ax, shrink=0.7, label="best (1) … worst (0)")
    fig.suptitle("Every model at every lead time, D+1 … D+7",
                 x=0.012, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


# --------------------------------------------------------------------------- #
# 4.  Figures that need per-run predictions
# --------------------------------------------------------------------------- #
def _reg_pairs(z, h=None):
    h = HZ if h is None else h
    m = z["m_reg"][:, h].astype(bool)
    return z["y_reg_raw"][m, h], z["reg_raw"][m, h]


def _cls_pairs(z, h=None, event_only=True):
    h = HZ if h is None else h
    m = z["m_cls"][:, h].astype(bool)
    if event_only:
        m = m & z["event_adj"].astype(bool)
    return z["y_cls"][m, h].astype(int), z["prob"][m, h]


def fig08_taylor(t, preds=None, **_):
    items = [(k, preds[k][0]) for k in KEYS if k in preds]
    if not items:
        return None
    fig = plt.figure(figsize=(8.4, 7.4))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_thetamin(0)
    ax.set_thetamax(90)

    ref_sd = None
    pts = []
    for k, p in items:
        y, yh = _reg_pairs(load_pred(p))
        if len(y) < 3:
            continue
        r = float(np.corrcoef(y, yh)[0, 1])
        sd = float(np.std(yh) / (np.std(y) or 1.0))
        ref_sd = 1.0
        pts.append((k, np.arccos(np.clip(r, -1, 1)), sd, r))
    if not pts:
        plt.close(fig)
        return None

    rmax = max(1.6, max(p[2] for p in pts) * 1.12)
    # centred-RMS difference contours around the reference point
    th = np.linspace(0, np.pi / 2, 200)
    for crms in np.arange(0.25, 2.01, 0.25):
        xx = ref_sd + crms * np.cos(np.linspace(0, 2 * np.pi, 400))
        yy = crms * np.sin(np.linspace(0, 2 * np.pi, 400))
        rr = np.sqrt(xx ** 2 + yy ** 2)
        tt = np.arctan2(yy, xx)
        ok = (tt >= 0) & (tt <= np.pi / 2) & (rr <= rmax)
        ax.plot(tt[ok], rr[ok], color=GRID, lw=0.8, zorder=1)
    ax.plot(0, ref_sd, marker="*", markersize=15, color=INK, zorder=5)
    ax.annotate("Observed", (0, ref_sd), textcoords="offset points",
                xytext=(6, 10), fontsize=8, color=INK)

    for k, theta, sd, r in pts:
        ax.plot(theta, sd, marker="o", color=COLOR[k],
                markersize=11 if k == HIGHLIGHT else 7,
                markeredgecolor=INK if k == HIGHLIGHT else SURFACE,
                markeredgewidth=1.4 if k == HIGHLIGHT else 0.8, zorder=4)
    for k, theta, sd, r in pts:
        if k in (HIGHLIGHT, "lstm", "gru", BASELINE):
            ax.annotate(LABEL[k], (theta, sd), textcoords="offset points",
                        xytext=(8, 5), fontsize=7.4, color=COLOR[k])

    ax.set_rmax(rmax)
    ticks = [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    ax.set_thetagrids(np.degrees(np.arccos(ticks)), labels=[f"{v:g}" for v in ticks])
    ax.set_xlabel("normalised standard deviation")
    ax.text(np.radians(45), rmax * 1.06, "correlation", ha="center", fontsize=9, color=INK_2)
    ax.set_title(f"Taylor diagram — D+{HORIZON} Chl-$a$ (grey arcs: centred RMS difference)",
                 loc="left", color=INK, pad=18)
    family_legend(ax, loc=(1.02, 0.02))
    fig.tight_layout()
    return fig


def fig09_scatter(t, preds=None, **_):
    keys = [k for k in [HIGHLIGHT, "lstm", "cnn_lstm", "gru", "histgb", BASELINE]
            if k in preds]
    if not keys:
        return None
    n = len(keys)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 4.2 * rows), squeeze=False)
    lim = None
    for ax, k in zip(axes.ravel(), keys):
        y, yh = _reg_pairs(load_pred(preds[k][0]))
        if lim is None:
            lim = [0, float(np.percentile(y, 99.5))]
        ax.scatter(y, yh, s=7, alpha=0.28, color=COLOR[k], edgecolor="none")
        ax.plot(lim, lim, ls="--", lw=1.0, color=INK_MUTED)
        ax.set_xlim(lim); ax.set_ylim(lim)
        rmse = float(np.sqrt(np.mean((yh - y) ** 2)))
        r2 = 1 - np.mean((yh - y) ** 2) / (np.var(y) or 1)
        ax.set_title(f"{LABEL[k]}" + (MARK if k == HIGHLIGHT else ""),
                     loc="left", color=INK, fontsize=9)
        ax.text(0.04, 0.93, f"RMSE {rmse:.2f}\n$R^2$ {r2:.2f}", transform=ax.transAxes,
                fontsize=7.6, va="top", color=INK_2)
        ax.set_xlabel("observed Chl-$a$ (µg/L)")
        ax.set_ylabel("predicted Chl-$a$ (µg/L)")
    for ax in axes.ravel()[len(keys):]:
        ax.axis("off")
    fig.suptitle(f"Observed vs predicted Chl-$a$, D+{HORIZON}, 2020 test year",
                 x=0.012, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def fig10_pr(t, preds=None, **_):
    keys = [k for k in KEYS if k in preds]
    if not keys:
        return None
    fig, ax = plt.subplots(figsize=(7.8, 6.2))
    base = None
    for k in keys:
        y, p = _cls_pairs(load_pred(preds[k][0]))
        if y.sum() == 0:
            continue
        base = y.mean()
        o = np.argsort(-p)
        tp = np.cumsum(y[o]); fp = np.cumsum(1 - y[o])
        rec = tp / y.sum()
        prec = tp / np.maximum(tp + fp, 1e-9)
        ax.plot(rec, prec, color=COLOR[k], alpha=1.0 if k == HIGHLIGHT else 0.6,
                lw=2.6 if k == HIGHLIGHT else 1.4,
                label=LABEL[k] + (MARK if k == HIGHLIGHT else ""))
    if base is not None:
        ax.axhline(base, ls="--", lw=1.0, color=INK_MUTED, label="no-skill rate")
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_title(f"Precision-recall, red-tide at D+{HORIZON} (event-adjacent subset)",
                 loc="left", color=INK)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout()
    return fig


def fig11_reliability(t, preds=None, **_):
    keys = [k for k in [HIGHLIGHT, "lstm", "cnn_lstm", BASELINE] if k in preds]
    if not keys:
        return None
    fig, axes = plt.subplots(1, len(keys), figsize=(3.9 * len(keys), 4.0), squeeze=False)
    bins = np.linspace(0, 1, 11)
    for ax, k in zip(axes.ravel(), keys):
        y, p = _cls_pairs(load_pred(preds[k][0]))
        cx, cy, cn = [], [], []
        for i in range(10):
            s = (p >= bins[i]) & (p < bins[i + 1])
            if s.sum() > 3:
                cx.append(p[s].mean()); cy.append(y[s].mean()); cn.append(s.sum())
        ax.plot([0, 1], [0, 1], ls="--", lw=1.0, color=INK_MUTED)
        if cx:
            ax.plot(cx, cy, marker="o", color=COLOR[k],
                    markersize=8 if k == HIGHLIGHT else 6,
                    lw=2.4 if k == HIGHLIGHT else 1.6)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("predicted probability")
        ax.set_ylabel("observed frequency")
        ax.set_title(LABEL[k] + (MARK if k == HIGHLIGHT else ""), loc="left",
                     color=INK, fontsize=9)
    fig.suptitle("Probability calibration (event-adjacent subset)",
                 x=0.012, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def fig12_errors(t, preds=None, **_):
    keys = [k for k in KEYS if k in preds]
    if not keys:
        return None
    data, cols = [], []
    for k in keys:
        y, yh = _reg_pairs(load_pred(preds[k][0]))
        data.append(yh - y)
        cols.append(k)
    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    bp = ax.boxplot(data, vert=True, patch_artist=True, showfliers=False,
                    widths=0.62, medianprops=dict(color=INK, lw=1.4),
                    whiskerprops=dict(color=INK_MUTED), capprops=dict(color=INK_MUTED))
    for patch, k in zip(bp["boxes"], cols):
        patch.set_facecolor(COLOR[k])
        patch.set_alpha(0.85)
        patch.set_hatch(HATCH[k])
        patch.set_edgecolor(INK if k == HIGHLIGHT else SURFACE)
        patch.set_linewidth(1.6 if k == HIGHLIGHT else 0.8)
    ax.axhline(0, color=INK_2, lw=1.0)
    ax.set_xticklabels([LABEL[k] + (MARK if k == HIGHLIGHT else "") for k in cols],
                       rotation=30, ha="right", fontsize=7.6)
    ax.set_ylabel("prediction error (predicted − observed, µg/L)")
    ax.set_title(f"Signed error distribution at D+{HORIZON} (whiskers 1.5 IQR, outliers hidden)",
                 loc="left", color=INK)
    fig.tight_layout(rect=[0, 0.075, 1, 1])
    family_legend_fig(fig, y=0.012)
    return fig


def fig13_timeseries(t, preds=None, **_):
    keys = [k for k in ["lstm", "cnn_lstm", HIGHLIGHT] if k in preds]
    if not keys:
        return None
    ref = load_pred(preds[keys[0]][0])
    pos = ref["y_cls"][:, HZ] > 0.5
    st = int(np.bincount(ref["station"][pos]).argmax()) if pos.any() else int(ref["station"][0])

    fig, axes = plt.subplots(2, 1, figsize=(13.0, 7.2), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1]})
    m = (ref["station"] == st) & ref["m_reg"][:, HZ].astype(bool)
    d = ref["date"][m].astype("datetime64[D]")
    axes[0].plot(d, ref["y_reg_raw"][m, HZ], color=INK, lw=1.8, label="Observed", zorder=5)
    for k in keys:
        z = load_pred(preds[k][0])
        mk = (z["station"] == st) & z["m_reg"][:, HZ].astype(bool)
        axes[0].plot(z["date"][mk].astype("datetime64[D]"), z["reg_raw"][mk, HZ],
                     color=COLOR[k], alpha=1.0 if k == HIGHLIGHT else 0.75,
                     lw=2.2 if k == HIGHLIGHT else 1.3,
                     label=LABEL[k] + (MARK if k == HIGHLIGHT else ""))
    axes[0].set_ylabel("Chl-$a$ (µg/L)")
    axes[0].legend(fontsize=7.6, ncol=4, loc="upper left")
    axes[0].set_title(f"2020 test year, D+{HORIZON} — station index {st} (most red-tide days)",
                      loc="left", color=INK)

    mc = (ref["station"] == st) & ref["m_cls"][:, HZ].astype(bool)
    dc = ref["date"][mc].astype("datetime64[D]")
    axes[1].fill_between(dc, 0, ref["y_cls"][mc, HZ], step="mid", color=INK_MUTED,
                         alpha=0.28, label="Observed red tide")
    for k in keys:
        z = load_pred(preds[k][0])
        mk = (z["station"] == st) & z["m_cls"][:, HZ].astype(bool)
        axes[1].plot(z["date"][mk].astype("datetime64[D]"), z["prob"][mk, HZ],
                     color=COLOR[k], alpha=1.0 if k == HIGHLIGHT else 0.7,
                     lw=2.0 if k == HIGHLIGHT else 1.1,
                     label=LABEL[k] + " probability")
    axes[1].set_ylabel("red-tide probability")
    axes[1].set_ylim(-0.02, 1.05)
    axes[1].legend(fontsize=7.2, ncol=4, loc="upper left")
    fig.tight_layout()
    return fig


def fig14_proposed(t, preds=None, res=None, **_):
    """The headline figure: how far the proposed model sits ahead of the field."""
    if HIGHLIGHT not in t.index or len(t) < 2:
        return None
    others = [k for k in t.index if k != HIGHLIGHT]

    fig = plt.figure(figsize=(14.2, 5.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.15, 1.0])

    # --- A: margin over the STRONGEST baseline on each metric --------------- #
    ax = fig.add_subplot(gs[0, 0])
    labs, marg, who = [], [], []
    for lab, col, better, _fmt in METRICS:
        vp = t.loc[HIGHLIGHT, col]
        vb = t.loc[others, col].astype(float)
        vb = vb.dropna()
        if np.isnan(vp) or vb.empty:
            continue
        best_k = vb.idxmin() if better < 0 else vb.idxmax()
        best_v = vb.loc[best_k]
        denom = abs(best_v) if best_v != 0 else 1.0
        labs.append(lab)
        marg.append((vp - best_v) / denom * 100.0 * better)
        who.append(best_k)
    y = np.arange(len(labs))
    # a single blown-up margin (MCC/TSS can swing by hundreds of percent when the
    # baseline is near zero) would flatten every other bar, so the axis is set on
    # the robust spread and the outliers are drawn clipped with the true value
    absm = np.abs(marg)
    vmax = float(np.percentile(absm, 75)) * 2.2 or 1.0
    vmax = max(vmax, 5.0)
    shown = np.clip(marg, -vmax, vmax)
    ax.barh(y, shown, height=0.68, edgecolor=INK, linewidth=0.9,
            color=[DIV_CMAP(0.5 + 0.5 * np.clip(v / vmax, -1, 1)) for v in marg])
    for i, (v, sv) in enumerate(zip(marg, shown)):
        if abs(v) > vmax:                      # clipped -> print the real number
            ax.text(sv - 0.02 * vmax * np.sign(v), y[i], f"{v:+.0f}%",
                    va="center", ha="right" if v > 0 else "left",
                    fontsize=6.4, color=SURFACE, fontweight="bold")
    ax.axvline(0, color=INK_2, lw=1.1)
    ax.set_yticks(y)
    # the beaten baseline goes INTO the tick label, so nothing can collide with
    # the bars regardless of sign
    ax.set_yticklabels([f"{lab}\n(best baseline: {LABEL[k]})"
                        for lab, k in zip(labs, who)], fontsize=7.0)
    ax.set_xlim(-1.25 * vmax, 1.25 * vmax)
    ax.set_xlabel("margin over the strongest baseline (%)")
    ax.set_title("A · Proposed vs the best baseline, metric by metric",
                 loc="left", color=INK)
    ax.grid(axis="y", visible=False)

    # --- B: pairwise RMSE gap against every baseline ------------------------ #
    ax2 = fig.add_subplot(gs[0, 1])
    col = f"full_year|D+{HORIZON}|RMSE"
    vp = t.loc[HIGHLIGHT, col]
    gaps = (t.loc[others, col].astype(float) - vp).dropna().sort_values()
    y2 = np.arange(len(gaps))
    for i, (k, g) in enumerate(gaps.items()):
        ax2.plot([0, g], [y2[i], y2[i]], color=GRID, lw=2.4, zorder=1,
                 solid_capstyle="round")
        ax2.scatter(g, y2[i], s=64, color=COLOR[k], edgecolor=SURFACE,
                    linewidth=0.8, zorder=3)
    ax2.axvline(0, color=COLOR[HIGHLIGHT], lw=2.0, zorder=2)
    ax2.set_yticks(y2)
    ax2.set_yticklabels([LABEL[k] for k in gaps.index], fontsize=7.4)
    ax2.set_xlabel("baseline RMSE − proposed RMSE (µg/L)")
    n_worse = int((gaps > 0).sum())
    n_better = int((gaps < 0).sum())
    if n_better == 0:
        sub = "every baseline is to the right of the proposed model"
    else:
        beat = ", ".join(LABEL[k] for k in gaps.index[gaps < 0])
        sub = (f"{n_worse} of {len(gaps)} baselines are behind; "
               f"ahead of the proposal: {beat}")
    ax2.set_title(f"B · RMSE gap to every baseline\n     {sub}",
                  loc="left", color=INK, fontsize=9.5)
    ax2.grid(axis="y", visible=False)
    ax2.text(0.012, 0.02, f"vertical line = {LABEL[HIGHLIGHT]}",
             transform=ax2.transAxes, fontsize=7, color=COLOR[HIGHLIGHT])

    # --- C: error distribution --------------------------------------------- #
    ax3 = fig.add_subplot(gs[0, 2])
    wanted = [HIGHLIGHT] + [k for k in ["lstm", "cnn_lstm", "gru"] if k in t.index]
    drawn = []
    for k in wanted:
        if not preds or k not in preds:
            continue
        y_, yh_ = _reg_pairs(load_pred(preds[k][0]))
        ax3.hist(yh_ - y_, bins=60, range=(-8, 8), histtype="step",
                 lw=2.4 if k == HIGHLIGHT else 1.2,
                 color=COLOR[k], label=LABEL[k],
                 linestyle="-" if k == HIGHLIGHT else "--", density=True,
                 zorder=3 if k == HIGHLIGHT else 2)
        drawn.append(k)
    if drawn:
        ax3.axvline(0, color=INK_2, lw=1.0)
        ax3.set_xlabel("prediction error (µg/L)")
        ax3.set_ylabel("density")
        ax3.set_title("C · Error distribution", loc="left", color=INK)
        ax3.legend(fontsize=7.2)
        miss = [k for k in wanted if k not in drawn]
        if miss:
            ax3.text(0.02, 0.02,
                     "not shown (preds_test.npz not archived): "
                     + ", ".join(LABEL[k] for k in miss),
                     transform=ax3.transAxes, fontsize=6.4, color=INK_MUTED)
    else:
        ax3.axis("off")
        ax3.text(0.5, 0.5, "prediction files not available",
                 ha="center", va="center", fontsize=8.5, color=INK_MUTED)

    fig.suptitle(f"MT-HAB-G3D (proposed) against the full baseline set — D+{HORIZON}",
                 x=0.012, ha="left", fontsize=12.5, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.935])
    return fig


# --------------------------------------------------------------------------- #
FIGURES = [
    ("f01", "fig01_regression", fig01_regression, False),
    ("f02", "fig02_detection", fig02_detection, False),
    ("f03", "fig03_heatmap", fig03_heatmap, False),
    ("f04", "fig04_skill", fig04_skill, False),
    ("f05", "fig05_complexity", fig05_complexity, False),
    ("f06", "fig06_rank", fig06_rank, False),
    ("f07", "fig07_horizons", fig07_horizons, False),
    ("f08", "fig08_taylor", fig08_taylor, True),
    ("f09", "fig09_scatter", fig09_scatter, True),
    ("f10", "fig10_pr", fig10_pr, True),
    ("f11", "fig11_reliability", fig11_reliability, True),
    ("f12", "fig12_errors", fig12_errors, True),
    ("f13", "fig13_timeseries", fig13_timeseries, True),
    ("f14", "fig14_proposed", fig14_proposed, False),
    ("f15", "fig15_horizon_matrix", fig15_horizon_matrix, False),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="outputs/results.csv")
    ap.add_argument("--results_ext", default="outputs/results_ext.csv")
    ap.add_argument("--runs", default="outputs/runs")
    ap.add_argument("--out", default="outputs/figures")
    ap.add_argument("--only", default="", help="e.g. f01,f05,f13")
    ap.add_argument("--horizon", type=int, default=1, choices=HORIZONS,
                    help="forecast lead time the D+h figures are rendered at "
                         "(f07 and f15 always show all seven)")
    ap.add_argument("--all_horizons", action="store_true",
                    help="render the whole set once per lead time into "
                         "<out>/D+1 ... <out>/D+7, plus the D+1 set in <out>")
    a = ap.parse_args()

    if a.all_horizons:
        for h in HORIZONS:
            sub = os.path.join(a.out, f"D+{h}")
            print(f"\n=================  D+{h}  =================")
            render(a, h, sub)
        print("\n=================  top level = D+1  =================")
        return render(a, 1, a.out)
    return render(a, a.horizon, a.out)


def render(a, horizon, outdir):
    set_horizon(horizon)
    set_style()
    os.makedirs(outdir, exist_ok=True)
    res = load_results([a.results, a.results_ext])
    t = agg_table(res)
    if t.empty:
        raise SystemExit("results contain none of the 14 Table-3 models")

    preds = {}
    for k in KEYS:
        f = find_preds(a.runs, k)
        if f:
            preds[k] = f
    print(f"horizon                 : D+{HORIZON}")
    print(f"models with metrics     : {len(t)} / {len(KEYS)}  -> {list(t.index)}")
    print(f"models with predictions : {len(preds)} / {len(KEYS)}  -> {list(preds)}")
    missing = [k for k in KEYS if k not in t.index]
    if missing:
        print(f"NOT PLOTTED (no results yet): {missing}")

    want = {s for s in a.only.split(",") if s}
    index_rows = []
    suffix = "" if HORIZON == 1 else f"_D{HORIZON}"
    all_pdf = os.path.join(outdir, f"figures_all{suffix}.pdf")
    with PdfPages(all_pdf) as pdf:
        for fid, stem, fn, needs_preds in FIGURES:
            if want and fid not in want:
                continue
            if needs_preds and not preds:
                print(f"  {fid} {stem:22s} SKIPPED - needs outputs/runs/*/preds_test.npz")
                continue
            try:
                fig = fn(t, preds=preds, res=res)
            except Exception as exc:                      # never lose the whole run
                print(f"  {fid} {stem:22s} FAILED  - {type(exc).__name__}: {exc}")
                continue
            if fig is None:
                print(f"  {fid} {stem:22s} SKIPPED - not enough data")
                continue
            path = os.path.join(outdir, stem + suffix + ".pdf")
            fig.savefig(path, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            index_rows.append({"figure": fid, "horizon": f"D+{HORIZON}",
                               "file": stem + suffix + ".pdf",
                               "caption": CAPTIONS.get(stem, "")})
            print(f"  {fid} {stem:22s} -> {path}")

    pd.DataFrame(index_rows).to_csv(
        os.path.join(outdir, f"figure_index{suffix}.csv"), index=False)
    print(f"\ncombined            -> {all_pdf}")
    print(f"caption index       -> {os.path.join(outdir, f'figure_index{suffix}.csv')}")


if __name__ == "__main__":
    main()
