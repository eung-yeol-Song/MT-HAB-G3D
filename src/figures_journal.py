"""
figures_journal.py
==================
Journal-style figures for the manuscript — conventional academic look, not the
dashboard look of ``figures_paper.py``.

    python -m src.figures_journal                    # everything
    python -m src.figures_journal --out outputs/journal
    python -m src.figures_journal --column single    # 3.5 in wide panels

What it produces
----------------
    <out>/journal_all.pdf        every figure, one per page
    <out>/figJ1_leadtime.pdf     skill vs lead time, D+1 … D+7
    <out>/figJ2_aggregate.pdf    D+1 … D+7 averages, ranked bars
    <out>/figJ3_rankplane.pdf    mean rank on regression vs detection
    <out>/figJ4_shortlong.pdf    short (D+1–3) vs long (D+4–7) horizon
    <out>/figJ5_matrix.pdf       model x lead-time grid, greyscale
    <out>/summary_D1_D7.csv      the aggregate statistics behind the figures
    <out>/summary_D1_D7.pdf      the same table, typeset

Style conventions (matched to what most journals print)
------------------------------------------------------
* serif type, 8 pt body, 300 dpi, Type-42 embedded fonts
* full axis box, ticks turned inward on all four sides, no background grid
* figure widths are the standard 3.5 in (single column) / 7.16 in (double)
* **greyscale-safe**: every baseline is grey and separated by marker shape and
  dash pattern; the proposed model is the only black, solid, filled series, so
  the figure still reads when printed without colour
* panels are lettered (a), (b), … as journals expect
* no chartjunk: no shadows, no gradients, no 3-D, no dual axes
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

from .figures_paper import KEYS, LABEL, HIGHLIGHT, HORIZONS, load_results

# --------------------------------------------------------------------------- #
SINGLE_COL, DOUBLE_COL = 3.5, 7.16          # inches, the two standard widths

PROPOSED_C = "#000000"
GREYS = ["#4d4d4d", "#7a7a7a", "#9e9e9e", "#b5b5b5", "#c9c9c9"]
MARKERS = ["o", "s", "^", "D", "v", "<", ">", "P", "X", "*", "h", "p"]
DASHES = [(4, 1.6), (1.4, 1.4), (5, 1.5, 1, 1.5), (3, 1.2, 1, 1.2), (7, 2)]

# the metrics the aggregate is computed over
AGG_SPECS = [
    ("RMSE",     "full_year|D+{h}|RMSE",       -1, "Chl-$a$ RMSE (µg/L)"),
    ("PeakRMSE", "full_year|D+{h}|PeakRMSE",   -1, "Peak-RMSE (µg/L)"),
    ("R2",       "full_year|D+{h}|R2",         +1, "$R^{2}$"),
    ("TA1",      "full_year|D+{h}|TA1",        +1, "TA$_{\\pm 1}$"),
    ("F1",       "event_adjacent|D+{h}|F1",    +1, "F1 (event-adjacent)"),
    ("AUPRC",    "event_adjacent|D+{h}|AUPRC", +1, "AUPRC (event-adjacent)"),
    ("MCC",      "event_adjacent|D+{h}|MCC",   +1, "MCC (event-adjacent)"),
    ("TSS",      "event_adjacent|D+{h}|TSS",   +1, "TSS (event-adjacent)"),
    ("Accuracy", "full_year|D+{h}|accuracy",   +1, "Accuracy"),
]
BETTER = {k: b for k, _t, b, _l in AGG_SPECS}
NICE = {k: l for k, _t, _b, l in AGG_SPECS}


def set_journal_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman", "serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
        "xtick.labelsize": 7.2, "ytick.labelsize": 7.2, "legend.fontsize": 6.8,
        "axes.linewidth": 0.8, "axes.edgecolor": "black", "axes.labelcolor": "black",
        "text.color": "black", "xtick.color": "black", "ytick.color": "black",
        "axes.spines.top": True, "axes.spines.right": True,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 3.2, "ytick.major.size": 3.2,
        "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "axes.grid": False,
        "lines.linewidth": 1.0, "lines.markersize": 3.8,
        "legend.frameon": True, "legend.edgecolor": "black",
        "legend.framealpha": 1.0, "legend.fancybox": False,
        "legend.borderpad": 0.35, "legend.handlelength": 2.4,
        "figure.dpi": 200, "savefig.dpi": 300,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def panel_tag(ax, letter, dx=-0.13, dy=1.04):
    ax.text(dx, dy, f"({letter})", transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="bottom", ha="left")


# --------------------------------------------------------------------------- #
def horizon_matrix(res: pd.DataFrame, metric_key: str):
    """models x D+1..D+7 mean values (and the seed s.d.)."""
    tmpl = dict((k, t) for k, t, _b, _l in AGG_SPECS)[metric_key]
    rows, sds, keys = [], [], []
    for k in KEYS:
        d = res[res["name"] == k]
        if not len(d):
            continue
        m, s = [], []
        for h in HORIZONS:
            c = tmpl.replace("{h}", str(h))
            v = pd.to_numeric(d[c], errors="coerce").dropna() if c in d else pd.Series(dtype=float)
            m.append(v.mean() if len(v) else np.nan)
            s.append(v.std(ddof=1) if len(v) > 1 else 0.0)
        rows.append(m); sds.append(s); keys.append(k)
    idx = [f"D+{h}" for h in HORIZONS]
    return (pd.DataFrame(rows, index=keys, columns=idx),
            pd.DataFrame(sds, index=keys, columns=idx))


def aggregate(res: pd.DataFrame) -> pd.DataFrame:
    """One row per model: the D+1..D+7 average, mean rank and win count for
    every metric, plus an overall mean rank across all metrics x horizons."""
    out, rank_stack = {}, []
    for key, _tmpl, better, _lab in AGG_SPECS:
        M, _ = horizon_matrix(res, key)
        if M.empty:
            continue
        rk = M.rank(axis=0, ascending=(better < 0))
        out[f"{key}_mean"] = M.mean(axis=1)
        out[f"{key}_rank"] = rk.mean(axis=1)
        out[f"{key}_wins"] = (rk == 1).sum(axis=1)
        rank_stack.append(rk)
    t = pd.DataFrame(out)
    if rank_stack:
        allr = pd.concat(rank_stack, axis=1)
        t["overall_rank"] = allr.mean(axis=1)
        t["overall_wins"] = sum((r == 1).sum(axis=1) for r in rank_stack)
        t["n_cells"] = len(rank_stack) * len(HORIZONS)
    t.insert(0, "Model", [LABEL.get(k, k) for k in t.index])
    return t.sort_values("overall_rank")


# --------------------------------------------------------------------------- #
def _series_style(i, k):
    if k == HIGHLIGHT:
        return dict(color=PROPOSED_C, lw=1.6, marker="o", markersize=4.6,
                    markerfacecolor=PROPOSED_C, markeredgecolor=PROPOSED_C,
                    zorder=5, linestyle="-")
    return dict(color=GREYS[i % len(GREYS)], lw=0.95,
                marker=MARKERS[i % len(MARKERS)], markersize=3.4,
                markerfacecolor="white", markeredgewidth=0.8,
                markeredgecolor=GREYS[i % len(GREYS)],
                dashes=DASHES[i % len(DASHES)], zorder=3)


def figJ1_leadtime(res, keys, width):
    specs = [("RMSE", "Chl-$a$ RMSE (µg/L)"), ("R2", "$R^{2}$"),
             ("F1", "F1 (event-adjacent)"), ("AUPRC", "AUPRC (event-adjacent)")]
    fig, axes = plt.subplots(2, 2, figsize=(width, width * 0.72))
    for ax, (letter, (mk, ylab)) in zip(axes.ravel(), zip("abcd", specs)):
        M, S = horizon_matrix(res, mk)
        gi = 0
        for k in keys:
            if k not in M.index:
                continue
            st = _series_style(gi, k)
            if k != HIGHLIGHT:
                gi += 1
            ax.errorbar(HORIZONS, M.loc[k], yerr=S.loc[k], capsize=1.6,
                        elinewidth=0.6, label=LABEL[k], **st)
        ax.set_xticks(HORIZONS)
        ax.set_xlabel("Forecast lead time (d)")
        ax.set_ylabel(ylab)
        panel_tag(ax, letter)
    axes[0, 0].legend(loc="best", ncol=1)
    fig.tight_layout(pad=0.6)
    return fig


def figJ2_aggregate(t, width):
    specs = [("RMSE", "Chl-$a$ RMSE (µg/L)", -1), ("R2", "$R^{2}$", +1),
             ("F1", "F1 (event-adjacent)", +1), ("AUPRC", "AUPRC (event-adjacent)", +1)]
    # 2 x 2 rather than 1 x 4: four horizontal bar charts side by side leave no
    # room for the model names and the axis labels collide
    fig, axes = plt.subplots(2, 2, figsize=(width, width * 0.80))
    for ax, (letter, (mk, lab, better)) in zip(axes.ravel(), zip("abcd", specs)):
        col = f"{mk}_mean"
        if col not in t:
            ax.axis("off"); continue
        d = t[col].dropna().sort_values(ascending=(better > 0))
        y = np.arange(len(d))
        for i, (k, v) in enumerate(d.items()):
            ax.barh(y[i], v, height=0.72,
                    color="black" if k == HIGHLIGHT else "white",
                    edgecolor="black", linewidth=0.8, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels([LABEL[k] for k in d.index], fontsize=6.0)
        ax.set_xlabel(lab)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.7, len(d) - 0.3)
        panel_tag(ax, letter, dx=-0.42, dy=1.02)
    fig.suptitle("Averages over the seven lead times D+1 … D+7", fontsize=8.5, y=1.0)
    fig.tight_layout(pad=0.7, rect=[0, 0, 1, 0.965])
    return fig


def figJ3_rankplane(t, width):
    if "RMSE_rank" not in t or "F1_rank" not in t:
        return None
    fig, ax = plt.subplots(figsize=(width, width * 0.86))
    # alternate the label side so the crowded middle of the plane stays legible
    offs = [(5, 3), (5, -8), (-5, 4), (-5, -9)]
    order = t.sort_values("RMSE_rank").index
    for i, k in enumerate(order):
        r = t.loc[k]
        prop = k == HIGHLIGHT
        ax.scatter(r["RMSE_rank"], r["F1_rank"],
                   s=52 if prop else 26,
                   facecolor="black" if prop else "white",
                   edgecolor="black", linewidth=0.9, zorder=4 if prop else 3)
        dx, dy = (6, 4) if prop else offs[i % len(offs)]
        ax.annotate(LABEL.get(k, k), (r["RMSE_rank"], r["F1_rank"]),
                    textcoords="offset points", xytext=(dx, dy),
                    ha="left" if dx > 0 else "right",
                    fontsize=5.9, fontweight="bold" if prop else "normal")
    n = len(t)
    ax.plot([1, n], [1, n], ls=":", lw=0.7, color="0.6", zorder=1)
    ax.margins(0.16)
    ax.set_xlabel("Mean rank, Chl-$a$ RMSE (1 = best)")
    ax.set_ylabel("Mean rank, red-tide F1 (1 = best)")
    ax.invert_xaxis(); ax.invert_yaxis()
    ax.set_title("Averaged over D+1 … D+7; upper-right corner is best", fontsize=7.6)
    fig.tight_layout(pad=0.6)
    return fig


def figJ4_shortlong(res, width):
    short, long_ = HORIZONS[:3], HORIZONS[3:]
    specs = [("RMSE", "Chl-$a$ RMSE (µg/L)", -1), ("F1", "F1 (event-adjacent)", +1)]
    fig, axes = plt.subplots(1, 2, figsize=(width, width * 0.40))
    for ax, (letter, (mk, lab, better)) in zip(axes, zip("ab", specs)):
        M, _ = horizon_matrix(res, mk)
        if M.empty:
            ax.axis("off"); continue
        s = M[[f"D+{h}" for h in short]].mean(axis=1)
        l = M[[f"D+{h}" for h in long_]].mean(axis=1)
        order = (s + l).sort_values(ascending=(better > 0)).index
        y = np.arange(len(order))
        ax.barh(y - 0.19, s.loc[order], height=0.35, color="0.75",
                edgecolor="black", linewidth=0.6, label="D+1 – D+3", zorder=3)
        ax.barh(y + 0.19, l.loc[order], height=0.35, color="0.35",
                edgecolor="black", linewidth=0.6, label="D+4 – D+7", zorder=3)
        for i, k in enumerate(order):
            if k == HIGHLIGHT:
                ax.axhspan(y[i] - 0.45, y[i] + 0.45, color="0.92", zorder=1)
        ax.set_yticks(y)
        ax.set_yticklabels([LABEL[k] for k in order], fontsize=6.2)
        ax.set_xlabel(lab)
        ax.tick_params(axis="y", length=0)
        ax.legend(loc="lower right")
        panel_tag(ax, letter, dx=-0.62)
    fig.tight_layout(pad=0.6)
    return fig


def figJ5_matrix(res, width):
    specs = [("RMSE", "Chl-$a$ RMSE (µg/L)", -1, "{:.2f}"),
             ("F1", "F1 (event-adjacent)", +1, "{:.2f}")]
    fig, axes = plt.subplots(1, 2, figsize=(width, width * 0.52))
    for ax, (letter, (mk, lab, better, fmt)) in zip(axes, zip("ab", specs)):
        M, _ = horizon_matrix(res, mk)
        if M.empty:
            ax.axis("off"); continue
        Z = M.rank(axis=0, ascending=(better < 0)).values
        Z = 1.0 - (Z - 1) / (Z.shape[0] - 1)                  # 1 = best
        ax.imshow(Z, cmap="Greys", vmin=0, vmax=1.35, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isnan(M.values[i, j]):
                    continue
                ax.text(j, i, fmt.format(M.values[i, j]), ha="center", va="center",
                        fontsize=5.6, color="white" if Z[i, j] > 0.72 else "black")
        for i, k in enumerate(M.index):
            if k == HIGHLIGHT:
                for yy in (i - 0.5, i + 0.5):
                    ax.axhline(yy, color="black", lw=1.1)
        ax.set_xticks(range(len(HORIZONS)))
        ax.set_xticklabels([f"D+{h}" for h in HORIZONS], fontsize=6.6)
        ax.set_yticks(range(len(M.index)))
        ax.set_yticklabels([LABEL[k] for k in M.index], fontsize=6.2)
        ax.tick_params(length=0)
        ax.set_title(lab, fontsize=7.8)
        panel_tag(ax, letter, dx=-0.62, dy=1.06)
    fig.suptitle("Shading = rank within each lead time (darker is better)",
                 fontsize=7.4, y=0.99)
    fig.tight_layout(pad=0.6, rect=[0, 0, 1, 0.95])
    return fig


def table_summary_pdf(t, width):
    cols = [("Model", "Model", None),
            ("RMSE_mean", "RMSE", "{:.2f}"), ("RMSE_rank", "rank", "{:.1f}"),
            ("R2_mean", "$R^{2}$", "{:.2f}"),
            ("F1_mean", "F1", "{:.2f}"), ("F1_rank", "rank", "{:.1f}"),
            ("AUPRC_mean", "AUPRC", "{:.2f}"),
            ("overall_rank", "overall rank", "{:.2f}"),
            ("overall_wins", "wins", "{:.0f}")]
    cols = [c for c in cols if c[0] in t.columns or c[0] == "Model"]
    text = []
    for k, r in t.iterrows():
        row = []
        for cn, _h, fmt in cols:
            v = r[cn]
            row.append(str(v) if fmt is None else
                       ("--" if pd.isna(v) else fmt.format(v)))
        text.append(row)
    fig, ax = plt.subplots(figsize=(width, 0.30 * len(text) + 1.0))
    ax.axis("off")
    tab = ax.table(cellText=text, colLabels=[c[1] for c in cols],
                   cellLoc="center", loc="upper center",
                   colWidths=[0.30] + [0.70 / (len(cols) - 1)] * (len(cols) - 1))
    tab.auto_set_font_size(False); tab.set_fontsize(6.6); tab.scale(1, 1.3)
    for (r, c), cl in tab.get_celld().items():
        cl.set_linewidth(0.0)
        if r == 0:
            cl.visible_edges = "BT"; cl.set_linewidth(0.9)
            cl.set_text_props(fontweight="bold")
        elif r == len(text):
            cl.visible_edges = "B"; cl.set_linewidth(0.9)
        if c == 0:
            cl.set_text_props(ha="left"); cl._text.set_x(0.02)
        if r > 0 and t.index[r - 1] == HIGHLIGHT:
            cl.set_text_props(fontweight="bold")
    ax.set_title("Aggregate performance over D+1 … D+7 "
                 "(mean of the seven lead times; rank 1 = best)",
                 fontsize=8, loc="left", pad=12)
    fig.tight_layout(pad=0.5)
    return fig


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="outputs/results.csv")
    ap.add_argument("--results_ext", default="outputs/results_ext.csv")
    ap.add_argument("--out", default="outputs/journal")
    ap.add_argument("--column", default="double", choices=["single", "double"],
                    help="panel width: 3.5 in (single) or 7.16 in (double)")
    ap.add_argument("--curve_models", default="climatology,gru,lstm,cnn_lstm,mthab_mse",
                    help="models drawn in the lead-time curves (keep it readable)")
    a = ap.parse_args()

    set_journal_style()
    os.makedirs(a.out, exist_ok=True)
    width = SINGLE_COL if a.column == "single" else DOUBLE_COL
    res = load_results([a.results, a.results_ext])
    t = aggregate(res)
    if t.empty:
        raise SystemExit("no Table-3 models found in the results")

    t.to_csv(os.path.join(a.out, "summary_D1_D7.csv"))
    print(f"aggregate over D+1..D+7 for {len(t)} models\n")
    show = [c for c in ["Model", "RMSE_mean", "RMSE_rank", "F1_mean", "F1_rank",
                        "overall_rank", "overall_wins"] if c in t.columns]
    print(t[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    keys = [k for k in a.curve_models.split(",") if k]
    figs = [
        ("figJ1_leadtime", figJ1_leadtime(res, keys, width)),
        ("figJ2_aggregate", figJ2_aggregate(t, width)),
        ("figJ3_rankplane", figJ3_rankplane(t, SINGLE_COL * 1.35)),
        ("figJ4_shortlong", figJ4_shortlong(res, width)),
        ("figJ5_matrix", figJ5_matrix(res, width)),
        ("summary_D1_D7", table_summary_pdf(t, width)),
    ]
    with PdfPages(os.path.join(a.out, "journal_all.pdf")) as pdf:
        for stem, fig in figs:
            if fig is None:
                print(f"  {stem}: skipped")
                continue
            path = os.path.join(a.out, stem + ".pdf")
            fig.savefig(path, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            print(f"  {stem:18s} -> {path}")
    print(f"\ncombined -> {os.path.join(a.out, 'journal_all.pdf')}")
    print(f"table    -> {os.path.join(a.out, 'summary_D1_D7.csv')}")


if __name__ == "__main__":
    main()
