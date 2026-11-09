"""
tables_pdf.py
=============
Renders the manuscript's result tables as **vector PDF** (no LaTeX required),
so the numbers can go straight into the paper or a slide next to the figures.

    python -m src.tables_pdf                     # every table, D+1 + all horizons
    python -m src.tables_pdf --horizon 3         # Table 3 rendered at D+3
    python -m src.tables_pdf --out outputs/tables

Output
------
    <out>/tables_all.pdf            every table, one per page
    <out>/table3_D1.pdf ...         Table 3 at each lead time D+1 … D+7
    <out>/table4_proposed.pdf       the proposed model across D+1 … D+7
    <out>/table4b_rmse.pdf          model x lead time, Chl-a RMSE
    <out>/table4b_f1.pdf            model x lead time, red-tide F1
    <out>/table_summary.csv         the same numbers in machine-readable form

Reading conventions used in every table
---------------------------------------
* cells are  mean ± s.d.  over seeds
* the best value in each column is **bold**
* the proposed model's row is tinted and marked with ▶
* an arrow in the header says which direction is better
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

from .figures_paper import (KEYS, LABEL, HIGHLIGHT, MARK, HORIZONS, METRIC_SPECS,
                            FAMILY_COLOR, INK, INK_2, INK_MUTED, GRID, SURFACE,
                            load_results)

ROW_HL = "#f7e6f1"        # tint for the proposed model's row
HEAD_BG = "#f2f1ec"


# --------------------------------------------------------------------------- #
def cell(v_mean, v_sd, fmt, n_seeds):
    if v_mean is None or (isinstance(v_mean, float) and np.isnan(v_mean)):
        return "--"
    if n_seeds > 1 and v_sd is not None and not np.isnan(v_sd):
        return f"{fmt.format(v_mean)} ± {fmt.format(v_sd)}"
    return fmt.format(v_mean)


def collect(res: pd.DataFrame, horizon: int):
    """Rows = models, columns = the nine reported metrics at D+`horizon`."""
    rows = []
    for k in KEYS:
        d = res[res["name"] == k]
        if not len(d):
            continue
        row = {"key": k, "Model": LABEL[k], "seeds": len(d)}
        raw = {}
        for lab, tmpl, better, fmt in METRIC_SPECS:
            col = tmpl.replace("{h}", str(horizon))
            v = pd.to_numeric(d[col], errors="coerce").dropna() if col in d else pd.Series(dtype=float)
            m = v.mean() if len(v) else np.nan
            sd = v.std(ddof=1) if len(v) > 1 else np.nan
            row[lab] = cell(m, sd, fmt, len(v))
            raw[lab] = m
        row["_raw"] = raw
        rows.append(row)
    return pd.DataFrame(rows)


def best_mask(t: pd.DataFrame):
    """Which cell is the winner in each metric column."""
    out = {}
    for lab, _tmpl, better, _fmt in METRIC_SPECS:
        vals = np.array([r.get(lab, np.nan) for r in t["_raw"]], float)
        if np.all(np.isnan(vals)):
            out[lab] = -1
            continue
        out[lab] = int(np.nanargmin(vals) if better < 0 else np.nanargmax(vals))
    return out


# --------------------------------------------------------------------------- #
def draw_table(col_labels, cell_text, row_keys, title, subtitle="",
               bold_cells=None, col_width=None, fontsize=8.0):
    """One table -> one matplotlib figure, sized to its content."""
    n_rows, n_cols = len(cell_text), len(col_labels)
    # the first column holds model names, so give it roughly three data columns
    if col_width is None:
        first = 3.0
        col_width = [first / (first + n_cols - 1)] + \
                    [1.0 / (first + n_cols - 1)] * (n_cols - 1)
    w = max(9.5, 1.15 * n_cols + 3.4)
    h = 1.45 + 0.34 * n_rows
    fig, ax = plt.subplots(figsize=(w, h))
    ax.axis("off")

    tab = ax.table(cellText=cell_text, colLabels=col_labels,
                   cellLoc="center", loc="upper center",
                   colWidths=col_width)
    tab.auto_set_font_size(False)
    tab.set_fontsize(fontsize)
    tab.scale(1, 1.42)

    for (r, c), cl in tab.get_celld().items():
        cl.set_edgecolor(GRID)
        cl.set_linewidth(0.6)
        if r == 0:                                   # header
            cl.set_facecolor(HEAD_BG)
            cl.set_text_props(color=INK, fontweight="bold")
            cl.set_height(cl.get_height() * 1.5)
            continue
        key = row_keys[r - 1] if r - 1 < len(row_keys) else None
        if key == HIGHLIGHT:
            cl.set_facecolor(ROW_HL)
            cl.set_text_props(color=INK, fontweight="bold")
        else:
            cl.set_facecolor(SURFACE)
            cl.set_text_props(color=INK_2)
        if c == 0:
            cl.set_text_props(ha="left")
            cl._text.set_x(0.03)
        if bold_cells and (r - 1, c) in bold_cells:
            cl.set_text_props(fontweight="bold", color=INK)

    ax.set_title(title, loc="left", color=INK, fontsize=11.5, pad=16)
    if subtitle:
        ax.text(0, 1.0, subtitle, transform=ax.transAxes, va="bottom",
                fontsize=8, color=INK_MUTED)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
def table3(res, horizon):
    t = collect(res, horizon)
    if t.empty:
        return None, t
    bm = best_mask(t)
    labs = [lab for lab, _t, _b, _f in METRIC_SPECS]
    arrows = {lab: ("↓" if better < 0 else "↑")
              for lab, _t, better, _f in METRIC_SPECS}
    col_labels = ["Model", "seeds"] + [f"{lab} {arrows[lab]}" for lab in labs]
    cell_text, bold = [], set()
    for i, (_, r) in enumerate(t.iterrows()):
        cell_text.append([r["Model"] + (MARK if r["key"] == HIGHLIGHT else ""),
                          str(r["seeds"])] + [r[lab] for lab in labs])
        for j, lab in enumerate(labs):
            if bm[lab] == i:
                bold.add((i, j + 2))
    fig = draw_table(
        col_labels, cell_text, list(t["key"]),
        f"Table 3 · Model comparison at D+{horizon} (2020 test year)",
        "Chl-a metrics on the full-year timeline; red-tide metrics on the "
        "event-adjacent subset.  ↑/↓ = direction of better.  "
        "Bold = best in column.  Cells are mean ± s.d. over seeds.",
        bold_cells=bold, fontsize=7.4)
    return fig, t


def table4_proposed(res, key=HIGHLIGHT):
    d = res[res["name"] == key]
    if not len(d):
        return None, pd.DataFrame()
    labs = [lab for lab, _t, _b, _f in METRIC_SPECS]
    rows, raws = [], []
    for h in HORIZONS:
        row = {"Horizon": f"D+{h}"}
        raw = {}
        for lab, tmpl, better, fmt in METRIC_SPECS:
            col = tmpl.replace("{h}", str(h))
            v = pd.to_numeric(d[col], errors="coerce").dropna() if col in d else pd.Series(dtype=float)
            m = v.mean() if len(v) else np.nan
            sd = v.std(ddof=1) if len(v) > 1 else np.nan
            row[lab] = cell(m, sd, fmt, len(v))
            raw[lab] = m
        rows.append(row)
        raws.append(raw)
    t = pd.DataFrame(rows)
    cell_text = [[r["Horizon"]] + [r[lab] for lab in labs] for _, r in t.iterrows()]
    fig = draw_table(
        ["Horizon"] + labs, cell_text, [HIGHLIGHT] * len(t),
        f"Table 4 · {LABEL[key]} across every lead time",
        "All seven heads come from ONE trained model (direct multi-horizon, "
        "T = 5-day input window, H = 7).  Cells are mean ± s.d. over seeds.",
        fontsize=7.4)
    return fig, t


def table4b(res, tmpl, fmt, better, title, subtitle):
    keys = [k for k in KEYS if (res["name"] == k).any()]
    if not keys:
        return None, pd.DataFrame()
    cell_text, raw = [], []
    for k in keys:
        d = res[res["name"] == k]
        line, rl = [LABEL[k] + (MARK if k == HIGHLIGHT else "")], []
        for h in HORIZONS:
            col = tmpl.replace("{h}", str(h))
            v = pd.to_numeric(d[col], errors="coerce").dropna() if col in d else pd.Series(dtype=float)
            m = v.mean() if len(v) else np.nan
            sd = v.std(ddof=1) if len(v) > 1 else np.nan
            line.append(cell(m, sd, fmt, len(v)))
            rl.append(m)
        cell_text.append(line)
        raw.append(rl)
    A = np.array(raw, float)
    bold = set()
    for j in range(A.shape[1]):
        colv = A[:, j]
        if np.all(np.isnan(colv)):
            continue
        i = int(np.nanargmin(colv) if better < 0 else np.nanargmax(colv))
        bold.add((i, j + 1))
    fig = draw_table(["Model"] + [f"D+{h}" for h in HORIZONS], cell_text, keys,
                     title, subtitle, bold_cells=bold, fontsize=7.2)
    return fig, pd.DataFrame(A, index=keys, columns=[f"D+{h}" for h in HORIZONS])


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="outputs/results.csv")
    ap.add_argument("--results_ext", default="outputs/results_ext.csv")
    ap.add_argument("--out", default="outputs/tables")
    ap.add_argument("--horizon", type=int, default=0, choices=[0] + HORIZONS,
                    help="0 = render Table 3 at every lead time (default)")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    res = load_results([a.results, a.results_ext])
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                         "font.size": 8, "figure.facecolor": SURFACE})

    hs = HORIZONS if a.horizon == 0 else [a.horizon]
    made, summary = [], []

    with PdfPages(os.path.join(a.out, "tables_all.pdf")) as pdf:
        for h in hs:
            fig, t = table3(res, h)
            if fig is None:
                print(f"  Table 3 D+{h}: no data")
                continue
            path = os.path.join(a.out, f"table3_D{h}.pdf")
            fig.savefig(path, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            made.append(path)
            print(f"  Table 3  D+{h:<2d} -> {path}")
            for _, r in t.iterrows():
                rec = {"table": "table3", "horizon": f"D+{h}",
                       "model": r["key"], "seeds": r["seeds"]}
                rec.update({lab: r["_raw"][lab] for lab in r["_raw"]})
                summary.append(rec)

        fig, _ = table4_proposed(res)
        if fig is not None:
            path = os.path.join(a.out, "table4_proposed.pdf")
            fig.savefig(path, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            made.append(path)
            print(f"  Table 4       -> {path}")

        for stem, tmpl, fmt, better, title, sub in [
            ("table4b_rmse", "full_year|D+{h}|RMSE", "{:.2f}", -1,
             "Table 4b · Chl-a RMSE (µg/L) at every lead time",
             "Lower is better; bold = best model at that lead time."),
            ("table4b_r2", "full_year|D+{h}|R2", "{:.2f}", +1,
             "Table 4c · R² at every lead time",
             "Higher is better; bold = best model at that lead time."),
            ("table4b_f1", "event_adjacent|D+{h}|F1", "{:.2f}", +1,
             "Table 4d · Red-tide F1 (event-adjacent) at every lead time",
             "Higher is better; bold = best model at that lead time."),
            ("table4b_auprc", "event_adjacent|D+{h}|AUPRC", "{:.2f}", +1,
             "Table 4e · Red-tide AUPRC (event-adjacent) at every lead time",
             "Higher is better; bold = best model at that lead time."),
        ]:
            fig, m = table4b(res, tmpl, fmt, better, title, sub)
            if fig is None:
                continue
            path = os.path.join(a.out, stem + ".pdf")
            fig.savefig(path, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            made.append(path)
            print(f"  {stem:14s} -> {path}")
            for k, row in m.iterrows():
                for hh, v in row.items():
                    summary.append({"table": stem, "horizon": hh,
                                    "model": k, "value": v})

    pd.DataFrame(summary).to_csv(os.path.join(a.out, "table_summary.csv"), index=False)
    print(f"\n  combined      -> {os.path.join(a.out, 'tables_all.pdf')}  ({len(made)} tables)")
    print(f"  machine-readable -> {os.path.join(a.out, 'table_summary.csv')}")


if __name__ == "__main__":
    main()
