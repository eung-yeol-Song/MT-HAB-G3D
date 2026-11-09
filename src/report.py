"""
report.py
=========
Turns outputs/results.csv + outputs/runs/*/preds_test.npz into paper-ready
artefacts:

    outputs/report/table3_extended.tex     model comparison at D+1 (replaces Table 3)
    outputs/report/table4_horizons.tex     multi-horizon D+1..D+7 (replaces Table 4)
    outputs/report/table5_ablation.tex     pathway / graph / objective ablations
    outputs/report/table6_uncertainty.tex  quantile & MC-dropout calibration
    outputs/report/table7_significance.tex paired statistical tests
    outputs/report/*.png                   figures 4-7
    outputs/report/summary.md              everything in Markdown for quick reading

Every cell is  mean +- sd  over seeds, exactly as the manuscript declares.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .metrics import classification_metrics, regression_metrics
from .stats_tests import compare_runs, holm

PRETTY = {
    "persistence": "Persistence (D+h = D)",
    "climatology": "Day-of-year climatology",
    "histgb": "Gradient-boosted trees",
    "mlp": "MLP",
    "lstm": "LSTM",
    "gru": "GRU",
    "cnn_lstm": "CNN-LSTM",
    "tcn": "Dilated TCN",
    "transformer": "Transformer",
    "stgcn": "ST-GCN (no G3D)",
    "mtgnn": "MTGNN",
    "am_mtgnn": "AM-MTGNN",
    "mtgnn_nograph": "MTGNN (no graph)",
    "mthab_ref": "MT-HAB-G3D (group-G protocol)",
    "mthab_mse": "MT-HAB-G3D (proposed)",
    "mthab_cls_only": "MT-HAB-G3D (classification only)",
    "mthab_reg_only": "MT-HAB-G3D (regression only)",
    "mthab_huber": "MT-HAB-G3D (Huber, withdrawn)",
    "mthab_huber_focal": "MT-HAB-G3D (Huber + Focal + resampling)",
    "mthab_huber_wbce": "MT-HAB-G3D (Huber + weighted BCE)",
    "lstm_focal": "LSTM (Focal + resampling)",
    "graph_identity": "graph = identity (no variable edges)",
    "graph_complete": "graph = complete",
    "graph_adaptive": "graph = learned (adaptive)",
    "graph_hybrid": "graph = correlation + learned (hybrid)",
    "no_g3d": "w/o G3D pathway",
    "no_factorized": "w/o factorised pathway",
    "weight_uncertainty": "homoscedastic-uncertainty weighting",
    "weight_dwa": "dynamic weight averaging",
    "peak_huber": "peak-weighted Huber",
    "quantile": "quantile (pinball) head",
    "mc_dropout": "MC-dropout (30 passes)",
    "residual": "MT-HAB-G3D + residual (persistence-anchored) head",
    "residual_peak": "residual head + peak-weighted Huber",
    "lstm_residual": "LSTM + residual head",
}


def ms(df, col, fmt="{:.3f}"):
    """mean +- sd string (tolerant of columns absent from partial results)."""
    if col not in df.columns:
        return "--"
    v = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(v) == 0:
        return "--"
    if len(v) == 1:
        return fmt.format(v.iloc[0])
    return f"{fmt.format(v.mean())} $\\pm$ {fmt.format(v.std(ddof=1))}"


def agg(res: pd.DataFrame, names, cols):
    rows = []
    for n in names:
        d = res[res["name"] == n]
        if len(d) == 0:
            continue
        row = {"Model": PRETTY.get(n, n), "_key": n, "seeds": len(d)}
        for label, col, fmt in cols:
            row[label] = ms(d, col, fmt)
        rows.append(row)
    return pd.DataFrame(rows)


def to_latex(df: pd.DataFrame, caption: str, label: str, drop=("_key",)):
    d = df.drop(columns=[c for c in drop if c in df.columns])
    body = d.to_latex(index=False, escape=False, column_format="l" + "c" * (len(d.columns) - 1))
    return (f"\\begin{{table*}}[t]\n\\centering\n\\caption{{{caption}}}\n"
            f"\\label{{{label}}}\n{body}\\end{{table*}}\n")


# --------------------------------------------------------------------------- #
def table_main(res, out):
    cols = [("Chl-\\emph{a} RMSE ($\\mu$g/L)", "full_year|D+1|RMSE", "{:.2f}"),
            ("Peak-RMSE", "full_year|D+1|PeakRMSE", "{:.1f}"),
            ("$R^2$", "full_year|D+1|R2", "{:.2f}"),
            ("TA$_{\\pm1}$ (\\%)", "full_year|D+1|TA1", "{:.3f}"),
            ("F1 (event-adj.)", "event_adjacent|D+1|F1", "{:.2f}"),
            ("AUPRC", "event_adjacent|D+1|AUPRC", "{:.2f}"),
            ("MCC", "event_adjacent|D+1|MCC", "{:.2f}"),
            ("TSS", "event_adjacent|D+1|TSS", "{:.2f}"),
            ("Acc. (\\%)", "full_year|D+1|accuracy", "{:.3f}")]
    # exactly the 13 rows of Table 3; ablation / variant runs have their own
    # tables (table5_ablation), so they are deliberately not mixed in here
    order = ["persistence", "climatology", "histgb", "mlp", "lstm", "gru", "cnn_lstm", "tcn",
             "transformer", "stgcn", "mtgnn", "am_mtgnn", "mtgnn_nograph",
             "mthab_mse"]
    order = [k for k in order if k in set(res["name"])] if "name" in res.columns else order
    t = agg(res, order, cols)
    open(os.path.join(out, "table3_extended.tex"), "w").write(to_latex(
        t, "Next-day (D+1) multi-task HAB forecasting performance. Chl-\\emph{a} "
           "metrics are computed on the full-year 2020 timeline; red-tide metrics "
           "on the event-adjacent subset. Values are mean $\\pm$ s.d. across seeds.",
        "tab:main"))
    open(os.path.join(out, "table3_extended.md"), "w").write(t.to_markdown(index=False))
    return t


def table_groups(res, out):
    """Extra tables for the MTGNN family, the aggregation protocol and the
    nutrient-augmented node set (experiment groups G, H, I)."""
    cols = [("Chl-\\emph{a} RMSE ($\\mu$g/L)", "full_year|D+1|RMSE", "{:.2f}"),
            ("Peak-RMSE", "full_year|D+1|PeakRMSE", "{:.1f}"),
            ("$R^2$", "full_year|D+1|R2", "{:.2f}"),
            ("F1 (event-adj.)", "event_adjacent|D+1|F1", "{:.2f}"),
            ("AUPRC", "event_adjacent|D+1|AUPRC", "{:.2f}"),
            ("TSS", "event_adjacent|D+1|TSS", "{:.2f}")]
    specs = [
        ("table_mtgnn", ["mtgnn_nograph", "mtgnn", "am_mtgnn", "mthab_ref"],
         "Comparison with the graph forecasting family of Ding et al. (2025), "
         "re-implemented on the present coastal dataset.", "tab:mtgnn"),
        ("table_protocol", ["agg_mean", "agg_max", "agg_mean_lstm", "agg_max_lstm"],
         "Daily aggregation statistic, evaluated on identical station-days. "
         "Absolute regression errors are not comparable across protocols because "
         "the target scale differs; $R^2$ and all red-tide metrics are.",
         "tab:protocol"),
        ("table_nodes", ["nodes9", "nodes13", "nodes13_hybrid", "nodes13_am_mtgnn"],
         "Nutrient-augmented variable graph (+TN, TP, NO$_3$-N, PO$_4$-P).",
         "tab:nodes"),
    ]
    made = {}
    for stem, order, cap, lab in specs:
        t = agg(res, order, cols)
        if t.empty:
            continue
        open(os.path.join(out, stem + ".tex"), "w").write(to_latex(t, cap, lab))
        open(os.path.join(out, stem + ".md"), "w").write(t.to_markdown(index=False))
        made[stem] = t
    return made


def table_horizons_compare(res, out, keys=None, ref="gru"):
    """Table 4b: every model across D+1..D+7 for the two headline metrics.

    The single-horizon Table 3 hides the main result — the proposed model's
    advantage grows with lead time — so this table reports the whole horizon
    sweep side by side.
    """
    keys = keys or ["climatology", "lstm", "gru", "cnn_lstm", "mthab_mse"]
    keys = [k for k in keys if (res["name"] == k).any()]
    if not keys:
        return pd.DataFrame()
    rows = []
    for metric, col_t, fmt in [("Chl-\\emph{a} RMSE ($\\mu$g/L)", "full_year|D+{h}|RMSE", "{:.2f}"),
                               ("F1 (event-adj.)", "event_adjacent|D+{h}|F1", "{:.2f}")]:
        for k in keys:
            d = res[res["name"] == k]
            row = {"Metric": metric, "Model": PRETTY.get(k, k)}
            for h in range(1, 8):
                row[f"D+{h}"] = ms(d, col_t.format(h=h), fmt)
            rows.append(row)
    t = pd.DataFrame(rows)
    open(os.path.join(out, "table4b_horizon_compare.tex"), "w").write(to_latex(
        t, "Lead-time sweep, D+1 to D+7, for the proposed model and the strongest "
           "baselines. All heads are trained jointly (direct multi-horizon), so a "
           "single trained model produces every column.", "tab:horizon_compare"))
    open(os.path.join(out, "table4b_horizon_compare.md"), "w").write(t.to_markdown(index=False))
    return t


def table_horizons(res, out, key="mthab_mse"):
    d = res[res["name"] == key]
    rows = []
    for h in range(1, 8):
        rows.append({
            "Horizon": f"D+{h}",
            "Red-tide acc. (\\%)": ms(d, f"full_year|D+{h}|accuracy", "{:.3f}"),
            "F1 (event-adj.)": ms(d, f"event_adjacent|D+{h}|F1", "{:.2f}"),
            "AUPRC": ms(d, f"event_adjacent|D+{h}|AUPRC", "{:.2f}"),
            "Chl-\\emph{a} RMSE": ms(d, f"full_year|D+{h}|RMSE", "{:.2f}"),
            "TA$_{\\pm1}$": ms(d, f"full_year|D+{h}|TA1", "{:.3f}"),
            "$R^2$": ms(d, f"full_year|D+{h}|R2", "{:.2f}"),
        })
    t = pd.DataFrame(rows)
    open(os.path.join(out, "table4_horizons.tex"), "w").write(to_latex(
        t, "Multi-horizon performance of the extended model (direct heads, D+1--D+7). "
           "No horizon is left blank, unlike Table 4 of the original manuscript.",
        "tab:horizon"))
    return t


def table_ablation(res, out):
    cols = [("Chl-\\emph{a} RMSE", "full_year|D+1|RMSE", "{:.2f}"),
            ("Peak-RMSE", "full_year|D+1|PeakRMSE", "{:.1f}"),
            ("$R^2$", "full_year|D+1|R2", "{:.2f}"),
            ("F1 (event-adj.)", "event_adjacent|D+1|F1", "{:.2f}"),
            ("AUPRC", "event_adjacent|D+1|AUPRC", "{:.2f}"),
            ("Params", "n_params", "{:.0f}")]
    order = ["mthab_huber_focal", "no_g3d", "no_factorized", "graph_identity",
             "graph_complete", "graph_adaptive", "graph_hybrid",
             "weight_uncertainty", "weight_dwa", "peak_huber",
             "residual", "residual_peak"]
    t = agg(res, order, cols)
    open(os.path.join(out, "table5_ablation.tex"), "w").write(to_latex(
        t, "Component ablations. The first row is the full extended model; every "
           "other row changes exactly one component.", "tab:ablation"))
    return t


def table_uncertainty(res, out):
    cols = [("RMSE", "full_year|D+1|RMSE", "{:.2f}"),
            ("PICP (80\\%)", "full_year|D+1|PICP", "{:.3f}"),
            ("MPIW ($\\mu$g/L)", "full_year|D+1|MPIW", "{:.2f}"),
            ("Pinball", "full_year|D+1|pinball", "{:.3f}"),
            ("CRPS", "full_year|D+1|CRPS", "{:.3f}"),
            ("Brier", "event_adjacent|D+1|Brier", "{:.3f}"),
            ("ECE", "event_adjacent|D+1|ECE", "{:.3f}")]
    t = agg(res, ["quantile", "mc_dropout", "mthab_huber_focal"], cols)
    open(os.path.join(out, "table6_uncertainty.tex"), "w").write(to_latex(
        t, "Predictive-uncertainty quality. A nominal 80\\% interval should give "
           "PICP $\\approx$ 0.80 at the smallest possible MPIW.", "tab:uncertainty"))
    return t


# --------------------------------------------------------------------------- #
def table_significance(runs_dir, out, pairs, horizon=0):
    rows, pvals = [], {}
    for (na, nb, la, lb) in pairs:
        a = sorted(glob.glob(os.path.join(runs_dir, f"*_{na}_s*", "preds_test.npz")))
        b = sorted(glob.glob(os.path.join(runs_dir, f"*_{nb}_s*", "preds_test.npz")))
        if not a or not b:
            continue
        try:
            r = compare_runs(a[0], b[0], horizon=horizon)
        except AssertionError:
            # runs evaluated on different bundles cannot be paired sample-by-sample
            continue
        key = f"{la} vs {lb}"
        rows.append({
            "Comparison": key,
            "$\\Delta$F1": f"{r['classification']['paired_bootstrap_F1']['delta']:+.3f}",
            "F1 95\\% CI": (f"[{r['classification']['paired_bootstrap_F1']['ci_lo']:+.3f}, "
                            f"{r['classification']['paired_bootstrap_F1']['ci_hi']:+.3f}]"),
            "DeLong $p$": f"{r['classification']['delong_AUROC']['p_value']:.3f}",
            "McNemar $p$": f"{r['classification']['mcnemar']['p_value']:.3f}",
            "$\\Delta$RMSE": (f"{r['regression']['RMSE_A'] - r['regression']['RMSE_B']:+.3f}"),
            "DM $p$": f"{r['regression']['diebold_mariano_RMSE']['p_value']:.3f}",
        })
        pvals[key + " (DM)"] = r["regression"]["diebold_mariano_RMSE"]["p_value"]
        pvals[key + " (DeLong)"] = r["classification"]["delong_AUROC"]["p_value"]
    t = pd.DataFrame(rows)
    if len(t):
        adj = holm(pvals)
        json.dump(adj, open(os.path.join(out, "holm_adjusted_pvalues.json"), "w"), indent=1)
        open(os.path.join(out, "table7_significance.tex"), "w").write(to_latex(
            t, "Paired statistical tests on the 2020 test set (seed 0 of each model). "
               "$\\Delta$ = first model minus second. Holm-corrected $p$-values are "
               "provided in the supplementary JSON.", "tab:stats"))
    return t


# --------------------------------------------------------------------------- #
def figures(runs_dir, out, key="mthab_huber_focal", ref="lstm"):
    f = sorted(glob.glob(os.path.join(runs_dir, f"*_{key}_s*", "preds_test.npz")))
    if not f:
        return
    z = np.load(f[0])
    thr = z["thr"][0]

    # --- Fig. A: time series of Chl-a for the station with most positives ---
    st = np.bincount(z["station"][z["y_cls"][:, 0] > 0.5]).argmax() if (z["y_cls"][:, 0] > 0.5).any() else 0
    m = (z["station"] == st) & z["m_reg"][:, 0]
    d = z["date"][m].astype("datetime64[D]")
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax[0].plot(d, z["y_reg_raw"][m, 0], lw=1.0, label="Ground truth")
    ax[0].plot(d, z["reg_raw"][m, 0], lw=1.0, label="Prediction")
    if z["quant_raw"].shape[-1] > 1:
        ax[0].fill_between(d, z["quant_raw"][m, 0, 0], z["quant_raw"][m, 0, -1],
                           alpha=0.25, label="80% interval")
    ax[0].set_ylabel("Chl-a (ug/L)")
    ax[0].legend(fontsize=8, ncol=3)
    mc = (z["station"] == st) & z["m_cls"][:, 0]
    dc = z["date"][mc].astype("datetime64[D]")
    ax[1].step(dc, z["y_cls"][mc, 0], where="mid", label="Observed red tide")
    ax[1].step(dc, (z["prob"][mc, 0] >= thr).astype(int) * 0.9, where="mid",
               label="Predicted")
    ax[1].plot(dc, z["prob"][mc, 0], lw=0.8, alpha=0.6, label="Probability")
    ax[1].set_ylabel("Red tide")
    ax[1].legend(fontsize=8, ncol=3)
    fig.suptitle("2020 test year, station index %d" % st)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_timeseries.png"), dpi=160)
    plt.close(fig)

    # --- Fig. B: precision-recall + reliability ----------------------------
    mm = z["m_cls"][:, 0].astype(bool) & z["event_adj"].astype(bool)
    y, p = z["y_cls"][mm, 0].astype(int), z["prob"][mm, 0]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    o = np.argsort(-p)
    tp = np.cumsum(y[o]); fp = np.cumsum(1 - y[o])
    ax[0].plot(tp / max(y.sum(), 1), tp / np.maximum(tp + fp, 1e-9))
    ax[0].axhline(y.mean(), ls="--", c="gray", label="no-skill")
    ax[0].set_xlabel("Recall"); ax[0].set_ylabel("Precision")
    ax[0].set_title("Precision-recall (event-adjacent)"); ax[0].legend(fontsize=8)

    bins = np.linspace(0, 1, 11)
    cx, cy = [], []
    for i in range(10):
        s = (p >= bins[i]) & (p < bins[i + 1])
        if s.sum() > 3:
            cx.append(p[s].mean()); cy.append(y[s].mean())
    ax[1].plot([0, 1], [0, 1], ls="--", c="gray")
    ax[1].plot(cx, cy, "o-")
    ax[1].set_xlabel("Predicted probability"); ax[1].set_ylabel("Observed frequency")
    ax[1].set_title("Reliability diagram")

    mr = z["m_reg"][:, 0].astype(bool)
    ax[2].scatter(z["y_reg_raw"][mr, 0], z["reg_raw"][mr, 0], s=4, alpha=0.3)
    lim = [0, np.percentile(z["y_reg_raw"][mr, 0], 99.5)]
    ax[2].plot(lim, lim, ls="--", c="gray")
    ax[2].set_xlim(lim); ax[2].set_ylim(lim)
    ax[2].set_xlabel("Observed Chl-a"); ax[2].set_ylabel("Predicted Chl-a")
    ax[2].set_title("Scatter (full-year)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_diagnostics.png"), dpi=160)
    plt.close(fig)

    # --- Fig. C: learned adjacency ----------------------------------------
    mj = os.path.join(os.path.dirname(f[0]), "metrics.json")
    hyb = sorted(glob.glob(os.path.join(runs_dir, "*_graph_hybrid_s*", "metrics.json")))
    src = hyb[0] if hyb else mj
    m = json.load(open(src))
    if "learned_adjacency" in m:
        A = np.array(m["learned_adjacency"])
        names = m.get("node_order", [str(i) for i in range(A.shape[0])])
        fig, ax = plt.subplots(figsize=(5.2, 4.4))
        im = ax.imshow(A, cmap="viridis")
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=60, ha="right")
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
        ax.set_title("Learned variable adjacency")
        fig.colorbar(im); fig.tight_layout()
        fig.savefig(os.path.join(out, "fig_adjacency.png"), dpi=160)
        plt.close(fig)

    # --- Fig. D: horizon degradation ---------------------------------------
    res = pd.read_csv(os.path.join(os.path.dirname(runs_dir), "results.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    for n in [key, ref, "persistence"]:
        d = res[res["name"] == n]
        if not len(d):
            continue
        hs = range(1, 8)
        ax[0].plot(list(hs), [pd.to_numeric(d[f"full_year|D+{h}|RMSE"]).mean() for h in hs],
                   marker="o", label=PRETTY.get(n, n))
        ax[1].plot(list(hs), [pd.to_numeric(d[f"event_adjacent|D+{h}|F1"]).mean() for h in hs],
                   marker="o", label=PRETTY.get(n, n))
    ax[0].set_xlabel("Horizon (days)"); ax[0].set_ylabel("Chl-a RMSE (ug/L)")
    ax[1].set_xlabel("Horizon (days)"); ax[1].set_ylabel("Red-tide F1 (event-adj.)")
    ax[0].legend(fontsize=7); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_horizons.png"), dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="outputs/results.csv")
    ap.add_argument("--results_ext", default="outputs/results_ext.csv")
    ap.add_argument("--runs", default="outputs/runs")
    ap.add_argument("--out", "--outdir", dest="out", default="outputs/report")
    ap.add_argument("--horizon_key", default="mthab_mse",
                    help="model used for the single-model horizon table (Table 4)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    frames = [pd.read_csv(f) for f in (a.results, a.results_ext) if os.path.exists(f)]
    if not frames:
        raise SystemExit("no results found - run the experiments first")
    res = pd.concat(frames, ignore_index=True)

    t_main = table_main(res, a.out)
    t_hor = table_horizons(res, a.out, key=a.horizon_key)
    t_horc = table_horizons_compare(res, a.out)
    t_abl = table_ablation(res, a.out)
    t_unc = table_uncertainty(res, a.out)
    t_sig = table_significance(a.runs, a.out, [
        ("mthab_mse", "lstm", "MT-HAB-G3D", "LSTM"),
        ("mthab_mse", "cnn_lstm", "MT-HAB-G3D", "CNN-LSTM"),
        ("mthab_mse", "gru", "MT-HAB-G3D", "GRU"),
        ("mthab_huber_focal", "mthab_huber", "+Focal", "BCE"),
        ("mthab_huber_focal", "persistence", "+Focal", "Persistence"),
        ("graph_hybrid", "graph_identity", "hybrid graph", "no graph"),
        ("mthab_huber_focal", "no_g3d", "full model", "w/o G3D"),
        ("mthab_ref", "am_mtgnn", "MT-HAB-G3D", "AM-MTGNN"),
        ("mthab_ref", "mtgnn", "MT-HAB-G3D", "MTGNN"),
        ("residual", "mthab_huber_focal", "residual head", "level head"),
        ("residual", "persistence", "residual head", "Persistence"),
        ("nodes13", "nodes9", "13 nodes", "9 nodes"),
    ])
    t_grp = table_groups(res, a.out)
    figures(a.runs, a.out)

    md = ["# Extended experimental results\n"]
    for name, t in [("Table 3 (extended) - model comparison at D+1", t_main),
                    ("Table 4 (extended) - multi-horizon", t_hor),
                    ("Table 4b - lead-time sweep, all models", t_horc),
                    ("Table 5 - ablations", t_abl),
                    ("Table 6 - uncertainty quality", t_unc),
                    ("Table 7 - statistical significance", t_sig)] + \
                   [(f"Extra - {k}", v) for k, v in t_grp.items()]:
        md.append(f"\n## {name}\n")
        md.append(t.drop(columns=[c for c in ("_key",) if c in t.columns])
                  .to_markdown(index=False) if len(t) else "_no runs yet_")
    open(os.path.join(a.out, "summary.md"), "w").write("\n".join(md))
    print("\n".join(md))
    print(f"\nartefacts -> {a.out}")


if __name__ == "__main__":
    main()
