"""R2-M2: multi-task MT-HAB-G3D vs otherwise-identical single-task controls.

  cls-only : same backbone/heads/budget/selection, lambda = 0 (no regression loss)
  reg-only : same, classification loss weight = 0

Red-tide metrics compare MT vs cls-only; Chl-a metrics compare MT vs reg-only.
Seed-paired Wilcoxon signed-rank tests on the D+1..D+7 mean (as in the paper),
Benjamini-Hochberg across the reported metrics, plus per-horizon medians.

`--demo` substitutes the spatial-extension run for both controls so the code path
can be checked before the single-task runs exist (output NOT reportable).
"""
import sys, os, argparse; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from scipy.stats import wilcoxon
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8})
pd.set_option("display.width", 250)

ap = argparse.ArgumentParser(); ap.add_argument("--demo", action="store_true"); a = ap.parse_args()
P = load_all(["MT-HAB-G3D (proposed)", "MT-HAB-G3D (cls only)", "MT-HAB-G3D (reg only)", "MT-HAB-G3D + spatial ext."])
mt = P["MT-HAB-G3D (proposed)"]
demo = a.demo or ("MT-HAB-G3D (cls only)" not in P) or ("MT-HAB-G3D (reg only)" not in P)
if demo:
    print("[DEMO] single-task runs not found - using a stand-in run; NOT reportable")
    if "MT-HAB-G3D + spatial ext." in P:
        stand_in = P["MT-HAB-G3D + spatial ext."]
    elif os.path.exists(f"{RUNS}/A_mthab_mse_s0/preds_test.npz") and (RUNS, "A_mthab_mse_s{}") != (FINAL_DIR, FINAL_FMT):
        stand_in = [load_pred(RUNS, f"A_mthab_mse_s{s}") for s in range(10) if os.path.exists(f"{RUNS}/A_mthab_mse_s{s}/preds_test.npz")]
    else:
        stand_in = mt
    cls_only = reg_only = stand_in
else:
    cls_only, reg_only = P["MT-HAB-G3D (cls only)"], P["MT-HAB-G3D (reg only)"]
ev_h = mt[0]["event_adj_h"] if "event_adj_h" in mt[0] else np.load(f"{OUT}/event_adj_per_horizon.npy")


def per_seed(plist):
    """(n_seeds, H) arrays of every metric."""
    out = {k: [] for k in ["AUPRC_ev", "F1_ev", "MCC_ev", "TSS_ev", "AUPRC_full", "F1_full", "RMSE", "R2", "PeakRMSE", "TA1"]}
    for p in plist:
        row = {k: [] for k in out}
        for h in range(H):
            me = p["m_cls"][:, h] & ev_h[:, h]; mf = p["m_cls"][:, h]
            ce = cls_metrics(p["y_cls"][me, h], p["prob"][me, h], p["thr"][h]); cf = cls_metrics(p["y_cls"][mf, h], p["prob"][mf, h], p["thr"][h])
            mr = p["m_reg"][:, h]; r = reg_metrics(p["y_reg_raw"][mr, h], p["reg_ugl"][mr, h])
            row["AUPRC_ev"].append(ce["AUPRC"]); row["F1_ev"].append(ce["F1"]); row["MCC_ev"].append(ce["MCC"]); row["TSS_ev"].append(ce["TSS"])
            row["AUPRC_full"].append(cf["AUPRC"]); row["F1_full"].append(cf["F1"])
            row["RMSE"].append(r["RMSE"]); row["R2"].append(r["R2"]); row["PeakRMSE"].append(r["PeakRMSE"]); row["TA1"].append(r["TA1"])
        for k in out: out[k].append(row[k])
    return {k: np.array(v) for k, v in out.items()}


M, C, R = per_seed(mt), per_seed(cls_only), per_seed(reg_only)
CLS = ["AUPRC_ev", "F1_ev", "MCC_ev", "TSS_ev", "AUPRC_full", "F1_full"]; REG = ["RMSE", "R2", "PeakRMSE", "TA1"]
rows = []
for k in CLS + REG:
    S = C if k in CLS else R
    n = min(len(M[k]), len(S[k]))
    a_, b_ = M[k][:n].mean(1), S[k][:n].mean(1)
    d = a_ - b_
    try:
        pval = wilcoxon(a_, b_).pvalue if n >= 5 and np.any(d != 0) else np.nan
    except ValueError:
        pval = np.nan
    rows.append(dict(metric=k, control="cls-only" if k in CLS else "reg-only", MT_median=np.median(a_), ST_median=np.median(b_),
                     delta_median=np.median(d), MT_better=(np.median(d) < 0) if k in ("RMSE", "PeakRMSE") else (np.median(d) > 0),
                     p_raw=pval, n_seeds=n))
t = pd.DataFrame(rows)
# Benjamini-Hochberg
pv = t.p_raw.values; m = np.sum(~np.isnan(pv)); order = np.argsort(np.where(np.isnan(pv), np.inf, pv))
adj = np.full(len(pv), np.nan); prev = 1.0
for rank, i in enumerate(order[::-1][m - m:], start=0): pass
ranked = [i for i in order if not np.isnan(pv[i])]
for j in range(len(ranked) - 1, -1, -1):
    i = ranked[j]; val = min(prev, pv[i] * m / (j + 1)); adj[i] = val; prev = val
t["p_BH"] = adj
print(f"=== multi-task vs single-task [{'DEMO' if demo else 'real'}], seed-paired, D+1..D+7 mean ===")
print(t.round(4).to_string(index=False))
t.to_csv(f"{OUT}/multitask_vs_single.csv", index=False)

# per-horizon medians
byh = pd.DataFrame({"horizon": [f"D+{h+1}" for h in range(H)],
                    "MT_AUPRC_ev": np.median(M["AUPRC_ev"], 0), "cls_only_AUPRC_ev": np.median(C["AUPRC_ev"], 0),
                    "MT_F1_ev": np.median(M["F1_ev"], 0), "cls_only_F1_ev": np.median(C["F1_ev"], 0),
                    "MT_RMSE": np.median(M["RMSE"], 0), "reg_only_RMSE": np.median(R["RMSE"], 0),
                    "MT_R2": np.median(M["R2"], 0), "reg_only_R2": np.median(R["R2"], 0)})
print("\n-- per horizon (median over seeds) --"); print(byh.round(3).to_string(index=False)); byh.to_csv(f"{OUT}/multitask_vs_single_by_horizon.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(11, 3.3)); x = np.arange(1, H + 1)
def band(ax, arr, c, lab):
    ax.plot(x, np.median(arr, 0), color=c, marker="o", ms=3, lw=1.2, label=lab)
    ax.fill_between(x, np.quantile(arr, .25, 0), np.quantile(arr, .75, 0), color=c, alpha=0.15, lw=0)
band(axes[0], M["RMSE"], "tab:red", "multi-task"); band(axes[0], R["RMSE"], "tab:blue", "regression only"); axes[0].set_title("(a) Chl-a RMSE (µg/L)", fontsize=8)
band(axes[1], M["AUPRC_ev"], "tab:red", "multi-task"); band(axes[1], C["AUPRC_ev"], "tab:green", "classification only"); axes[1].set_title("(b) red-tide AUPRC (event-adjacent)", fontsize=8)
band(axes[2], M["F1_ev"], "tab:red", "multi-task"); band(axes[2], C["F1_ev"], "tab:green", "classification only"); axes[2].set_title("(c) red-tide F1 (event-adjacent)", fontsize=8)
for ax in axes: ax.set_xlabel("forecast lead time (days)"); ax.legend(fontsize=7)
fig.suptitle("Multi-task vs otherwise-identical single-task MT-HAB-G3D (median, IQR over seeds)" + (" - DEMO" if demo else ""), fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/figH_multitask_vs_single.png", dpi=170); fig.savefig(f"{OUT}/figH_multitask_vs_single.pdf")
print("done ->", OUT)
