"""Refit experiment: reference vs fixed-budget control vs train+validation refit (run_refit.sh).

Three protocols, identical settings, same seeds, same 2020 test rows:
  ref   : train 2015-2018, early stopping / checkpoint selection on 2019     (outputs/)
  fixed : train 2015-2018, fixed epoch budget (median best epoch of ref), last weights
  refit : train 2015-2019, same fixed budget, thresholds carried over from `fixed`
refit - fixed = effect of the extra year; fixed - ref = cost of dropping checkpoint selection.

Per model and seed (mean over D+1..D+7): RMSE, R2, PeakRMSE, AUPRC (event-adjacent and full
year), F1 / MCC / TSS (event-adjacent, carried thresholds).  Then
  * median per protocol, paired differences with two-sided Wilcoxon over seeds
  * rank of the proposed model inside each protocol
  * proposed vs every baseline inside each protocol (seed-paired Wilcoxon)
    python analysis/14_refit_compare.py
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import json
from scipy.stats import wilcoxon

ROOT = os.environ.get("MTHAB_REFIT_ROOT", "outputs/refit")
TREES = {"ref": "outputs", "fixed": f"{ROOT}/fixed", "refit": f"{ROOT}/refit"}
FMT = {k: (v[1], v[2]) for k, v in MODELS.items() if k != "MT-HAB-G3D + spatial ext."}
FMT["MT-HAB-G3D (proposed)"] = ("A_mthab_mse_s{}", 10)
PROP = "MT-HAB-G3D (proposed)"
METS = ["RMSE", "R2", "PeakRMSE", "AUPRC_ea", "AUPRC_full", "F1_ea", "MCC_ea", "TSS_ea"]
LOWER = {"RMSE", "PeakRMSE"}                       # lower is better
REG, CLS = {"RMSE", "R2", "PeakRMSE"}, set(METS) - {"RMSE", "R2", "PeakRMSE"}


def load(root, tag):
    f = f"{root}/run_standard/runs/{tag}/preds_test.npz"
    if not os.path.exists(f):
        return None
    z = np.load(f); d = {k: z[k] for k in z.files}
    d["reg_ugl"] = to_ugl(d["reg_raw"])
    return d


def metrics(d):
    out = {k: [] for k in METS}
    for h in range(H):
        m = d["m_cls"][:, h]
        out["AUPRC_full"].append(auprc(d["y_cls"][m, h], d["prob"][m, h]))
        me = m & d["event_adj_h"][:, h]
        c = cls_metrics(d["y_cls"][me, h], d["prob"][me, h], d["thr"][h])
        out["AUPRC_ea"].append(c["AUPRC"]); out["F1_ea"].append(c["F1"])
        out["MCC_ea"].append(c["MCC"]); out["TSS_ea"].append(c["TSS"])
        mr = d["m_reg"][:, h]
        r = reg_metrics(d["y_reg_raw"][mr, h], d["reg_ugl"][mr, h])
        out["RMSE"].append(r["RMSE"]); out["R2"].append(r["R2"]); out["PeakRMSE"].append(r["PeakRMSE"])
    return {k: float(np.nanmean(v)) for k, v in out.items()}


rows = []
for name, (fmt, ns) in FMT.items():
    for s in range(ns):
        for prot, root in TREES.items():
            d = load(root, fmt.format(s))
            if d is None:
                continue
            m = metrics(d)
            dead = DEAD_HEAD.get(name)
            for k in METS:
                if (dead == "reg" and k in REG) or (dead == "cls" and k in CLS):
                    m[k] = np.nan
            rows.append(dict(model=name, seed=s, protocol=prot, **m))
df = pd.DataFrame(rows); df.to_csv(f"{ROOT}/refit_per_seed.csv", index=False)


def pval(d):
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    return float(wilcoxon(d).pvalue) if (len(d) > 5 and not np.allclose(d, 0)) else np.nan


# ------------------------------------------------------------------ 1. protocol effect per model
summ = []
for name in FMT:
    g = df[df.model == name].pivot(index="seed", columns="protocol", values=METS)
    for k in METS:
        if k not in g or not {"ref", "fixed", "refit"} <= set(g[k].columns):
            continue
        a, b, c = g[k]["ref"].values, g[k]["fixed"].values, g[k]["refit"].values
        if not np.isfinite(a).any():
            continue
        summ.append(dict(model=name, metric=k, n=len(a), ref=np.nanmedian(a), fixed=np.nanmedian(b), refit=np.nanmedian(c),
                         d_fixed_ref=np.nanmedian(b - a), p_fixed_ref=pval(b - a),
                         d_refit_fixed=np.nanmedian(c - b), p_refit_fixed=pval(c - b),
                         d_refit_ref=np.nanmedian(c - a), p_refit_ref=pval(c - a),
                         sd_ref=np.nanstd(a, ddof=1) if len(a) > 1 else 0.0,
                         sd_fixed=np.nanstd(b, ddof=1) if len(b) > 1 else 0.0,
                         sd_refit=np.nanstd(c, ddof=1) if len(c) > 1 else 0.0))
S = pd.DataFrame(summ)


def budget(name):                                  # fixed epoch budget of a trained row (0: no epoch loop)
    f = f"{TREES['fixed']}/run_standard/runs/{FMT[name][0].format(0)}/metrics.json"
    return int(json.load(open(f)).get("config", {}).get("stop_epoch", 0) or 0) if os.path.exists(f) else 0


S["epochs"] = S.model.map(budget); S.to_csv(f"{ROOT}/refit_summary.csv", index=False)

pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
for k in METS:
    t = S[S.metric == k].set_index("model")[["ref", "fixed", "refit", "d_fixed_ref", "p_fixed_ref",
                                             "d_refit_fixed", "p_refit_fixed", "d_refit_ref", "p_refit_ref",
                                             "sd_ref", "sd_refit"]].round(4)
    print(f"\n=== {k}: median over seeds; d = paired median difference; p = Wilcoxon over seeds ===")
    print(t.to_string())

# ------------------------------------------------------------------ 2. rank of the proposed model
MAIN = [m for m in FMT if m not in DEAD_HEAD]
rk = []
for prot in TREES:
    for k in METS:
        med = S[(S.metric == k) & S.model.isin(MAIN)].set_index("model")[prot]
        order = med.sort_values(ascending=(k in LOWER))
        best = order.index[0]
        rk.append(dict(protocol=prot, metric=k, rank=int(list(order.index).index(PROP)) + 1, n_models=len(order),
                       proposed=med[PROP], best_model=best, best=order.iloc[0]))
RK = pd.DataFrame(rk); RK.to_csv(f"{ROOT}/refit_rank.csv", index=False)
print("\n=== rank of the proposed model among the 14 Table-2 rows (median over seeds) ===")
print(RK.pivot(index="metric", columns="protocol", values="rank").loc[METS, list(TREES)].to_string())
print(RK.round(4).to_string(index=False))

# ------------------------------------------------------------------ 3. proposed vs each baseline
pw = []
for prot in TREES:
    P = df[(df.model == PROP) & (df.protocol == prot)].set_index("seed")
    for name in FMT:
        if name == PROP:
            continue
        Bm = df[(df.model == name) & (df.protocol == prot)].set_index("seed")
        for k in METS:
            if Bm[k].isna().all():
                continue
            if len(Bm) == 1:                       # deterministic reference: compare with its single value
                d = P[k].values - Bm[k].values[0]
            else:
                j = P.index.intersection(Bm.index); d = P.loc[j, k].values - Bm.loc[j, k].values
            gain = -d if k in LOWER else d         # > 0 : proposed better
            pw.append(dict(protocol=prot, baseline=name, metric=k, gain=np.median(gain), p=pval(gain)))
PW = pd.DataFrame(pw); PW.to_csv(f"{ROOT}/refit_pairwise.csv", index=False)
PW["verdict"] = np.where(PW.p < 0.05, np.where(PW.gain > 0, "for", "against"), "ns")
main_pw = PW[~PW.baseline.isin(DEAD_HEAD)]
print("\n=== proposed vs 13 baselines x 8 metrics (seed-paired Wilcoxon, raw p<0.05) ===")
print(main_pw.groupby(["protocol", "verdict"]).size().unstack(fill_value=0).loc[list(TREES)].to_string())
print("\n--- by metric (for / against) ---")
t = main_pw.groupby(["metric", "protocol", "verdict"]).size().unstack(fill_value=0)
for c in ("for", "against", "ns"):
    if c not in t:
        t[c] = 0
print((t["for"].astype(str) + "/" + t["against"].astype(str)).unstack().loc[METS, list(TREES)].to_string())
print("\n--- comparisons against the proposed model ---")
print(main_pw[main_pw.verdict == "against"].round(4).to_string(index=False))
print("\n--- single-task controls (gain > 0 : multi-task better) ---")
print(PW[PW.baseline.isin(DEAD_HEAD)].round(4).to_string(index=False))
print("written", ROOT)
