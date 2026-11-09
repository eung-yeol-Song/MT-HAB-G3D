"""Operating-point sensitivity (R2-M1): alternative decision-threshold rules, all
selected on the 2019 VALIDATION predictions only, scored on the 2020 test set.

Rules
  f1max   : argmax_p F1_val(p)            (manuscript rule, Eq. 16)
  f2max   : argmax_p F2_val(p)            (recall-weighted)
  tssmax  : argmax_p TSS_val(p)           (Youden's J)
  rate_k  : alert rate on validation = k x validation prevalence   (k = 2, 3, 5)
  pooled  : median over horizons of the f1max thresholds

Needs preds_val.npz (written by train.py --save_val_preds).  If it is absent the
script runs in DEMO mode on the test predictions (clearly labelled) so the code
path can be checked; demo numbers are NOT reportable.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)

base, fmt, n = MODELS["MT-HAB-G3D (proposed)"]
ev_h_path = f"{OUT}/event_adj_per_horizon.npy"
GRID = np.unique(np.round(np.linspace(0.01, 0.99, 99), 3))


def fbeta(y, p, t, beta):
    c = cls_metrics(y, p, t); pr, rc = c["precision"], c["recall"]
    return (1 + beta ** 2) * pr * rc / (beta ** 2 * pr + rc) if pr + rc else 0.0


def rules_from_val(yv, pv):
    f1 = GRID[np.argmax([cls_metrics(yv, pv, t)["F1"] for t in GRID])]
    f2 = GRID[np.argmax([fbeta(yv, pv, t, 2.0) for t in GRID])]
    tss = GRID[np.argmax([cls_metrics(yv, pv, t)["TSS"] for t in GRID])]
    prev = yv.mean()
    out = {"f1max": f1, "f2max": f2, "tssmax": tss}
    for k in (2, 3, 5):
        q = 1 - min(k * prev, 0.5)
        out[f"rate_{k}x"] = float(np.quantile(pv, q))
    return out


rows = []
demo = False
for s in range(n):
    pt = load_pred(base, fmt.format(s))
    fv = f"{base}/{fmt.format(s)}/preds_val.npz"
    if os.path.exists(fv):
        zv = np.load(fv); val = {k: zv[k] for k in zv.files}
    else:
        demo = True; val = pt                                   # DEMO: test used as pseudo-validation
    ev_h = pt["event_adj_h"] if "event_adj_h" in pt else np.load(ev_h_path)
    thr_by_rule = {}
    for h in range(H):
        mv = val["m_cls"][:, h]
        thr_by_rule[h] = rules_from_val(val["y_cls"][mv, h].astype(int), val["prob"][mv, h])
    pooled = float(np.median([thr_by_rule[h]["f1max"] for h in range(H)]))
    for h in range(H):
        thr_by_rule[h]["pooled"] = pooled
        thr_by_rule[h]["manuscript(stored)"] = float(pt["thr"][h])
        me = pt["m_cls"][:, h] & ev_h[:, h]; mf = pt["m_cls"][:, h]
        for rule, t in thr_by_rule[h].items():
            ce = cls_metrics(pt["y_cls"][me, h], pt["prob"][me, h], t)
            cf = cls_metrics(pt["y_cls"][mf, h], pt["prob"][mf, h], t)
            rows.append(dict(seed=s, horizon=h + 1, rule=rule, thr=t, F1_ev=ce["F1"], POD_ev=ce["POD"], FAR_ev=ce["FAR"],
                             MCC_ev=ce["MCC"], TSS_ev=ce["TSS"], F1_full=cf["F1"], POD_full=cf["POD"],
                             FP_full=cf["FP"], FA_station_month=cf["FP"] / (17 * 12), TSS_full=cf["TSS"]))
df = pd.DataFrame(rows)
tag = "DEMO (test used as validation - NOT reportable)" if demo else "validation-selected"
print(f"=== threshold rules [{tag}], median over seeds ===")
agg = df.groupby(["rule", "horizon"]).median(numeric_only=True).reset_index()
for metric in ["thr", "F1_ev", "POD_ev", "FAR_ev", "FP_full"]:
    print(f"\n-- {metric} --")
    print(agg.pivot(index="rule", columns="horizon", values=metric).round(3).to_string())
summary = df.groupby(["seed", "rule"]).mean(numeric_only=True).reset_index().groupby("rule").median(numeric_only=True)
print("\n-- D+1..D+7 mean, median over seeds --")
print(summary[["F1_ev", "POD_ev", "FAR_ev", "MCC_ev", "TSS_ev", "F1_full", "FP_full", "FA_station_month"]].round(3).to_string())
df.to_csv(f"{OUT}/threshold_rules_raw.csv", index=False); summary.to_csv(f"{OUT}/threshold_rules_summary.csv")
print("demo mode:", demo)
