"""R2-m5 / R2-M1: re-anchor the event-adjacent subset on each horizon's own
TARGET day, recompute Table 2 classification metrics for all 13 models, and
add the AUPRC-vs-prevalence (lift) view.

Current manuscript definition (dataset.py line 346): a sample belongs to the
subset iff its D+1 target day lies within +-2 days of a positive day. The
label scored at horizon h is the one at D+h, so for h>=4 the scored day can be
up to 8 days after the event -> positives fall 44 -> 14.

New definition: sample (station s, origin o) is event-adjacent FOR HORIZON h
iff the target day o+h lies within +-2 days of a positive day at station s.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

df = load_daily()
df = df[df.year.isin([2015, 2016, 2017, 2018, 2019, 2020])]
pos_days = {s: set(g.loc[g.redtide_30km == 1, "date"].values.astype("datetime64[D]"))
            for s, g in df.groupby("station")}
b = load_bundle(BUNDLE_9)
stations = b["stations"]

preds = load_all()
ref = preds["MT-HAB-G3D (proposed)"][0]
date_d1 = ref["date"]                     # D+1 target day
st = ref["station"]
n = len(date_d1)

# per-horizon event-adjacent mask (target-day anchored), radius 2
ev_h = np.zeros((n, H), bool)
for h in range(H):
    tgt = date_d1 + np.timedelta64(h, "D")
    for i in range(n):
        pd_ = pos_days[stations[st[i]]]
        ev_h[i, h] = any((tgt[i] + np.timedelta64(k, "D")) in pd_ for k in range(-2, 3))
ev_old = ref["event_adj"]
np.save(f"{OUT}/event_adj_per_horizon.npy", ev_h)

print("=== event-adjacent subset size / positives per horizon ===")
rows = []
for h in range(H):
    m_old = ref["m_cls"][:, h] & ev_old
    m_new = ref["m_cls"][:, h] & ev_h[:, h]
    rows.append(dict(horizon=f"D+{h+1}",
                     n_old=int(m_old.sum()), pos_old=int(ref["y_cls"][m_old, h].sum()),
                     n_new=int(m_new.sum()), pos_new=int(ref["y_cls"][m_new, h].sum()),
                     prev_old=ref["y_cls"][m_old, h].mean(), prev_new=ref["y_cls"][m_new, h].mean(),
                     n_full=int(ref["m_cls"][:, h].sum()), pos_full=int(ref["y_cls"][ref["m_cls"][:, h], h].sum())))
sub = pd.DataFrame(rows); print(sub.round(3).to_string(index=False))
sub.to_csv(f"{OUT}/eventadj_subset_counts.csv", index=False)


def eval_model(plist, anchor):
    """median over seeds of mean over horizons for F1/AUPRC/MCC/TSS (+ full-year reg)."""
    per_seed = []
    for p in plist:
        hs = []
        for h in range(H):
            m = p["m_cls"][:, h] & (ev_h[:, h] if anchor == "target" else ev_old)
            c = cls_metrics(p["y_cls"][m, h], p["prob"][m, h], p["thr"][h])
            mr = p["m_reg"][:, h]
            r = reg_metrics(p["y_reg_raw"][mr, h], p["reg_ugl"][mr, h])
            c.update({k: r[k] for k in ["RMSE", "PeakRMSE", "R2", "TA1"]})
            c["lift"] = c["AUPRC"] / c["prevalence"] if c["prevalence"] > 0 else np.nan
            hs.append(c)
        per_seed.append(pd.DataFrame(hs))
    mean_over_h = pd.DataFrame([d.mean(numeric_only=True) for d in per_seed])
    med = mean_over_h.median(); sd = mean_over_h.std(ddof=1) if len(per_seed) > 1 else mean_over_h.iloc[0] * 0
    byh = pd.concat(per_seed).groupby(level=0).median()      # median over seeds per horizon
    return med, sd, byh


print("\n=== Table 2 (classification part) under both anchorings — median over seeds of D+1..D+7 mean ===")
tab = []
byh_store = {}
for name, plist in preds.items():
    mo, so, _ = eval_model(plist, "d1")
    mn, sn, byh = eval_model(plist, "target")
    byh_store[name] = byh
    tab.append(dict(model=name,
                    F1_old=mo.F1, F1_new=mn.F1, F1_new_sd=sn.F1,
                    AUPRC_old=mo.AUPRC, AUPRC_new=mn.AUPRC, AUPRC_new_sd=sn.AUPRC,
                    prev_new=mn.prevalence, lift_new=mn.lift,
                    MCC_old=mo.MCC, MCC_new=mn.MCC, TSS_old=mo.TSS, TSS_new=mn.TSS,
                    POD_new=mn.POD, FAR_new=mn.FAR,
                    RMSE=mn.RMSE, PeakRMSE=mn.PeakRMSE, R2=mn.R2, TA1=mn.TA1))
tab = pd.DataFrame(tab)
# single-task controls: blank the head that received no training signal
for nm, dead in DEAD_HEAD.items():
    cols = (["RMSE", "PeakRMSE", "R2", "TA1"] if dead == "reg"
            else [c for c in tab.columns if c.split("_")[0] in ("F1", "AUPRC", "MCC", "TSS", "POD", "FAR", "prev", "lift")])
    tab.loc[tab.model == nm, cols] = np.nan
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(tab.round(3).to_string(index=False))
tab.to_csv(f"{OUT}/table2_reanchored.csv", index=False)

print("\n=== MT-HAB-G3D per horizon, target-anchored (median over seeds) ===")
byh = byh_store["MT-HAB-G3D (proposed)"]
byh.index = [f"D+{h+1}" for h in range(H)]
print(byh[["n", "n_pos", "prevalence", "F1", "AUPRC", "lift", "POD", "FAR", "MCC", "TSS", "RMSE", "R2", "TA1"]].round(3).to_string())
byh.to_csv(f"{OUT}/table4_reanchored_mthab.csv")

print("\n=== climatology / GBT / MLP per horizon F1 & AUPRC (target-anchored) ===")
for nm in [m for m in ["Day-of-year climatology", "Gradient-boosted trees", "MLP", "LSTM", "Persistence", "MT-HAB-G3D (cls only)"] if m in byh_store]:
    bh = byh_store[nm]; bh.index = [f"D+{h+1}" for h in range(H)]
    print(nm); print(bh[["F1", "AUPRC", "lift", "POD", "FAR"]].round(3).T.to_string())
