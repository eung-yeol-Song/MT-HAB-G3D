"""Onset detection (persistence has zero skill by construction), per-station
results (R1-s12), data-flow accounting (R1-s10), summary statistics (R1-s8),
winsorisation (R2-m3) and forward-fill (R2-m2) proportions, QC flags (R2-m1)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)

b = load_bundle(BUNDLE_9); stations = b["stations"]
te = np.where(b["split"] == "test")[0]
preds = load_all(["MT-HAB-G3D (proposed)", "MT-HAB-G3D (cls only)", "MLP", "Day-of-year climatology", "Gradient-boosted trees", "LSTM", "MTGNN (no graph)", "ST-GCN (no G3D)"])
ref = preds["MT-HAB-G3D (proposed)"][0]
ev = pd.read_csv(f"{OUT}/events_2020.csv", parse_dates=["start", "end"])
df = load_daily(); d20 = df[df.year == 2020]
CHL, RT = NODE_VARS.index("chl"), NODE_VARS.index("redtide")
rt_last = b["X"][te, -1, RT, 0]

# ---------------------------------------------------------------- onset detection
def onset_hits(prob, m_cls, thr, h, lead=2):
    """event onset (first positive day) is detected if an alert is issued for any target day in
    [onset-lead, onset]; only onsets with >=1 scorable forecast in that window are counted."""
    tgt = ref["date"] + np.timedelta64(h, "D"); alert = (prob[:, h] >= thr) & m_cls[:, h]
    n = hit = 0
    for _, r in ev.iterrows():
        si = stations.index(r.station)
        win = (ref["station"] == si) & (tgt >= np.datetime64(r.start - pd.Timedelta(days=lead))) & (tgt <= np.datetime64(r.start))
        if not m_cls[win, h].any(): continue
        n += 1; hit += int(alert[win].any())
    return n, hit
print("=== ONSET detection (first day of each station-level event, alert within 2 days before or on onset) ===")
rows = []
for name, plist in preds.items():
    for h in range(H):
        v = [onset_hits(p["prob"], p["m_cls"], p["thr"][h], h) for p in plist]
        rows.append(dict(model=name, horizon=f"D+{h+1}", onsets=int(np.median([a for a, _ in v])), hits=float(np.median([b_ for _, b_ in v])),
                         onset_POD=float(np.median([b_ / a for a, b_ in v]))))
for h in range(H):
    n, hit = onset_hits(np.repeat(rt_last[:, None], H, 1), ref["m_cls"], 0.5, h)
    rows.append(dict(model="Persistence", horizon=f"D+{h+1}", onsets=n, hits=hit, onset_POD=hit / n))
on = pd.DataFrame(rows); on.to_csv(f"{OUT}/onset_detection.csv", index=False)
print(on.pivot(index="model", columns="horizon", values="onset_POD").round(2).to_string())
print("(onsets evaluable per horizon:", on[on.model == "MT-HAB-G3D (proposed)"].onsets.tolist(), ")")

# ---------------------------------------------------------------- per-station results (final model, median over seeds)
print("\n=== per-station results, MT-HAB-G3D, D+1..D+7 mean, median over 10 seeds ===")
rows = []
for si, s in enumerate(stations):
    per = []
    for p in preds["MT-HAB-G3D (proposed)"]:
        hs = []
        for h in range(H):
            m = (p["station"] == si)
            mr = m & p["m_reg"][:, h]; mc = m & p["m_cls"][:, h]
            d = {}
            if mr.sum() > 10:
                d.update({k: v for k, v in reg_metrics(p["y_reg_raw"][mr, h], p["reg_ugl"][mr, h]).items() if k in ["RMSE", "R2", "TA1"]})
            d["n_reg"] = int(mr.sum()); d["n_cls"] = int(mc.sum()); d["pos"] = int(p["y_cls"][mc, h].sum())
            d["TP"] = int(((p["prob"][mc, h] >= p["thr"][h]) & (p["y_cls"][mc, h] == 1)).sum()); d["FP"] = int(((p["prob"][mc, h] >= p["thr"][h]) & (p["y_cls"][mc, h] == 0)).sum())
            hs.append(d)
        per.append(pd.DataFrame(hs).mean(numeric_only=True))
    md = pd.DataFrame(per).median()
    chl_mean = d20.loc[d20.station == s, "chl"].mean()
    rows.append(dict(station=s, n_days_cls=int(md.n_cls), n_days_chl=int(md.n_reg), chl_mean_2020=chl_mean,
                     RMSE=md.get("RMSE", np.nan), R2=md.get("R2", np.nan), TA1=md.get("TA1", np.nan),
                     pos_station_days=round(md.pos, 1), TP=round(md.TP, 1), FP=round(md.FP, 1)))
ps = pd.DataFrame(rows); print(ps.round(3).to_string(index=False)); ps.to_csv(f"{OUT}/per_station_results.csv", index=False)

# ---------------------------------------------------------------- data-flow accounting
print("\n=== DATA FLOW (R1-s10) ===")
d = df[df.year.between(2015, 2020)].copy()
wq = ["ec", "temp", "ph", "do", "turbidity", "salinity", "chl", "cod"]
req = ["ec", "temp", "ph", "do", "turbidity", "salinity", "cod"]          # chl optional
flow = []
flow.append(("station-days in the 2015-2020 archive (17 stations)", len(d)))
flow.append(("  ... with all 7 required variables passing the 70% coverage rule", int(d[req].notna().all(1).sum())))
flow.append(("  ... with Chl-a also available", int(d[wq].notna().all(1).sum())))
# after ffill (<=3 d) as implemented in the bundle
def ffill_station(g):
    g = g.sort_values("date").set_index("date").reindex(pd.date_range(g.date.min(), g.date.max(), freq="D"))
    raw = g[wq]; filled = raw.ffill(limit=3)
    return raw, filled
tot_raw_missing = tot_filled = tot_cells = 0; per_var_filled = pd.Series(0.0, index=wq); per_var_cells = pd.Series(0.0, index=wq)
rows_after_ffill = 0
for s, g in d.groupby("station"):
    raw, filled = ffill_station(g)
    imputed = raw.isna() & filled.notna()
    per_var_filled += imputed.sum(); per_var_cells += len(raw)
    rows_after_ffill += int(filled[req].notna().all(1).sum())
flow.append(("  ... required variables complete AFTER forward-fill (<=3 d), calendar grid", rows_after_ffill))
n_samples = len(b["X"]); flow.append(("forecast samples (10-day window fully valid + D+1 target), all splits", n_samples))
for sp in ["train", "val", "test"]:
    i = np.where(b["split"] == sp)[0]
    flow.append((f"  {sp}: samples / with valid red-tide label at D+1 / with valid Chl-a at D+1", f"{len(i)} / {int(b['m_cls'][i,0].sum())} / {int(b['m_reg'][i,0].sum())}"))
for k, v in flow: print(f"{k:80s} {v}")
pd.DataFrame(flow, columns=["stage", "n"]).to_csv(f"{OUT}/data_flow.csv", index=False)
print("\nforward-filled share of input cells (per variable, % of calendar station-days 2015-2020):")
print((per_var_filled / per_var_cells * 100).round(2).to_string())
# share of model-input cells that were forward-filled: compare bundle X values against raw daily values is complex;
# approximate with the per-variable rates above.

# ---------------------------------------------------------------- summary statistics (R1-s8)
print("\n=== SUMMARY STATISTICS (daily values, 2015-2020, 17 stations) ===")
st = d[wq].describe(percentiles=[.05, .5, .95]).T[["count", "mean", "std", "min", "5%", "50%", "95%", "max"]]
st["missing_%"] = (1 - d[wq].notna().mean()) * 100
st.loc["redtide_30km", ["count", "mean"]] = [int(d.redtide_30km.notna().sum()), d.redtide_30km.mean()]
print(st.round(3).to_string()); st.to_csv(f"{OUT}/table1_summary_stats.csv")
# the same table after the physical QC screen that the models actually see (R2-m1 / R1 Table 1):
# the raw minima (pH 0, DO 0, Chl-a < 0) are sensor faults, not environmental values
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.data.dataset import apply_physical_qc
dq = apply_physical_qc(d.copy())
stq = dq[wq].describe(percentiles=[.05, .5, .95]).T[["count", "mean", "std", "min", "5%", "50%", "95%", "max"]]
stq["missing_%"] = (1 - dq[wq].notna().mean()) * 100
print("\n--- after physical QC (values used for modelling) ---"); print(stq.round(3).to_string())
stq.to_csv(f"{OUT}/table1_summary_stats_postqc.csv")
print("\nper-split Chl-a and positives:")
for yrs, nm in [((2015, 2016, 2017, 2018), "train"), ((2019,), "val"), ((2020,), "test")]:
    x = d[d.year.isin(yrs)]
    print(f"  {nm:5s}: station-days {len(x):5d}  chl mean {x.chl.mean():.2f} median {x.chl.median():.2f} p95 {x.chl.quantile(.95):.1f} max {x.chl.max():.1f}  | red-tide positive days {int(x.redtide_30km.sum())} ({x.redtide_30km.mean()*100:.2f}%)")

# ---------------------------------------------------------------- winsorisation (R2-m3)
print("\n=== WINSORISATION at +-6 SD (train-period per-station statistics), applied to INPUT features only ===")
tr = d[d.year.between(2015, 2018)]
cnt = pd.Series(0, index=wq); tot = pd.Series(0, index=wq); hi = pd.Series(0, index=wq); chl_hi_vals = []
for s, g in d.groupby("station"):
    t = tr[tr.station == s]; mu = t[wq].mean(); sdv = t[wq].std().replace(0, np.nan)
    gmu, gsd = tr[wq].mean(), tr[wq].std(); mu = mu.fillna(gmu); sdv = sdv.fillna(gsd).replace(0, 1.0)
    z = (g[wq] - mu) / sdv
    cnt += (z.abs() > 6).sum(); hi += (z > 6).sum(); tot += z.notna().sum()
    chl_hi_vals += g.loc[z.chl > 6, "chl"].tolist()
w = pd.DataFrame({"n_valid": tot, "n_clipped": cnt, "pct": cnt / tot * 100, "n_clipped_high": hi})
print(w.round(3).to_string())
print(f"Chl-a values clipped as inputs (>+6 SD): n={len(chl_hi_vals)}, min {np.min(chl_hi_vals) if chl_hi_vals else np.nan:.1f} max {np.max(chl_hi_vals) if chl_hi_vals else np.nan:.1f} ug/L")
print("NOTE: the Chl-a TARGET is log1p-transformed and standardised WITHOUT clipping (dataset.py l.264-268); y_reg_raw is the raw value.")
w.to_csv(f"{OUT}/winsorisation.csv")

# ---------------------------------------------------------------- QC flags (R2-m1)
print("\n=== physically implausible daily values present in the archive (QC gap) ===")
q = dict(ph_lt5=(d.ph < 5).sum(), ph_gt10=(d.ph > 10).sum(), do_eq0=(d["do"] <= 0).sum(), salinity_eq0=(d.salinity <= 0).sum(),
         ec_eq0=(d.ec <= 0).sum(), chl_neg=(d.chl < 0).sum(), turb_gt200=(d.turbidity > 200).sum(), temp_lt0=(d.temp < 0).sum())
print(q)
