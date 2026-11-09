"""R2-m6 persistence baselines, full-year threshold-free skill (AUPRC/AUROC
vs prevalence), threshold sweep (POD-FAR), and calendar-based event-level
detection with regional merging (R2-M1)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
b = load_bundle(BUNDLE_9)
stations = b["stations"]
te = np.where(b["split"] == "test")[0]
preds = load_all()
ref = preds["MT-HAB-G3D (proposed)"][0]
assert np.array_equal(b["date"][te], ref["date"]) and np.array_equal(b["station"][te], ref["station"])
ev_h = np.load(f"{OUT}/event_adj_per_horizon.npy")
CHL, RT = NODE_VARS.index("chl"), NODE_VARS.index("redtide")

# ------------------------------------------------------------------ persistence
# Chl-a: last observed (possibly forward-filled) value on the forecast origin, in ug/L.
# X[...,0] is the per-station z-score of the raw value (clipped at +-6); invert with scaler.
mu = np.array([b["scaler"][s]["mu"]["chl"] for s in stations]); sd = np.array([b["scaler"][s]["sd"]["chl"] for s in stations])
z_last = b["X"][te, -1, CHL, 0]; ok_last = b["X"][te, -1, CHL, 1] > 0.5
chl_last = z_last * sd[b["station"][te]] + mu[b["station"][te]]
rt_last = b["X"][te, -1, RT, 0]
pers = dict(prob=np.repeat(rt_last[:, None], H, 1), reg_ugl=np.repeat(chl_last[:, None], H, 1),
            y_cls=ref["y_cls"], m_cls=ref["m_cls"], y_reg_raw=ref["y_reg_raw"],
            m_reg=ref["m_reg"] & ok_last[:, None], station=ref["station"], date=ref["date"],
            thr=np.full(H, 0.5))
print(f"persistence: Chl-a anchor available for {ok_last.sum()} / {len(te)} test samples "
      f"({ref['m_reg'][:,0].sum()} have a valid D+1 Chl-a target)")

rows = []
for h in range(H):
    mr = pers["m_reg"][:, h]
    r = reg_metrics(pers["y_reg_ugl"][mr, h] if "y_reg_ugl" in pers else pers["y_reg_raw"][mr, h], pers["reg_ugl"][mr, h])
    me = pers["m_cls"][:, h] & ev_h[:, h]
    c = cls_metrics(pers["y_cls"][me, h], pers["prob"][me, h], 0.5)
    mf = pers["m_cls"][:, h]
    cf = cls_metrics(pers["y_cls"][mf, h], pers["prob"][mf, h], 0.5)
    # the proposed model on the SAME regression samples (so the comparison is paired)
    pm = []
    for p in preds["MT-HAB-G3D (proposed)"]:
        pm.append(reg_metrics(p["y_reg_raw"][mr, h], p["reg_ugl"][mr, h])["RMSE"])
    gb = [reg_metrics(p["y_reg_raw"][mr, h], p["reg_ugl"][mr, h])["RMSE"] for p in preds["Gradient-boosted trees"]]
    rows.append(dict(horizon=f"D+{h+1}", n_reg=int(mr.sum()), pers_RMSE=r["RMSE"], pers_R2=r["R2"], pers_PeakRMSE=r["PeakRMSE"], pers_TA1=r["TA1"],
                     mthab_RMSE_same_n=np.median(pm), gbt_RMSE_same_n=np.median(gb),
                     pers_F1_ev=c["F1"], pers_POD_ev=c["POD"], pers_FAR_ev=c["FAR"], pers_MCC_ev=c["MCC"], pers_TSS_ev=c["TSS"],
                     pers_F1_full=cf["F1"], pers_POD_full=cf["POD"], pers_FAR_full=cf["FAR"], pers_FP_full=cf["FP"], pers_TSS_full=cf["TSS"]))
pt = pd.DataFrame(rows)
print("\n=== PERSISTENCE baseline per horizon ===")
print(pt.round(3).to_string(index=False))
pt.to_csv(f"{OUT}/persistence_by_horizon.csv", index=False)
print("\nmean over horizons: pers RMSE %.3f | MT-HAB %.3f | GBT %.3f ;  pers ev-F1 %.3f  POD %.3f FAR %.3f ; pers R2 %.3f" % (
    pt.pers_RMSE.mean(), pt.mthab_RMSE_same_n.mean(), pt.gbt_RMSE_same_n.mean(), pt.pers_F1_ev.mean(), pt.pers_POD_ev.mean(), pt.pers_FAR_ev.mean(), pt.pers_R2.mean()))

# event-adjacent F1 of MT-HAB at horizon 1..7 vs persistence, per seed, paired
print("\n=== MT-HAB-G3D vs persistence, event-adjacent F1 per horizon (median over seeds) ===")
for name in [m for m in ["MT-HAB-G3D (proposed)", "MT-HAB-G3D (cls only)", "MLP", "Day-of-year climatology"] if m in preds]:
    f = np.array([[cls_metrics(p["y_cls"][p["m_cls"][:, h] & ev_h[:, h], h], p["prob"][p["m_cls"][:, h] & ev_h[:, h], h], p["thr"][h])["F1"] for h in range(H)] for p in preds[name]])
    print(f"{name:26s}", np.round(np.median(f, 0), 3))
print(f"{'persistence':26s}", np.round(pt.pers_F1_ev.values, 3))

# ------------------------------------------------------------------ full-year threshold-free skill
print("\n=== FULL-YEAR classification skill (prevalence ~1.2%): median over seeds of D+1..D+7 mean ===")
rows = []
for name, plist in preds.items():
    per = []
    for p in plist:
        hs = []
        for h in range(H):
            m = p["m_cls"][:, h]
            c = cls_metrics(p["y_cls"][m, h], p["prob"][m, h], p["thr"][h])
            c["AUROC"] = auroc(p["y_cls"][m, h], p["prob"][m, h])
            c["lift"] = c["AUPRC"] / c["prevalence"]
            c["FA_per_station_month"] = c["FP"] / station_months(p, h)   # monitored station-months, not 17 x 12
            hs.append(c)
        per.append(pd.DataFrame(hs).mean(numeric_only=True))
    d = pd.DataFrame(per)
    rows.append(dict(model=name, AUPRC_full=d.AUPRC.median(), lift_full=d.lift.median(), AUROC_full=d.AUROC.median(),
                     F1_full=d.F1.median(), POD_full=d.POD.median(), FAR_full=d.FAR.median(), MCC_full=d.MCC.median(), TSS_full=d.TSS.median(),
                     FP_per_h=d.FP.median(), FA_station_month=d.FA_per_station_month.median()))
fy = pd.DataFrame(rows); print(fy.round(3).to_string(index=False)); fy.to_csv(f"{OUT}/fullyear_skill.csv", index=False)

# ------------------------------------------------------------------ threshold sweep (proposed vs climatology / MLP), D+1 and D+7
print("\n=== threshold sweep: event-adjacent POD/FAR and full-year FP at matched POD (D+1), seed-pooled median curve ===")
def sweep(plist, h, grid=np.linspace(0.01, 0.99, 99)):
    out = []
    for t in grid:
        vals = []
        for p in plist:
            me = p["m_cls"][:, h] & ev_h[:, h]; mf = p["m_cls"][:, h]
            ce = cls_metrics(p["y_cls"][me, h], p["prob"][me, h], t); cf = cls_metrics(p["y_cls"][mf, h], p["prob"][mf, h], t)
            vals.append((ce["POD"], ce["FAR"], ce["F1"], cf["FP"], cf["POD"], cf["precision"]))
        v = np.median(np.array(vals), 0)
        out.append(dict(thr=t, POD_ev=v[0], FAR_ev=v[1], F1_ev=v[2], FP_full=v[3], POD_full=v[4], prec_full=v[5]))
    return pd.DataFrame(out)
sw = {}
for name in [m for m in ["MT-HAB-G3D (proposed)", "MLP", "Day-of-year climatology", "Gradient-boosted trees"] if m in preds]:
    for h in [0, 3, 6]:
        s = sweep(preds[name], h); s["model"] = name; s["horizon"] = f"D+{h+1}"; sw[(name, h)] = s
allsw = pd.concat(sw.values()); allsw.to_csv(f"{OUT}/threshold_sweep.csv", index=False)
# POD-matched comparison: at the threshold where MT-HAB reaches climatology's POD (per horizon)
for h in [0, 3, 6]:
    clim = sw[("Day-of-year climatology", h)]; cpod = clim.POD_ev.max()
    # climatology at its validation-chosen threshold
    p = preds["Day-of-year climatology"][0]; me = p["m_cls"][:, h] & ev_h[:, h]; mf = p["m_cls"][:, h]
    cc = cls_metrics(p["y_cls"][me, h], p["prob"][me, h], p["thr"][h]); cfc = cls_metrics(p["y_cls"][mf, h], p["prob"][mf, h], p["thr"][h])
    m = sw[("MT-HAB-G3D (proposed)", h)]
    cand = m[m.POD_ev >= cc["POD"] - 1e-9]
    if len(cand):
        r = cand.iloc[-1]   # largest threshold still reaching that POD
        print(f"D+{h+1}: climatology thr={p['thr'][h]:.2f} POD={cc['POD']:.3f} FAR={cc['FAR']:.3f} FP_full={cfc['FP']}  ||  MT-HAB at thr={r.thr:.2f}: POD={r.POD_ev:.3f} FAR={r.FAR_ev:.3f} FP_full={r.FP_full:.0f} F1={r.F1_ev:.3f}")
    else:
        print(f"D+{h+1}: MT-HAB cannot reach climatology POD {cc['POD']:.3f} (max {m.POD_ev.max():.3f})")
# best achievable F1 (oracle threshold on test) per horizon, to show the gap between val-selected and oracle
print("\noracle (test-selected) event-adjacent F1 vs validation-selected, MT-HAB-G3D, median over seeds:")
for h in range(H):
    orc = []; val = []
    for p in preds["MT-HAB-G3D (proposed)"]:
        me = p["m_cls"][:, h] & ev_h[:, h]; y = p["y_cls"][me, h]; pr = p["prob"][me, h]
        orc.append(max(cls_metrics(y, pr, t)["F1"] for t in np.linspace(0.01, 0.99, 99))); val.append(cls_metrics(y, pr, p["thr"][h])["F1"])
    print(f"  D+{h+1}: val-thr F1 {np.median(val):.3f}  oracle F1 {np.median(orc):.3f}   (val thr median {np.median([p['thr'][h] for p in preds['MT-HAB-G3D (proposed)']]):.2f})")

# ------------------------------------------------------------------ calendar-based event-level detection
print("\n=== EVENTS in the 2020 test year (calendar-contiguous positive days per station; gap<=1 day merged) ===")
df = load_daily(); d20 = df[df.year == 2020]
CLUSTER = {s: s[:2] for s in stations}          # 광양/금강/낙동/마산/부산/새만금/시화/영산/울산/인천
events = []
for s, g in d20.groupby("station"):
    g = g.sort_values("date"); pos = g.loc[g.redtide_30km == 1, "date"].dt.normalize().tolist()
    if not pos: continue
    cur = [pos[0]]
    for d in pos[1:]:
        if (d - cur[-1]).days <= 2: cur.append(d)
        else: events.append((s, cur[0], cur[-1])); cur = [d]
    events.append((s, cur[0], cur[-1]))
ev = pd.DataFrame(events, columns=["station", "start", "end"]); ev["days"] = (ev.end - ev.start).dt.days + 1; ev["cluster"] = ev.station.map(CLUSTER)
print(ev.sort_values(["start", "station"]).to_string(index=False))
print(f"\nstation-level events: {len(ev)}  (positive station-days in 2020: {int((d20.redtide_30km==1).sum())}; mean duration {ev.days.mean():.1f} d, max {ev.days.max()})")
# regional events: merge overlapping station events within the same cluster, then across all stations (any overlap in time)
def merge(evs):
    evs = sorted(evs, key=lambda x: x[0]); out = []
    for s, e in evs:
        if out and s <= out[-1][1] + pd.Timedelta(days=2): out[-1] = (out[-1][0], max(out[-1][1], e))
        else: out.append((s, e))
    return out
reg = sum(len(merge(list(zip(g.start, g.end)))) for _, g in ev.groupby("cluster"))
nat = len(merge(list(zip(ev.start, ev.end))))
print(f"regional (same-cluster merged) events: {reg};  nationwide time-merged episodes: {nat}")
ev.to_csv(f"{OUT}/events_2020.csv", index=False)

# event detection for the forecast models: an event (s,[a,b]) at horizon h is HIT if any alert is issued for a
# target day in [a-2, b] at station s (alerts issued before onset or during the event count; tolerance 2 days before onset).
def event_hits(p, h, thr=None, lead_window=2):
    thr = p["thr"][h] if thr is None else thr
    tgt = p["date"] + np.timedelta64(h, "D"); alert = (p["prob"][:, h] >= thr) & p["m_cls"][:, h]
    hits = 0; early = 0; evaluable = 0
    for _, r in ev.iterrows():
        si = stations.index(r.station)
        win = (p["station"] == si) & (tgt >= np.datetime64(r.start - pd.Timedelta(days=lead_window))) & (tgt <= np.datetime64(r.end))
        if not p["m_cls"][win, h].any(): continue      # no scorable forecast in the window
        evaluable += 1
        if alert[win].any():
            hits += 1
            if alert[win & (tgt < np.datetime64(r.start))].any(): early += 1
    # false-alarm episodes: contiguous alert runs (per station, calendar) not within 2 days of any positive day
    fa = 0
    pos_days = {s: set(g.loc[g.redtide_30km == 1, "date"].values.astype("datetime64[D]")) for s, g in d20.groupby("station")}
    for si, s in enumerate(stations):
        m = (p["station"] == si) & alert
        days = np.sort(tgt[m])
        if len(days) == 0: continue
        runs = []; cur = [days[0]]
        for d in days[1:]:
            if (d - cur[-1]).astype(int) <= 2: cur.append(d)
            else: runs.append(cur); cur = [d]
        runs.append(cur)
        for run in runs:
            if not any((d + np.timedelta64(k, "D")) in pos_days[s] for d in run for k in range(-2, 3)):
                fa += 1
    return dict(events=evaluable, hits=hits, early=early, false_alarm_episodes=fa,
                event_POD=hits / evaluable if evaluable else np.nan,
                event_FAR=fa / (hits + fa) if hits + fa else np.nan)

print("\n=== event-level detection per model (median over seeds), validation thresholds ===")
rows = []
for name in [m for m in ["MT-HAB-G3D (proposed)", "MT-HAB-G3D (cls only)", "MLP", "Day-of-year climatology", "Gradient-boosted trees", "LSTM", "ST-GCN (no G3D)", "MTGNN (no graph)"] if m in preds]:
    for h in [0, 1, 2, 3, 4, 5, 6]:
        vals = pd.DataFrame([event_hits(p, h) for p in preds[name]]).median()
        rows.append(dict(model=name, horizon=f"D+{h+1}", **vals.to_dict()))
evt = pd.DataFrame(rows); evt.to_csv(f"{OUT}/event_level_detection.csv", index=False)
print(evt.pivot(index="model", columns="horizon", values="event_POD").round(2).to_string())
print("\nfalse-alarm episodes (full year, per model):")
print(evt.pivot(index="model", columns="horizon", values="false_alarm_episodes").round(1).to_string())
print("\nhits (events):")
print(evt.pivot(index="model", columns="horizon", values="hits").round(1).to_string())
# persistence at event level
pv = [event_hits(pers, h, thr=0.5) for h in range(H)]
print("\npersistence event-level:", [(f"D+{h+1}", round(v["event_POD"], 2), v["false_alarm_episodes"]) for h, v in enumerate(pv)])
