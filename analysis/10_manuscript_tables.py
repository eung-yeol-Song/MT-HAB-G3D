"""Emit the LaTeX table bodies of the revised manuscript (Tables 2-7) directly from
the final run directories, so every number in the tex can be traced to a run.

Conventions (unchanged from the submitted manuscript unless stated):
  * value = median over seeds of each run's mean over D+1..D+7, "+-" = seed s.d.
  * red-tide metrics on the TARGET-anchored event-adjacent subset (revision, R2-m5)
  * Chl-a metrics on the full-year 2020 timeline
  * paired tests = two-sided Wilcoxon signed-rank over the 10 seed-level means
Output: $MTHAB_ANALYSIS_OUT/manuscript_tables.tex (+ key_numbers.txt)
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from scipy.stats import wilcoxon

ev_h = np.load(f"{OUT}/event_adj_per_horizon.npy")
preds = load_all()
b = load_bundle(BUNDLE_9)
out_tex = open(f"{OUT}/manuscript_tables.tex", "w")
key = open(f"{OUT}/key_numbers.txt", "w")


def emit(s=""):
    out_tex.write(s + "\n")


def per_seed(plist, ev=ev_h):
    """DataFrame (seeds x metrics): mean over horizons of EA classification + full-year regression."""
    rows = []
    for p in plist:
        # runs on another bundle (T=5) have their own sample set: use the target-anchored mask stored with the run
        ev = ev if ev.shape[0] == len(p["m_cls"]) else p["event_adj_h"]
        hs = []
        for h in range(H):
            m = p["m_cls"][:, h] & ev[:, h]
            c = cls_metrics(p["y_cls"][m, h], p["prob"][m, h], p["thr"][h])
            mr = p["m_reg"][:, h]
            c.update(reg_metrics(p["y_reg_raw"][mr, h], p["reg_ugl"][mr, h]))
            hs.append(c)
        rows.append(pd.DataFrame(hs).mean(numeric_only=True))
    return pd.DataFrame(rows)


def per_seed_h(plist, metric, ev=ev_h):
    """array (seeds x H) of one metric."""
    out = []
    for p in plist:
        ev = ev if ev.shape[0] == len(p["m_cls"]) else p["event_adj_h"]
        v = []
        for h in range(H):
            if metric in ("RMSE", "R2", "PeakRMSE", "TA1"):
                mr = p["m_reg"][:, h]; v.append(reg_metrics(p["y_reg_raw"][mr, h], p["reg_ugl"][mr, h])[metric])
            else:
                m = p["m_cls"][:, h] & ev[:, h]; v.append(cls_metrics(p["y_cls"][m, h], p["prob"][m, h], p["thr"][h])[metric])
        out.append(v)
    return np.array(out)


def pw(a, b_):
    """paired Wilcoxon p (NaN if fewer than 2 pairs or all-zero differences)."""
    n = min(len(a), len(b_))
    if n < 2: return np.nan
    d = np.asarray(a[:n]) - np.asarray(b_[:n])
    if np.allclose(d, 0): return 1.0
    return float(wilcoxon(a[:n], b_[:n]).pvalue)


def fmt(v, nd=3):
    return "---" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def fp(p):
    s = fmt(p); return r"\textbf{" + s + "}" if (not np.isnan(p) and p < 0.05) else s


PS = {name: per_seed(pl) for name, pl in preds.items()}
for nm, dead in DEAD_HEAD.items():
    if nm in PS:
        cols = ["RMSE", "PeakRMSE", "R2", "TA1", "MAE", "bias"] if dead == "reg" else ["F1", "AUPRC", "MCC", "TSS", "POD", "FAR", "CSI", "precision", "recall", "specificity"]
        PS[nm][cols] = np.nan

# ------------------------------------------------------------------ dataset counts
te = b["split"] == "test"; tr = b["split"] == "train"; va = b["split"] == "val"
for nm, msk in [("train", tr), ("val", va), ("test", te)]:
    n = int(msk.sum()); lab = b["m_cls"][msk, 0]; pos = int(b["y_cls"][msk, 0][lab].sum())
    key.write(f"{nm}: samples {n}, labelled D+1 {int(lab.sum())}, positives {pos} ({100*pos/lab.sum():.2f}%), chl-valid D+1 {int(b['m_reg'][msk,0].sum())}\n")
sub = pd.read_csv(f"{OUT}/eventadj_subset_counts.csv"); key.write("EA subset per horizon n/pos: " + ", ".join(f"D+{r.horizon[2:]} {r.n_new}/{r.pos_new}" for r in sub.itertuples()) + "\n")

# ------------------------------------------------------------------ Table 2
ORDER2 = ["Persistence", "Day-of-year climatology", "Gradient-boosted trees", "MLP", "LSTM", "GRU", "CNN-LSTM", "Dilated TCN",
          "Transformer", "ST-GCN (no G3D)", "MTGNN", "AM-MTGNN", "MTGNN (no graph)", "MT-HAB-G3D (proposed)",
          "MT-HAB-G3D (cls only)", "MT-HAB-G3D (reg only)"]
COLS2 = [("RMSE", False), ("PeakRMSE", False), ("R2", True), ("F1", True), ("AUPRC", True), ("MCC", True), ("TSS", True)]
med = {n: PS[n].median() for n in ORDER2 if n in PS}; sd = {n: (PS[n].std(ddof=1) if len(PS[n]) > 1 else None) for n in ORDER2 if n in PS}
best = {}
for c, hi in COLS2:
    vals = {n: med[n][c] for n in med if not np.isnan(med[n][c])}
    best[c] = max(vals, key=vals.get) if hi else min(vals, key=vals.get)
emit("% ---------------- Table 2 (tab:main) body ----------------")
for n in ORDER2:
    if n not in med: continue
    cells = []
    for c, _ in COLS2:
        v = med[n][c]
        if np.isnan(v): cells.append("---"); continue
        s = fmt(v) + (f" $\\pm$ {sd[n][c]:.3f}" if sd[n] is not None else "")
        cells.append(r"\textbf{" + s + "}" if best[c] == n else s)
    emit(n + " & " + " & ".join(cells) + r" \\")
key.write("Table2 best per column: " + ", ".join(f"{c}:{best[c]}" for c, _ in COLS2) + "\n")
# rank of the proposed model per column (among all rows with a value)
for c, hi in COLS2:
    vals = sorted([(med[n][c], n) for n in med if not np.isnan(med[n][c])], reverse=hi)
    rk = [n for _, n in vals].index("MT-HAB-G3D (proposed)") + 1
    key.write(f"  proposed rank on {c}: {rk}/{len(vals)}  (best {vals[0][1]} {vals[0][0]:.3f})\n")

# ------------------------------------------------------------------ Table 3 (D+1 operating point)
emit("\n% ---------------- Table 3 (tab:operating) body ----------------")
ref = preds["MT-HAB-G3D (proposed)"][0]
# persistence row built as in 02_persistence_fullyear_events.py
RT = NODE_VARS.index("redtide"); tei = np.where(b["split"] == "test")[0]
rt_last = b["X"][tei, -1, RT, 0]
pers = dict(prob=np.repeat(rt_last[:, None], H, 1), y_cls=ref["y_cls"], m_cls=ref["m_cls"], station=ref["station"], date=ref["date"], thr=np.full(H, 0.5))
rows3 = [("Persistence", [pers])] + [(n, preds[n]) for n in ["Day-of-year climatology", "Gradient-boosted trees", "MLP", "LSTM", "MT-HAB-G3D (proposed)"] if n in preds]
for n, pl in rows3:
    v = []
    for p in pl:
        me = p["m_cls"][:, 0] & ev_h[:, 0]; c = cls_metrics(p["y_cls"][me, 0], p["prob"][me, 0], p["thr"][0])
        mf = p["m_cls"][:, 0]; cf = cls_metrics(p["y_cls"][mf, 0], p["prob"][mf, 0], p["thr"][0])
        v.append([c["TP"], c["n_pos"], c["POD"], c["FAR"], c["CSI"], cf["FP"] / station_months(p, 0), cf["FP"]])
    v = np.median(np.array(v, float), 0)
    emit(f"{n} & {v[0]:.0f}/{v[1]:.0f} & {fmt(v[2])} & {fmt(v[3])} & {fmt(v[4])} & {v[5]:.2f} \\\\")
    key.write(f"Table3 {n}: TP {v[0]:.1f}/{v[1]:.0f} POD {v[2]:.3f} FAR {v[3]:.3f} CSI {v[4]:.3f} FA/station-month {v[5]:.3f} (FP full-year D+1 {v[6]:.1f}; station-months {station_months(ref,0)})\n")

# ------------------------------------------------------------------ Table 4 (horizon, proposed)
emit("\n% ---------------- Table 4 (tab:horizon) body ----------------")
t4 = pd.read_csv(f"{OUT}/table4_reanchored_mthab.csv", index_col=0)
for h, r in t4.iterrows():
    emit(f"{h} & {r.RMSE:.3f} & {r.R2:.3f} & {100*r.TA1:.1f}\\% & {r.F1:.3f} & {r.AUPRC:.3f} & {r.MCC:.3f} & {int(r.n)} & {int(r.n_pos)} \\\\")

# ------------------------------------------------------------------ Table 5 (paired tests)
emit("\n% ---------------- Table 5 (tab:pairs) body ----------------")
t5 = pd.read_csv(f"{OUT}/table5_pairs.csv")
ORDER5 = ["Gradient-boosted trees", "MLP", "LSTM", "GRU", "CNN-LSTM", "Dilated TCN", "Transformer", "ST-GCN (no G3D)", "MTGNN", "AM-MTGNN", "MTGNN (no graph)", "MT-HAB-G3D + spatial ext."]
for n in ORDER5:
    g = t5[t5.comparison == n].set_index("metric")
    if g.empty: continue
    cells = []
    for m in ["RMSE", "F1", "AUPRC"]:
        d = g.loc[m, "delta_mean"]; p = g.loc[m, "p"]; q = g.loc[m, "q_BH"]
        sgn = f"{d:+.3f}"
        cells += [sgn, fp(p) + ("$^{\\ast}$" if q < 0.05 else "")]
    lab = r"\quad + spatial extension" if n.startswith("MT-HAB") else n
    emit(lab + " & " + " & ".join(cells) + r" \\")
cnt = pd.read_csv(f"{OUT}/table5_counts.csv").iloc[0]
key.write(f"Table5 counts: {cnt.to_dict()}\n")
agn = t5[(~t5.favours_proposed) & (t5.p < 0.05)]; key.write("significant against proposed (raw): " + "; ".join(f"{r.comparison} {r.metric} d={r.delta_mean:+.4f} p={r.p:.3f} q={r.q_BH:.3f}" for r in agn.itertuples()) + "\n")

# ------------------------------------------------------------------ Table 6 (reliability)
emit("\n% ---------------- Table 6 (tab:reliability) body ----------------")
ORDER6 = ["Gradient-boosted trees", "MLP", "LSTM", "GRU", "CNN-LSTM", "Dilated TCN", "Transformer", "ST-GCN (no G3D)", "MTGNN", "AM-MTGNN", "MTGNN (no graph)",
          "MT-HAB-G3D (proposed)", "MT-HAB-G3D (cls only)", "MT-HAB-G3D + spatial ext."]
for n in ORDER6:
    if n not in preds: continue
    f1 = per_seed_h(preds[n], "F1"); r2 = per_seed_h(preds[n], "R2")
    div = int((r2.min(1) < -1).sum()) if n != "MT-HAB-G3D (cls only)" else None
    fails = int((f1 == 0).sum()); seeds_fail = int(((f1 == 0).sum(1) > 0).sum()); worst = f1.mean(1).min()
    lab = r"\quad + spatial extension" if "spatial" in n else (r"\quad classification-only control" if "cls only" in n else n)
    emit(f"{lab} & {'---' if div is None else div} & {fails} & {seeds_fail} & {worst:.3f} \\\\")
    key.write(f"Table6 {n}: diverged {div} fails {fails}/70 seeds_fail {seeds_fail} worst {worst:.3f}  (sd of mean F1 {f1.mean(1).std(ddof=1):.3f})\n")

# ------------------------------------------------------------------ Table 7 (ablation / revision candidates vs final)
emit("\n% ---------------- Table 7 (tab:ablation) body ----------------")
refP = PS["MT-HAB-G3D (proposed)"]
ABLROWS = [("plus_station_graph", "+ spatial extension (two nearest stations as nodes)"),
           ("no_station_emb", "$-$ station embedding"),
           ("no_log_target", "$-$ $\\log(1+y)$ target transform"),
           ("window_5", "five-day input window ($T=5$)$^{\\ddagger}$"),
           ("lam_0.2", "$\\lambda=0.2$"), ("lam_2.0", "$\\lambda=2.0$"),
           ("residual_head", "persistence-anchored Chl-\\emph{a} head"),
           ("wbce", "class-weighted BCE"),
           ("plus_doy", "+ day-of-year channels"), ("plus_rthist", "+ red-tide history channels"), ("plus_doy_rthist", "+ day-of-year and history channels")]
vs = pd.read_csv(f"{OUT}/validation_selection.csv").set_index("config") if os.path.exists(f"{OUT}/validation_selection.csv") else None
emit(f"Final configuration (reference) & {refP.RMSE.median():.3f} & --- & --- & {refP.F1.median():.3f} & --- & --- & {refP.AUPRC.median():.3f} & {refP.RMSE.std(ddof=1):.3f} & --- \\\\")
emit(r"\midrule")
for cfg, lab in ABLROWS:
    pl = [load_pred(ABL, f"{cfg}_s{s}") for s in range(10) if os.path.exists(f"{ABL}/{cfg}_s{s}/preds_test.npz")]
    if not pl: key.write(f"Table7 {cfg}: no runs\n"); continue
    P = per_seed(pl)
    n = min(len(P), len(refP))
    dR = (P.RMSE.values[:n] - refP.RMSE.values[:n]).mean(); pR = pw(P.RMSE.values, refP.RMSE.values)
    dF = (P.F1.values[:n] - refP.F1.values[:n]).mean(); pF = pw(P.F1.values, refP.F1.values)
    val = "---"
    if vs is not None and cfg in vs.index:
        r = vs.loc[cfg]; val = f"{r.dS_mean:+.3f} ({fmt(r.p_S)})"
    emit(f"{lab} & {P.RMSE.median():.3f} & {dR:+.3f} & {fp(pR)} & {P.F1.median():.3f} & {dF:+.3f} & {fp(pF)} & {P.AUPRC.median():.3f} & {P.RMSE.std(ddof=1):.3f} & {val} \\\\")
    key.write(f"Table7 {cfg}: n={len(pl)} RMSE {P.RMSE.median():.3f} dRMSE {dR:+.3f} p {pR:.3f} | F1 {P.F1.median():.3f} dF1 {dF:+.3f} p {pF:.3f} | AUPRC {P.AUPRC.median():.3f} (d {P.AUPRC.median()-refP.AUPRC.median():+.3f}, p {pw(P.AUPRC.values, refP.AUPRC.values):.3f}) | val {val} | eval n_reg {int(pl[0]['m_reg'][:,0].sum())}\n")

# ------------------------------------------------------------------ horizon-wise p-values vs GBT / persistence (text of 4.4)
key.write("\nhorizon-wise paired Wilcoxon, proposed vs GBT RMSE (p):  ")
g = per_seed_h(preds["Gradient-boosted trees"], "RMSE"); m_ = per_seed_h(preds["MT-HAB-G3D (proposed)"], "RMSE")
key.write(", ".join(f"D+{h+1} d={np.mean(m_[:,h]-g[:,h]):+.3f} p={pw(m_[:,h], g[:,h]):.3f}" for h in range(H)) + "\n")
ml = per_seed_h(preds["MLP"], "RMSE")
key.write("proposed vs MLP RMSE (p): " + ", ".join(f"D+{h+1} d={np.mean(m_[:,h]-ml[:,h]):+.3f} p={pw(m_[:,h], ml[:,h]):.3f}" for h in range(H)) + "\n")
cl = per_seed_h(preds["Day-of-year climatology"], "F1"); mf = per_seed_h(preds["MT-HAB-G3D (proposed)"], "F1")
key.write("EA F1 by horizon  proposed: " + ", ".join(f"{np.median(mf[:,h]):.3f}" for h in range(H)) + "\n")
key.write("EA F1 by horizon  climatology: " + ", ".join(f"{cl[0,h]:.3f}" for h in range(H)) + "\n")
pe = pd.read_csv(f"{OUT}/persistence_by_horizon.csv"); key.write("EA F1 by horizon  persistence: " + ", ".join(f"{v:.3f}" for v in pe.pers_F1_ev) + "\n")
key.write("EA AUPRC by horizon proposed: " + ", ".join(f"{np.median(per_seed_h(preds['MT-HAB-G3D (proposed)'],'AUPRC')[:,h]):.3f}" for h in range(H)) + "\n")
key.write("EA AUPRC by horizon climatology: " + ", ".join(f"{per_seed_h(preds['Day-of-year climatology'],'AUPRC')[0,h]:.3f}" for h in range(H)) + "\n")
key.write("EA AUPRC by horizon LSTM: " + ", ".join(f"{np.median(per_seed_h(preds['LSTM'],'AUPRC')[:,h]):.3f}" for h in range(H)) + "\n")
# multitask vs single-task summary
mt = pd.read_csv(f"{OUT}/multitask_vs_single.csv"); key.write("\nmultitask vs single:\n" + mt.round(3).to_string(index=False) + "\n")
ci = pd.read_csv(f"{OUT}/sample_ci.csv"); key.write("\nsample CI:\n" + ci.round(3).to_string(index=False) + "\n")
on = pd.read_csv(f"{OUT}/onset_detection.csv"); key.write("\nonset POD proposed: " + str(on[on.model == "MT-HAB-G3D (proposed)"].onset_POD.round(3).tolist()) + "  persistence: " + str(on[on.model == "Persistence"].onset_POD.round(3).tolist()) + "\n")
out_tex.close(); key.close()
print(open(f"{OUT}/manuscript_tables.tex").read()); print(open(f"{OUT}/key_numbers.txt").read())
