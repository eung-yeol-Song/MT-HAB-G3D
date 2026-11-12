"""Supplementary material for the revised manuscript (tables S1-S8, figures S1-S5).
S1-S3  Table-2 equivalents of the full comparison matrix re-trained on the three augmented inputs
S4     threshold rules (validation F1-max vs alternatives, oracle) for the final model
S5     onset / event-level detection per model and lead time
S6     per-station results of the final model (2020)
S7-S8  comparison matrix refitted on the training + validation years, and the effect of the added year
Figures: S1 obs-vs-pred, S2 permutation importance, S3 residual adjacency, S4 station embedding, S5 multitask-vs-single.
Writes $SUPP_DIR/supplement_v4.tex and copies the figure PDFs there."""
import os, shutil, sys
import numpy as np, pandas as pd
OUTD = os.environ.get("MTHAB_ANALYSIS_OUT", "outputs/analysis")
SUPP = os.environ.get("SUPP_DIR", "../manuscript_v4/supplement"); os.makedirs(SUPP, exist_ok=True)
EN = {"인천송도": "Incheon-Songdo", "시화조력": "Sihwa-tidal", "시화반월": "Sihwa-Banwol", "금강하구": "Geum estuary", "새만금": "Saemangeum",
      "영산영암": "Yeongsan-Yeongam", "영산목포": "Yeongsan-Mokpo", "광양망덕": "Gwangyang-Mangdeok", "광양초남": "Gwangyang-Chonam",
      "광양적량": "Gwangyang-Jeokryang", "마산양덕": "Masan-Yangdeok", "마산봉암": "Masan-Bongam", "마산삼귀": "Masan-Samgwi",
      "울산온산": "Ulsan-Onsan", "부산수영": "Busan-Suyeong", "낙동을숙": "Nakdong-Eulsuk", "낙동명지": "Nakdong-Myeongji"}
L = []
def w(s=""): L.append(s)
def f(v, nd=3): return "---" if pd.isna(v) else f"{v:.{nd}f}"

w(r"""\documentclass[11pt]{article}
\usepackage[a4paper,margin=2cm]{geometry}\usepackage{booktabs,graphicx,amsmath}\usepackage[table]{xcolor}
\renewcommand{\thetable}{S\arabic{table}}\renewcommand{\thefigure}{S\arabic{figure}}
\title{Supplementary Material\\Multi-Task Spatiotemporal Graph Learning for Coastal Harmful Algal Bloom Forecasting}
\date{}\begin{document}\maketitle
All numbers are medians over ten seeds of each run's mean over D+1--D+7 unless stated otherwise; red-tide metrics use the target-anchored event-adjacent subset and Chl-$a$ metrics the full-year 2020 timeline, exactly as in the main text.
""")
# ---------------- S1-S3
ORDER = ["Persistence", "Day-of-year climatology", "Gradient-boosted trees", "MLP", "LSTM", "GRU", "CNN-LSTM", "Dilated TCN", "Transformer",
         "ST-GCN (no G3D)", "MTGNN", "AM-MTGNN", "MTGNN (no graph)", "MT-HAB-G3D (proposed)", "MT-HAB-G3D (cls only)", "MT-HAB-G3D (reg only)"]
for i, (feats, title) in enumerate([("doy", "day-of-year channels"), ("rthist", "red-tide history channels"), ("doy-rthist", "day-of-year and red-tide history channels")], 1):
    p = f"outputs/stage2_{feats}/analysis/table2_reanchored.csv"
    if not os.path.exists(p): continue
    t = pd.read_csv(p).set_index("model"); c = pd.read_csv(f"outputs/stage2_{feats}/analysis/table5_counts.csv").iloc[0]
    on = pd.read_csv(f"outputs/stage2_{feats}/analysis/onset_detection.csv"); op = on[on.model == "MT-HAB-G3D (proposed)"].onset_POD.values
    w(r"\begin{table}[h]\centering\small")
    w(rf"\caption{{Full comparison matrix re-trained with {title} added to the input of every model (142 runs). Seed-paired comparisons favouring MT-HAB-G3D: {int(c.favour_p05)} of 96 at $p<0.05$ ({int(c.favour_BH)} after BH), {int(c.against_p05)} against. Onset detection of the proposed model at D+1/D+2/D+3: {op[0]:.2f}/{op[1]:.2f}/{op[2]:.2f} (final model: 0.00/0.00/0.00).}}")
    w(r"\begin{tabular}{lccccccc}\toprule Model & RMSE & Peak-RMSE & $R^2$ & F1 & AUPRC & MCC & TSS\\\midrule")
    for m in ORDER:
        if m not in t.index: continue
        r = t.loc[m]; lab = m.replace("MT-HAB-G3D (cls only)", r"\quad classification-only control").replace("MT-HAB-G3D (reg only)", r"\quad regression-only control")
        w(f"{lab} & {f(r.RMSE)} & {f(r.PeakRMSE)} & {f(r.R2)} & {f(r.F1_new)} & {f(r.AUPRC_new)} & {f(r.MCC_new)} & {f(r.TSS_new)}\\\\")
    w(r"\bottomrule\end{tabular}\end{table}")
# ---------------- S4 threshold rules
tr = pd.read_csv(f"{OUTD}/threshold_rules_summary.csv")
w(r"\begin{table}[h]\centering\small\caption{Sensitivity of the final model to the decision-threshold rule (means over seeds and lead times). f1max is the rule used in the main text (per-horizon F1 maximisation on the 2019 validation year); f2max maximises F2; pooled uses one threshold for all horizons; tssmax maximises TSS; rate\_$k$x sets the threshold so that the validation alert rate equals $k$ times the training prevalence. FP is the number of full-year false alarms per lead time.}")
w(r"\begin{tabular}{lcccccccc}\toprule Rule & thr. & F1 (EA) & POD (EA) & FAR (EA) & MCC (EA) & F1 (full) & FP (full) & FA/station-month\\\midrule")
for r in tr.itertuples():
    if r.rule == "manuscript(stored)": continue
    rule = r.rule.replace("_", "\\_")
    w(f"{rule} & {f(r.thr)} & {f(r.F1_ev)} & {f(r.POD_ev)} & {f(r.FAR_ev)} & {f(r.MCC_ev)} & {f(r.F1_full)} & {r.FP_full:.0f} & {f(r.FA_station_month)}\\\\")
w(r"\midrule oracle (test-selected, per lead time) & --- & 0.27--0.39 & --- & --- & --- & --- & --- & ---\\\bottomrule\end{tabular}\end{table}")
# ---------------- S5 onset detection
on = pd.read_csv(f"{OUTD}/onset_detection.csv"); ev = pd.read_csv(f"{OUTD}/event_level_detection.csv")
w(r"\begin{table}[h]\centering\small\caption{Event-level detection in the 2020 test year (medians over seeds, validation thresholds). Onset POD: share of station-level event onsets (30--32 evaluable per lead time) with an alert for a target day within two days before or on the first reported day. Event POD: share of events with an alert for any target day from two days before onset to the last reported day; FA episodes: contiguous alert runs not within two days of any reported day.}")
w(r"\begin{tabular}{lccccccc}\toprule Model & \multicolumn{7}{c}{Onset POD, D+1 \ldots D+7}\\\midrule")
for m in ["Persistence", "Day-of-year climatology", "Gradient-boosted trees", "MLP", "LSTM", "MTGNN (no graph)", "MT-HAB-G3D (proposed)", "MT-HAB-G3D (cls only)"]:
    g = on[on.model == m]
    if g.empty: continue
    w(f"{m} & " + " & ".join(f"{v:.2f}" for v in g.onset_POD) + r"\\")
w(r"\midrule & \multicolumn{7}{c}{Event POD / false-alarm episodes, D+1 \ldots D+7}\\\midrule")
for m in ["Day-of-year climatology", "Gradient-boosted trees", "MLP", "MT-HAB-G3D (proposed)", "MT-HAB-G3D (cls only)"]:
    g = ev[ev.model == m]
    if g.empty: continue
    w(f"{m} & " + " & ".join(f"{a:.2f}/{b:.0f}" for a, b in zip(g.event_POD, g.false_alarm_episodes)) + r"\\")
w(r"\bottomrule\end{tabular}\end{table}")
# ---------------- S6 per station
ps = pd.read_csv(f"{OUTD}/per_station_results.csv")
w(r"\begin{table}[h]\centering\small\caption{Per-station results of the final model in 2020 (mean over D+1--D+7, median over seeds). Stations without labelled test days are omitted; positives, TP and FP are event-adjacent station-days.}")
w(r"\begin{tabular}{lrrrrrrrr}\toprule Station & labelled days & Chl-$a$ days & RMSE & $R^2$ & TA$_{\pm1}$ & positives & TP & FP\\\midrule")
for r in ps.itertuples():
    if r.n_days_cls == 0: continue
    w(f"{EN.get(r.station, r.station)} & {r.n_days_cls} & {r.n_days_chl} & {f(r.RMSE)} & {f(r.R2)} & {f(r.TA1)} & {r.pos_station_days:.1f} & {r.TP:.1f} & {r.FP:.1f}\\\\")
w(r"\bottomrule\end{tabular}\end{table}")
# ---------------- S7-S8 refit on the training + validation years (run_refit.sh, analysis/14_refit_compare.py)
RF = "outputs/refit"
if os.path.exists(f"{RF}/refit_summary.csv"):
    def counts88(root):                            # Table-5 style counts over the eleven trained rows (88 comparisons)
        u = pd.read_csv(f"{root}/table5_pairs.csv"); u = u[~u.comparison.str.contains("spatial ext")]
        p = u.p.fillna(1).values; o = np.argsort(p); q = np.empty(len(p))
        q[o] = np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
        fav = u.favours_proposed.astype(bool).values
        return int(((p < 0.05) & fav).sum()), int(((q < 0.05) & fav).sum()), int(((p < 0.05) & ~fav).sum())
    S = pd.read_csv(f"{RF}/refit_summary.csv").set_index(["model", "metric"])
    def g(m, k, c): return S.loc[(m, k), c] if (m, k) in S.index else np.nan
    t = pd.read_csv(f"{RF}/refit/analysis/table2_reanchored.csv").set_index("model")
    c0, c1 = counts88(OUTD), counts88(f"{RF}/refit/analysis")
    on = pd.read_csv(f"{RF}/refit/analysis/onset_detection.csv"); op = on[on.model == "MT-HAB-G3D (proposed)"].onset_POD.values
    w(r"\begin{table}[h]\centering\small")
    w(rf"\caption{{Comparison matrix refitted on the training and validation years (2015--2019; 142 runs, 101 instead of 66 D+1 positive training station-days) and evaluated on 2020. Settings are those of the main text. Each model is trained for a fixed number of epochs, the median best validation epoch of its ten reference runs (Table~S8), and the last weights are used. Decision thresholds are carried over from the fixed-budget control trained on 2015--2018, so F1, MCC and TSS are not re-tuned; RMSE and AUPRC do not depend on a threshold. Persistence does not depend on the training period. Seed-paired comparisons with the eleven trained comparison models favouring MT-HAB-G3D: {c1[0]} of 88 at $p<0.05$ ({c1[1]} after BH), {c1[2]} against (reference protocol: {c0[0]}, {c0[1]} and {c0[2]}). Onset detection of the proposed model at D+1/D+2/D+3: {op[0]:.2f}/{op[1]:.2f}/{op[2]:.2f}.}}")
    w(r"\begin{tabular}{lcccccccc}\toprule Model & RMSE & Peak-RMSE & $R^2$ & F1 & AUPRC & AUPRC (full year) & MCC & TSS\\\midrule")
    for m in ORDER:
        if m not in t.index: continue
        r = t.loc[m]; lab = m.replace("MT-HAB-G3D (cls only)", r"\quad classification-only control").replace("MT-HAB-G3D (reg only)", r"\quad regression-only control")
        w(f"{lab} & {f(r.RMSE)} & {f(r.PeakRMSE)} & {f(r.R2)} & {f(r.F1_new)} & {f(r.AUPRC_new)} & {f(g(m, 'AUPRC_full', 'refit'))} & {f(r.MCC_new)} & {f(r.TSS_new)}\\\\")
    w(r"\bottomrule\end{tabular}\end{table}")
    P = "MT-HAB-G3D (proposed)"; KS = ["RMSE", "AUPRC_ea", "AUPRC_full", "MCC_ea"]
    pmin = min(g(P, k, "p_fixed_ref") for k in ["RMSE", "R2", "PeakRMSE", "AUPRC_ea", "AUPRC_full", "F1_ea", "MCC_ea", "TSS_ea"])
    w(r"\begin{table}[h]\centering\scriptsize\setlength{\tabcolsep}{3pt}")
    w(rf"\caption{{Effect of adding the validation year to the training data (2020 test year). Control: trained on 2015--2018 for the fixed epoch budget (Ep.) without checkpoint selection; refit: trained on 2015--2019 with the same budget. Medians over ten seeds; $p$: two-sided seed-paired Wilcoxon test of refit against control. AUPRC and MCC use the event-adjacent subset. Under the reference protocol of the main text (early stopping on 2019) the proposed model has RMSE {f(g(P, 'RMSE', 'ref'))}, AUPRC {f(g(P, 'AUPRC_ea', 'ref'))}, full-year AUPRC {f(g(P, 'AUPRC_full', 'ref'))} and MCC {f(g(P, 'MCC_ea', 'ref'))}; its control differs from the reference in no metric ($p\geq{pmin:.2f}$). Gradient-boosted trees have no epoch budget, and their control is identical to the reference row; with the added year the Chl-\emph{{a}} training sample exceeds 10,000, above which scikit-learn applies internal early stopping to the regressor as well, so that refit also differs from the control in this respect.}}")
    w(r"\begin{tabular}{lc ccc ccc ccc ccc}\toprule & & \multicolumn{3}{c}{RMSE} & \multicolumn{3}{c}{AUPRC} & \multicolumn{3}{c}{AUPRC (full year)} & \multicolumn{3}{c}{MCC}\\")
    w(r"Model & Ep. & control & refit & $p$ & control & refit & $p$ & control & refit & $p$ & control & refit & $p$\\\midrule")
    for m in ORDER[1:]:
        ep = g(m, "AUPRC_ea", "epochs"); ep = g(m, "RMSE", "epochs") if pd.isna(ep) else ep
        lab = m.replace("MT-HAB-G3D (cls only)", r"\quad classification-only control").replace("MT-HAB-G3D (reg only)", r"\quad regression-only control")
        w(f"{lab} & {'---' if not ep else int(ep)} & " + " & ".join(f"{f(g(m, k, 'fixed'))} & {f(g(m, k, 'refit'))} & {f(g(m, k, 'p_refit_fixed'))}" for k in KS) + r"\\")
    w(r"\bottomrule\end{tabular}\end{table}")
# ---------------- figures
FIGS = [("figA_obs_vs_pred_timeseries", "Observed and predicted Chl-$a$ (seed-median, D+1 and D+7) and D+1 red-tide probability at three stations in 2020; shaded bands are reported red-tide days."),
        ("figE_permutation_importance", "Permutation importance of the nine input nodes for the final model on the 2020 test year (loss in event-adjacent AUPRC and increase in Chl-$a$ RMSE when the node's inputs are permuted; mean over seeds)."),
        ("figF_learned_graph", "Correlation-based 2-NN input graph and the learned residual adjacency $|\\mathbf{A}^{\\mathrm{res}}|$ (mean over ten seeds)."),
        ("figG_station_embedding", "Principal components of the learned station embedding."),
        ("figH_multitask_vs_single", "Multi-task model versus single-task controls, per metric and lead time (medians over seeds).")]
for i, (fn, cap) in enumerate(FIGS, 1):
    src = f"{OUTD}/{fn}.pdf"
    if os.path.exists(src):
        shutil.copy(src, f"{SUPP}/Fig_S{i}.pdf")
        w(rf"\begin{{figure}}[h]\centering\includegraphics[width=0.95\textwidth]{{Fig_S{i}.pdf}}\caption{{{cap}}}\end{{figure}}")
w(r"\end{document}")
open(f"{SUPP}/supplement_v4.tex", "w").write("\n".join(L))
print("written", SUPP, os.listdir(SUPP))
