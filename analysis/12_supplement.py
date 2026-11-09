"""Supplementary material for the revised manuscript (tables S1-S6, figures S1-S5).
S1-S3  Table-2 equivalents of the full comparison matrix re-trained on the three augmented inputs
S4     threshold rules (validation F1-max vs alternatives, oracle) for the final model
S5     onset / event-level detection per model and lead time
S6     per-station results of the final model (2020)
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
