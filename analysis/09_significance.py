"""Statistical support for the revised main table (revision v13).

Three layers, all on the target-anchored event-adjacent subset for
classification and the full-year 2020 timeline for regression:

  1. seed-paired Wilcoxon signed-rank tests, MT-HAB-G3D vs every trained row,
     8 metrics x 12 rows = 96 comparisons, with Benjamini-Hochberg correction
     over ALL 96 (manuscript Table 5 / Fig. 4);
  2. sample-level uncertainty of the headline numbers (reviewer 2: "sample-level
     confidence intervals"): moving-block bootstrap (block = 7 days within a
     station-ordered series) of the D+1..D+7 mean RMSE and AUPRC, computed per
     seed and summarised by the median seed;
  3. Diebold-Mariano tests of equal squared-error accuracy, MT-HAB-G3D vs the
     strongest regression rows (persistence, GBT, MLP), per horizon and seed,
     reported as the share of seeds with p < 0.05 and the median p.

Outputs: table5_pairs.csv / .tex, table5_counts.csv, sample_ci.csv, dm_tests.csv
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from scipy.stats import wilcoxon
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.stats_tests import diebold_mariano, bootstrap_ci

METS = ["RMSE", "PeakRMSE", "R2", "F1", "AUPRC", "MCC", "TSS", "acc"]
LOWER = {"RMSE", "PeakRMSE"}
preds = load_all()
ref = preds["MT-HAB-G3D (proposed)"][0]
ev_h = ref["event_adj_h"]


def seed_means(plist):
    rows = []
    for p in plist:
        hs = []
        for h in range(H):
            m = p["m_cls"][:, h] & ev_h[:, h]
            c = cls_metrics(p["y_cls"][m, h], p["prob"][m, h], p["thr"][h])
            mf = p["m_cls"][:, h]
            yhat = (p["prob"][mf, h] >= p["thr"][h]).astype(int)
            c["acc"] = float((yhat == p["y_cls"][mf, h].astype(int)).mean())
            mr = p["m_reg"][:, h]
            r = reg_metrics(p["y_reg_raw"][mr, h], p["reg_ugl"][mr, h])
            c.update({k: r[k] for k in ["RMSE", "PeakRMSE", "R2", "TA1"]})
            hs.append(c)
        rows.append(pd.DataFrame(hs).mean(numeric_only=True))
    return pd.DataFrame(rows)


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1); q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n); out[o] = np.minimum(q, 1.0); return out


# ---------------------------------------------------------------- 1. seed-paired Wilcoxon, BH over 96
M = {name: seed_means(pl) for name, pl in preds.items() if len(pl) == 10}
pm = M["MT-HAB-G3D (proposed)"]
comps = [n for n in M if n not in ("MT-HAB-G3D (proposed)", "MT-HAB-G3D (cls only)", "MT-HAB-G3D (reg only)")]
rows = []
for n in comps:
    for k in METS:
        d = pm[k].values - M[n][k].values
        p = 1.0 if np.allclose(d, 0) else wilcoxon(pm[k], M[n][k]).pvalue
        rows.append(dict(comparison=n, metric=k, delta_mean=d.mean(), p=p,
                         favours_proposed=bool((d.mean() < 0) if k in LOWER else (d.mean() > 0))))
t5 = pd.DataFrame(rows); t5["q_BH"] = bh(t5.p)
t5.to_csv(f"{OUT}/table5_pairs.csv", index=False)
sig = t5[t5.p < 0.05]; sigq = t5[t5.q_BH < 0.05]
counts = dict(n_comparisons=len(t5),
              favour_p05=int(sig.favours_proposed.sum()), against_p05=int((~sig.favours_proposed).sum()),
              favour_BH=int(sigq.favours_proposed.sum()), against_BH=int((~sigq.favours_proposed).sum()),
              not_significant=int((t5.p >= 0.05).sum()))
pd.DataFrame([counts]).to_csv(f"{OUT}/table5_counts.csv", index=False)
print("=== seed-paired Wilcoxon, MT-HAB-G3D vs trained rows (target-anchored EA) ===")
print(t5.pivot(index="comparison", columns="metric", values="p").round(3).loc[comps, METS].to_string())
print("\ncounts:", counts)
print("significant AGAINST the proposed model:")
print(t5[(t5.p < 0.05) & ~t5.favours_proposed].round(4).to_string(index=False))

# LaTeX for Table 5 (RMSE / F1 / AUPRC like the manuscript, plus BH flag)
def cell(r):
    star = r"$^{\ast}$" if r.q_BH < 0.05 else ""
    p = r"\textbf{%.3f}" % r.p if r.p < 0.05 else "%.3f" % r.p
    return f"{r.delta_mean:+.3f} & {p}{star}"
lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
         r"Comparison & $\Delta$RMSE & $p$ & $\Delta$F1 & $p$ & $\Delta$AUPRC & $p$ \\", r"\midrule"]
for n in comps:
    sub = t5[t5.comparison == n].set_index("metric")
    lines.append(f"{n} & " + " & ".join(cell(sub.loc[k]) for k in ["RMSE", "F1", "AUPRC"]) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}",
          r"% $\Delta$ = MT-HAB-G3D minus the comparison row (seed-paired mean over D+1--D+7); bold: raw $p<0.05$; $^{\ast}$: BH $q<0.05$ over all 96 model--metric comparisons."]
open(f"{OUT}/table5_pairs.tex", "w").write("\n".join(lines))

# ---------------------------------------------------------------- 2. sample-level block-bootstrap CIs
print("\n=== sample-level 95% CIs (moving-block bootstrap, block=7, median seed) ===")
def mean_rmse_fn(y, yh):
    return float(np.sqrt(np.mean((y - yh) ** 2)))
ci_rows = []
for name in ["MT-HAB-G3D (proposed)", "Persistence", "Gradient-boosted trees", "MLP", "LSTM", "Day-of-year climatology"]:
    if name not in preds:
        continue
    per_seed = []
    for p in preds[name]:
        # order by station then date so that blocks are contiguous in time
        o = np.lexsort((p["date"], p["station"]))
        rm, au = [], []
        for h in range(H):
            mr = p["m_reg"][:, h]; idx = o[mr[o]]
            rm.append(bootstrap_ci(mean_rmse_fn, p["y_reg_raw"][idx, h], p["reg_ugl"][idx, h], n_boot=500, block=7, seed=h))
            me = (p["m_cls"][:, h] & ev_h[:, h]); idx = o[me[o]]
            au.append(bootstrap_ci(lambda y, q: auprc(y, q), p["y_cls"][idx, h], p["prob"][idx, h], n_boot=500, block=7, seed=h))
        per_seed.append(dict(RMSE=np.mean([r[0] for r in rm]), RMSE_lo=np.mean([r[1] for r in rm]), RMSE_hi=np.mean([r[2] for r in rm]),
                             AUPRC=np.mean([a[0] for a in au]), AUPRC_lo=np.mean([a[1] for a in au]), AUPRC_hi=np.mean([a[2] for a in au])))
    d = pd.DataFrame(per_seed); med = d.iloc[(d.RMSE - d.RMSE.median()).abs().argsort().iloc[0]]
    ci_rows.append(dict(model=name, n_seeds=len(d), **med.to_dict()))
    print(f"  {name:26s} RMSE {med.RMSE:.3f} [{med.RMSE_lo:.3f}, {med.RMSE_hi:.3f}]   AUPRC_ea {med.AUPRC:.3f} [{med.AUPRC_lo:.3f}, {med.AUPRC_hi:.3f}]")
pd.DataFrame(ci_rows).to_csv(f"{OUT}/sample_ci.csv", index=False)

# ---------------------------------------------------------------- 3. Diebold-Mariano, per horizon and seed
print("\n=== Diebold-Mariano (squared error), MT-HAB-G3D vs regression rows: share of seeds with p<0.05 / median p ===")
dm_rows = []
for name in ["Persistence", "Gradient-boosted trees", "MLP", "LSTM"]:
    if name not in preds:
        continue
    for h in range(H):
        ps, signs = [], []
        for s, p in enumerate(preds["MT-HAB-G3D (proposed)"]):
            q = preds[name][min(s, len(preds[name]) - 1)]
            mr = p["m_reg"][:, h] & q["m_reg"][:, h]
            o = np.lexsort((p["date"][mr], p["station"][mr]))
            y = p["y_reg_raw"][mr, h][o]; a = p["reg_ugl"][mr, h][o]; b_ = q["reg_ugl"][mr, h][o]
            r = diebold_mariano(y, a, b_, h=h + 1)
            ps.append(r["p_value"]); signs.append(np.sign(r["mean_loss_diff"]))
        ps = np.array(ps); signs = np.array(signs)
        dm_rows.append(dict(comparison=name, horizon=f"D+{h+1}", share_p05=float(np.mean(ps < 0.05)),
                            share_p05_proposed_better=float(np.mean((ps < 0.05) & (signs < 0))),
                            share_p05_comparison_better=float(np.mean((ps < 0.05) & (signs > 0))),
                            median_p=float(np.median(ps))))
dm = pd.DataFrame(dm_rows); dm.to_csv(f"{OUT}/dm_tests.csv", index=False)
print(dm.pivot(index="comparison", columns="horizon", values="share_p05_proposed_better").round(2).to_string())
print("(share of seeds where the COMPARISON row is significantly better:)")
print(dm.pivot(index="comparison", columns="horizon", values="share_p05_comparison_better").round(2).to_string())
print("done ->", OUT)
