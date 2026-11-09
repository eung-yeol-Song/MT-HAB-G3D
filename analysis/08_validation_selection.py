"""Stage-1 model selection on the 2019 VALIDATION split only (revision v13).

Every candidate configuration of the proposed model is scored with the same
quantity train.py uses for checkpoint selection:

    S = 0.5 * mean_h AUPRC_h(val, full timeline) + 0.5 * mean_h max(R2_h(val), -1)

The winner is the configuration whose seed-paired S improves on the reference
(`full`) with Wilcoxon p < 0.05; ties keep the reference.  Test-set numbers are
printed alongside for the record but play no part in the decision.

    MTHAB_OUT=outputs python analysis/08_validation_selection.py [--candidates a,b,c]
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from scipy.stats import wilcoxon

ap = argparse.ArgumentParser()
ap.add_argument("--candidates", default="plus_doy,plus_rthist,plus_doy_rthist,residual_head,"
                                        "lam_0.2,lam_2.0,wbce,window_5,no_station_emb,no_log_target")
ap.add_argument("--seeds", type=int, default=10)
a = ap.parse_args()

REF_DIR, REF_FMT = FINAL_DIR, FINAL_FMT


def split_scores(base, fmt, split):
    """per-seed (S, AUPRC_full, AUPRC_ea, R2, RMSE) on one split."""
    rows = []
    for s in range(a.seeds):
        f = f"{base}/{fmt.format(s)}/preds_{split}.npz"
        if not os.path.exists(f):
            continue
        z = np.load(f); d = {k: z[k] for k in z.files}
        ap_full, ap_ea, r2s, rm = [], [], [], []
        for h in range(H):
            m = d["m_cls"][:, h]
            ap_full.append(auprc(d["y_cls"][m, h], d["prob"][m, h]))
            me = m & d["event_adj_h"][:, h]
            ap_ea.append(auprc(d["y_cls"][me, h], d["prob"][me, h]))
            mr = d["m_reg"][:, h]
            r = reg_metrics(d["y_reg_raw"][mr, h], to_ugl(d["reg_raw"][mr, h]))
            r2s.append(max(r["R2"], -1.0)); rm.append(r["RMSE"])
        rows.append(dict(seed=s, S=0.5 * np.nanmean(ap_full) + 0.5 * np.mean(r2s),
                         AUPRC_full=np.nanmean(ap_full), AUPRC_ea=np.nanmean(ap_ea),
                         R2=np.mean(r2s), RMSE=np.mean(rm)))
    return pd.DataFrame(rows).set_index("seed")


ref_v = split_scores(REF_DIR, REF_FMT, "val")
ref_t = split_scores(REF_DIR, REF_FMT, "test")
out = []
print(f"reference `full`: VAL S={ref_v.S.median():.4f}  AUPRC_full={ref_v.AUPRC_full.median():.3f} "
      f"R2={ref_v.R2.median():.3f} RMSE={ref_v.RMSE.median():.3f}   (n_seeds={len(ref_v)})")
for key in a.candidates.split(","):
    cv = split_scores(ABL, key + "_s{}", "val")
    if cv.empty:
        print(f"  {key:18s} (no runs yet)"); continue
    ct = split_scores(ABL, key + "_s{}", "test")
    common = cv.index.intersection(ref_v.index)
    dS = (cv.loc[common, "S"] - ref_v.loc[common, "S"])
    p = wilcoxon(cv.loc[common, "S"], ref_v.loc[common, "S"]).pvalue if len(common) > 1 and not np.allclose(dS, 0) else 1.0
    pa = wilcoxon(cv.loc[common, "AUPRC_full"], ref_v.loc[common, "AUPRC_full"]).pvalue if len(common) > 1 else 1.0
    pr = wilcoxon(cv.loc[common, "RMSE"], ref_v.loc[common, "RMSE"]).pvalue if len(common) > 1 else 1.0
    verdict = "SELECT over reference" if (dS.mean() > 0 and p < 0.05) else ("worse (p<.05)" if (dS.mean() < 0 and p < 0.05) else "no significant change")
    out.append(dict(config=key, n_seeds=len(common), val_S_median=cv.S.median(), dS_mean=dS.mean(), p_S=p,
                    val_AUPRC_full=cv.AUPRC_full.median(), p_AUPRC=pa, val_AUPRC_ea=cv.AUPRC_ea.median(),
                    val_R2=cv.R2.median(), val_RMSE=cv.RMSE.median(), p_RMSE=pr, verdict=verdict,
                    test_AUPRC_ea=ct.AUPRC_ea.median() if not ct.empty else np.nan,
                    test_RMSE=ct.RMSE.median() if not ct.empty else np.nan))
    print(f"  {key:18s} VAL S={cv.S.median():.4f} dS={dS.mean():+.4f} p={p:.3f} | AUPRC_full {cv.AUPRC_full.median():.3f} (p={pa:.3f}) "
          f"R2 {cv.R2.median():.3f} RMSE {cv.RMSE.median():.3f} (p={pr:.3f})  -> {verdict}")
df = pd.DataFrame(out)
df.to_csv(f"{OUT}/validation_selection.csv", index=False)
print(f"\n-> {OUT}/validation_selection.csv   (decision uses VAL columns only; test columns are for the record)")
