"""T=5 versus T=10 protocol, model by model (seed-paired, same seed index in both protocols).

For every model with ten seeds in both matrices:
  * VALIDATION 2019, full split      -> S = 0.5*AUPRC_full + 0.5*R2 (the pre-registered selection score),
                                         AUPRC_ea, R2, RMSE
  * VALIDATION 2019, common dates    -> same metrics on the (station, date) pairs present in both protocols
  * TEST 2020, common dates          -> RMSE, R2, AUPRC_ea, F1_ea, MCC_ea
Differences are T5 - T10 per seed; two-sided Wilcoxon over seeds.
    MTHAB_T10=outputs MTHAB_T5=outputs/t5_full python analysis/13_t5_vs_t10.py
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from scipy.stats import wilcoxon

T10 = os.environ.get("MTHAB_T10", "outputs"); T5 = os.environ.get("MTHAB_T5", "outputs/t5_full")
OUTD = f"{T5}/analysis"; os.makedirs(OUTD, exist_ok=True)
FMT = {k: v[1] for k, v in MODELS.items() if k not in ("Persistence", "Day-of-year climatology", "MT-HAB-G3D + spatial ext.")}
FMT["MT-HAB-G3D (proposed)"] = "A_mthab_mse_s{}"


def load(root, fmt, s, split):
    f = f"{root}/run_standard/runs/{fmt.format(s)}/preds_{split}.npz"
    if not os.path.exists(f):
        return None
    z = np.load(f); d = {k: z[k] for k in z.files}
    d["reg_ugl"] = to_ugl(d["reg_raw"]); d["date"] = d["date"].astype("datetime64[D]")
    d["key"] = np.array([f"{a}|{b}" for a, b in zip(d["station"], d["date"])])
    if "thr" not in d:  # validation files carry no thresholds; take them from the test file
        zt = np.load(f.replace(f"preds_{split}", "preds_test")); d["thr"] = zt["thr"]
    return d


def metrics(d, idx):
    """mean over horizons on rows idx."""
    out = dict(S=0, AUPRC_full=[], AUPRC_ea=[], F1_ea=[], MCC_ea=[], R2=[], RMSE=[])
    for h in range(H):
        m = idx & d["m_cls"][:, h]
        out["AUPRC_full"].append(auprc(d["y_cls"][m, h], d["prob"][m, h]))
        me = m & d["event_adj_h"][:, h]
        c = cls_metrics(d["y_cls"][me, h], d["prob"][me, h], d["thr"][h])
        out["AUPRC_ea"].append(c["AUPRC"]); out["F1_ea"].append(c["F1"]); out["MCC_ea"].append(c["MCC"])
        mr = idx & d["m_reg"][:, h]
        r = reg_metrics(d["y_reg_raw"][mr, h], d["reg_ugl"][mr, h])
        out["R2"].append(max(r["R2"], -1.0)); out["RMSE"].append(r["RMSE"])
    o = {k: float(np.nanmean(v)) for k, v in out.items() if k != "S"}
    o["S"] = 0.5 * o["AUPRC_full"] + 0.5 * o["R2"]
    return o


rows = []
for name, fmt in FMT.items():
    for split in ("val", "test"):
        per_seed = []
        for s in range(10):
            a, b = load(T10, fmt, s, split), load(T5, fmt, s, split)
            if a is None or b is None:
                continue
            common = np.intersect1d(a["key"], b["key"])
            ia, ib = np.isin(a["key"], common), np.isin(b["key"], common)
            rec = dict(model=name, split=split, seed=s, n_T10=len(a["key"]), n_T5=len(b["key"]), n_common=len(common))
            for scope, (ma, mb) in {"full": (np.ones(len(a["key"]), bool), np.ones(len(b["key"]), bool)), "common": (ia, ib)}.items():
                A, B = metrics(a, ma), metrics(b, mb)
                for k in A:
                    rec[f"{scope}|{k}|T10"] = A[k]; rec[f"{scope}|{k}|T5"] = B[k]
            per_seed.append(rec)
        rows += per_seed
df = pd.DataFrame(rows); df.to_csv(f"{OUTD}/t5_vs_t10_per_seed.csv", index=False)

# ------------------------------------------------------------- summary: median per protocol, paired diff, p
summ = []
for (name, split), g in df.groupby(["model", "split"], sort=False):
    for scope in ("full", "common"):
        for k in ("S", "AUPRC_full", "AUPRC_ea", "F1_ea", "MCC_ea", "R2", "RMSE"):
            a, b = g[f"{scope}|{k}|T10"].values, g[f"{scope}|{k}|T5"].values
            d = b - a; p = wilcoxon(d).pvalue if (len(d) > 1 and not np.allclose(d, 0)) else 1.0
            summ.append(dict(model=name, split=split, scope=scope, metric=k, n_seeds=len(g), T10=np.median(a), T5=np.median(b),
                             delta=np.median(d), p=p, n_common=int(g.n_common.iloc[0]), n_T10=int(g.n_T10.iloc[0]), n_T5=int(g.n_T5.iloc[0])))
S = pd.DataFrame(summ); S.to_csv(f"{OUTD}/t5_vs_t10_summary.csv", index=False)

pd.set_option("display.width", 200)
def show(split, scope, mets):
    sub = S[(S.split == split) & (S.scope == scope) & S.metric.isin(mets)]
    piv = sub.pivot(index="model", columns="metric", values=["T10", "T5", "delta", "p"])
    print(f"\n=== {split.upper()} / {scope} rows (n_T10={sub.n_T10.iloc[0]}, n_T5={sub.n_T5.iloc[0]}, common={sub.n_common.iloc[0]}) : median over seeds; delta = T5 - T10; p = Wilcoxon over seeds ===")
    for m in mets:
        t = piv.xs(m, axis=1, level=1); t = t.loc[[k for k in FMT if k in t.index]].round(4)
        t.columns = [f"{m}_{c}" for c in t.columns]
        print(t.to_string())

show("val", "full", ["S", "AUPRC_ea", "R2", "RMSE"])
show("val", "common", ["S", "AUPRC_ea", "RMSE"])
show("test", "full", ["RMSE", "AUPRC_ea"])
show("test", "common", ["RMSE", "R2", "AUPRC_ea", "F1_ea", "MCC_ea"])

# ------------------------------------------------------------- selection verdict for the proposed model
g = S[(S.model == "MT-HAB-G3D (proposed)") & (S.split == "val") & (S.metric == "S")]
print("\n=== pre-registered rule on the proposed model (VAL S, adopt T=5 iff delta>0 and p<0.05) ===")
print(g[["scope", "T10", "T5", "delta", "p"]].to_string(index=False))
print("written", OUTD)
