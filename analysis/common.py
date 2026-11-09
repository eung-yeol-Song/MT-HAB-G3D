"""Shared loaders + metrics for the ECOINF revision analysis (re-implemented
from the authors' src/metrics.py so results are reproducible without torch)."""
from __future__ import annotations
import json, os
import numpy as np, pandas as pd

# Paths can be overridden so the same scripts run on the authors' machine:
#   MTHAB_OUT=outputs MTHAB_FINAL_DIR=outputs/run_standard/runs MTHAB_FINAL_FMT=A_mthab_mse_s{} \
#   MTHAB_BUNDLE=outputs/bundle_main.npz python 01_reanchor_eventadj.py
U = os.environ.get("MTHAB_OUT", "outputs")
RUNS = f"{U}/run_standard/runs"
ABL = f"{U}/ablation/runs"
FINAL_DIR = os.environ.get("MTHAB_FINAL_DIR", ABL)
FINAL_FMT = os.environ.get("MTHAB_FINAL_FMT", "no_station_graph_s{}")
BUNDLE_9 = os.environ.get("MTHAB_BUNDLE", f"{U}/bundle_ab_k2_s0_l1_T10.npz")
OUT = os.environ.get("MTHAB_ANALYSIS_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
os.makedirs(OUT, exist_ok=True)

NODE_VARS = ["ec", "temp", "ph", "do", "turbidity", "salinity", "chl", "cod", "redtide"]
CHL_MAX = 500.0
H = 7

# station names as printed in manuscript Fig. 1 (figures must not depend on a CJK font)
EN = {"인천송도": "Incheon-Songdo", "시화조력": "Sihwa-tidal", "시화반월": "Sihwa-Banwol", "금강하구": "Geum estuary",
      "새만금": "Saemangeum", "영산영암": "Yeongsan-Yeongam", "영산목포": "Yeongsan-Mokpo",
      "광양망덕": "Gwangyang-Mangdeok", "광양초남": "Gwangyang-Chonam", "광양적량": "Gwangyang-Jeokryang",
      "마산양덕": "Masan-Yangdeok", "마산봉암": "Masan-Bongam", "마산삼귀": "Masan-Samgwi", "울산온산": "Ulsan-Onsan",
      "부산수영": "Busan-Suyeong", "낙동을숙": "Nakdong-Eulsuk", "낙동명지": "Nakdong-Myeongji"}


def station_months(p, h):
    """Number of distinct (station, calendar month) pairs that have at least one
    scorable red-tide label at horizon h - the denominator of 'false alarms per
    station-month'.  The manuscript divided by a fixed constant; in 2020 only 15
    stations report and several have gaps, so the fixed constant overstates the
    monitored exposure and understates the false-alarm rate."""
    m = p["m_cls"][:, h]
    tgt = (p["date"] + np.timedelta64(h, "D")).astype("datetime64[M]")
    return len(set(zip(p["station"][m].tolist(), tgt[m].astype(int).tolist())))

# Table-2 row label -> (run directory, tag format)
MODELS = {
    "Persistence":             (RUNS, "A_persistence_s{}", 1),
    "Day-of-year climatology": (RUNS, "A_climatology_s{}", 1),
    "Gradient-boosted trees":  (RUNS, "A_histgb_s{}", 10),
    "MLP":                     (RUNS, "A_mlp_s{}", 10),
    "LSTM":                    (RUNS, "A_lstm_s{}", 10),
    "GRU":                     (RUNS, "A_gru_s{}", 10),
    "CNN-LSTM":                (RUNS, "A_cnn_lstm_s{}", 10),
    "Dilated TCN":             (RUNS, "A_tcn_s{}", 10),
    "Transformer":             (RUNS, "A_transformer_s{}", 10),
    "ST-GCN (no G3D)":         (RUNS, "A_stgcn_s{}", 10),
    "MTGNN":                   (RUNS, "G_mtgnn_s{}", 10),
    "AM-MTGNN":                (RUNS, "G_am_mtgnn_s{}", 10),
    "MTGNN (no graph)":        (RUNS, "G_mtgnn_nograph_s{}", 10),
    "MT-HAB-G3D (proposed)":   (FINAL_DIR, FINAL_FMT, 10),
    # revision rows (R2-M2 single-task controls; spatial-extension reference from the ablation)
    "MT-HAB-G3D (cls only)":   (RUNS, "A_mthab_cls_only_s{}", 10),
    "MT-HAB-G3D (reg only)":   (RUNS, "A_mthab_reg_only_s{}", 10),
    "MT-HAB-G3D + spatial ext.": (ABL, "plus_station_graph_s{}", 10),
}
# rows whose regression (cls-only) or classification (reg-only) head received no training signal
DEAD_HEAD = {"MT-HAB-G3D (cls only)": "reg", "MT-HAB-G3D (reg only)": "cls"}


def to_ugl(reg_log):
    """preds_test.npz stores de-standardised log1p(Chl-a); invert as train.inv_chl."""
    v = np.clip(reg_log, 0.0, np.log1p(CHL_MAX))
    return np.expm1(v)


def load_pred(base, tag):
    z = np.load(f"{base}/{tag}/preds_test.npz")
    d = {k: z[k] for k in z.files}
    d["reg_ugl"] = to_ugl(d["reg_raw"])
    d["date"] = d["date"].astype("datetime64[D]")
    return d


def load_all(models=None, verbose=True):
    """Load every available run; rows whose directories do not exist yet are skipped with a note,
    so the scripts also work on a partially finished matrix."""
    out = {}
    for name, (base, fmt, n) in MODELS.items():
        if models and name not in models:
            continue
        runs = []
        for s in range(n):
            f = f"{base}/{fmt.format(s)}/preds_test.npz"
            if os.path.exists(f):
                runs.append(load_pred(base, fmt.format(s)))
        if runs:
            out[name] = runs
            if verbose and len(runs) < n:
                print(f"[note] {name}: {len(runs)}/{n} seeds available")
        elif verbose:
            print(f"[note] {name}: no runs found under {base}/{fmt} - skipped")
    # drop rows evaluated on a different sample set than the final model (e.g. T=10 ablation rows
    # when MTHAB_OUT points at a T=5 matrix); every analysis script indexes them with one mask
    ref = out.get("MT-HAB-G3D (proposed)")
    if ref:
        n_ref = len(ref[0]["m_cls"])
        for name in [k for k, v in out.items() if len(v[0]["m_cls"]) != n_ref]:
            if verbose:
                print(f"[note] {name}: {len(out[name][0]['m_cls'])} test samples vs {n_ref} for the final model - skipped")
            del out[name]
    return out


def load_bundle(path):
    z = np.load(path, allow_pickle=True)
    b = {k: z[k] for k in z.files}
    b["date"] = b["date"].astype("datetime64[D]")
    b["cfg"] = json.loads(str(z["cfg"]))
    b["scaler"] = json.loads(str(z["scaler"]))
    b["stations"] = list(z["stations"])
    return b


def load_daily():
    df = pd.read_csv(f"{U}/daily_station_day.csv", parse_dates=["date"])
    return df


# ----------------------------------------------------------------- metrics
def auprc(y, p):
    y = np.asarray(y).astype(int); p = np.asarray(p, float)
    if y.sum() == 0:
        return np.nan
    o = np.argsort(-p, kind="mergesort"); y, ps = y[o], p[o]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    last = np.r_[ps[1:] != ps[:-1], True]
    tp, fp = tp[last], fp[last]
    prec = tp / np.maximum(tp + fp, 1e-12); rec = tp / y.sum()
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def auroc(y, p):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    return float(roc_auc_score(y, p))


def cls_metrics(y, p, thr):
    y = np.asarray(y).astype(int); p = np.asarray(p, float)
    yh = (p >= thr).astype(int)
    tp = int(((y == 1) & (yh == 1)).sum()); fp = int(((y == 0) & (yh == 1)).sum())
    fn = int(((y == 1) & (yh == 0)).sum()); tn = int(((y == 0) & (yh == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / den if den > 0 else 0.0
    return dict(F1=f1, precision=prec, recall=rec, POD=rec, FAR=(fp / (tp + fp) if tp + fp else np.nan),
                CSI=(tp / (tp + fp + fn) if tp + fp + fn else np.nan),
                specificity=spec, MCC=mcc, TSS=rec + spec - 1.0,
                AUPRC=auprc(y, p), TP=tp, FP=fp, FN=fn, TN=tn, n=len(y), n_pos=int(y.sum()),
                prevalence=float(y.mean()) if len(y) else np.nan)


def reg_metrics(y, yh, tol=1.0, peak_q=0.9):
    y = np.asarray(y, float); yh = np.asarray(yh, float)
    e = yh - y
    rmse = float(np.sqrt(np.mean(e ** 2)))
    var = float(np.var(y))
    r2 = float(1 - np.mean(e ** 2) / var) if var > 0 else np.nan
    thr = np.quantile(y, peak_q); pk = y >= thr
    return dict(RMSE=rmse, MAE=float(np.mean(np.abs(e))), R2=r2,
                PeakRMSE=float(np.sqrt(np.mean(e[pk] ** 2))) if pk.sum() else np.nan,
                TA1=float(np.mean(np.abs(e) <= tol)), bias=float(e.mean()), n=len(y))


def episodes(mask):
    ep, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            ep.append((start, i - 1)); start = None
    if start is not None:
        ep.append((start, len(mask) - 1))
    return ep
