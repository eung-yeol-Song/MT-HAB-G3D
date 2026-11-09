"""
metrics.py
==========
Every number that appears in the extended result tables.

Classification (red tide)
    F1, precision, recall, accuracy, balanced accuracy,
    MCC, TSS (= Youden's J, standard in ecological forecasting),
    AUROC, AUPRC (average precision), Brier score, ECE,
    event-wise POD / FAR / CSI  and  onset lead time

Regression (Chl-a)
    RMSE, MAE, R2, Peak-RMSE (top-10 % days), TA+-1 ug/L tolerance accuracy,
    NSE, KGE, bias, and the log-space RMSE (robust to the skewed Chl-a scale)

Uncertainty (for the quantile / MC-dropout variants)
    PICP, MPIW, pinball loss, CRPS (quantile approximation)
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# classification
# --------------------------------------------------------------------------- #
def confusion(y, yhat):
    y, yhat = np.asarray(y).astype(int), np.asarray(yhat).astype(int)
    tp = int(((y == 1) & (yhat == 1)).sum())
    fp = int(((y == 0) & (yhat == 1)).sum())
    fn = int(((y == 1) & (yhat == 0)).sum())
    tn = int(((y == 0) & (yhat == 0)).sum())
    return tp, fp, fn, tn


def auroc(y, p):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), float)
    sp = p[order]
    i = 0
    r = np.arange(1, len(p) + 1, dtype=float)
    while i < len(p):                                   # average ranks for ties
        j = i
        while j + 1 < len(p) and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i:j + 1]] = r[i:j + 1].mean()
        i = j + 1
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def auprc(y, p):
    """
    Average precision with correct handling of tied scores: precision/recall are
    evaluated once per *distinct* threshold, as in sklearn.average_precision_score.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    if y.sum() == 0:
        return float("nan")
    o = np.argsort(-p, kind="mergesort")
    y, ps = y[o], p[o]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    last = np.r_[ps[1:] != ps[:-1], True]          # last index of each tie group
    tp, fp = tp[last], fp[last]
    prec = tp / np.maximum(tp + fp, 1e-12)
    rec = tp / y.sum()
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def ece(y, p, n_bins=10):
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0, 1, n_bins + 1)
    e, n = 0.0, len(y)
    for i in range(n_bins):
        m = (p > edges[i]) & (p <= edges[i + 1]) if i else (p >= 0) & (p <= edges[1])
        if m.sum() == 0:
            continue
        e += m.sum() / n * abs(y[m].mean() - p[m].mean())
    return float(e)


def classification_metrics(y, p, thr=0.5) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    yhat = (p >= thr).astype(int)
    tp, fp, fn, tn = confusion(y, yhat)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / den if den > 0 else 0.0
    return {
        "F1": f1, "precision": prec, "recall": rec, "specificity": spec,
        "accuracy": (tp + tn) / max(len(y), 1),
        "balanced_accuracy": 0.5 * (rec + spec),
        "MCC": mcc, "TSS": rec + spec - 1.0,
        "AUROC": auroc(y, p), "AUPRC": auprc(y, p),
        "Brier": float(np.mean((p - y) ** 2)), "ECE": ece(y, p),
        "n": int(len(y)), "n_pos": int(y.sum()), "threshold": float(thr),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }


def best_threshold(y, p, metric="F1", grid=None):
    """Validation-selected decision threshold (Eq. 16)."""
    grid = grid if grid is not None else np.unique(np.round(np.linspace(0.01, 0.99, 99), 3))
    best, bt = -1.0, 0.5
    for t in grid:
        m = classification_metrics(y, p, t)[metric]
        if m > best:
            best, bt = m, float(t)
    return bt, best


# --------------------------------------------------------------------------- #
# event-wise skill (contiguous bloom episodes rather than single days)
# --------------------------------------------------------------------------- #
def _episodes(mask: np.ndarray):
    ep, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            ep.append((start, i - 1))
            start = None
    if start is not None:
        ep.append((start, len(mask) - 1))
    return ep


def event_metrics(y, yhat, group, tol=1):
    """
    POD / FAR / CSI at the level of contiguous events, allowing a +-`tol` day
    matching window, computed per station (`group`) and pooled.
    Also returns the mean onset lead time (days) for detected events.
    """
    hits = misses = fas = 0
    leads = []
    for g in np.unique(group):
        m = group == g
        yt, yp = np.asarray(y)[m].astype(bool), np.asarray(yhat)[m].astype(bool)
        for (a, b) in _episodes(yt):
            lo, hi = max(0, a - tol), min(len(yp) - 1, b + tol)
            if yp[lo:hi + 1].any():
                hits += 1
                first = lo + int(np.argmax(yp[lo:hi + 1]))
                leads.append(a - first)                     # >0 -> warned early
            else:
                misses += 1
        for (a, b) in _episodes(yp):
            lo, hi = max(0, a - tol), min(len(yt) - 1, b + tol)
            if not yt[lo:hi + 1].any():
                fas += 1
    pod = hits / (hits + misses) if hits + misses else float("nan")
    far = fas / (hits + fas) if hits + fas else float("nan")
    csi = hits / (hits + misses + fas) if hits + misses + fas else float("nan")
    return {"event_POD": pod, "event_FAR": far, "event_CSI": csi,
            "event_hits": hits, "event_misses": misses, "event_false_alarms": fas,
            "mean_lead_days": float(np.mean(leads)) if leads else float("nan")}


# --------------------------------------------------------------------------- #
# regression
# --------------------------------------------------------------------------- #
def regression_metrics(y, yhat, tol=1.0, peak_q=0.9) -> dict:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    if len(y) == 0:
        return {}
    e = yhat - y
    rmse = float(np.sqrt(np.mean(e ** 2)))
    mae = float(np.mean(np.abs(e)))
    var = float(np.var(y))
    r2 = float(1 - np.mean(e ** 2) / var) if var > 0 else float("nan")
    thr = np.quantile(y, peak_q)
    pk = y >= thr
    peak_rmse = float(np.sqrt(np.mean(e[pk] ** 2))) if pk.sum() else float("nan")
    nse = r2                                              # identical definition
    r = np.corrcoef(y, yhat)[0, 1] if len(y) > 2 else np.nan
    beta = yhat.mean() / y.mean() if y.mean() != 0 else np.nan
    gamma = ((yhat.std() / yhat.mean()) / (y.std() / y.mean())
             if yhat.mean() != 0 and y.mean() != 0 and y.std() != 0 else np.nan)
    kge = float(1 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)) \
        if np.all(np.isfinite([r, beta, gamma])) else float("nan")
    logrmse = float(np.sqrt(np.mean((np.log1p(np.clip(yhat, 0, None))
                                     - np.log1p(np.clip(y, 0, None))) ** 2)))
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "PeakRMSE": peak_rmse,
            f"TA{tol:g}": float(np.mean(np.abs(e) <= tol)),
            "NSE": nse, "KGE": kge, "bias": float(e.mean()),
            "logRMSE": logrmse, "n": int(len(y)), "peak_thr": float(thr)}


# --------------------------------------------------------------------------- #
# uncertainty
# --------------------------------------------------------------------------- #
def interval_metrics(y, lo, hi) -> dict:
    y, lo, hi = np.asarray(y, float), np.asarray(lo, float), np.asarray(hi, float)
    cov = float(np.mean((y >= lo) & (y <= hi)))
    width = float(np.mean(hi - lo))
    return {"PICP": cov, "MPIW": width}


def pinball(y, q_pred, quantiles) -> float:
    y = np.asarray(y, float)[:, None]
    q = np.asarray(quantiles, float)[None, :]
    e = y - np.asarray(q_pred, float)
    return float(np.mean(np.maximum(q * e, (q - 1) * e)))


def crps_from_quantiles(y, q_pred, quantiles) -> float:
    """CRPS approximated by the average pinball loss x 2 (Gneiting & Raftery)."""
    return 2.0 * pinball(y, q_pred, quantiles)


# --------------------------------------------------------------------------- #
def summarize(y_cls, p_cls, y_reg, yhat_reg, station=None, thr=0.5,
              tol=1.0, q_pred=None, quantiles=None) -> dict:
    out = {}
    if y_cls is not None and len(y_cls):
        out.update(classification_metrics(y_cls, p_cls, thr))
        if station is not None:
            out.update(event_metrics(y_cls, (np.asarray(p_cls) >= thr).astype(int), station))
    if y_reg is not None and len(y_reg):
        out.update(regression_metrics(y_reg, yhat_reg, tol=tol))
    if q_pred is not None and quantiles is not None:
        out.update(interval_metrics(y_reg, q_pred[:, 0], q_pred[:, -1]))
        out["pinball"] = pinball(y_reg, q_pred, quantiles)
        out["CRPS"] = crps_from_quantiles(y_reg, q_pred, quantiles)
    return out
