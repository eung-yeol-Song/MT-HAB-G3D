"""
reference.py
============
Non-learned / classical reference forecasters (persistence, day-of-year climatology,
histogram gradient boosting).  Pure numpy / scikit-learn: importable without torch,
so these rows can be produced on a CPU-only machine (revision pipeline).
"""
from __future__ import annotations

import numpy as np


def persistence(bundle, idx, horizons):
    """D+h prediction = value observed on day D (last input frame).

    Revision fix: the regression output must live in the SAME standardised space as the
    training target so that train.inv_chl() inverts it.  With `log_target` the target is
    standardised log1p(Chl-a) while X holds the per-station z-score of RAW Chl-a; the
    previous implementation returned the raw z-score and was therefore inverted wrongly.
    The red-tide probability is the label observed on day D (0/1)."""
    from ..data.dataset import CHL_IDX, RT_IDX
    X = bundle.X[idx]
    st = bundle.station[idx]
    z_raw = X[:, -1, CHL_IDX, 0]
    mu = np.array([bundle.scaler[s]["mu"]["chl"] for s in bundle.stations], np.float32)[st]
    sd = np.array([bundle.scaler[s]["sd"]["chl"] for s in bundle.stations], np.float32)[st]
    chl_raw = z_raw * sd + mu
    if bundle.cfg.get("log_target"):
        lmu = np.array([bundle.scaler[s]["log_mu"] for s in bundle.stations], np.float32)[st]
        lsd = np.array([bundle.scaler[s]["log_sd"] for s in bundle.stations], np.float32)[st]
        chl_last = (np.log1p(np.clip(chl_raw, 0, None)) - lmu) / lsd
    else:
        chl_last = z_raw
    rt_last = X[:, -1, RT_IDX, 0]
    H = len(horizons)
    return {"reg": np.repeat(chl_last[:, None].astype(np.float32), H, 1),
            "prob": np.repeat(rt_last[:, None].astype(np.float32), H, 1)}


def persistence_valid_anchor(bundle, idx):
    """True where the Chl-a anchor on day D was actually observed (mask channel = 1).
    Samples without an anchor fall back to the station mean and should be excluded
    from the persistence regression score (reported in the table footnote)."""
    from ..data.dataset import CHL_IDX
    X = bundle.X[idx]
    if X.shape[-1] > 1:
        return X[:, -1, CHL_IDX, 1] > 0.5
    return np.ones(len(idx), bool)

def climatology(bundle, idx_train, idx_eval, horizons):
    """Day-of-year climatology computed on the training split."""
    doy_tr = bundle.date[idx_train].astype("datetime64[D]").astype(object)
    doy_tr = np.array([d.timetuple().tm_yday for d in doy_tr])
    doy_ev = bundle.date[idx_eval].astype("datetime64[D]").astype(object)
    doy_ev = np.array([d.timetuple().tm_yday for d in doy_ev])

    reg = np.zeros((len(idx_eval), len(horizons)), np.float32)
    prob = np.zeros_like(reg)
    for hi, _h in enumerate(horizons):
        yr = bundle.y_reg[idx_train][:, hi]
        yc = bundle.y_cls[idx_train][:, hi]
        tab_r = np.zeros(367, np.float32) + yr.mean()
        tab_c = np.zeros(367, np.float32) + yc.mean()
        for d in range(1, 367):
            w = np.abs(((doy_tr - d + 182) % 365) - 182) <= 7    # +-7 day window
            if w.sum() > 5:
                tab_r[d] = yr[w].mean()
                tab_c[d] = yc[w].mean()
        reg[:, hi] = tab_r[doy_ev]
        prob[:, hi] = tab_c[doy_ev]
    return {"reg": reg, "prob": prob}


def histgb(bundle, idx_train, idx_eval, horizons, seed=0):
    """Histogram gradient boosting on flattened windows (one model per task)."""
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    Xtr = bundle.X[idx_train].reshape(len(idx_train), -1)
    Xev = bundle.X[idx_eval].reshape(len(idx_eval), -1)
    reg = np.zeros((len(idx_eval), len(horizons)), np.float32)
    prob = np.zeros_like(reg)
    for hi, _h in enumerate(horizons):
        mr = bundle.m_reg[idx_train][:, hi]
        r = HistGradientBoostingRegressor(random_state=seed, max_iter=300, learning_rate=0.05)
        r.fit(Xtr[mr], bundle.y_reg[idx_train][mr, hi])
        reg[:, hi] = r.predict(Xev)

        mc = bundle.m_cls[idx_train][:, hi]
        ytr = bundle.y_cls[idx_train][mc, hi]
        if ytr.sum() >= 5:
            c = HistGradientBoostingClassifier(random_state=seed, max_iter=300,
                                               learning_rate=0.05,
                                               class_weight="balanced")
            c.fit(Xtr[mc], ytr)
            prob[:, hi] = c.predict_proba(Xev)[:, 1]
        else:
            prob[:, hi] = ytr.mean() if len(ytr) else 0.0
    return {"reg": reg, "prob": prob}


