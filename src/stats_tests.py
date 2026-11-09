"""
stats_tests.py
==============
Verification layer.  The manuscript reports single numbers; a reviewer will ask
whether the differences are real.  This module provides the standard tests.

    bootstrap_ci            percentile bootstrap CI for any metric
    paired_bootstrap        paired bootstrap test for a metric difference
    delong_test             DeLong test for two correlated AUROCs
    mcnemar_test            exact / chi2 McNemar test for two classifiers
    diebold_mariano         DM test for equal squared-error forecast accuracy
    wilcoxon_seeds          Wilcoxon signed-rank across seeds
    holm                    Holm-Bonferroni multiple-comparison correction

All functions work on plain numpy arrays so they can be applied to the
`preds_test.npz` dumps written by train.py.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .metrics import classification_metrics, regression_metrics, auroc


# --------------------------------------------------------------------------- #
def bootstrap_ci(fn, *arrays, n_boot=2000, alpha=0.05, seed=0, block=None):
    """Percentile bootstrap CI.  `block` enables a moving-block bootstrap for
    autocorrelated daily series (recommended for time-series metrics)."""
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    stat = []
    for _ in range(n_boot):
        if block:
            starts = rng.integers(0, max(n - block, 1), size=int(np.ceil(n / block)))
            idx = np.concatenate([np.arange(s, min(s + block, n)) for s in starts])[:n]
        else:
            idx = rng.integers(0, n, n)
        try:
            stat.append(fn(*[a[idx] for a in arrays]))
        except Exception:
            continue
    stat = np.asarray(stat, float)
    stat = stat[np.isfinite(stat)]
    return (float(np.mean(stat)),
            float(np.quantile(stat, alpha / 2)),
            float(np.quantile(stat, 1 - alpha / 2)))


def paired_bootstrap(metric_fn, y, pred_a, pred_b, n_boot=2000, seed=0, block=None):
    """Two-sided paired bootstrap test of  metric(A) - metric(B)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    obs = metric_fn(y, pred_a) - metric_fn(y, pred_b)
    diffs = []
    for _ in range(n_boot):
        if block:
            starts = rng.integers(0, max(n - block, 1), size=int(np.ceil(n / block)))
            idx = np.concatenate([np.arange(s, min(s + block, n)) for s in starts])[:n]
        else:
            idx = rng.integers(0, n, n)
        try:
            diffs.append(metric_fn(y[idx], pred_a[idx]) - metric_fn(y[idx], pred_b[idx]))
        except Exception:
            continue
    d = np.asarray(diffs, float)
    d = d[np.isfinite(d)]
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return {"delta": float(obs), "ci_lo": float(np.quantile(d, 0.025)),
            "ci_hi": float(np.quantile(d, 0.975)), "p_value": float(min(p, 1.0))}


# --------------------------------------------------------------------------- #
def _midrank(x):
    J = np.argsort(x)
    z = x[J]
    N = len(x)
    t = np.zeros(N, float)
    i = 0
    while i < N:
        j = i
        while j < N - 1 and z[j + 1] == z[i]:
            j += 1
        t[i:j + 1] = 0.5 * (i + j) + 1
        i = j + 1
    out = np.empty(N, float)
    out[J] = t
    return out


def delong_test(y, p1, p2):
    """
    DeLong test for two correlated ROC curves
    (DeLong et al. 1988; fast implementation of Sun & Xu 2014).

    The score matrix is re-ordered as [positives | negatives] before the
    mid-rank computations, which is what the structural-component formulation
    requires.
    """
    y = np.asarray(y).astype(int)
    pos = y == 1
    neg = ~pos
    m, n = int(pos.sum()), int(neg.sum())
    if m == 0 or n == 0:
        return {"auc1": np.nan, "auc2": np.nan, "z": np.nan, "p_value": np.nan}

    preds = np.vstack([np.asarray(p1, float), np.asarray(p2, float)])
    ordered = np.hstack([preds[:, pos], preds[:, neg]])          # (k, m + n)
    k = ordered.shape[0]

    tx = np.array([_midrank(ordered[r, :m]) for r in range(k)])
    ty = np.array([_midrank(ordered[r, m:]) for r in range(k)])
    tz = np.array([_midrank(ordered[r]) for r in range(k)])

    aucs = (tz[:, :m].sum(1) - m * (m + 1) / 2.0) / (m * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    s01 = np.cov(v01, ddof=1)
    s10 = np.cov(v10, ddof=1)
    S = np.atleast_2d(s01) / m + np.atleast_2d(s10) / n
    L = np.array([[1.0, -1.0]])
    var = float(L @ S @ L.T)
    if var <= 0:
        return {"auc1": float(aucs[0]), "auc2": float(aucs[1]), "z": np.nan,
                "p_value": 1.0}
    z = float((aucs[0] - aucs[1]) / np.sqrt(var))
    return {"auc1": float(aucs[0]), "auc2": float(aucs[1]), "z": z,
            "p_value": float(2 * stats.norm.sf(abs(z)))}


def mcnemar_test(y, yhat_a, yhat_b, exact=True):
    """Compare two hard classifiers on the same samples."""
    a = (np.asarray(yhat_a) == np.asarray(y))
    b = (np.asarray(yhat_b) == np.asarray(y))
    n01 = int((~a & b).sum())          # A wrong, B right
    n10 = int((a & ~b).sum())          # A right, B wrong
    if n01 + n10 == 0:
        return {"n01": n01, "n10": n10, "p_value": 1.0}
    if exact and n01 + n10 < 25:
        p = float(2 * stats.binom.cdf(min(n01, n10), n01 + n10, 0.5))
    else:
        chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
        p = float(stats.chi2.sf(chi2, 1))
    return {"n01": n01, "n10": n10, "p_value": float(min(p, 1.0))}


def diebold_mariano(y, pred_a, pred_b, h=1, power=2):
    """
    Diebold-Mariano test of equal predictive accuracy with the
    Harvey-Leybourne-Newbold small-sample correction.
    Negative statistic -> model A has smaller loss.
    """
    y, a, b = map(lambda v: np.asarray(v, float), (y, pred_a, pred_b))
    d = np.abs(y - a) ** power - np.abs(y - b) ** power
    T = len(d)
    dbar = d.mean()
    gam = [np.mean((d[k:] - dbar) * (d[:T - k] - dbar)) for k in range(h)]
    var = (gam[0] + 2 * sum(gam[1:])) / T
    if var <= 0:
        return {"dm": np.nan, "p_value": np.nan, "mean_loss_diff": float(dbar)}
    dm = dbar / np.sqrt(var)
    corr = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm *= corr
    return {"dm": float(dm), "p_value": float(2 * stats.t.sf(abs(dm), T - 1)),
            "mean_loss_diff": float(dbar)}


def wilcoxon_seeds(a, b):
    """Paired comparison of per-seed scores."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return {"p_value": np.nan, "note": "n_seeds < 3, report mean +- sd only",
                "mean_diff": float(np.mean(a - b))}
    s, p = stats.wilcoxon(a, b)
    return {"stat": float(s), "p_value": float(p), "mean_diff": float(np.mean(a - b))}


def holm(pvals: dict) -> dict:
    """Holm-Bonferroni step-down correction over a dict {name: p}."""
    items = sorted(pvals.items(), key=lambda kv: (np.inf if np.isnan(kv[1]) else kv[1]))
    m = len(items)
    out, prev = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(prev, (m - i) * (p if np.isfinite(p) else 1.0)))
        out[k] = adj
        prev = adj
    return out


# --------------------------------------------------------------------------- #
def compare_runs(npz_a: str, npz_b: str, horizon=0, event_adjacent=True, seed=0):
    """
    Full head-to-head verification between two saved runs
    (classification on the event-adjacent subset, regression on the full year).
    """
    A, B = np.load(npz_a), np.load(npz_b)
    assert np.array_equal(A["date"], B["date"]), "runs must share the evaluation set"

    mc = A["m_cls"][:, horizon].astype(bool)
    if event_adjacent:
        mc &= A["event_adj"].astype(bool)
    y = A["y_cls"][mc, horizon].astype(int)
    pa, pb = A["prob"][mc, horizon], B["prob"][mc, horizon]
    ta, tb = float(A["thr"][horizon]), float(B["thr"][horizon])

    mr = A["m_reg"][:, horizon].astype(bool)
    yr = A["y_reg_raw"][mr, horizon]
    ra, rb = A["reg_raw"][mr, horizon], B["reg_raw"][mr, horizon]

    f1 = lambda yy, pp: classification_metrics(yy, pp, ta)["F1"]
    rmse = lambda yy, pp: regression_metrics(yy, pp)["RMSE"]

    return {
        "classification": {
            "delong_AUROC": delong_test(y, pa, pb),
            "mcnemar": mcnemar_test(y, (pa >= ta).astype(int), (pb >= tb).astype(int)),
            "paired_bootstrap_F1": paired_bootstrap(f1, y, pa, pb, seed=seed),
            "AUPRC_A": classification_metrics(y, pa, ta)["AUPRC"],
            "AUPRC_B": classification_metrics(y, pb, tb)["AUPRC"],
        },
        "regression": {
            "diebold_mariano_RMSE": diebold_mariano(yr, ra, rb),
            "paired_bootstrap_RMSE": paired_bootstrap(rmse, yr, ra, rb, seed=seed, block=7),
            "RMSE_A": rmse(yr, ra), "RMSE_B": rmse(yr, rb),
        },
        "n_cls": int(mc.sum()), "n_reg": int(mr.sum()),
    }


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="preds_test.npz of model A")
    ap.add_argument("--b", required=True, help="preds_test.npz of model B")
    ap.add_argument("--horizon", type=int, default=0)
    ap.add_argument("--full_year", action="store_true")
    a = ap.parse_args()
    print(json.dumps(compare_runs(a.a, a.b, a.horizon, not a.full_year), indent=1))
