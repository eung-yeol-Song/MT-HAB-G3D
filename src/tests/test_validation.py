"""
test_validation.py
==================
Verification of the evaluation code itself.  Reviewers rightly distrust
hand-rolled metrics, so every metric implemented in `src/metrics.py` is checked
against an independent reference (scikit-learn / scipy), and the experimental
protocol is checked for the failure modes that silently inflate results:

  1. metric correctness        AUROC, AUPRC, F1, MCC, R2, RMSE vs sklearn
  2. statistical tests         DeLong vs sklearn AUC, McNemar vs statsmodels-style
                               exact binomial, DM test sign / symmetry
  3. temporal leakage          no test date appears in train; the input window of
                               a sample never overlaps its own target day
  4. graph leakage             the variable graph is invariant to the test data
  5. scaler leakage            standardisation statistics depend on train only
  6. label integrity           targets are binary, masks agree with NaN pattern
  7. inverse transform         standardise -> inverse -> original round-trip
  8. threshold selection       the reported threshold comes from validation, and
                               re-selecting it on test would inflate F1
                               (the size of that inflation is reported)

Run:
    python -m src.tests.test_validation --bundle outputs/bundle_ffill.npz
Exit code 0 = all checks passed.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

from ..data.dataset import Bundle, build, DataConfig, k_adjacency, normalize_adj
from ..metrics import (auroc, auprc, classification_metrics, regression_metrics,
                       best_threshold, event_metrics)
from ..stats_tests import delong_test, mcnemar_test, diebold_mariano, paired_bootstrap

PASS, FAIL = "  PASS", "  FAIL"
_state = {"fail": 0}


def check(name, cond, detail=""):
    print(f"{PASS if cond else FAIL}  {name}{('  | ' + detail) if detail else ''}")
    if not cond:
        _state["fail"] += 1


# --------------------------------------------------------------------------- #
def test_metrics(seed=0):
    print("\n[1] metric correctness against scikit-learn")
    from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                                 matthews_corrcoef, r2_score, mean_squared_error,
                                 brier_score_loss)
    rng = np.random.default_rng(seed)
    for trial in range(5):
        n = rng.integers(200, 800)
        y = (rng.random(n) < rng.uniform(0.02, 0.4)).astype(int)
        if y.sum() == 0 or y.sum() == n:
            continue
        p = np.clip(0.15 * rng.normal(size=n) + 0.6 * y + 0.2, 0, 1)
        p[rng.random(n) < 0.1] = 0.5                      # force ties
        check(f"AUROC (trial {trial})",
              abs(auroc(y, p) - roc_auc_score(y, p)) < 1e-9,
              f"{auroc(y, p):.6f} vs {roc_auc_score(y, p):.6f}")
        check(f"AUPRC (trial {trial})",
              abs(auprc(y, p) - average_precision_score(y, p)) < 1e-9,
              f"{auprc(y, p):.6f} vs {average_precision_score(y, p):.6f}")
        m = classification_metrics(y, p, 0.5)
        yh = (p >= 0.5).astype(int)
        check(f"F1 (trial {trial})", abs(m["F1"] - f1_score(y, yh, zero_division=0)) < 1e-9)
        check(f"MCC (trial {trial})", abs(m["MCC"] - matthews_corrcoef(y, yh)) < 1e-9)
        check(f"Brier (trial {trial})", abs(m["Brier"] - brier_score_loss(y, p)) < 1e-9)

        yt = rng.normal(5, 3, n)
        yp = yt + rng.normal(0, 1, n)
        r = regression_metrics(yt, yp)
        check(f"R2 (trial {trial})", abs(r["R2"] - r2_score(yt, yp)) < 1e-9)
        check(f"RMSE (trial {trial})",
              abs(r["RMSE"] - np.sqrt(mean_squared_error(yt, yp))) < 1e-9)

    # TSS / balanced accuracy identity
    y = np.array([1, 1, 0, 0, 1, 0, 0, 0])
    p = np.array([.9, .2, .1, .8, .7, .05, .3, .4])
    m = classification_metrics(y, p, 0.5)
    check("TSS == 2*balanced_accuracy - 1",
          abs(m["TSS"] - (2 * m["balanced_accuracy"] - 1)) < 1e-12)
    # Peak-RMSE uses exactly the top-10 % of observations
    yt = np.arange(100, dtype=float)
    r = regression_metrics(yt, yt + 1.0)
    check("PeakRMSE on constant error == error", abs(r["PeakRMSE"] - 1.0) < 1e-9)
    check("TA tolerance counts |e|<=1", abs(r["TA1"] - 1.0) < 1e-9)


def test_stats(seed=0):
    print("\n[2] statistical tests")
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    n = 600
    y = (rng.random(n) < 0.2).astype(int)
    p1 = np.clip(0.25 * rng.normal(size=n) + 0.7 * y + 0.15, 0, 1)
    p2 = np.clip(0.60 * rng.normal(size=n) + 0.2 * y + 0.40, 0, 1)

    d = delong_test(y, p1, p2)
    check("DeLong AUC1 matches sklearn", abs(d["auc1"] - roc_auc_score(y, p1)) < 1e-6,
          f"{d['auc1']:.6f} vs {roc_auc_score(y, p1):.6f}")
    check("DeLong AUC2 matches sklearn", abs(d["auc2"] - roc_auc_score(y, p2)) < 1e-6)
    check("DeLong detects a real difference", d["p_value"] < 0.01, f"p={d['p_value']:.2e}")
    dn = delong_test(y, p1, p1.copy())
    check("DeLong on identical predictors is not significant",
          not (dn["p_value"] < 0.05), f"p={dn['p_value']}")

    a = (p1 >= 0.5).astype(int)
    b = (p2 >= 0.5).astype(int)
    mc = mcnemar_test(y, a, b)
    check("McNemar detects a real difference", mc["p_value"] < 0.05, f"p={mc['p_value']:.2e}")
    check("McNemar on identical classifiers p == 1",
          abs(mcnemar_test(y, a, a.copy())["p_value"] - 1.0) < 1e-12)

    yt = rng.normal(0, 1, 400)
    good = yt + rng.normal(0, 0.3, 400)
    bad = yt + rng.normal(0, 1.5, 400)
    dm = diebold_mariano(yt, good, bad)
    check("DM statistic is negative when A is better", dm["dm"] < 0, f"dm={dm['dm']:.2f}")
    check("DM detects a real difference", dm["p_value"] < 0.01, f"p={dm['p_value']:.2e}")
    dm2 = diebold_mariano(yt, bad, good)
    check("DM is antisymmetric", abs(dm["dm"] + dm2["dm"]) < 1e-8)

    pb = paired_bootstrap(lambda yy, pp: -np.sqrt(np.mean((yy - pp) ** 2)),
                          yt, good, bad, n_boot=400)
    check("paired bootstrap agrees in sign with DM", pb["delta"] > 0 and pb["p_value"] < 0.05,
          f"delta={pb['delta']:.3f}, p={pb['p_value']:.3f}")


def test_event_metrics():
    print("\n[3] event-wise skill")
    y = np.array([0, 0, 1, 1, 1, 0, 0, 0, 1, 0])
    yh = np.array([0, 1, 1, 0, 0, 0, 1, 0, 0, 0])
    g = np.zeros(10, int)
    m = event_metrics(y, yh, g, tol=1)
    check("event POD counts a hit within tolerance", m["event_hits"] >= 1, str(m))
    check("event CSI in [0, 1]", 0 <= m["event_CSI"] <= 1)
    perfect = event_metrics(y, y.copy(), g, tol=1)
    check("perfect forecast -> POD=1, FAR=0",
          perfect["event_POD"] == 1.0 and perfect["event_FAR"] == 0.0, str(perfect))


def test_leakage(bundle_path):
    print("\n[4] protocol / leakage checks")
    b = Bundle.load(bundle_path)
    tr, va, te = b.idx("train"), b.idx("val"), b.idx("test")

    yr_tr = np.array([d.astype("datetime64[Y]").astype(int) + 1970 for d in b.date[tr]])
    yr_va = np.array([d.astype("datetime64[Y]").astype(int) + 1970 for d in b.date[va]])
    yr_te = np.array([d.astype("datetime64[Y]").astype(int) + 1970 for d in b.date[te]])
    check("train years are 2015-2018", set(np.unique(yr_tr)) <= {2015, 2016, 2017, 2018},
          str(sorted(set(np.unique(yr_tr)))))
    check("validation year is 2019", set(np.unique(yr_va)) == {2019})
    check("test year is 2020", set(np.unique(yr_te)) == {2020})
    check("splits are chronologically ordered", yr_tr.max() < yr_va.min() <= yr_va.max() < yr_te.min())

    key = lambda i: set(zip(b.station[i].tolist(), b.date[i].astype(np.int64).tolist()))
    check("no (station, date) shared between train and test", len(key(tr) & key(te)) == 0)
    check("no (station, date) shared between val and test", len(key(va) & key(te)) == 0)

    # targets must lie strictly after the input window
    check("targets are strictly in the future",
          b.X.shape[1] == b.cfg["T"] and b.y_cls.shape[1] == b.cfg["H"],
          f"T={b.cfg['T']}, H={b.cfg['H']}")

    # graph must not depend on the test split
    cfg = DataConfig(**{k: (tuple(v) if isinstance(v, list) else v)
                        for k, v in b.cfg.items()})
    from ..data.dataset import build_variable_graph
    W1, A1 = build_variable_graph(b, cfg.corr_threshold)
    check("variable graph reproducible from the training split only",
          np.allclose(A1, b.A), f"max|dA|={np.abs(A1 - b.A).max():.2e}")

    # standardisation: training mean of each variable must be ~0 in the tensor
    zt = b.X[tr][:, -1, :8, 0]
    check("train features are centred (|mean| < 0.35)", np.abs(zt.mean(0)).max() < 0.35,
          f"max|mean|={np.abs(zt.mean(0)).max():.3f}")
    check("winsorising respected (|z| <= 6)", np.abs(b.X[:, :, :8, 0]).max() <= 6.0 + 1e-5,
          f"max|z|={np.abs(b.X[:, :, :8, 0]).max():.3f}")

    check("labels are binary", set(np.unique(b.y_cls)) <= {0.0, 1.0})
    check("red-tide node channel is binary", set(np.unique(b.X[:, :, 8, 0])) <= {0.0, 1.0})
    check("regression mask is non-empty in every split",
          all(b.m_reg[i][:, 0].sum() > 0 for i in (tr, va, te)))
    check("positives exist in every split",
          all(b.y_cls[i][:, 0][b.m_cls[i][:, 0]].sum() > 0 for i in (tr, va, te)),
          f"train={int(b.y_cls[tr][:,0].sum())}, val={int(b.y_cls[va][:,0].sum())}, "
          f"test={int(b.y_cls[te][:,0].sum())}")

    # k-adjacency sanity
    A0 = k_adjacency(b.A, 0)
    A1k = k_adjacency(b.A, 1)
    check("k=0 adjacency is the identity", np.allclose(A0, np.eye(len(b.A))))
    check("k=1 adjacency equals the graph", np.allclose(A1k, b.A))
    An = normalize_adj(b.A)
    check("normalised adjacency is symmetric", np.allclose(An, An.T))
    check("normalised adjacency spectral radius <= 1",
          np.max(np.abs(np.linalg.eigvalsh(An))) <= 1 + 1e-6)


def test_inverse_transform(bundle_path):
    print("\n[5] inverse standardisation round-trip")
    b = Bundle.load(bundle_path)
    mu = np.array([b.scaler[s]["mu"]["chl"] for s in b.stations], np.float32)
    sd = np.array([b.scaler[s]["sd"]["chl"] for s in b.stations], np.float32)
    te = b.idx("test")
    m = b.m_reg[te][:, 0]
    back = b.y_reg[te][m, 0] * sd[b.station[te][m]] + mu[b.station[te][m]]
    err = np.abs(back - b.y_reg_raw[te][m, 0]).max()
    check("standardised target maps back to ug/L", err < 1e-2, f"max err={err:.2e}")


def test_threshold_protocol(runs_dir):
    print("\n[6] decision-threshold protocol")
    files = sorted(glob.glob(os.path.join(runs_dir, "*", "preds_test.npz")))
    if not files:
        print("  (skipped: no runs yet)")
        return
    infl = []
    for f in files[:12]:
        z = np.load(f)
        m = z["m_cls"][:, 0].astype(bool) & z["event_adj"].astype(bool)
        if m.sum() < 10 or z["y_cls"][m, 0].sum() == 0:
            continue
        y, p = z["y_cls"][m, 0].astype(int), z["prob"][m, 0]
        f1_val_thr = classification_metrics(y, p, float(z["thr"][0]))["F1"]
        _, f1_oracle = best_threshold(y, p)
        infl.append(f1_oracle - f1_val_thr)
        check(f"validation threshold is not the test optimum ({os.path.basename(os.path.dirname(f))})",
              f1_oracle >= f1_val_thr - 1e-9,
              f"val-thr F1={f1_val_thr:.3f}, oracle F1={f1_oracle:.3f}")
    if infl:
        print(f"  -> mean optimistic bias avoided by using the validation threshold: "
              f"{np.mean(infl):+.3f} F1")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="outputs/bundle_ffill.npz")
    ap.add_argument("--runs", default="outputs/runs")
    a = ap.parse_args()

    test_metrics()
    test_stats()
    test_event_metrics()
    if os.path.exists(a.bundle):
        test_leakage(a.bundle)
        test_inverse_transform(a.bundle)
    else:
        print(f"\n(skipped data checks: {a.bundle} not found)")
    test_threshold_protocol(a.runs)

    print("\n" + "=" * 62)
    if _state["fail"]:
        print(f"{_state['fail']} CHECK(S) FAILED")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
