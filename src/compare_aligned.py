"""
compare_aligned.py
==================
Fair head-to-head comparison of two runs that were trained on *different*
bundles (e.g. the 9-node and the 13-node nutrient-augmented graph, whose valid
sample sets differ because the nutrient channels have their own coverage rule).

`src/stats_tests.compare_runs` deliberately refuses to compare such runs, since
metrics computed on different station-days are not comparable. This module
restores comparability by intersecting the two evaluation sets on the
(station, date) key and recomputing every metric — and every paired test — on
the common subset only.

Usage
-----
python -m src.compare_aligned \
       --a outputs/runs/I_nodes9_s0/preds_test.npz \
       --b outputs/runs/I_nodes13_s0/preds_test.npz \
       --label_a "9 nodes" --label_b "13 nodes (+nutrients)"
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from .metrics import classification_metrics, regression_metrics, best_threshold
from .stats_tests import delong_test, mcnemar_test, diebold_mariano, paired_bootstrap


def _key(z):
    return np.array([f"{s}|{d}" for s, d in zip(z["station"], z["date"])])


def align(npz_a: str, npz_b: str):
    A, B = np.load(npz_a), np.load(npz_b)
    ka, kb = _key(A), _key(B)
    common = np.intersect1d(ka, kb)
    ia = np.array([np.where(ka == k)[0][0] for k in common])
    ib = np.array([np.where(kb == k)[0][0] for k in common])
    return A, B, ia, ib, common


def compare(npz_a, npz_b, horizon=0, event_adjacent=True, seed=0):
    A, B, ia, ib, common = align(npz_a, npz_b)

    # ---- classification on the common event-adjacent station-days ---------- #
    mc = A["m_cls"][ia, horizon].astype(bool) & B["m_cls"][ib, horizon].astype(bool)
    if event_adjacent:
        mc &= A["event_adj"][ia].astype(bool)
    y = A["y_cls"][ia, horizon][mc].astype(int)
    pa, pb = A["prob"][ia, horizon][mc], B["prob"][ib, horizon][mc]
    ta, tb = float(A["thr"][horizon]), float(B["thr"][horizon])

    # ---- regression on the common full-year station-days ------------------ #
    mr = A["m_reg"][ia, horizon].astype(bool) & B["m_reg"][ib, horizon].astype(bool)
    yr = A["y_reg_raw"][ia, horizon][mr]
    ra, rb = A["reg_raw"][ia, horizon][mr], B["reg_raw"][ib, horizon][mr]

    f1 = lambda yy, pp: classification_metrics(yy, pp, ta)["F1"]
    rmse = lambda yy, pp: regression_metrics(yy, pp)["RMSE"]

    out = {
        "n_common_station_days": int(len(common)),
        "n_classification": int(mc.sum()), "n_positive": int(y.sum()),
        "n_regression": int(mr.sum()),
        "A": {"classification": classification_metrics(y, pa, ta),
              "regression": regression_metrics(yr, ra)},
        "B": {"classification": classification_metrics(y, pb, tb),
              "regression": regression_metrics(yr, rb)},
        "tests": {
            "delong_AUROC": delong_test(y, pa, pb),
            "mcnemar": mcnemar_test(y, (pa >= ta).astype(int), (pb >= tb).astype(int)),
            "paired_bootstrap_F1": paired_bootstrap(f1, y, pa, pb, seed=seed),
            "diebold_mariano_RMSE": diebold_mariano(yr, ra, rb),
            "paired_bootstrap_RMSE": paired_bootstrap(rmse, yr, ra, rb, seed=seed,
                                                      block=7),
        },
    }
    return out


def summary_line(r, la, lb):
    a, b = r["A"], r["B"]
    t = r["tests"]
    return (
        f"aligned on {r['n_common_station_days']} common station-days "
        f"({r['n_regression']} with Chl-a, {r['n_classification']} scored for "
        f"red tide, {r['n_positive']} positive)\n"
        f"  {la:24s} RMSE={a['regression']['RMSE']:.3f}  "
        f"PeakRMSE={a['regression']['PeakRMSE']:.3f}  R2={a['regression']['R2']:.3f}  "
        f"AUPRC={a['classification']['AUPRC']:.3f}  F1={a['classification']['F1']:.3f}\n"
        f"  {lb:24s} RMSE={b['regression']['RMSE']:.3f}  "
        f"PeakRMSE={b['regression']['PeakRMSE']:.3f}  R2={b['regression']['R2']:.3f}  "
        f"AUPRC={b['classification']['AUPRC']:.3f}  F1={b['classification']['F1']:.3f}\n"
        f"  RMSE difference {t['paired_bootstrap_RMSE']['delta']:+.3f} "
        f"[{t['paired_bootstrap_RMSE']['ci_lo']:+.3f}, "
        f"{t['paired_bootstrap_RMSE']['ci_hi']:+.3f}], "
        f"Diebold-Mariano p={t['diebold_mariano_RMSE']['p_value']:.4f}\n"
        f"  AUROC {t['delong_AUROC']['auc1']:.3f} vs {t['delong_AUROC']['auc2']:.3f}, "
        f"DeLong p={t['delong_AUROC']['p_value']:.4f}; "
        f"McNemar p={t['mcnemar']['p_value']:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--label_a", default="A")
    ap.add_argument("--label_b", default="B")
    ap.add_argument("--horizon", type=int, default=0)
    ap.add_argument("--full_year", action="store_true")
    ap.add_argument("--json_out", default="")
    g = ap.parse_args()
    r = compare(g.a, g.b, g.horizon, not g.full_year)
    print(summary_line(r, g.label_a, g.label_b))
    if g.json_out:
        json.dump(r, open(g.json_out, "w"), indent=1, default=float)
