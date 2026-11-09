"""
run_reference.py  (revision, torch-free)
========================================
Produces the non-neural rows of the main table — persistence, day-of-year
climatology and histogram gradient boosting — on ANY machine, with exactly the
evaluation protocol of `train.py` (validation-selected thresholds, full-year and
target-anchored event-adjacent scoring, identical preds_test.npz / metrics.json
layout) so that `run_parallel.collect()` and the analysis scripts treat them like
every other run.

    python -m src.run_reference --bundle outputs/bundle_main.npz --outdir outputs/run_standard/runs \
           --models persistence,climatology,histgb --seeds 10 --results outputs/run_standard/results.csv

Why this exists: the revision asks for a persistence baseline and the authors'
`train.py` imports torch at module level; this file depends on numpy / pandas /
scikit-learn only.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from .data.dataset import Bundle, NODE_VARS
from .metrics import summarize, best_threshold
from .models.reference import persistence, climatology, histgb, persistence_valid_anchor

CHL_MAX_UGL = 500.0


def flatten(name, group, seed, m):
    row = {"name": name, "group": group, "seed": seed,
           "n_params": m.get("n_params"), "runtime_s": m.get("runtime_s")}
    for proto in ("full_year", "event_adjacent"):
        for hk, hv in m[proto].items():
            if not hk.startswith("D+"):
                continue
            for k, v in hv.items():
                row[f"{proto}|{hk}|{k}"] = v
    return row


def chl_scale(bundle: Bundle, idx: np.ndarray):
    if bundle.cfg.get("log_target"):
        mu = np.array([bundle.scaler[s]["log_mu"] for s in bundle.stations], np.float32)
        sd = np.array([bundle.scaler[s]["log_sd"] for s in bundle.stations], np.float32)
    else:
        mu = np.array([bundle.scaler[s]["mu"]["chl"] for s in bundle.stations], np.float32)
        sd = np.array([bundle.scaler[s]["sd"]["chl"] for s in bundle.stations], np.float32)
    return mu[bundle.station[idx]], sd[bundle.station[idx]]


def inv_chl(bundle: Bundle, z, mu, sd):
    v = z * sd + mu
    if bundle.cfg.get("log_target"):
        v = np.clip(v, 0.0, float(np.log1p(CHL_MAX_UGL)))
        return np.expm1(v)
    return np.clip(v, -CHL_MAX_UGL, CHL_MAX_UGL)


def evaluate_split(b: Bundle, idx, pred, thr, horizons, event_only=False, tag=""):
    mu, sd = chl_scale(b, idx)
    res = {}
    for hi, h in enumerate(horizons):
        m = b.m_cls[idx][:, hi]
        if event_only:
            ea = getattr(b, "event_adj_h", None)
            m = m & (ea[idx][:, hi] if ea is not None else b.event_adj[idx])
        y_c, p_c, st = b.y_cls[idx][m, hi], pred["prob"][m, hi], b.station[idx][m]
        mr = b.m_reg[idx][:, hi]
        y_r = b.y_reg_raw[idx][mr, hi]
        p_r = inv_chl(b, pred["reg"][mr, hi], mu[mr], sd[mr])
        res[f"D+{h}"] = summarize(y_c, p_c, y_r, p_r, station=st, thr=thr[hi])
    res["_tag"] = tag
    return res


def run_one(b: Bundle, model: str, seed: int, outdir: str, tag: str):
    t0 = time.time()
    H = b.y_cls.shape[1]
    horizons = list(range(1, H + 1))
    tr, va, te = b.idx("train"), b.idx("val"), b.idx("test")
    if model == "persistence":
        pv, pt = persistence(b, va, horizons), persistence(b, te, horizons)
        b = Bundle(**{**b.__dict__})                       # shallow copy so the mask edit is local
        b.m_reg = b.m_reg.copy()
        b.m_reg[va] &= persistence_valid_anchor(b, va)[:, None]
        b.m_reg[te] &= persistence_valid_anchor(b, te)[:, None]
    elif model == "climatology":
        pv, pt = climatology(b, tr, va, horizons), climatology(b, tr, te, horizons)
    elif model == "histgb":
        pv, pt = histgb(b, tr, va, horizons, seed=seed), histgb(b, tr, te, horizons, seed=seed)
    else:
        raise ValueError(model)
    thr = []
    for hi in range(H):
        m = b.m_cls[va][:, hi]
        t_, _ = best_threshold(b.y_cls[va][m, hi], pv["prob"][m, hi])
        thr.append(t_)
    metrics = {
        "full_year": evaluate_split(b, te, pt, thr, horizons, False, "full_year"),
        "event_adjacent": evaluate_split(b, te, pt, thr, horizons, True, "event_adjacent"),
        "config": {"model": model, "seed": seed, "bundle": b.cfg.get("daily_path"), "task": "both",
                   "runner": "run_reference (torch-free)"},
        "thresholds": thr, "n_params": 0, "task": "both",
        "runtime_s": round(time.time() - t0, 1),
    }
    os.makedirs(outdir, exist_ok=True)
    mu, sd = chl_scale(b, te)
    np.savez_compressed(
        os.path.join(outdir, "preds_test.npz"),
        prob=pt["prob"], reg_raw=pt["reg"] * sd[:, None] + mu[:, None],
        quant_raw=(pt["reg"] * sd[:, None] + mu[:, None])[..., None],
        y_cls=b.y_cls[te], m_cls=b.m_cls[te], y_reg_raw=b.y_reg_raw[te],
        m_reg=b.m_reg[te], station=b.station[te], event_adj=b.event_adj[te],
        event_adj_h=getattr(b, "event_adj_h", np.repeat(b.event_adj[:, None], H, 1))[te],
        date=b.date[te].astype("datetime64[D]").astype(np.int64), thr=np.array(thr))
    json.dump(metrics, open(os.path.join(outdir, "metrics.json"), "w"), indent=1)
    d1f, d1e = metrics["full_year"]["D+1"], metrics["event_adjacent"]["D+1"]
    print(f"[{tag}] D+1 RMSE={d1f['RMSE']:.3f} R2={d1f['R2']:.3f} TA1={d1f['TA1']*100:.1f}% "
          f"| ev F1={d1e['F1']:.3f} AUPRC={d1e['AUPRC']:.3f} | {metrics['runtime_s']}s", flush=True)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="outputs/bundle_main.npz")
    ap.add_argument("--outdir", default="outputs/run_standard/runs")
    ap.add_argument("--results", default="outputs/run_standard/results.csv")
    ap.add_argument("--models", default="persistence,climatology,histgb")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--group", default="A")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    b = Bundle.load(a.bundle)
    rows = pd.read_csv(a.results).to_dict("records") if os.path.exists(a.results) else []
    for model in a.models.split(","):
        n = 1 if model in ("persistence", "climatology") else a.seeds
        for sd in range(n):
            tag = f"{a.group}_{model}_s{sd}"
            od = os.path.join(a.outdir, tag)
            mj = os.path.join(od, "metrics.json")
            if os.path.exists(mj) and not a.force:
                m = json.load(open(mj))
                print(f"[{tag}] reused")
            else:
                m = run_one(b, model, sd, od, tag)
            rows = [r for r in rows if not (r.get("name") == model and r.get("seed") == sd)]
            rows.append(flatten(model, a.group, sd, m))
            os.makedirs(os.path.dirname(a.results) or ".", exist_ok=True)
            pd.DataFrame(rows).to_csv(a.results, index=False)
    print(f"{len(rows)} rows -> {a.results}")


if __name__ == "__main__":
    main()
