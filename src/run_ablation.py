"""
run_ablation.py
===============
Contribution ablation for the **proposed model only**.

The v11 configuration changed four things at once relative to v9 — the station
(spatial) graph, the log1p target, the 10-day input window and the learned
station embedding — so no result from it can be attributed to any one of them.
Reviewers ask for exactly this decomposition, and without it the "spatial graph
is what helps" claim in the manuscript is unsupported.

Each row turns **one** ingredient off and leaves everything else at the v11
setting, so the difference is attributable:

    key                config                                     bundle
    ----------------   ---------------------------------------   ---------------
    full               knn=2 station_knn=2 log T=10 emb=8        (reference)
    no_station_graph   station_knn=0                             new bundle
    no_log_target      raw Chl-a target                          new bundle
    window_5           T=5                                       new bundle
    no_station_emb     --station_emb 0                           same as full

Test-set caveat that must go in the caption: `window_5` is the only row that
changes the evaluation sample count (a shorter window yields more usable days),
so its absolute numbers are not directly comparable to the others.  The other
three keep the identical test set — turning off the station graph or the log
target changes the inputs/scaling, not which days are scorable.

Usage
-----
    # reuse the 10-seed main run for the `full` row, so only 4x10 runs are new
    python -m src.run_ablation --device cuda --seeds 10 \
           --reuse_main outputs/run_standard/runs

    python -m src.run_ablation --dry_run          # plan + bundle list only
    python -m src.run_ablation --only no_log_target --seeds 10
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time

import numpy as np
import pandas as pd

from .run_experiments import flatten
from .run_table3 import COMPACT_WIDTH, seed_guard, write_manifest

H = [f"D+{h}" for h in range(1, 8)]

# dataset-side knobs -> bundle;  model-side knobs -> train.py flags
# Revision (v13): the REFERENCE is the final 9-node model (= main-table row, reused);
# every other row changes exactly one ingredient relative to it.  Model-side knobs
# (station_emb, lam, residual) are train.py flags; dataset-side knobs rebuild the bundle.
CONFIGS = {
    "full":              dict(knn=2, station_knn=0, log_target=True,  T=10, station_emb=8),
    "plus_station_graph": dict(knn=2, station_knn=2, log_target=True,  T=10, station_emb=8),
    "no_log_target":     dict(knn=2, station_knn=0, log_target=False, T=10, station_emb=8),
    "window_5":          dict(knn=2, station_knn=0, log_target=True,  T=5,  station_emb=8),
    "no_station_emb":    dict(knn=2, station_knn=0, log_target=True,  T=10, station_emb=0),
    # optional rows (reviewer-driven): multi-task weight and a persistence-anchored head
    "lam_0.2":           dict(knn=2, station_knn=0, log_target=True,  T=10, station_emb=8, lam=0.2),
    "lam_2.0":           dict(knn=2, station_knn=0, log_target=True,  T=10, station_emb=8, lam=2.0),
    "residual_head":     dict(knn=2, station_knn=0, log_target=True,  T=10, station_emb=8, residual=True),
    # stage-1 candidates (revision v13): extra input channels and class weighting.
    # Chosen on the 2019 validation split only (analysis/08_validation_selection.py).
    "plus_doy":          dict(knn=2, station_knn=0, log_target=True,  T=10, station_emb=8, feats="doy"),
    "plus_rthist":       dict(knn=2, station_knn=0, log_target=True,  T=10, station_emb=8, feats="rthist"),
    "plus_doy_rthist":   dict(knn=2, station_knn=0, log_target=True,  T=10, station_emb=8, feats="doy,rthist"),
    "wbce":              dict(knn=2, station_knn=0, log_target=True,  T=10, station_emb=8, cls_loss="wbce"),
}
LABEL = {
    "full": "final configuration (reference)",
    "plus_station_graph": "+ neighbouring-station nodes (spatial extension)",
    "no_log_target": "− log1p target",
    "window_5": "− 10-day window (T=5)",
    "no_station_emb": "− station embedding",
    "lam_0.2": "λ = 0.2 (reference 0.8)",
    "lam_2.0": "λ = 2.0 (reference 0.8)",
    "residual_head": "Chl-a head predicts increment w.r.t. day D (persistence-anchored)",
    "plus_doy": "+ day-of-year sin/cos input channels (explicit seasonality)",
    "plus_rthist": "+ red-tide history channels (days since last positive, 30-day count)",
    "plus_doy_rthist": "+ seasonality + red-tide history channels",
    "wbce": "class-weighted BCE (pos_weight = N-/N+)",
}
CORE = ["full", "plus_station_graph", "no_log_target", "window_5", "no_station_emb"]
STAGE1 = ["plus_doy", "plus_rthist", "plus_doy_rthist", "residual_head", "lam_0.2", "lam_2.0", "wbce"]
DIVERGE_R2 = -1.0
COLLAPSE_F1 = 0.02


def bundle_path(cfg, outdir="outputs"):
    key = (f"bundle_ab_k{cfg['knn']}_s{cfg['station_knn']}"
           f"_l{int(cfg['log_target'])}_T{cfg['T']}_qc")
    feats = cfg.get("feats", "")
    if feats:
        key += "_f" + feats.replace(",", "-")
    elif cfg["station_knn"] == 0 and cfg["log_target"] and cfg["T"] == 10 and cfg["knn"] == 2:
        return os.path.join(outdir, "bundle_main.npz")       # identical to the main-table bundle
    return os.path.join(outdir, key + ".npz")


def ensure_bundle(cfg, daily="outputs/daily_station_day.csv", dry=False):
    p = bundle_path(cfg)
    if os.path.exists(p):
        return p, False
    if not os.path.exists(daily):
        pq = daily.replace(".csv", ".parquet")
        if os.path.exists(pq) and not dry:
            pd.read_parquet(pq).to_csv(daily, index=False)
        elif not os.path.exists(pq):
            raise SystemExit(f"ERROR: {daily} 도 {pq} 도 없습니다")
    cmd = [sys.executable, "-m", "src.data.dataset", "--daily", daily,
           "--out", p, "--impute", "ffill", "--mask_channel", "--qc", "physical",
           "--T", str(cfg["T"])]
    cmd += ["--knn", str(cfg["knn"])] if cfg["knn"] > 0 else ["--tau", "0.30"]
    if cfg["station_knn"] > 0:
        cmd += ["--station_knn", str(cfg["station_knn"])]
    if cfg["log_target"]:
        cmd.append("--log_target")
    if cfg.get("feats"):
        cmd += ["--extra_feats", cfg["feats"]]
    if dry:
        print("   would build:", " ".join(cmd))
        return p, True
    print(f"[bundle] {os.path.basename(p)}")
    subprocess.run(cmd, check=True)
    return p, True


def job_cmd(key, cfg, seed, a, bundle, device):
    cmd = [sys.executable, "-m", "src.train",
           "--bundle", bundle, "--outdir", a.outdir, "--tag", f"{key}_s{seed}",
           "--model", "mthab", "--reg_loss", "mse", "--cls_loss", cfg.get("cls_loss", "bce"),
           "--channels", a.channels, "--epochs", str(a.epochs),
           "--patience", str(a.patience), "--batch_size", str(a.batch_size),
           "--select_metric", "auprc", "--seed", str(seed),
           "--station_emb", str(cfg["station_emb"]), "--device", device,
           "--save_state", "--save_val_preds", "--perm_importance", "5"]
    if "lam" in cfg:
        cmd += ["--lam", str(cfg["lam"])]
    if cfg.get("residual"):
        cmd.append("--residual")
    return cmd


# --------------------------------------------------------------------------- #
def summarise(rows):
    out = []
    for key in CONFIGS:
        g = [r for r in rows if r["cfg"] == key]
        if not g:
            continue
        rmse = np.array([np.mean([r["m"]["full_year"][h]["RMSE"] for h in H]) for r in g])
        r2min = np.array([min(r["m"]["full_year"][h]["R2"] for h in H) for r in g])
        f1 = np.array([np.mean([r["m"]["event_adjacent"][h]["F1"] for h in H]) for r in g])
        aup = np.array([np.mean([r["m"]["event_adjacent"][h]["AUPRC"] for h in H]) for r in g])
        r2 = np.array([np.mean([r["m"]["full_year"][h]["R2"] for h in H]) for r in g])
        sr = [r["m"].get("spread_ratio_mean") for r in g]
        sr = [x for x in sr if x is not None]
        n_eval = g[0]["m"]["full_year"]["D+1"].get("n_reg",
                 g[0]["m"]["full_year"]["D+1"].get("n"))
        out.append(dict(
            config=key, label=LABEL[key], n_seeds=len(g),
            RMSE_mean=rmse.mean(), RMSE_median=float(np.median(rmse)),
            RMSE_sd=rmse.std(ddof=1) if len(rmse) > 1 else 0.0,
            R2_mean=r2.mean(),
            F1ea_mean=f1.mean(), F1ea_median=float(np.median(f1)), F1ea_min=f1.min(),
            AUPRCea_mean=aup.mean(),
            n_diverged=int((r2min < DIVERGE_R2).sum()),
            n_collapsed=int((f1 < COLLAPSE_F1).sum()),
            spread_ratio=float(np.mean(sr)) if sr else np.nan,
            n_eval_samples=n_eval))
    s = pd.DataFrame(out)
    # `full` 행이 없으면(--only 로 일부만 돌린 경우) 델타는 정의되지 않지만,
    # 열 자체는 항상 있어야 출력 코드가 깨지지 않습니다.
    if "full" in set(s.config):
        base = s[s.config == "full"].iloc[0]
        s["dRMSE_median"] = s.RMSE_median - base.RMSE_median
        s["dF1ea_median"] = s.F1ea_median - base.F1ea_median
    else:
        s["dRMSE_median"] = np.nan
        s["dF1ea_median"] = np.nan
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="outputs/ablation/runs")
    ap.add_argument("--out", default="outputs/ablation")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--channels", default=COMPACT_WIDTH)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=2,
                    help="concurrent training processes")
    ap.add_argument("--only", default="", help="comma-separated config keys")
    ap.add_argument("--reuse_main", default="",
                    help="run dir of the main 10-seed experiment (e.g. "
                         "outputs/run_standard/runs). Its A_mthab_mse_s* results "
                         "are used for the `full` row, saving 10 runs.")
    ap.add_argument("--force", action="store_true", help="retrain even if present")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--extended", action="store_true",
                    help="also run the optional rows (lambda sweep, residual head)")
    ap.add_argument("--stage1", action="store_true",
                    help="run the revision stage-1 candidate rows only (validation-selected)")
    a = ap.parse_args()
    seed_guard(a.seeds, "run_ablation")

    if a.only:
        keys = [k for k in CONFIGS if k in a.only.split(",")]
    elif a.stage1:
        keys = ["full"] + STAGE1
    else:
        keys = [k for k in CONFIGS if (k in CORE or a.extended)]
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.outdir, exist_ok=True)

    # ---- bundles -------------------------------------------------------
    bundles = {}
    for k in keys:
        bundles[k], built = ensure_bundle(CONFIGS[k], dry=a.dry_run)

    # ---- job list ------------------------------------------------------
    jobs, reused = [], []
    for k in keys:
        for sd in range(a.seeds):
            mine = os.path.join(a.outdir, f"{k}_s{sd}", "metrics.json")
            if os.path.exists(mine) and not a.force:
                reused.append((k, sd, mine)); continue
            if k == "full" and a.reuse_main:
                alt = os.path.join(a.reuse_main, f"A_mthab_mse_s{sd}", "metrics.json")
                if os.path.exists(alt):
                    reused.append((k, sd, alt)); continue
            jobs.append((k, sd))

    print("=" * 70)
    print(" MT-HAB-G3D ablation (proposed model only)")
    print(f"   configs : {', '.join(keys)}")
    print(f"   seeds   : {a.seeds}")
    print(f"   to run  : {len(jobs)}    reused: {len(reused)}")
    print(f"   workers : {a.workers}   device={a.device}")
    print("=" * 70)
    for k in keys:
        print(f"   {k:18s} {os.path.basename(bundles[k])}"
              + ("   station_emb=0" if CONFIGS[k]["station_emb"] == 0 else ""))
    if a.dry_run:
        print("\n(dry run - nothing trained)")
        return

    # ---- run -----------------------------------------------------------
    jq = queue.Queue()
    for j in jobs:
        jq.put(j)
    state = {"done": 0, "total": len(jobs), "failed": [], "lock": threading.Lock()}
    devs = queue.Queue()
    for i in range(max(a.workers, 1)):
        devs.put(a.device)

    def work():
        while True:
            try:
                k, sd = jq.get_nowait()
            except queue.Empty:
                return
            dev = devs.get()
            t0 = time.time()
            try:
                r = subprocess.run(job_cmd(k, CONFIGS[k], sd, a, bundles[k], dev),
                                   capture_output=True, text=True)
                with state["lock"]:
                    state["done"] += 1
                    n = state["done"]
                if r.returncode == 0:
                    tail = [l for l in r.stdout.strip().splitlines() if l.startswith("[")]
                    print(f"  [{n:3d}/{state['total']}] OK   {k}_s{sd:<3d} "
                          f"{time.time()-t0:6.0f}s  {tail[-1][:80] if tail else ''}",
                          flush=True)
                else:
                    state["failed"].append(f"{k}_s{sd}")
                    err = (r.stderr.strip().splitlines() or ["?"])[-1]
                    print(f"  [{n:3d}/{state['total']}] FAIL {k}_s{sd:<3d} {err[:90]}",
                          flush=True)
            finally:
                devs.put(dev)
                jq.task_done()

    ts = [threading.Thread(target=work, daemon=True) for _ in range(max(a.workers, 1))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()

    # ---- collect -------------------------------------------------------
    rows = []
    for k in keys:
        for sd in range(a.seeds):
            for p in (os.path.join(a.outdir, f"{k}_s{sd}", "metrics.json"),
                      os.path.join(a.reuse_main, f"A_mthab_mse_s{sd}", "metrics.json")
                      if (k == "full" and a.reuse_main) else ""):
                if p and os.path.exists(p):
                    rows.append(dict(cfg=k, seed=sd, m=json.load(open(p))))
                    break
    if not rows:
        print("no metrics collected"); return

    s = summarise(rows)
    s.to_csv(os.path.join(a.out, "ablation_summary.csv"), index=False)
    flat = [flatten(r["cfg"], "AB", r["seed"], r["m"]) for r in rows]
    pd.DataFrame(flat).to_csv(os.path.join(a.out, "ablation_results.csv"), index=False)

    cols = [c for c in ["label", "n_seeds", "RMSE_median", "dRMSE_median",
                        "R2_mean", "F1ea_median", "dF1ea_median", "n_diverged",
                        "n_collapsed", "n_eval_samples"] if c in s.columns]
    if s.dRMSE_median.isna().all():
        print("  (참고: `full` 행이 없어 델타 열은 비어 있습니다 - "
              "--only 없이 전체를 돌리거나 --reuse_main 을 주십시오)")
    print("\n" + "=" * 100)
    print(" Ablation — MT-HAB-G3D only; one ingredient removed per row")
    print("=" * 100)
    print(s[cols].to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    print("\n  주의: window_5 행만 평가 표본 수가 다릅니다 (n_eval_samples 열 확인).")
    print("        나머지 3행은 full 과 동일한 테스트셋이므로 직접 비교 가능합니다.")
    write_manifest(a.out, a.seeds, "ablation", "multiple",
                   extra=dict(configs=keys, failed=state["failed"]))
    print(f"\n-> {a.out}/ablation_summary.csv")
    if state["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
