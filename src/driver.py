"""
driver.py
=========
One robust, restartable driver for the entire experiment matrix.

Each configuration is launched as an isolated subprocess.  If a single run
crashes (out-of-memory, numerical blow-up, container hiccup) the driver logs it
and moves on instead of taking the whole queue down with it.  Finished runs
(those that already have a metrics.json) are skipped, so the driver can be killed
and relaunched at any time.

Usage
-----
python -m src.driver                 # run everything that is not yet done
python -m src.driver --only G,H,I,J  # a subset of groups
python -m src.driver --dry_run       # just print the plan
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import pandas as pd

BUNDLES = {
    "paper": "outputs/bundle_v2_paper.npz",
    "max": "outputs/bundle_v2_max.npz",
    "nutrient": "outputs/bundle_v2_nutrient.npz",
}
COMMON = ["--epochs", "16", "--patience", "6", "--channels", "32,64,128",
          "--batch_size", "256"]

BASE = ["--reg_loss", "huber", "--cls_loss", "focal", "--pos_sampler"]


def plan(seeds, seeds_small):
    """List of (tag, bundle_key, [extra cli args], seeds)."""
    P = []

    # ---------------- A: baselines & manuscript reproduction --------------- #
    P += [("A_persistence", "paper", ["--model", "persistence"], [0]),
          ("A_climatology", "paper", ["--model", "climatology"], [0]),
          ("A_histgb", "paper", ["--model", "histgb"], seeds_small),
          ("A_mlp", "paper", ["--model", "mlp"], seeds_small),
          ("A_lstm", "paper", ["--model", "lstm", "--reg_loss", "mse"], seeds),
          ("A_gru", "paper", ["--model", "gru", "--reg_loss", "mse"], seeds_small),
          ("A_tcn", "paper", ["--model", "tcn", "--reg_loss", "mse"], seeds_small),
          ("A_transformer", "paper", ["--model", "transformer", "--reg_loss", "mse"],
           seeds_small),
          ("A_stgcn", "paper", ["--model", "stgcn", "--reg_loss", "mse"], seeds_small),
          ("A_mthab_mse", "paper", ["--model", "mthab", "--reg_loss", "mse"], seeds),
          ("A_mthab_huber", "paper", ["--model", "mthab", "--reg_loss", "huber"], seeds)]

    # ---------------- B: imbalance ----------------------------------------- #
    P += [("B_mthab_huber_focal", "paper", ["--model", "mthab"] + BASE, seeds),
          ("B_mthab_huber_wbce", "paper",
           ["--model", "mthab", "--reg_loss", "huber", "--cls_loss", "wbce"], seeds_small),
          ("B_lstm_focal", "paper",
           ["--model", "lstm", "--reg_loss", "mse", "--cls_loss", "focal",
            "--pos_sampler"], seeds_small)]

    # ---------------- C: graph --------------------------------------------- #
    for g in ["identity", "complete", "adaptive", "hybrid"]:
        extra = ["--model", "mthab", "--graph", g] + BASE
        if g != "identity":
            extra += ["--graph_l1", "1e-3"]
        P.append((f"C_graph_{g}", "paper", extra, seeds_small))

    # ---------------- D: pathway ------------------------------------------- #
    P += [("D_no_g3d", "paper", ["--model", "mthab", "--no_g3d"] + BASE, seeds_small),
          ("D_no_factorized", "paper",
           ["--model", "mthab", "--no_factorized"] + BASE, seeds_small)]

    # ---------------- E: objective ----------------------------------------- #
    P += [("E_weight_uncertainty", "paper",
           ["--model", "mthab", "--weighting", "uncertainty"] + BASE, seeds_small),
          ("E_weight_dwa", "paper",
           ["--model", "mthab", "--weighting", "dwa"] + BASE, seeds_small),
          ("E_peak_huber", "paper",
           ["--model", "mthab", "--reg_loss", "peak_huber", "--peak_gamma", "2.0",
            "--cls_loss", "focal", "--pos_sampler"], seeds_small)]

    # ---------------- F: uncertainty --------------------------------------- #
    P += [("F_quantile", "paper",
           ["--model", "mthab", "--reg_loss", "quantile", "--n_quantiles", "3"]
           + ["--cls_loss", "focal", "--pos_sampler"], seeds_small),
          ("F_mc_dropout", "paper",
           ["--model", "mthab", "--mc_dropout", "30", "--dropout", "0.2"] + BASE,
           seeds_small)]

    # ---------------- G: MTGNN family -------------------------------------- #
    P += [("G_mtgnn", "paper", ["--model", "mtgnn"] + BASE, seeds),
          ("G_am_mtgnn", "paper", ["--model", "am_mtgnn"] + BASE, seeds),
          ("G_mtgnn_nograph", "paper", ["--model", "mtgnn_nograph"] + BASE, seeds_small),
          ("G_mthab_ref", "paper", ["--model", "mthab"] + BASE, seeds)]

    # ---------------- H: aggregation protocol ------------------------------ #
    P += [("H_agg_mean", "paper", ["--model", "mthab"] + BASE, seeds_small),
          ("H_agg_max", "max", ["--model", "mthab"] + BASE, seeds_small),
          ("H_agg_mean_lstm", "paper",
           ["--model", "lstm", "--reg_loss", "mse", "--cls_loss", "focal",
            "--pos_sampler"], seeds_small),
          ("H_agg_max_lstm", "max",
           ["--model", "lstm", "--reg_loss", "mse", "--cls_loss", "focal",
            "--pos_sampler"], seeds_small)]

    # ---------------- I: nutrient nodes ------------------------------------ #
    P += [("I_nodes9", "paper", ["--model", "mthab"] + BASE, seeds_small),
          ("I_nodes13", "nutrient", ["--model", "mthab"] + BASE, seeds),
          ("I_nodes13_hybrid", "nutrient",
           ["--model", "mthab", "--graph", "hybrid", "--graph_l1", "1e-3"] + BASE,
           seeds_small),
          ("I_nodes13_am_mtgnn", "nutrient", ["--model", "am_mtgnn"] + BASE,
           seeds_small)]

    # ---------------- J: residual (persistence-anchored) head -------------- #
    P += [("J_residual", "paper", ["--model", "mthab", "--residual"] + BASE, seeds),
          ("J_residual_peak", "paper",
           ["--model", "mthab", "--residual", "--reg_loss", "peak_huber",
            "--peak_gamma", "2.0", "--cls_loss", "focal", "--pos_sampler"], seeds_small),
          ("J_lstm_residual", "paper",
           ["--model", "lstm", "--residual", "--reg_loss", "huber",
            "--cls_loss", "focal", "--pos_sampler"], seeds_small)]
    return P


# names used by report.py, keyed on tag prefix
def report_name(tag: str) -> str:
    return tag.split("_", 1)[1]


def flatten(name, group, seed, m):
    row = {"name": name, "group": group, "seed": seed,
           "n_params": m.get("n_params"), "runtime_s": m.get("runtime_s")}
    for proto in ("full_year", "event_adjacent"):
        for hk, hv in m.get(proto, {}).items():
            if not hk.startswith("D+"):
                continue
            for k, v in hv.items():
                row[f"{proto}|{hk}|{k}"] = v
    return row


def collect(runs_dir, plan_list, out_main, out_ext):
    main_rows, ext_rows = [], []
    for tag, _b, _e, sds in plan_list:
        group = tag[0]
        name = report_name(tag)
        for sd in sds:
            mj = os.path.join(runs_dir, f"{tag}_s{sd}", "metrics.json")
            if not os.path.exists(mj):
                continue
            try:
                m = json.load(open(mj))
            except Exception:
                continue
            row = flatten(name, group, sd, m)
            (ext_rows if group in "GHI" else main_rows).append(row)
    if main_rows:
        pd.DataFrame(main_rows).to_csv(out_main, index=False)
    if ext_rows:
        pd.DataFrame(ext_rows).to_csv(out_ext, index=False)
    return len(main_rows), len(ext_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="outputs/runs")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--seeds_small", default="0,1")
    ap.add_argument("--only", default="A,B,C,D,E,F,G,H,I,J")
    ap.add_argument("--results", default="outputs/results.csv")
    ap.add_argument("--results_ext", default="outputs/results_ext.csv")
    ap.add_argument("--timeout", type=int, default=1800, help="per-run seconds")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--budget", type=int, default=0,
                    help="stop launching new runs after this many seconds (0 = no limit)")
    a = ap.parse_args()

    seeds = [int(s) for s in a.seeds.split(",")]
    seeds_small = [int(s) for s in a.seeds_small.split(",")]
    groups = set(a.only.split(","))
    P = [p for p in plan(seeds, seeds_small) if p[0][0] in groups]

    todo = []
    for tag, bkey, extra, sds in P:
        for sd in sds:
            done = os.path.exists(os.path.join(a.outdir, f"{tag}_s{sd}", "metrics.json"))
            if not done:
                todo.append((tag, bkey, extra, sd))

    total = sum(len(sds) for _t, _b, _e, sds in P)
    print(f"plan: {total} runs, {total - len(todo)} done, {len(todo)} to run", flush=True)
    if a.dry_run:
        for tag, bkey, extra, sd in todo:
            print(f"  {tag}_s{sd}  [{bkey}]  {' '.join(extra)}")
        return

    t_start = time.time()
    for i, (tag, bkey, extra, sd) in enumerate(todo, 1):
        if a.budget and time.time() - t_start > a.budget:
            print(f"budget reached, {len(todo) - i + 1} runs left", flush=True)
            break
        cmd = [sys.executable, "-m", "src.train",
               "--bundle", BUNDLES[bkey], "--outdir", a.outdir,
               "--tag", f"{tag}_s{sd}", "--seed", str(sd)] + COMMON + extra
        env = dict(os.environ, OMP_NUM_THREADS="1")
        t0 = time.time()
        print(f"[{i}/{len(todo)}] {tag}_s{sd}", flush=True)
        try:
            r = subprocess.run(cmd, env=env, timeout=a.timeout,
                               capture_output=True, text=True)
            tail = r.stdout.strip().splitlines()[-1:] if r.stdout else []
            if r.returncode != 0:
                print(f"    FAILED rc={r.returncode}: {r.stderr.strip().splitlines()[-1:]}",
                      flush=True)
            else:
                print("    " + (tail[0] if tail else f"ok {time.time()-t0:.0f}s"),
                      flush=True)
        except subprocess.TimeoutExpired:
            print(f"    TIMEOUT after {a.timeout}s - skipped", flush=True)
        # refresh result tables after every run so partial progress is usable
        collect(a.outdir, P, a.results, a.results_ext)

    nm, ne = collect(a.outdir, P, a.results, a.results_ext)
    print(f"\ncollected {nm} main + {ne} ext rows", flush=True)


if __name__ == "__main__":
    main()
