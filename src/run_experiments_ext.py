"""
run_experiments_ext.py
======================
Second experiment matrix, covering the questions raised by the reference
repository (github.com/haoshilong-0920/Dissolved-oxygen-prediction) and by the
author-supplied `gt_day` files.

Groups
------
G_graphbase   MTGNN, AM-MTGNN (indicator attention), MTGNN w/o graph,
              all on the manuscript protocol -> answers "is MS-G3D better than
              the graph model already published in this journal?"
H_aggregation daily-mean(+median turbidity) protocol vs. daily-maximum protocol
              on *identical station-days*  -> resolves the Sec. 2.1.1 vs.
              gt_day inconsistency
I_nodes       9-node graph (Table 1) vs. 13-node nutrient-augmented graph
              (+TN, TP, NO3-N, PO4-P) -> tests the driver variables the
              introduction names but the model never sees

Every group re-runs the reference model on the same bundle so that each
comparison is internally controlled.

Usage
-----
python -m src.run_experiments_ext --groups G,H,I --seeds 0,1,2
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from .train import build_parser, run
from .run_experiments import flatten

BUNDLES = {
    "paper": "outputs/bundle_v2_paper.npz",      # mean/median-turbidity, 9 nodes
    "max": "outputs/bundle_v2_max.npz",          # daily maximum, 9 nodes, same days
    "nutrient": "outputs/bundle_v2_nutrient.npz",  # 13 nodes
}

BASE = dict(reg_loss="huber", cls_loss="focal", pos_sampler=True)


def matrix(seeds, seeds_small):
    M = []
    # ---- G: published graph baselines from the reference repository -------- #
    M += [("G", "mtgnn", seeds, {"bundle": "paper", "model": "mtgnn", **BASE}),
          ("G", "am_mtgnn", seeds, {"bundle": "paper", "model": "am_mtgnn", **BASE}),
          ("G", "mtgnn_nograph", seeds_small,
           {"bundle": "paper", "model": "mtgnn_nograph", **BASE}),
          ("G", "mthab_ref", seeds, {"bundle": "paper", "model": "mthab", **BASE})]

    # ---- H: daily aggregation protocol ------------------------------------ #
    M += [("H", "agg_mean", seeds_small, {"bundle": "paper", "model": "mthab", **BASE}),
          ("H", "agg_max", seeds_small, {"bundle": "max", "model": "mthab", **BASE}),
          ("H", "agg_mean_lstm", seeds_small,
           {"bundle": "paper", "model": "lstm", **BASE}),
          ("H", "agg_max_lstm", seeds_small, {"bundle": "max", "model": "lstm", **BASE})]

    # ---- I: nutrient-augmented variable graph ----------------------------- #
    M += [("I", "nodes9", seeds_small, {"bundle": "nutrient_ctrl", "model": "mthab",
                                        **BASE}),
          ("I", "nodes13", seeds, {"bundle": "nutrient", "model": "mthab", **BASE}),
          ("I", "nodes13_hybrid", seeds_small,
           {"bundle": "nutrient", "model": "mthab", "graph": "hybrid",
            "graph_l1": 1e-3, **BASE}),
          ("I", "nodes13_am_mtgnn", seeds_small,
           {"bundle": "nutrient", "model": "am_mtgnn", **BASE})]
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="outputs/runs")
    ap.add_argument("--results", default="outputs/results_ext.csv")
    ap.add_argument("--groups", default="G,H,I")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--seeds_small", default="0,1")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--channels", default="32,64,128")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--bo_params", default="outputs/bo_best.json",
                    help="if present, apply the Bayesian-optimised hyper-parameters")
    a = ap.parse_args()

    bo = {}
    if os.path.exists(a.bo_params):
        bo = json.load(open(a.bo_params))["best_params"]
        print("applying BO hyper-parameters:", bo)

    groups = a.groups.split(",")
    seeds = [int(s) for s in a.seeds.split(",")]
    seeds_small = [int(s) for s in a.seeds_small.split(",")]

    rows = pd.read_csv(a.results).to_dict("records") if os.path.exists(a.results) else []
    done = {(r["name"], r["seed"]) for r in rows}

    for group, name, sds, over in matrix(seeds, seeds_small):
        if group not in groups:
            continue
        for sd in sds:
            if (name, sd) in done:
                print(f"skip {name} s{sd}")
                continue
            tag = f"{group}_{name}_s{sd}"
            mj = os.path.join(a.outdir, tag, "metrics.json")
            if os.path.exists(mj):
                m = json.load(open(mj))
            else:
                p = build_parser().parse_args([])
                bkey = over.pop("bundle", "paper")
                # "nutrient_ctrl": the 9-node control that shares the 13-node sample set
                p.bundle = BUNDLES.get(bkey, BUNDLES["paper"]) if bkey != "nutrient_ctrl" \
                    else BUNDLES["paper"]
                p.outdir, p.tag, p.seed = a.outdir, tag, sd
                p.epochs, p.patience = a.epochs, a.patience
                p.channels, p.batch_size = a.channels, a.batch_size
                for k, v in bo.items():
                    setattr(p, k, v)
                for k, v in over.items():
                    setattr(p, k, v)
                over["bundle"] = bkey
                m = run(p)
            rows.append(flatten(name, group, sd, m))
            pd.DataFrame(rows).to_csv(a.results, index=False)

    print(f"\nwrote {len(rows)} rows -> {a.results}")


if __name__ == "__main__":
    main()
