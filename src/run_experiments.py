"""
run_experiments.py
==================
Executes the full experiment matrix and appends one row per run to
outputs/results.csv.  Runs that already have a metrics.json are skipped, so the
script is restartable.

Groups
------
A_baselines   persistence / climatology / HistGB / LSTM / GRU / TCN /
              Transformer / ST-GCN-lite / MT-HAB-G3D(MSE) / MT-HAB-G3D(Huber)
B_imbalance   focal + positive resampling, class-weighted BCE
C_graph       correlation vs. identity vs. complete vs. adaptive vs. hybrid
D_pathway     w/o G3D pathway, w/o factorised pathway
E_objective   fixed lambda vs. homoscedastic-uncertainty vs. DWA weighting,
              Huber vs. peak-weighted Huber
F_uncertainty quantile head (pinball) and MC-dropout

Usage
-----
python -m src.run_experiments --groups A,B,C,D,E,F --seeds 0,1,2
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd

# torch is imported lazily so that --plan / --dry_run and run_parallel's scheduling work on
# machines without PyTorch (revision pipeline).
def _train_api():
    from .train import build_parser, run
    return build_parser, run


def cfg(**kw):
    return kw


def matrix(seeds, seeds_small):
    """Returns a list of (group, name, seeds, overrides)."""
    M = []

    # ---------------- A: baselines & manuscript reproduction --------------- #
    M += [("A", "persistence", [0], cfg(model="persistence")),
          ("A", "climatology", [0], cfg(model="climatology")),
          ("A", "histgb", seeds_small, cfg(model="histgb")),
          ("A", "mlp", seeds_small, cfg(model="mlp")),
          ("A", "lstm", seeds, cfg(model="lstm", reg_loss="mse")),
          ("A", "gru", seeds_small, cfg(model="gru", reg_loss="mse")),
          ("A", "tcn", seeds_small, cfg(model="tcn", reg_loss="mse")),
          ("A", "transformer", seeds_small, cfg(model="transformer", reg_loss="mse")),
          ("A", "stgcn", seeds_small, cfg(model="stgcn", reg_loss="mse")),
          ("A", "mthab_mse", seeds, cfg(model="mthab", reg_loss="mse")),
          ("A", "mthab_huber", seeds, cfg(model="mthab", reg_loss="huber"))]

    # ---------------- B: class-imbalance handling -------------------------- #
    M += [("B", "mthab_huber_focal", seeds,
           cfg(model="mthab", reg_loss="huber", cls_loss="focal", pos_sampler=True)),
          ("B", "mthab_huber_wbce", seeds_small,
           cfg(model="mthab", reg_loss="huber", cls_loss="wbce")),
          ("B", "lstm_focal", seeds_small,
           cfg(model="lstm", reg_loss="mse", cls_loss="focal", pos_sampler=True))]

    # ---------------- C: graph construction -------------------------------- #
    base = cfg(model="mthab", reg_loss="huber", cls_loss="focal", pos_sampler=True)
    for g in ["identity", "complete", "adaptive", "hybrid"]:
        M.append(("C", f"graph_{g}", seeds_small, {**base, "graph": g,
                                                   "graph_l1": 1e-3 if g != "identity" else 0.0}))

    # ---------------- D: architectural pathways ---------------------------- #
    M += [("D", "no_g3d", seeds_small, {**base, "no_g3d": True}),
          ("D", "no_factorized", seeds_small, {**base, "no_factorized": True})]

    # ---------------- E: multi-task objective ------------------------------ #
    M += [("E", "weight_uncertainty", seeds_small, {**base, "weighting": "uncertainty"}),
          ("E", "weight_dwa", seeds_small, {**base, "weighting": "dwa"}),
          ("E", "peak_huber", seeds_small, {**base, "reg_loss": "peak_huber",
                                            "peak_gamma": 2.0})]

    # ---------------- G: persistence-anchored residual head ---------------- #
    M += [("G", "residual", seeds, {**base, "residual": True}),
          ("G", "residual_peak", seeds_small, {**base, "residual": True,
                                               "reg_loss": "peak_huber",
                                               "peak_gamma": 2.0}),
          ("G", "lstm_residual", seeds_small,
           cfg(model="lstm", reg_loss="mse", cls_loss="focal", pos_sampler=True,
               residual=True))]

    # ---------------- J: persistence-anchored (residual) target ------------ #
    M += [("J", "residual", seeds,
           {**base, "residual": True}),
          ("J", "residual_peak", seeds_small,
           {**base, "residual": True, "reg_loss": "peak_huber", "peak_gamma": 2.0}),
          ("J", "lstm_residual", seeds_small,
           {"model": "lstm", "reg_loss": "huber", "cls_loss": "focal",
            "pos_sampler": True, "residual": True})]

    # ---------------- F: predictive uncertainty ---------------------------- #
    M += [("F", "quantile", seeds_small, {**base, "reg_loss": "quantile",
                                          "n_quantiles": 3}),
          ("F", "mc_dropout", seeds_small, {**base, "mc_dropout": 30, "dropout": 0.2})]
    return M


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="outputs/bundle_ffill.npz")
    ap.add_argument("--outdir", default="outputs/runs")
    ap.add_argument("--results", default="outputs/results.csv")
    ap.add_argument("--groups", default="A,B,C,D,E,F,G")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--seeds_small", default="0,1")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--channels", default="32,64,128")
    ap.add_argument("--batch_size", type=int, default=256)
    a = ap.parse_args()

    groups = a.groups.split(",")
    seeds = [int(s) for s in a.seeds.split(",")]
    seeds_small = [int(s) for s in a.seeds_small.split(",")]

    rows = []
    if os.path.exists(a.results):
        rows = pd.read_csv(a.results).to_dict("records")
    done = {(r["name"], r["seed"]) for r in rows}

    for group, name, sds, over in matrix(seeds, seeds_small):
        if group not in groups:
            continue
        for sd in sds:
            if (name, sd) in done:
                print(f"skip {name} seed={sd}")
                continue
            tag = f"{group}_{name}_s{sd}"
            mj = os.path.join(a.outdir, tag, "metrics.json")
            if os.path.exists(mj):
                m = json.load(open(mj))
            else:
                build_parser, run = _train_api()
                p = build_parser().parse_args([])
                p.bundle, p.outdir, p.tag, p.seed = a.bundle, a.outdir, tag, sd
                p.epochs, p.patience = a.epochs, a.patience
                p.channels, p.batch_size = a.channels, a.batch_size
                for k, v in over.items():
                    setattr(p, k, v)
                m = run(p)
            rows.append(flatten(name, group, sd, m))
            pd.DataFrame(rows).to_csv(a.results, index=False)

    print(f"\nwrote {len(rows)} rows -> {a.results}")


if __name__ == "__main__":
    main()
