"""
run_table3.py
=============
Runs *exactly* the 13 models of the manuscript's extended Table 3 and nothing
else.  MT-HAB-G3D is the proposed model; the Huber variant was withdrawn from
the comparison, so only the MSE-trained proposal remains.

    #   key               row label in the table
    --  ----------------  -----------------------------------------
     1  climatology       Day-of-year climatology
     2  histgb            Gradient-boosted trees
     3  mlp               MLP
     4  lstm              LSTM (manuscript baseline)
     5  gru               GRU
     6  cnn_lstm          CNN-LSTM                       <-- NEW
     7  tcn               Dilated TCN
     8  transformer       Transformer
     9  stgcn             ST-GCN (no G3D)
    10  mtgnn             MTGNN
    11  am_mtgnn          AM-MTGNN
    12  mtgnn_nograph     MTGNN w/o graph
    13  mthab_mse         MT-HAB-G3D (proposed)

Restartable: a (name, seed) pair that already appears in the results CSV, or
that already has an outputs/runs/<tag>/metrics.json, is skipped.  So on an
existing checkout this script effectively trains **only CNN-LSTM** and leaves
the 13 published rows untouched.

Two bundles are involved, exactly as in the original experiment scripts:
  * rows 1-9, 13     : outputs/bundle_ffill.npz      (group A protocol)
  * rows 10-12       : outputs/bundle_v2_paper.npz   (group G protocol)
CNN-LSTM is placed in group A next to LSTM/GRU/TCN/Transformer and therefore
uses the same bundle, the same losses and the same seed count as the LSTM row
it is meant to be compared against.

Usage
-----
    python -m src.run_table3                        # CPU
    python -m src.run_table3 --device cuda          # single GPU
    python -m src.run_table3 --only cnn_lstm        # just the new model
    python -m src.run_table3 --force cnn_lstm       # retrain even if present
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
from .run_experiments import flatten

# --------------------------------------------------------------------------- #
# Revision pipeline (v13): ONE bundle for every row - 9 variable nodes of the centre station,
# physical QC, <=3-day forward-fill + missingness channel, log1p target, T = 10, 2-NN graph.
BUNDLE_A = "outputs/bundle_main.npz"
BUNDLE_G = BUNDLE_A

SEQ = dict(reg_loss="mse", cls_loss="bce")                       # group-A sequence models
# Revision: the MTGNN family is trained under the SAME objective as every other neural model
# (the manuscript states so; the v12 runs used focal + Huber + oversampling for these rows).
GRAPHREF = dict(**SEQ)

# Capacity note
# -----------------
# v5 restored the published width 96,192,384 (3.68 M parameters).  On this
# 12,701-sample training set that width proved unstable: 1 seed in 10 diverged
# (D+6 R2 = -5.26) and dragged the D+1..D+7 mean RMSE from 3.56 (median) to
# 4.08.  v6 therefore defaults back to the compact width, which was the best
# configuration measured so far, and exposes --proposed_width for the capacity
# ablation rather than hard-coding a guess.
COMPACT_WIDTH = "32,64,128"
PUBLISHED_WIDTH = "96,192,384"
DEEP = dict(channels=COMPACT_WIDTH, epochs=60, patience=15)

# --------------------------------------------------------------------------- #
MIN_PUBLISHABLE_SEEDS = 10


def seed_guard(seeds: int, where: str = "") -> None:
    """Refuse to let a short run be mistaken for the real thing.

    The v11_2 results looked like a large improvement over v11_1 (proposal mean
    RMSE 3.33 vs 3.81) purely because it had been launched with 3 seeds instead
    of 10: on this dataset 1-2 seeds in 10 diverge, so a 3-seed mean routinely
    misses them.  Nothing in the outputs said "3 seeds", so the comparison went
    unnoticed for a whole round of analysis.  This prints the warning that was
    missing, and run_manifest.json records the count next to the numbers.
    """
    if seeds and seeds < MIN_PUBLISHABLE_SEEDS:
        print("\n" + "!" * 68)
        print(f"!! WARNING: seeds = {seeds}  (< {MIN_PUBLISHABLE_SEEDS})"
              + (f"   [{where}]" if where else ""))
        print("!! On this dataset 1-2 seeds in 10 diverge (R2 << 0) and 8 of 12")
        print("!! learned models lose the red-tide task entirely on at least one")
        print("!! seed.  A mean over few seeds is dominated by which failures it")
        print("!! happened to miss and MUST NOT be reported in the manuscript.")
        print("!! Use it for smoke tests only.")
        print("!" * 68 + "\n")


def write_manifest(outdir: str, seeds: int, mode: str, bundle: str, extra=None):
    """Stamp every run directory with what produced it."""
    import datetime
    man = dict(seeds=seeds, mode=mode, bundle=bundle,
               min_publishable_seeds=MIN_PUBLISHABLE_SEEDS,
               publishable=bool(seeds >= MIN_PUBLISHABLE_SEEDS),
               created=datetime.datetime.now().isoformat(timespec="seconds"),
               version=open("VERSION").readline().strip() if os.path.exists("VERSION") else "")
    if extra:
        man.update(extra)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "run_manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    return man


# (order, group, name, results_file, bundle, n_seeds, overrides)
DETERMINISTIC = ("persistence", "climatology")      # no training, one seed is enough

TABLE3 = [
    (0,  "A", "persistence",   "results",     BUNDLE_A, 1, dict(model="persistence")),
    (1,  "A", "climatology",   "results",     BUNDLE_A, 1, dict(model="climatology")),
    (2,  "A", "histgb",        "results",     BUNDLE_A, 10, dict(model="histgb")),
    (3,  "A", "mlp",           "results",     BUNDLE_A, 10, dict(model="mlp", **SEQ)),   # revision: v12 trained MLP with the default Huber loss
    (4,  "A", "lstm",          "results",     BUNDLE_A, 10, dict(model="lstm", **SEQ)),
    (5,  "A", "gru",           "results",     BUNDLE_A, 10, dict(model="gru", **SEQ)),
    (6,  "A", "cnn_lstm",      "results",     BUNDLE_A, 10, dict(model="cnn_lstm", **SEQ)),
    (7,  "A", "tcn",           "results",     BUNDLE_A, 10, dict(model="tcn", **SEQ)),
    (8,  "A", "transformer",   "results",     BUNDLE_A, 10, dict(model="transformer", **SEQ)),
    (9,  "A", "stgcn",         "results",     BUNDLE_A, 10, dict(model="stgcn", **SEQ)),
    (10, "G", "mtgnn",         "results_ext", BUNDLE_G, 10, dict(model="mtgnn", **GRAPHREF)),
    (11, "G", "am_mtgnn",      "results_ext", BUNDLE_G, 10, dict(model="am_mtgnn", **GRAPHREF)),
    (12, "G", "mtgnn_nograph", "results_ext", BUNDLE_G, 10, dict(model="mtgnn_nograph", **GRAPHREF)),
    (13, "A", "mthab_mse",     "results",     BUNDLE_A, 10, dict(model="mthab", reg_loss="mse", **DEEP)),
    # revision, R2-M2: otherwise-identical single-task controls
    (14, "A", "mthab_cls_only", "results",    BUNDLE_A, 10, dict(model="mthab", reg_loss="mse", task="cls", **DEEP)),
    (15, "A", "mthab_reg_only", "results",    BUNDLE_A, 10, dict(model="mthab", reg_loss="mse", task="reg", **DEEP)),
]


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="outputs/runs")
    ap.add_argument("--results", default="outputs/results.csv")
    ap.add_argument("--results_ext", default="outputs/results_ext.csv")
    ap.add_argument("--only", default="",
                    help="comma-separated subset of model keys (default: all 13)")
    ap.add_argument("--force", default="",
                    help="comma-separated keys to retrain even if results exist")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--channels", default="32,64,128")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--device", default="cpu", help="cpu | cuda | cuda:0 | auto")
    ap.add_argument("--single_bundle", action="store_true",
                    help="evaluate EVERY row on outputs/bundle_ffill.npz. Fixes the "
                         "protocol inconsistency in the published results, where the "
                         "MTGNN family and some seeds were scored on a different "
                         "test set (see PROTOCOL_AUDIT_KO.md). Use together with "
                         "--force all so every row is recomputed under it.")
    ap.add_argument("--proposed_width", default="",
                    help="channels for MT-HAB-G3D, e.g. 32,64,128 (default) or "
                         "96,192,384. Use for the capacity ablation.")
    ap.add_argument("--select_metric", default="auprc", choices=["auprc", "f1"],
                    help="validation metric for early stopping (v6 default auprc)")
    ap.add_argument("--select_legacy", action="store_true",
                    help="reproduce the v5 selection rule (D+1 only, F1)")
    ap.add_argument("--station_emb", type=int, default=0,
                    help="width of the learned per-station embedding (0 = off)")
    ap.add_argument("--pos_weight_cap", type=float, default=0.0,
                    help="cap on the positive-oversampling weight (0 = uncapped)")
    ap.add_argument("--cls_loss_all", default="",
                    choices=["", "focal", "wbce"],
                    help="apply this classification loss to EVERY learned model "
                         "WITHOUT touching the sampler. Milder than --imbalance: "
                         "it reweights the loss but leaves the training "
                         "distribution (and therefore the regression head) alone.")
    ap.add_argument("--imbalance", action="store_true",
                    help="apply --cls_loss focal --pos_sampler to EVERY learned "
                         "model (never to the proposal alone). The red-tide task "
                         "has ~81 positives in training, so plain BCE collapses "
                         "MCC/TSS to zero for most models.")
    ap.add_argument("--seeds", type=int, default=0,
                    help="override the per-row seed count "
                         "(0 = use the table default: 10, or 1 for the "
                         "deterministic climatology reference)")
    ap.add_argument("--nondeterministic", action="store_true",
                    help="pass --nondeterministic to src.train (results stop "
                         "being reproducible run-to-run)")
    ap.add_argument("--dry_run", action="store_true",
                    help="print the run plan and exit without training")
    a = ap.parse_args()
    seed_guard(a.seeds, "run_table3")

    only = {s for s in a.only.split(",") if s}
    force = {s for s in a.force.split(",") if s}
    if "all" in force:
        force = {t[2] for t in TABLE3}

    paths = {"results": a.results, "results_ext": a.results_ext}
    tables = {k: (pd.read_csv(v).to_dict("records") if os.path.exists(v) else [])
              for k, v in paths.items()}
    done = {k: {(r["name"], r["seed"]) for r in v} for k, v in tables.items()}

    todo = [t for t in TABLE3 if not only or t[2] in only]
    if a.single_bundle:
        # one test set for every row; MTGNN rows move from BUNDLE_G to BUNDLE_A
        todo = [(o, g, n, "results", BUNDLE_A, k, ov) for o, g, n, _r, _b, k, ov in todo]
    if a.seeds > 0:
        # climatology has no random component - repeating it N times would just
        # write N identical rows, so it stays at one seed whatever is requested
        todo = [(o, g, n, r, b, (1 if n in DETERMINISTIC else a.seeds), ov)
                for o, g, n, r, b, _k, ov in todo]
    # persistence / climatology have no training step and histgb already uses
    # class_weight="balanced", so none is touched by either option
    SKIP_IMB = ("persistence", "climatology", "histgb")
    if a.imbalance:
        todo = [(o, g, n, r, b, k,
                 ({**ov, "cls_loss": "focal", "pos_sampler": True}
                  if n not in SKIP_IMB else ov))
                for o, g, n, r, b, k, ov in todo]
    elif a.cls_loss_all:
        todo = [(o, g, n, r, b, k,
                 ({**ov, "cls_loss": a.cls_loss_all} if n not in SKIP_IMB else ov))
                for o, g, n, r, b, k, ov in todo]
    print(f"Table 3 runner - {len(todo)} of {len(TABLE3)} rows selected, "
          f"device={a.device}"
          + ("  [single-bundle]" if a.single_bundle else "")
          + ("  [imbalance: focal + resampling for all]" if a.imbalance else "")
          + (f"  [cls_loss={a.cls_loss_all} for all, no resampling]"
             if a.cls_loss_all and not a.imbalance else "")
          + "\n")

    if a.dry_run:
        total = sum(k for *_x, k, _o in [(o, g, n, r, b, k, ov)
                                         for o, g, n, r, b, k, ov in todo])
        print(f"{'#':>3}  {'model':14s} {'seeds':>5}  bundle")
        for o, g, n, r, b, k, ov in todo:
            print(f"{o:3d}  {n:14s} {k:5d}  {os.path.basename(b)}"
                  + (f"   channels={ov['channels']}" if "channels" in ov else ""))
        print(f"\ntotal runs: {total}")
        return

    for order, group, name, rfile, bundle, n_seeds, over in todo:
        for sd in range(n_seeds):
            if (name, sd) in done[rfile] and name not in force:
                print(f"  [{order:2d}] skip   {name:14s} seed={sd}  (already in {rfile}.csv)")
                continue
            tag = f"{group}_{name}_s{sd}"
            if name in force:
                # a forced re-run REPLACES the stored row instead of appending a
                # duplicate, so the CSV never holds two rows for one (name, seed)
                tables[rfile] = [r for r in tables[rfile]
                                 if not (r["name"] == name and r["seed"] == sd)]
            mj = os.path.join(a.outdir, tag, "metrics.json")
            if os.path.exists(mj) and name not in force:
                print(f"  [{order:2d}] reuse  {name:14s} seed={sd}  ({tag}/metrics.json)")
                m = json.load(open(mj))
            else:
                if not os.path.exists(bundle):
                    print(f"  [{order:2d}] SKIP   {name:14s} seed={sd}  "
                          f"-- bundle missing: {bundle}")
                    continue
                print(f"  [{order:2d}] train  {name:14s} seed={sd}  -> {tag}")
                build_parser, run = _train_api()
                p = build_parser().parse_args([])
                p.bundle, p.outdir, p.tag, p.seed = bundle, a.outdir, tag, sd
                p.epochs, p.patience = a.epochs, a.patience
                p.channels, p.batch_size = a.channels, a.batch_size
                p.device = a.device
                p.select_metric = a.select_metric
                p.nondeterministic = a.nondeterministic
                p.select_legacy = a.select_legacy
                p.pos_weight_cap = a.pos_weight_cap
                p.station_emb = a.station_emb
                if a.proposed_width and name == "mthab_mse":
                    over = {**over, "channels": a.proposed_width}
                for k, v in over.items():
                    setattr(p, k, v)
                m = run(p)
            tables[rfile].append(flatten(name, group, sd, m))
            pd.DataFrame(tables[rfile]).to_csv(paths[rfile], index=False)

    write_manifest(a.outdir, a.seeds or 10,
                   "imbalance" if a.imbalance else "standard",
                   BUNDLE_A if a.single_bundle else "mixed",
                   extra=dict(launcher="run_table3", device=a.device,
                              epochs=a.epochs, patience=a.patience,
                              deterministic=not a.nondeterministic))
    for k, v in paths.items():
        print(f"\n{len(tables[k])} rows -> {v}")
    print("\nNow build the tables and figures:")
    print("  python -m src.report")
    print("  python -m src.figures_paper")


if __name__ == "__main__":
    main()
