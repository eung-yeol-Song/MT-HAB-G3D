"""
run_parallel.py
===============
Parallel replacement for `run_table3`.  Same experiment matrix, same outputs —
but the (model, seed) jobs run concurrently instead of one after another.

    # 4090 + 16-core CPU, everything at once
    python -m src.run_parallel --single_bundle --force all --seeds 10 \
           --gpu_workers 4 --cpu_workers 4

    python -m src.run_parallel --plan            # show the schedule, run nothing
    python -m src.run_parallel --auto            # pick worker counts from this machine

Why processes and not threads
-----------------------------
Each job is a full training run.  Python threads would serialise on the GIL and
share one CUDA context; separate processes give real parallelism, isolate
crashes, and let CPU and GPU jobs run side by side.  A failed job therefore
costs one row, not the whole matrix.

Two forms of parallelism are exploited
--------------------------------------
1. **Across jobs.**  121 runs (13 models x 10 seeds) are independent.
2. **Across devices.**  The models here are small (63 k - 535 k parameters,
   batch 256), so a single 4090 sits mostly idle on one job.  Several jobs share
   the GPU, and the CPU runs more jobs at the same time.

Scheduling
----------
Longest-job-first.  MT-HAB-G3D is ~20x slower per epoch than every baseline, so
starting it last would leave the machine idle waiting for one straggler.  Heavy
jobs are queued first and land on the GPU; the light ones fill the CPU workers.

Thread budget
-------------
N CPU workers each grabbing every core is oversubscription and runs *slower*
than N workers with cores/N threads.  Each CPU job is launched with
`--threads` and OMP/MKL pinned accordingly.

Resume
------
A job whose `metrics.json` already exists is skipped, so an interrupted matrix
can be restarted with the same command.  `--force` re-runs regardless.
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

import pandas as pd

from .run_table3 import (TABLE3, BUNDLE_A, COMPACT_WIDTH,
                          seed_guard, write_manifest, MIN_PUBLISHABLE_SEEDS)
from .run_experiments import flatten

# relative cost, only used to order the queue (measured: 567 s vs ~25 s / epoch)
COST = {"mthab_mse": 100, "mthab_cls_only": 100, "mthab_reg_only": 100,
        "cnn_lstm": 6, "transformer": 5, "am_mtgnn": 4,
        "mtgnn": 4, "mtgnn_nograph": 4, "stgcn": 4, "tcn": 4, "lstm": 4,
        "gru": 4, "mlp": 3, "histgb": 3, "climatology": 1, "persistence": 1}
NO_TRAIN = {"persistence", "climatology", "histgb"}

_print_lock = threading.Lock()


def log(msg):
    with _print_lock:
        print(msg, flush=True)


# --------------------------------------------------------------------------- #
def build_jobs(a):
    """Same selection logic as run_table3, flattened to one entry per seed."""
    only = {s for s in a.only.split(",") if s}
    force = {s for s in a.force.split(",") if s}
    if "all" in force:
        force = {t[2] for t in TABLE3}

    todo = [t for t in TABLE3 if not only or t[2] in only]
    if a.single_bundle or a.bundle:
        one = a.bundle or BUNDLE_A
        todo = [(o, g, n, "results", one, k, ov) for o, g, n, _r, _b, k, ov in todo]
    if a.seeds > 0:
        todo = [(o, g, n, r, b, (1 if n in ("climatology", "persistence") else a.seeds), ov)
                for o, g, n, r, b, _k, ov in todo]

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

    jobs = []
    for order, group, name, rfile, bundle, n_seeds, over in todo:
        if a.proposed_width and name == "mthab_mse":
            over = {**over, "channels": a.proposed_width}
        for sd in range(n_seeds):
            tag = f"{group}_{name}_s{sd}"
            mj = os.path.join(a.outdir, tag, "metrics.json")
            if os.path.exists(mj) and name not in force:
                jobs.append(dict(tag=tag, name=name, group=group, seed=sd,
                                 rfile=rfile, bundle=bundle, over=over,
                                 done=True, cost=COST.get(name, 5)))
                continue
            jobs.append(dict(tag=tag, name=name, group=group, seed=sd,
                             rfile=rfile, bundle=bundle, over=over,
                             done=False, cost=COST.get(name, 5)))
    if a.epochs_from:
        import numpy as np
        ep_tab = {}
        for j in jobs:
            if j["name"] in NO_TRAIN:
                continue
            key = (j["group"], j["name"])
            if key not in ep_tab:
                best = []
                for sd in range(a.epochs_from_seeds):
                    hp = os.path.join(a.epochs_from, f"{j['group']}_{j['name']}_s{sd}", "history.json")
                    vs = json.load(open(hp))["val_score"]
                    b_, e_ = -1e9, 0
                    for i, v in enumerate(vs):          # same rule as the training loop
                        if v > b_ + 1e-5:
                            b_, e_ = v, i + 1
                    best.append(e_)
                ep_tab[key] = (int(np.median(best) + 0.5), best)
            j["stop_epoch"] = ep_tab[key][0]
        a._ep_tab = ep_tab
    jobs.sort(key=lambda j: -j["cost"])              # longest job first
    return jobs


def job_cmd(j, a, device, threads):
    cmd = [sys.executable, "-m", "src.train",
           "--bundle", j["bundle"], "--outdir", a.outdir, "--tag", j["tag"],
           "--seed", str(j["seed"]), "--device", device,
           "--epochs", str(a.epochs), "--patience", str(a.patience),
           "--batch_size", str(a.batch_size), "--channels", a.channels,
           "--select_metric", a.select_metric]
    if a.select_legacy:
        cmd.append("--select_legacy")
    if a.pos_weight_cap:
        cmd += ["--pos_weight_cap", str(a.pos_weight_cap)]
    if a.station_emb:
        cmd += ["--station_emb", str(a.station_emb)]
    if getattr(a, "nondeterministic", False):
        cmd.append("--nondeterministic")
    if threads:
        cmd += ["--threads", str(threads)]
    if j["name"] not in NO_TRAIN:                       # revision: artefacts for interpretation
        cmd += ["--save_state", "--save_val_preds"]
        if j["name"].startswith("mthab"):
            cmd += ["--perm_importance", "5"]
    # refit protocol: fixed epoch budget per model (median best epoch of the reference
    # seeds), optionally train + validation, thresholds from the matching train-only run
    if j.get("stop_epoch") and j["name"] not in NO_TRAIN:
        cmd += ["--stop_epoch", str(j["stop_epoch"])]
    if a.refit:
        cmd.append("--refit")
    if a.thr_from:
        cmd += ["--thr_from", os.path.join(a.thr_from, j["tag"], "metrics.json")]
    for k, v in j["over"].items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd += [f"--{k}", str(v)]
    return cmd


# --------------------------------------------------------------------------- #
def worker(slots, jobs_q, a, state):
    while True:
        try:
            j = jobs_q.get_nowait()
        except queue.Empty:
            return
        slot = slots.get()                       # (device, threads)
        device, threads = slot
        t0 = time.time()
        env = dict(os.environ)
        if device == "cpu" and threads:
            env.update(OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
        try:
            r = subprocess.run(job_cmd(j, a, device, threads if device == "cpu" else 0),
                               capture_output=True, text=True, env=env, cwd=os.getcwd())
            dt = time.time() - t0
            with state["lock"]:
                state["finished"] += 1
                k = state["finished"]
            if r.returncode == 0:
                tail = [l for l in r.stdout.strip().splitlines() if l.startswith("[")]
                log(f"  [{k:3d}/{state['total']}] OK   {j['tag']:22s} {device:5s} "
                    f"{dt:6.0f}s   {tail[-1][:88] if tail else ''}")
            else:
                state["failed"].append(j["tag"])
                err = (r.stderr.strip().splitlines() or ["?"])[-1]
                log(f"  [{k:3d}/{state['total']}] FAIL {j['tag']:22s} {device:5s} "
                    f"{dt:6.0f}s   {err[:100]}")
        finally:
            slots.put(slot)
            jobs_q.task_done()


def collect(jobs, a):
    """Read every metrics.json and rebuild the result CSVs."""
    tables = {}
    for rfile, path in (("results", a.results), ("results_ext", a.results_ext)):
        tables[rfile] = (pd.read_csv(path).to_dict("records")
                         if os.path.exists(path) and not a.fresh_csv else [])
    for j in jobs:
        mj = os.path.join(a.outdir, j["tag"], "metrics.json")
        if not os.path.exists(mj):
            continue
        m = json.load(open(mj))
        rows = tables[j["rfile"]]
        rows[:] = [r for r in rows
                   if not (r.get("name") == j["name"] and r.get("seed") == j["seed"])]
        rows.append(flatten(j["name"], j["group"], j["seed"], m))
    for rfile, path in (("results", a.results), ("results_ext", a.results_ext)):
        if tables[rfile]:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            pd.DataFrame(tables[rfile]).to_csv(path, index=False)
            log(f"{len(tables[rfile]):4d} rows -> {path}")


def autodetect():
    """Conservative worker counts for this machine."""
    cores = os.cpu_count() or 2
    gpus, gpu_mem = 0, 0
    try:
        import torch
        if torch.cuda.is_available():
            gpus = torch.cuda.device_count()
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    except Exception:
        pass
    if gpus:
        # ~1.2 GB per concurrent job (CUDA context + these small models)
        gpu_w = max(1, min(6, int(gpu_mem // 3), cores))
        cpu_w = max(0, min(4, (cores - gpu_w) // 2))
    else:
        gpu_w = 0
        cpu_w = max(1, min(8, cores // 2))
    return gpu_w, cpu_w, cores


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="outputs/runs")
    ap.add_argument("--results", default="outputs/results.csv")
    ap.add_argument("--results_ext", default="outputs/results_ext.csv")
    ap.add_argument("--only", default="")
    ap.add_argument("--force", default="")
    ap.add_argument("--seeds", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--channels", default=COMPACT_WIDTH)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--single_bundle", action="store_true")
    ap.add_argument("--bundle", default="",
                    help="use this bundle for every row (stage 2 of the revision: "
                         "re-run all baselines on the validation-selected input)")
    ap.add_argument("--imbalance", action="store_true")
    ap.add_argument("--cls_loss_all", default="", choices=["", "focal", "wbce"])
    ap.add_argument("--proposed_width", default="")
    ap.add_argument("--select_metric", default="auprc", choices=["auprc", "f1"])
    ap.add_argument("--select_legacy", action="store_true")
    ap.add_argument("--pos_weight_cap", type=float, default=0.0)
    ap.add_argument("--station_emb", type=int, default=0,
                    help="width of the learned per-station embedding (0 = off)")
    ap.add_argument("--nondeterministic", action="store_true",
                    help="pass --nondeterministic to src.train (results stop "
                         "being reproducible run-to-run)")
    ap.add_argument("--gpu_workers", type=int, default=-1,
                    help="concurrent jobs on the GPU (-1 = autodetect)")
    ap.add_argument("--cpu_workers", type=int, default=-1,
                    help="concurrent jobs on the CPU (-1 = autodetect)")
    ap.add_argument("--threads_per_cpu_worker", type=int, default=0,
                    help="0 = cores / cpu_workers")
    ap.add_argument("--gpu_ids", default="0", help="e.g. 0 or 0,1")
    ap.add_argument("--auto", action="store_true", help="autodetect worker counts")
    ap.add_argument("--plan", action="store_true", help="print the schedule and exit")
    ap.add_argument("--epochs_from", default="",
                    help="reference run directory: every trained model gets --stop_epoch = median "
                         "best validation epoch of its reference seeds (fixed budget, no selection)")
    ap.add_argument("--epochs_from_seeds", type=int, default=10)
    ap.add_argument("--refit", action="store_true", help="fit on train + validation (see src.train)")
    ap.add_argument("--thr_from", default="",
                    help="run directory whose per-run decision thresholds are reused (same tags)")
    ap.add_argument("--fresh_csv", action="store_true",
                    help="rebuild the CSVs from metrics.json only, ignoring old rows")
    a = ap.parse_args()
    seed_guard(a.seeds, "run_parallel")

    gpu_w, cpu_w, cores = autodetect()
    if not a.auto:
        if a.gpu_workers >= 0:
            gpu_w = a.gpu_workers
        if a.cpu_workers >= 0:
            cpu_w = a.cpu_workers
    if gpu_w + cpu_w == 0:
        cpu_w = 1
    thr = a.threads_per_cpu_worker or max(1, cores // max(cpu_w, 1))

    jobs = build_jobs(a)
    pending = [j for j in jobs if not j["done"]]
    skipped = len(jobs) - len(pending)

    gpu_ids = [g for g in a.gpu_ids.split(",") if g]
    print("=" * 70)
    print(" MT-HAB parallel runner")
    print(f"   CPU cores        : {cores}")
    print(f"   GPU workers      : {gpu_w}" + (f"  (cuda:{','.join(gpu_ids)})" if gpu_w else ""))
    print(f"   CPU workers      : {cpu_w}   ({thr} threads each)")
    print(f"   jobs             : {len(pending)} to run, {skipped} already done")
    print(f"   seeds/model      : {a.seeds or 'table default (10)'}"
          + ("" if (a.seeds or 10) >= MIN_PUBLISHABLE_SEEDS else "   <-- NOT PUBLISHABLE"))
    print(f"   determinism      : {'on' if not a.nondeterministic else 'OFF'}")
    print("=" * 70)

    if getattr(a, "_ep_tab", None):
        print(" fixed epoch budget per model (median best epoch of the reference seeds):")
        for (g, n), (e, best) in a._ep_tab.items():
            print(f"   {g}_{n:16s} stop_epoch={e:3d}   reference best epochs {best}")
    if a.plan:
        print(f"\n{'#':>4}  {'tag':24s} {'cost':>5}")
        for i, j in enumerate(pending, 1):
            print(f"{i:4d}  {j['tag']:24s} {j['cost']:5d}")
        seq = sum(j["cost"] for j in pending)
        par = seq / max(gpu_w + cpu_w, 1)
        print(f"\nrelative cost: sequential {seq}, ideal parallel ~{par:.0f} "
              f"({gpu_w + cpu_w} workers)")
        print("longest single job sets the floor:", max((j['cost'] for j in pending),
                                                        default=0))
        return

    if not pending:
        print("\nnothing to run - rebuilding the CSVs from existing metrics.json")
        collect(jobs, a)
        return

    slots = queue.Queue()
    for i in range(gpu_w):
        slots.put((f"cuda:{gpu_ids[i % len(gpu_ids)]}", 0))
    for _ in range(cpu_w):
        slots.put(("cpu", thr))

    jq = queue.Queue()
    for j in pending:
        jq.put(j)

    state = {"total": len(pending), "finished": 0, "failed": [],
             "lock": threading.Lock()}
    t0 = time.time()
    threads = [threading.Thread(target=worker, args=(slots, jq, a, state), daemon=True)
               for _ in range(gpu_w + cpu_w)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    dt = time.time() - t0
    print("\n" + "=" * 70)
    print(f" {state['finished']} jobs in {dt / 60:.1f} min "
          f"({dt / max(state['finished'], 1):.0f} s/job wall-clock)")
    if state["failed"]:
        print(f" FAILED: {', '.join(state['failed'])}")
    print("=" * 70)
    collect(jobs, a)
    write_manifest(a.outdir, a.seeds or 10,
                   "imbalance" if a.imbalance else (a.cls_loss_all or "standard"),
                   BUNDLE_A if a.single_bundle else "mixed",
                   extra=dict(launcher="run_parallel", gpu_workers=gpu_w,
                              cpu_workers=cpu_w, epochs=a.epochs,
                              patience=a.patience, station_emb=a.station_emb,
                              deterministic=not a.nondeterministic,
                              failed=state["failed"]))
    if state["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
