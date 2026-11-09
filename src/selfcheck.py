"""
selfcheck.py
============
Run this BEFORE a multi-hour experiment.  It exercises every stage of the
pipeline on a tiny budget and reports what would break, so a 3-hour job does not
die at hour 2 on a missing package or a shape bug.

    python -m src.selfcheck                 # ~3-6 min on CPU, ~1 min on GPU
    python -m src.selfcheck --device cuda
    python -m src.selfcheck --quick         # skip the per-architecture training

Checks
------
  1  python packages           torch / pandas / sklearn / matplotlib / tabulate ...
  2  compute device            CUDA visible? which GPU? how much memory?
  3  data                      bundle (built on the spot if missing), shapes,
                               split sizes, class balance
  4  every model architecture  one forward+backward pass, 1 epoch, all 13 keys
  4b every launcher flag       run_parallel's command builder vs src.train's parser
  4c determinism               the same seed run twice must give the same numbers
  5  reporting stack           report / tables_pdf / figures_paper / figures_journal
  6  disk space                enough room for the run you are about to launch

Exit code is 0 only when every check passes, so it can gate a shell script:

    python -m src.selfcheck --device cuda && bash run_local_full.sh cuda 10
"""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
import subprocess
import sys
import tempfile
import time

OK, WARN, FAIL = "  OK  ", " WARN ", " FAIL "
_state = {"fail": 0, "warn": 0}


def line(status, name, detail=""):
    if status == FAIL:
        _state["fail"] += 1
    elif status == WARN:
        _state["warn"] += 1
    print(f"[{status}] {name:34s} {detail}")


def section(t):
    print(f"\n{'-' * 72}\n{t}\n{'-' * 72}")


# --------------------------------------------------------------------------- #
def check_packages():
    section("1. Python packages")
    print(f"       python {sys.version.split()[0]}")
    required = ["torch", "numpy", "pandas", "scipy", "sklearn", "matplotlib"]
    optional = ["tabulate", "pyarrow"]
    for m in required:
        try:
            mod = importlib.import_module(m)
            line(OK, m, getattr(mod, "__version__", ""))
        except ImportError as e:
            line(FAIL, m, f"missing - pip install {m}  ({e})")
    for m in optional:
        try:
            mod = importlib.import_module(m)
            line(OK, m + " (optional)", getattr(mod, "__version__", ""))
        except ImportError:
            line(WARN, m + " (optional)",
                 "missing - markdown tables / parquet output will be skipped")


def check_device(want):
    section("2. Compute device")
    try:
        import torch
    except ImportError:
        line(FAIL, "torch", "not installed")
        return "cpu"
    avail = torch.cuda.is_available()
    line(OK, "cuda available", str(avail))
    if avail:
        i = torch.cuda.current_device()
        p = torch.cuda.get_device_properties(i)
        line(OK, "gpu", f"{p.name}  {p.total_memory / 1e9:.1f} GB  sm_{p.major}{p.minor}")
    if want.startswith("cuda") and not avail:
        line(FAIL, "requested device", f"{want} but CUDA is not available")
        return "cpu"
    if want == "cpu" and avail:
        line(WARN, "requested device", "cpu, though a GPU is available")
    return want


def dataset_args_from_env():
    """The same knobs run_local_full.sh reads, so a standalone selfcheck builds
    exactly the bundle the next experiment will train on."""
    knn = int(os.environ.get("KNN", "2"))
    sknn = int(os.environ.get("STATION_KNN", "2"))
    logt = os.environ.get("LOG_TARGET", "1") == "1"
    win = int(os.environ.get("WINDOW", "10"))
    a = ["--impute", "ffill", "--mask_channel"]
    a += ["--knn", str(knn)] if knn > 0 else ["--tau", "0.30"]
    if sknn > 0:
        a += ["--station_knn", str(sknn)]
    if logt:
        a += ["--log_target"]
    a += ["--T", str(win)]
    return a


def ensure_bundle(bundle, build=True):
    """Build the bundle when it is missing.  The package ships only the daily
    table (a prebuilt bundle would silently override the ablation the user
    selected), so a standalone selfcheck has to be able to make one."""
    if os.path.exists(bundle):
        return True
    daily_csv = "outputs/daily_station_day.csv"
    daily_pq = "outputs/daily_station_day.parquet"
    if not build:
        line(FAIL, "bundle", f"{bundle} not found")
        return False
    if not os.path.exists(daily_csv):
        if os.path.exists(daily_pq):
            import pandas as pd
            pd.read_parquet(daily_pq).to_csv(daily_csv, index=False)
            line(OK, "daily table", f"{daily_pq} -> csv")
        else:
            line(FAIL, "bundle", f"{bundle} missing and no daily table; "
                                 f"run src.data.build_daily on data/raw first")
            return False
    args = dataset_args_from_env()
    line(OK, "bundle", f"not found - building it now  ({' '.join(args)})")
    r = subprocess.run([sys.executable, "-m", "src.data.dataset",
                        "--daily", daily_csv, "--out", bundle, *args],
                       capture_output=True, text=True, cwd=os.getcwd())
    if r.returncode != 0:
        line(FAIL, "bundle build",
             (r.stderr.strip().splitlines() or ["?"])[-1][:120])
        return False
    for ln in r.stdout.splitlines():
        if ln.startswith(("station graph", "samples total", "nodes ")):
            line(OK, "  " + ln.split(":")[0][:30], ln)
    return True


def check_data(bundle, build=True):
    section("3. Data")
    if not ensure_bundle(bundle, build):
        return None
    import numpy as np
    from .data.dataset import Bundle
    b = Bundle.load(bundle)
    line(OK, "bundle", f"{bundle}  ({os.path.getsize(bundle) / 1e6:.1f} MB)")
    line(OK, "X shape", f"{b.X.shape}   (n, T, nodes, channels)")
    line(OK, "horizons", f"D+1 .. D+{b.y_cls.shape[1]}")
    for sp in ("train", "val", "test"):
        i = b.idx(sp)
        if len(i) == 0:
            line(FAIL, f"split {sp}", "empty")
            continue
        pos = int(b.y_cls[i][:, 0][b.m_cls[i][:, 0]].sum())
        chl = int(b.m_reg[i][:, 0].sum())
        line(OK, f"split {sp}", f"n={len(i):6d}  positives={pos:4d}  chl-valid={chl:5d}")
    tr = b.idx("train")
    y = b.y_cls[tr][:, 0]
    npos = max(int(y.sum()), 1)
    line(WARN if npos < 200 else OK, "class balance",
         f"{npos} positives in training ({npos / len(y) * 100:.2f}%)"
         + ("  - selection on F1 is noisy at this size" if npos < 200 else ""))
    line(OK, "variable graph", f"{int(b.A.sum() / 2)} undirected edges over {b.A.shape[0]} nodes")
    return b


def check_models(bundle, device, quick):
    section("4. Model architectures (1 epoch each)")
    if quick:
        line(WARN, "skipped", "--quick was passed")
        return
    from .run_table3 import TABLE3
    from .train import build_parser, run
    keys = [t[2] for t in TABLE3]
    over = {t[2]: t[6] for t in TABLE3}
    tmp = tempfile.mkdtemp(prefix="selfcheck_")
    try:
        for k in keys:
            p = build_parser().parse_args([])
            p.bundle, p.outdir, p.tag, p.seed = bundle, tmp, f"chk_{k}", 0
            p.epochs, p.patience, p.batch_size = 1, 1, 256
            p.channels, p.device = "16,32,64", device
            for kk, vv in over[k].items():
                if kk == "channels":
                    continue                      # keep the check cheap
                setattr(p, kk, vv)
            t0 = time.time()
            try:
                import contextlib, io
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    m = run(p)
                r = m["full_year"]["D+1"]
                line(OK, k, f"RMSE={r['RMSE']:.2f}  params={m.get('n_params', 0):,}"
                            f"  {time.time() - t0:.0f}s/epoch")
            except Exception as e:
                line(FAIL, k, f"{type(e).__name__}: {e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_reporting(bundle, quick):
    section("5. Reporting stack")
    from .run_table3 import TABLE3
    from .train import build_parser, run
    tmp = tempfile.mkdtemp(prefix="selfcheck_rep_")
    try:
        import contextlib, io, pandas as pd
        from .run_experiments import flatten
        rows = []
        # two cheap models so the tables have something to rank
        for k, ov in [("climatology", dict(model="climatology")),
                      ("mlp", dict(model="mlp"))]:
            p = build_parser().parse_args([])
            p.bundle, p.outdir, p.tag, p.seed = bundle, os.path.join(tmp, "runs"), f"A_{k}_s0", 0
            p.epochs, p.patience, p.channels = 1, 1, "16,32,64"
            for kk, vv in ov.items():
                setattr(p, kk, vv)
            with contextlib.redirect_stdout(io.StringIO()):
                rows.append(flatten(k, "A", 0, run(p)))
        res = os.path.join(tmp, "results.csv")
        pd.DataFrame(rows).to_csv(res, index=False)
        line(OK, "synthetic results.csv", f"{len(rows)} rows")

        env = dict(os.environ, MPLBACKEND="Agg")
        stages = [
            ("src.report", ["--results", res, "--runs", os.path.join(tmp, "runs"),
                            "--out", os.path.join(tmp, "report")]),
            ("src.tables_pdf", ["--results", res, "--out", os.path.join(tmp, "tables")]),
            ("src.figures_paper", ["--results", res, "--runs", os.path.join(tmp, "runs"),
                                   "--out", os.path.join(tmp, "figures")]),
            ("src.figures_journal", ["--results", res, "--out", os.path.join(tmp, "journal")]),
        ]
        for mod, args in stages:
            r = subprocess.run([sys.executable, "-m", mod, *args],
                               capture_output=True, text=True, env=env, cwd=os.getcwd())
            if r.returncode == 0:
                n = sum(len([f for f in fs if f.endswith(".pdf")])
                        for _d, _s, fs in os.walk(tmp))
                line(OK, mod, f"exit 0   ({n} PDFs so far)")
            else:
                line(FAIL, mod, (r.stderr.strip().splitlines() or ["?"])[-1][:110])
        if not quick:
            r = subprocess.run([sys.executable, "-m", "src.figures_paper",
                                "--results", res, "--runs", os.path.join(tmp, "runs"),
                                "--out", os.path.join(tmp, "fig_all"), "--all_horizons"],
                               capture_output=True, text=True, env=env, cwd=os.getcwd())
            line(OK if r.returncode == 0 else FAIL, "figures_paper --all_horizons",
                 "exit 0" if r.returncode == 0 else
                 (r.stderr.strip().splitlines() or ["?"])[-1][:110])
    except Exception as e:
        line(FAIL, "reporting stack", f"{type(e).__name__}: {e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_cli():
    """Every flag run_parallel forwards must exist in src.train's parser.

    This check exists because v11 shipped with --station_emb defined in
    run_parallel's *command builder* but missing from its *argparse*, so the
    first real run died on `unrecognized arguments: --station_emb 8`.  A CLI
    mismatch between the launcher and the trainer is invisible to a syntax check
    and to any run that does not use the new flag, so it gets its own test.
    """
    section("4b. CLI consistency (launcher <-> trainer)")
    try:
        from .train import build_parser
        from . import run_parallel as RP
        from .run_table3 import TABLE3, BUNDLE_A

        # every option the parallel launcher itself accepts
        rp_args = RP.main.__globals__  # noqa: F841  (kept for clarity)
        import argparse
        probe = argparse.ArgumentParser()
        # rebuild run_parallel's parser by running it with --plan on a dry job set
        # and instead simply introspect the command it would emit:
        class A:  # minimal stand-in for the parsed args run_parallel uses
            outdir = "/tmp"; epochs = 1; patience = 1; batch_size = 8
            channels = "16,32,64"; select_metric = "auprc"; select_legacy = False
            pos_weight_cap = 0.0; station_emb = 8; nondeterministic = False
        job = dict(tag="probe", name="mthab", group="A", seed=0,
                   bundle=BUNDLE_A, over={t[2]: t[6] for t in TABLE3}["mthab_mse"])
        cmd = RP.job_cmd(job, A(), "cpu", 1)
        flags = [c for c in cmd if c.startswith("--")]
        tp = build_parser()
        known = {a.option_strings[0] for a in tp._actions if a.option_strings}
        missing = [f for f in flags if f not in known]
        if missing:
            line(FAIL, "run_parallel -> train", f"train.py rejects: {', '.join(missing)}")
        else:
            line(OK, "run_parallel -> train", f"{len(flags)} flags all accepted")

        # the launcher must also accept what run_local_full.sh passes to it
        import subprocess as sp
        h = sp.run([sys.executable, "-m", "src.run_parallel", "--help"],
                   capture_output=True, text=True, cwd=os.getcwd()).stdout
        for f in ("--station_emb", "--single_bundle", "--seeds", "--gpu_workers",
                  "--cpu_workers", "--imbalance", "--cls_loss_all",
                  "--nondeterministic"):
            line(OK if f in h else FAIL, f"run_parallel {f}",
                 "" if f in h else "not in --help")
    except Exception as e:
        line(FAIL, "CLI consistency", f"{type(e).__name__}: {e}")


def check_determinism(bundle, device):
    """Two identical short runs must produce identical numbers.

    This is the check that would have caught the v11_1 / v11_2 confusion: the
    two runs differed only in seed count, but because training was not
    reproducible there was no way to prove that from the outputs.  A regression
    here (someone adds an unseeded shuffle, or a non-deterministic kernel slips
    in) silently makes every comparison in the manuscript unverifiable, so it
    gets its own gate.
    """
    section("4c. Determinism (same seed -> same numbers)")
    import tempfile as tf
    from .train import build_parser, run
    try:
        with tf.TemporaryDirectory() as td:
            outs = []
            for rep in range(2):
                p = build_parser().parse_args([])
                p.bundle, p.outdir, p.tag = bundle, td, f"det_rep{rep}"
                p.model, p.reg_loss, p.cls_loss = "lstm", "mse", "bce"
                p.seed, p.epochs, p.patience = 0, 1, 1
                p.batch_size, p.channels, p.device = 64, "16,32,64", device
                p.verbose = False
                m = run(p)
                outs.append(m["full_year"]["D+1"])
            keys = ["RMSE", "R2", "AUPRC"]
            diffs = {k: abs(float(outs[0][k]) - float(outs[1][k])) for k in keys
                     if k in outs[0] and outs[0][k] is not None}
            worst = max(diffs.values()) if diffs else 0.0
            det = "  ".join(f"{k}={outs[0][k]:.6f}/{outs[1][k]:.6f}" for k in diffs)
            if worst == 0.0:
                line(OK, "repeat run identical", det)
            elif worst < 1e-6:
                line(OK, "repeat run identical", f"max |diff| = {worst:.2e}   {det}")
            else:
                line(FAIL, "repeat run differs",
                     f"max |diff| = {worst:.3e} - training is NOT reproducible; "
                     f"do not compare two runs of this build.  {det}")
    except Exception as e:
        line(WARN, "determinism", f"{type(e).__name__}: {e}")


def check_disk():
    section("6. Disk")
    st = shutil.disk_usage(".")
    free = st.free / 1e9
    line(FAIL if free < 2 else (WARN if free < 5 else OK), "free space",
         f"{free:.1f} GB   (a 13-model x 10-seed run writes ~1-2 GB of preds)")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="outputs/bundle_ffill.npz")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no_build", action="store_true",
                    help="fail instead of building a missing bundle")
    ap.add_argument("--quick", action="store_true",
                    help="skip the per-architecture training pass")
    ap.add_argument("--no_determinism_check", action="store_true",
                    help="skip 4c (it trains two 1-epoch LSTMs, ~1 min on CPU)")
    a = ap.parse_args()

    print("=" * 72)
    print(" MT-HAB self-check   —   run this before a long experiment")
    print("=" * 72)
    check_packages()
    dev = check_device(a.device)
    b = check_data(a.bundle, build=not a.no_build)
    if b is not None:
        check_models(a.bundle, dev, a.quick)
        check_cli()
        if not a.no_determinism_check:
            check_determinism(a.bundle, dev)
        check_reporting(a.bundle, a.quick)
    check_disk()

    print("\n" + "=" * 72)
    if _state["fail"]:
        print(f" 실패 {_state['fail']}건, 경고 {_state['warn']}건  —  위 FAIL 항목을 먼저 해결하세요")
        sys.exit(1)
    print(f" 전체 통과 (경고 {_state['warn']}건)  —  본 실험을 시작해도 됩니다:")
    print(f"     bash run_local_full.sh {dev} 10")
    print("=" * 72)


if __name__ == "__main__":
    main()
