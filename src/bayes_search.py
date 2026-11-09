"""
bayes_search.py
===============
Bayesian hyper-parameter optimisation for the multi-task objective.

Motivation
----------
The manuscript fixes lambda = 0.8, delta = 1.5 and tau_c without saying how they
were chosen, which is the kind of thing a reviewer asks about.  The reference
work of Ding et al. (2025) (BO-AM-MTGNN) answers the same question with
`bayes_opt.BayesianOptimization` over (learning rate, weight decay, gradient
clip), 5 initial points + 25 iterations, each trial trained for 9 epochs and
scored by the mean validation loss.

This module reproduces that protocol with two corrections and no extra
dependency:

  * the search is scored on the **2019 validation split only** - never on 2020 -
    so no test information can leak into the hyper-parameters;
  * the search space also contains the two multi-task-specific knobs the
    manuscript leaves unjustified (lambda and the Huber delta) plus dropout.

Implementation: Gaussian-process surrogate (Matern 5/2) with Expected
Improvement acquisition, maximised over a Sobol-like random candidate set.
Continuous parameters are searched in log space where appropriate.

Usage
-----
python -m src.bayes_search --bundle outputs/bundle_v2_paper.npz \
       --model mthab --init 5 --iters 25 --trial_epochs 9
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

from .train import build_parser, run

# name -> (low, high, log_scale)
SPACE = {
    "lr":      (1e-5, 1e-2, True),
    "wd":      (1e-6, 1e-2, True),
    "clip":    (0.5, 5.0, False),
    "lam":     (0.1, 2.0, False),
    "delta":   (0.5, 3.0, False),
    "dropout": (0.05, 0.4, False),
}


def _to_unit(x: dict) -> np.ndarray:
    v = []
    for k, (lo, hi, lg) in SPACE.items():
        a = np.log(x[k]) if lg else x[k]
        l, h = (np.log(lo), np.log(hi)) if lg else (lo, hi)
        v.append((a - l) / (h - l))
    return np.array(v)


def _from_unit(u: np.ndarray) -> dict:
    out = {}
    for i, (k, (lo, hi, lg)) in enumerate(SPACE.items()):
        l, h = (np.log(lo), np.log(hi)) if lg else (lo, hi)
        a = l + float(np.clip(u[i], 0, 1)) * (h - l)
        out[k] = float(np.exp(a)) if lg else float(a)
    return out


def objective(params: dict, args, trial_epochs: int, seed: int = 0) -> float:
    """Validation score of one configuration (higher is better)."""
    p = build_parser().parse_args([])
    p.bundle, p.outdir = args.bundle, os.path.join(args.outdir, "_bo_trials")
    p.model, p.graph = args.model, args.graph
    p.cls_loss, p.reg_loss = args.cls_loss, args.reg_loss
    p.channels, p.batch_size = args.channels, args.batch_size
    p.pos_sampler = args.pos_sampler
    p.epochs, p.patience, p.seed = trial_epochs, trial_epochs, seed
    p.tag = "trial_" + "_".join(f"{k}{v:.4g}" for k, v in params.items())[:80]
    for k, v in params.items():
        setattr(p, k, v)
    m = run(p)
    h = json.load(open(os.path.join(p.outdir, p.tag, "history.json")))
    return float(max(h["val_score"])) if h.get("val_score") else -1e9


def expected_improvement(gp, X, best, xi=0.01):
    mu, sd = gp.predict(X, return_std=True)
    sd = np.maximum(sd, 1e-9)
    z = (mu - best - xi) / sd
    return (mu - best - xi) * norm.cdf(z) + sd * norm.pdf(z)


def search(args):
    rng = np.random.default_rng(args.seed)
    d = len(SPACE)
    X, y, records = [], [], []

    for i in range(args.init):                       # random initial design
        u = rng.random(d)
        p = _from_unit(u)
        s = objective(p, args, args.trial_epochs, args.seed)
        X.append(u), y.append(s)
        records.append({"iter": i, "stage": "init", "params": p, "score": s})
        print(f"[BO init {i+1}/{args.init}] score={s:.4f}  {p}", flush=True)

    kernel = (ConstantKernel(1.0) * Matern(length_scale=np.ones(d), nu=2.5)
              + WhiteKernel(1e-3))
    for i in range(args.iters):
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                      n_restarts_optimizer=2, random_state=args.seed)
        gp.fit(np.array(X), np.array(y))
        cand = rng.random((2048, d))
        ei = expected_improvement(gp, cand, max(y))
        u = cand[int(np.argmax(ei))]
        p = _from_unit(u)
        s = objective(p, args, args.trial_epochs, args.seed)
        X.append(u), y.append(s)
        records.append({"iter": args.init + i, "stage": "bo", "params": p, "score": s})
        print(f"[BO {i+1}/{args.iters}] score={s:.4f} best={max(y):.4f}  {p}", flush=True)

    best = int(np.argmax(y))
    out = {"best_params": _from_unit(X[best]), "best_score": float(y[best]),
           "history": records, "space": {k: list(v) for k, v in SPACE.items()},
           "protocol": "scored on the 2019 validation split only"}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print("\nbest:", json.dumps(out["best_params"], indent=1))
    print(f"-> {args.out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="outputs/bundle_v2_paper.npz")
    ap.add_argument("--outdir", default="outputs/runs")
    ap.add_argument("--out", default="outputs/bo_best.json")
    ap.add_argument("--model", default="mthab")
    ap.add_argument("--graph", default="corr")
    ap.add_argument("--cls_loss", default="focal")
    ap.add_argument("--reg_loss", default="huber")
    ap.add_argument("--channels", default="32,64,128")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--pos_sampler", action="store_true", default=True)
    ap.add_argument("--init", type=int, default=5)
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--trial_epochs", type=int, default=9)
    ap.add_argument("--seed", type=int, default=0)
    search(ap.parse_args())
