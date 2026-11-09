"""
train.py
========
One experiment = one (model, loss, graph, seed) configuration.

Protocol (identical for every model, so the comparison is fair):
    train  = 2015-2018        val = 2019        test = 2020
    * early stopping on the validation split
    * red-tide decision threshold selected on validation (Eq. 16)
    * Chl-a metrics computed after inverse standardisation (ug/L)
    * red-tide F1 reported on the event-adjacent subset,
      accuracy / regression on the full-year timeline

Outputs (per run, under outputs/runs/<tag>/):
    preds_test.npz   probabilities, point + quantile predictions, targets, meta
    metrics.json     every metric, every horizon, both evaluation protocols
    history.json     per-epoch training curves
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .data.dataset import Bundle, WQ_VARS, NODE_VARS, CHL_IDX
from .losses import build_objective
from .metrics import summarize, best_threshold
from .models.baselines import build_model, persistence, climatology, histgb, persistence_valid_anchor


# --------------------------------------------------------------------------- #
def set_seed(s: int, deterministic: bool = True):
    """Seed every RNG this code path touches, and pin the kernels.

    v11 and earlier seeded only numpy and torch, and left cuDNN free to pick
    non-deterministic kernels.  Two runs of the same commit with the same seed
    therefore produced different numbers, which made it impossible to tell a
    real configuration change from run-to-run drift: the v11_1 (10 seeds) and
    v11_2 (3 seeds) results could not be compared at all.  With this in place a
    given (config, seed) reproduces bit-for-bit, so any difference between two
    runs is a difference in the configuration.

    `warn_only=True` on use_deterministic_algorithms: a few ops have no
    deterministic CUDA implementation.  Erroring out would abort the run; the
    warning names the op instead, and the remaining ~99% of the graph is still
    pinned.  --nondeterministic restores the old free-for-all if a kernel ever
    becomes a hard blocker.
    """
    os.environ["PYTHONHASHSEED"] = str(s)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    if not deterministic:
        torch.backends.cudnn.benchmark = True
        return
    # cuBLAS needs this set before the first CUDA context; run_local_full.sh
    # exports it, and this is the fallback for a bare `python -m src.train`.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def chl_scale(bundle: Bundle, idx: np.ndarray):
    """Per-sample (mu, sd) used to map predictions back to the physical scale.

    With `log_target` the bundle standardises log1p(Chl-a), so the statistics are
    the log ones and `inv_chl` applies expm1 after de-standardising.
    """
    if bundle.cfg.get("log_target"):
        mu = np.array([bundle.scaler[s]["log_mu"] for s in bundle.stations], np.float32)
        sd = np.array([bundle.scaler[s]["log_sd"] for s in bundle.stations], np.float32)
    else:
        mu = np.array([bundle.scaler[s]["mu"]["chl"] for s in bundle.stations], np.float32)
        sd = np.array([bundle.scaler[s]["sd"]["chl"] for s in bundle.stations], np.float32)
    return mu[bundle.station[idx]], sd[bundle.station[idx]]


# Chl-a never exceeds a few hundred ug/L in this network; anything past this is
# an un-converged model, not a bloom.  The bound matters specifically for the
# log target: expm1 turns a wild standardised output into an astronomically
# large concentration (an early-epoch run produced RMSE 364 and R2 = -7019
# before this guard), which would wreck both the metrics and model selection.
CHL_MAX_UGL = 500.0


def inv_chl(bundle: Bundle, z, mu, sd):
    """Standardised model output -> Chl-a in ug/L."""
    v = z * sd + mu
    if bundle.cfg.get("log_target"):
        v = np.clip(v, 0.0, float(np.log1p(CHL_MAX_UGL)))
        return np.expm1(v)
    return np.clip(v, -CHL_MAX_UGL, CHL_MAX_UGL)


def make_loader(b: Bundle, idx, batch_size, shuffle, pos_sampler=False, horizon0=0,
                pos_weight_cap=0.0, seed=0):
    """
    WARNING - the sampler distorts BOTH tasks.
    ------------------------------------------
    `pos_sampler` exists for the red-tide classification head, but the DataLoader
    is shared with the Chl-a regression head, so oversampling bloom days also
    changes the distribution the regressor is fitted on.  On this dataset the
    uncapped weight is ~156x, which lifts the expected training Chl-a from
    6.9 to 8.5 ug/L while the 2020 test mean is 4.0 ug/L - a built-in
    over-prediction bias that drives R2 negative.

    `pos_weight_cap` bounds the weight (e.g. 10) so the classifier still sees a
    workable positive rate without moving the regression target distribution as
    violently.  0 = uncapped (the original behaviour).
    """
    ds = TensorDataset(
        torch.tensor(b.X[idx]), torch.tensor(b.y_cls[idx]), torch.tensor(b.m_cls[idx]),
        torch.tensor(b.y_reg[idx]), torch.tensor(b.m_reg[idx]),
        torch.tensor(b.station[idx]))
    # Shuffling and the weighted sampler both draw from torch's *global* RNG
    # unless a generator is supplied.  Without it, run order depends on however
    # many random numbers were consumed earlier, so the same seed does not
    # reproduce the same batches.
    g = torch.Generator()
    g.manual_seed(int(seed))
    if pos_sampler and shuffle:
        y = b.y_cls[idx][:, horizon0]
        w_pos = max((y < 0.5).sum() / max(y.sum(), 1), 1.0)
        if pos_weight_cap and pos_weight_cap > 0:
            w_pos = min(w_pos, float(pos_weight_cap))
        w = np.where(y > 0.5, w_pos, 1.0)
        sampler = WeightedRandomSampler(torch.tensor(w, dtype=torch.double),
                                        num_samples=len(idx), replacement=True,
                                        generator=g)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                          drop_last=False, generator=g)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      drop_last=False, generator=g)


def chl_anchor(b: Bundle, idx=None):
    """Standardised Chl-a observed on day D - the persistence anchor used by the
    residual ('delta') forecasting head.  EXTENSION P2b."""
    a = b.X[:, -1, CHL_IDX, 0]
    return a if idx is None else a[idx]


def add_anchor(pred, b, idx, on: bool):
    """Map residual predictions back to absolute (standardised) Chl-a."""
    if not on:
        return pred
    a = chl_anchor(b, idx)[:, None]
    pred = dict(pred)
    pred["reg"] = pred["reg"] + a
    pred["quant"] = pred["quant"] + a[..., None]
    return pred


def resolve_device(name: str = "cpu") -> torch.device:
    """'auto' -> cuda when available, otherwise cpu.  Default stays 'cpu' so the
    numbers already in outputs/results.csv remain byte-for-byte reproducible."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA requested but not available - falling back to CPU")
        name = "cpu"
    return torch.device(name)


def to_batch(t, device=None):
    x, yc, mc, yr, mr, st = t
    if device is not None:
        x, yc, mc, yr, mr, st = (v.to(device, non_blocking=True)
                                 for v in (x, yc, mc, yr, mr, st))
    return {"x": x, "y_cls": yc, "m_cls": mc, "y_reg": yr, "m_reg": mr, "station": st}


@torch.no_grad()
def predict(model, b: Bundle, idx, batch_size=512, mc_dropout=0, device=None):
    model.eval()
    dev = device or next(model.parameters()).device
    P, R, Q, S = [], [], [], []
    for i in range(0, len(idx), batch_size):
        sl = idx[i:i + batch_size]
        x = torch.tensor(b.X[sl]).to(dev)
        st = torch.tensor(b.station[sl]).to(dev)
        if mc_dropout:
            o = model.mc_predict(x, mc_dropout, st)
            pm, rm, rs = (o["p_mean"].cpu().numpy(), o["r_mean"].cpu().numpy(),
                          o["r_std"].cpu().numpy())
            P.append(pm)
            R.append(rm)
            S.append(rs)
            Q.append(np.stack([rm - 1.2816 * rs, rm, rm + 1.2816 * rs], -1))
        else:
            o = model(x, st)
            P.append(torch.sigmoid(o["logit"]).cpu().numpy())
            q = o["reg"].cpu().numpy()
            R.append(q[..., q.shape[-1] // 2])
            Q.append(q)
    out = {"prob": np.concatenate(P), "reg": np.concatenate(R),
           "quant": np.concatenate(Q)}
    if S:
        out["reg_std"] = np.concatenate(S)
    return out


# --------------------------------------------------------------------------- #
def evaluate_split(b: Bundle, idx, pred, thr, horizons, quantiles=None,
                   event_only=False, tag="", anchor=None):
    """Metrics for every horizon; classification optionally on event-adjacent subset."""
    mu, sd = chl_scale(b, idx)
    an = np.zeros(len(idx), np.float32) if anchor is None else anchor[idx]
    res = {}
    for hi, h in enumerate(horizons):
        # ---- classification ------------------------------------------------
        m = b.m_cls[idx][:, hi]
        if event_only:
            ea = getattr(b, "event_adj_h", None)      # revision: each horizon scored on its own event window
            m = m & (ea[idx][:, hi] if ea is not None else b.event_adj[idx])
        y_c = b.y_cls[idx][m, hi]
        p_c = pred["prob"][m, hi]
        st = b.station[idx][m]

        # ---- regression (full-year, ug/L) ----------------------------------
        mr = b.m_reg[idx][:, hi]
        y_r = b.y_reg_raw[idx][mr, hi]
        p_r = inv_chl(b, pred["reg"][mr, hi] + an[mr], mu[mr], sd[mr])
        q_r = None
        if quantiles is not None and pred["quant"].shape[-1] > 1:
            q_r = inv_chl(b, pred["quant"][mr, hi] + an[mr, None],
                          mu[mr, None], sd[mr, None])

        res[f"D+{h}"] = summarize(y_c, p_c, y_r, p_r, station=st, thr=thr[hi],
                                  q_pred=q_r, quantiles=quantiles)
    res["_tag"] = tag
    return res


def _event_adj_h(b: Bundle, H: int):
    """Per-horizon event membership; falls back to the legacy D+1-anchored mask."""
    ea = getattr(b, "event_adj_h", None)
    return ea if ea is not None else np.repeat(b.event_adj[:, None], H, 1)


# --------------------------------------------------------------------------- #
def permutation_importance(model, b: Bundle, idx, horizons, repeats=5, seed=0, device=None,
                           batch_size=512):
    """Model-agnostic importance of each VARIABLE NODE (revision, R1-G2).

    The node's whole input block (all T frames, all channels) is permuted across
    samples, the model re-predicts, and the loss of full-year AUPRC (red tide) and
    the increase of RMSE in ug/L (Chl-a) are recorded per horizon.  Only the centre
    station's nodes are permuted when neighbour blocks are present.
    """
    from .metrics import regression_metrics, auprc
    rng = np.random.default_rng(seed)
    mu, sd = chl_scale(b, idx)

    def score(pred):
        au, rm = [], []
        for hi in range(len(horizons)):
            m = b.m_cls[idx][:, hi]
            a_ = auprc(b.y_cls[idx][m, hi], pred["prob"][m, hi])
            au.append(0.0 if np.isnan(a_) else float(a_))
            mr = b.m_reg[idx][:, hi]
            rm.append(regression_metrics(b.y_reg_raw[idx][mr, hi],
                                         inv_chl(b, pred["reg"][mr, hi], mu[mr], sd[mr]))["RMSE"])
        return np.array(au, float), np.array(rm, float)

    au0, rm0 = score(predict(model, b, idx, batch_size, device=device))
    X0 = b.X[idx].copy()
    n_nodes = min(b.X.shape[2], len(NODE_VARS))
    out = {"base_AUPRC": au0.tolist(), "base_RMSE": rm0.tolist(), "repeats": int(repeats),
           "node_order": list(NODE_VARS[:n_nodes]), "per_node": {}}
    try:
        for v in range(n_nodes):
            d_au, d_rm = [], []
            for _ in range(repeats):
                Xp = X0.copy()
                perm = rng.permutation(len(idx))
                Xp[:, :, v, :] = X0[perm][:, :, v, :]
                b.X[idx] = Xp
                au, rm = score(predict(model, b, idx, batch_size, device=device))
                d_au.append(au0 - au)
                d_rm.append(rm - rm0)
            out["per_node"][NODE_VARS[v]] = {"dAUPRC": np.mean(d_au, 0).tolist(),
                                             "dAUPRC_sd": np.std(d_au, 0).tolist(),
                                             "dRMSE": np.mean(d_rm, 0).tolist(),
                                             "dRMSE_sd": np.std(d_rm, 0).tolist()}
    finally:
        b.X[idx] = X0
    return out


def export_learned_params(model):
    """Interpretable parameters (revision, R1-G2 / R2-m4): per-block residual adjacency masks
    (Eq. 3, `res_mask`), temporal-lag gates (Eq. 7, `alpha_t`) and the station embedding."""
    out = {}
    for name, prm in model.named_parameters():
        if name.endswith("res_mask") or name.endswith("alpha_t") or name.endswith("emb.weight"):
            out[name] = prm.detach().cpu().numpy().tolist()
    return out


# --------------------------------------------------------------------------- #
def run(args):
    t0 = time.time()
    if getattr(args, "threads", 0) and args.threads > 0:
        torch.set_num_threads(int(args.threads))
    set_seed(args.seed, deterministic=not getattr(args, "nondeterministic", False))
    b = Bundle.load(args.bundle)
    H = b.y_cls.shape[1] if args.horizons <= 0 else args.horizons
    horizons = list(range(1, H + 1))
    quantiles = tuple(float(q) for q in args.quantiles.split(",")) if args.n_quantiles > 1 else None

    # the anchor is applied to the *predictions* by add_anchor(); evaluate_split
    # therefore receives a zero offset and stays a pure level-space evaluator
    anchor = np.zeros(len(b.X), np.float32)

    tr, va, te = b.idx("train"), b.idx("val"), b.idx("test")
    if args.residual:                      # learn the day-to-day increment
        b.y_reg = b.y_reg - chl_anchor(b)[:, None]
    C = b.X.shape[-1]
    tag = args.tag or f"{args.model}_{args.reg_loss}_{args.graph}_s{args.seed}"
    outdir = os.path.join(args.outdir, tag)
    os.makedirs(outdir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # non-parametric references need no training
    # ------------------------------------------------------------------ #
    if args.model in {"persistence", "climatology", "histgb"}:
        if args.model == "persistence":
            pv, pt = persistence(b, va, horizons), persistence(b, te, horizons)
            # score Chl-a persistence only where the day-D anchor was observed
            b.m_reg[va] &= persistence_valid_anchor(b, va)[:, None]
            b.m_reg[te] &= persistence_valid_anchor(b, te)[:, None]
        elif args.model == "climatology":
            pv = climatology(b, tr, va, horizons)
            pt = climatology(b, tr, te, horizons)
        else:
            pv = histgb(b, tr, va, horizons, seed=args.seed)
            pt = histgb(b, tr, te, horizons, seed=args.seed)
        for p in (pv, pt):
            p["quant"] = p["reg"][..., None]
        # persistence / climatology / gradient boosting already output levels
        thr = []
        for hi in range(len(horizons)):
            m = b.m_cls[va][:, hi]
            t_, _ = best_threshold(b.y_cls[va][m, hi], pv["prob"][m, hi])
            thr.append(t_)
        history = {}
        model = None
    else:
        # -------------------------------------------------------------- #
        pos = b.y_cls[tr][:, 0][b.m_cls[tr][:, 0]]
        pos_weight = float((len(pos) - pos.sum()) / max(pos.sum(), 1.0))
        model_kwargs = dict(graph=args.graph, use_g3d=not args.no_g3d,
                            use_factorized=not args.no_factorized,
                            channels=tuple(int(c) for c in args.channels.split(",")),
                            n_quantiles=args.n_quantiles,
                            quantiles=quantiles or (0.5,), topk=args.topk)
        emb_dim = int(getattr(args, "station_emb", 0) or 0)
        model = build_model(args.model, b.A, in_channels=C + emb_dim,
                            num_nodes=b.X.shape[2],
                            H=H, Q=args.n_quantiles, T=b.X.shape[1],
                            dropout=args.dropout, model_cfg_kwargs=model_kwargs)
        from .models.station_embed import StationEmbed
        model = StationEmbed(model, n_stations=len(b.stations), dim=emb_dim)
        dev = resolve_device(getattr(args, "device", "cpu"))
        model.to(dev)
        hw = None
        if args.horizon_weight_decay > 0:
            hw = torch.tensor([args.horizon_weight_decay ** (h - 1) for h in horizons],
                              dtype=torch.float32)
        # revision (R2-M2): single-task controls share the backbone, heads, budget and
        # selection protocol; only the loss weights (and the selection score) change.
        lam_eff, cls_w = args.lam, 1.0
        if args.task == "cls":
            lam_eff = 0.0
        elif args.task == "reg":
            cls_w = 0.0
        obj = build_objective(args.cls_loss, args.reg_loss, args.weighting, lam_eff,
                              args.delta, pos_weight, quantiles or (0.5,),
                              args.peak_gamma, args.graph_l1, hw, cls_weight=cls_w)
        f1_weight = {"both": args.f1_weight, "cls": 1.0, "reg": 0.0}[args.task]
        obj.to(dev)

        params = list(model.parameters()) + list(obj.parameters())
        opt = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

        dl_tr = make_loader(b, tr, args.batch_size, True, args.pos_sampler,
                            pos_weight_cap=getattr(args, "pos_weight_cap", 0.0),
                            seed=args.seed)
        history = {"train_loss": [], "val_loss": [], "val_score": []}
        best, best_state, patience = -1e9, None, 0

        for ep in range(args.epochs):
            model.train()
            agg = []
            for batch in dl_tr:
                bt = to_batch(batch, dev)
                out = model(bt["x"], bt["station"])
                loss, parts = obj(out, bt, model)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, args.clip)
                opt.step()
                agg.append((float(loss.detach()), parts["cls"], parts["reg"]))
            sched.step()
            a = np.mean(agg, 0)
            obj.update_dwa(a[1], a[2])

            pv = add_anchor(predict(model, b, va, args.batch_size, device=dev),
                            b, va, args.residual)
            from .metrics import classification_metrics, regression_metrics, auprc

            # --------------------------------------------------------------
            # Model selection (v6).  Two defects were found in the v5 runs and
            # both are fixed here; --select_legacy restores the old behaviour.
            #
            # (1) HORIZON BLINDNESS.  The score used only D+1, so a model that
            #     was healthy one day ahead but diverging at D+3..D+7 still
            #     looked like the best checkpoint.  Seed 6 of MT-HAB-G3D shipped
            #     exactly that: D+1 R2 = 0.62 while D+6 R2 = -5.26.
            #     Fix: average the score over every trained horizon.
            #
            # (2) SELECTION ON NOISE.  F1 depends on a threshold re-fitted every
            #     epoch on 54 validation positives, so it swings 0.09-0.41 with
            #     no trend (coefficient of variation ~0.6 across seeds) and the
            #     "best" epoch is picked by luck.  AUPRC is threshold-free and
            #     five times more stable (CV ~0.13).
            #     Fix: select on AUPRC by default.
            # --------------------------------------------------------------
            sel_h = [0] if args.select_legacy else list(range(len(horizons)))
            mu, sd = chl_scale(b, va)
            cls_scores, reg_scores, f1_log = [], [], []
            for hi in sel_h:
                mc_h = b.m_cls[va][:, hi]
                mr_h = b.m_reg[va][:, hi]
                y_h, p_h = b.y_cls[va][mc_h, hi], pv["prob"][mc_h, hi]
                _, f1_h = best_threshold(y_h, p_h)
                f1_log.append(f1_h)
                if args.select_metric == "f1" or args.select_legacy:
                    cls_scores.append(f1_h)
                else:
                    a_h = auprc(y_h, p_h)
                    cls_scores.append(0.0 if np.isnan(a_h) else float(a_h))
                rm_h = regression_metrics(b.y_reg_raw[va][mr_h, hi],
                                          inv_chl(b, pv["reg"][mr_h, hi], mu[mr_h], sd[mr_h]))
                reg_scores.append(max(rm_h.get("R2", -1.0), -1.0))
            cls_v = float(np.mean(cls_scores)) if cls_scores else 0.0
            reg_v = float(np.mean(reg_scores)) if reg_scores else -1.0
            score = f1_weight * cls_v + (1 - f1_weight) * reg_v

            mr0 = b.m_reg[va][:, 0]
            rm = regression_metrics(b.y_reg_raw[va][mr0, 0],
                                    inv_chl(b, pv["reg"][mr0, 0], mu[mr0], sd[mr0]))
            f1v = f1_log[0]
            history["train_loss"].append(float(a[0]))
            history["val_score"].append(float(score))
            history["val_loss"].append({"F1": f1v, "RMSE": rm["RMSE"], "R2": rm["R2"],
                                        "cls_sel": cls_v, "reg_sel": reg_v})

            if score > best + 1e-5:
                best, patience = score, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= args.patience:
                    break
            if args.verbose:
                print(f"  ep{ep:03d} loss={a[0]:.4f} valF1={f1v:.3f} "
                      f"valRMSE={rm['RMSE']:.3f} R2={rm['R2']:.3f}", flush=True)

        if best_state is not None:
            model.load_state_dict(best_state)

        pv = add_anchor(predict(model, b, va, args.batch_size, args.mc_dropout, dev),
                        b, va, args.residual)
        thr = []
        for hi in range(len(horizons)):
            m = b.m_cls[va][:, hi]
            t_, _ = best_threshold(b.y_cls[va][m, hi], pv["prob"][m, hi])
            thr.append(t_)
        pt = add_anchor(predict(model, b, te, args.batch_size, args.mc_dropout, dev),
                        b, te, args.residual)

    if args.residual:
        b.y_reg = b.y_reg + chl_anchor(b)[:, None]

    # ------------------------------------------------------------------ #
    metrics = {
        "full_year": evaluate_split(b, te, pt, thr, horizons, quantiles, False,
                                    "full_year", anchor),
        "event_adjacent": evaluate_split(b, te, pt, thr, horizons, quantiles, True,
                                         "event_adjacent", anchor),
        "config": vars(args),
        "thresholds": thr,
        "n_params": (model.n_params() if hasattr(model, "n_params")
                     else (sum(p.numel() for p in model.parameters()) if model else 0)),
        "runtime_s": round(time.time() - t0, 1),
    }
    # ------------------------------------------------------------------ #
    # Prediction-spread diagnostics.
    #
    # A model whose regression head collapses onto a near-constant value scores
    # R2 <= 0 while its *classification* head can still look fine, which reads
    # as a mysterious implementation bug in the results table.  ST-GCN did
    # exactly this in the v11_2 run (R2 -0.21 / -2.29 / -0.62 across seeds,
    # while its event-adjacent F1 was the highest of all 13 models).  These
    # numbers let that be stated in one sentence instead of guessed at:
    #   spread_ratio ~ 0   -> the head is predicting a constant
    #   spread_ratio >> 1  -> it is over-dispersed / diverging
    diag = {}
    _mu_d, _sd_d = chl_scale(b, te)
    for hi, h in enumerate(horizons):
        mr = b.m_reg[te][:, hi]
        if mr.sum() < 2:
            continue
        yv = b.y_reg_raw[te][mr, hi].astype(np.float64)
        pv_ = inv_chl(b, pt["reg"][mr, hi], _mu_d[mr], _sd_d[mr]).astype(np.float64)
        o_sd, p_sd = float(np.std(yv)), float(np.std(pv_))
        diag[f"D+{h}"] = {
            "obs_sd": round(o_sd, 4), "pred_sd": round(p_sd, 4),
            "spread_ratio": round(p_sd / o_sd, 4) if o_sd > 0 else None,
            "pred_mean": round(float(np.mean(pv_)), 4),
            "obs_mean": round(float(np.mean(yv)), 4),
            "bias": round(float(np.mean(pv_ - yv)), 4),
            "pred_unique_frac": round(float(len(np.unique(np.round(pv_, 3))) / len(pv_)), 4),
        }
    metrics["pred_diagnostics"] = diag
    _sr = [v["spread_ratio"] for v in diag.values() if v["spread_ratio"] is not None]
    metrics["spread_ratio_mean"] = round(float(np.mean(_sr)), 4) if _sr else None
    if _sr and np.mean(_sr) < 0.15:
        print(f"[{tag}] WARNING: regression head is near-constant "
              f"(pred sd / obs sd = {np.mean(_sr):.3f}). R2 <= 0 is expected; "
              f"this is a collapse, not a metric bug.", flush=True)

    if model is not None and hasattr(model, "learned_adjacency"):
        metrics["learned_adjacency"] = model.learned_adjacency().tolist()
        metrics["node_order"] = NODE_VARS
    metrics["task"] = args.task
    if model is not None:
        metrics["learned_params"] = export_learned_params(model)
        if args.save_state:
            torch.save(model.state_dict(), os.path.join(outdir, "model_state.pt"))
        if args.save_val_preds:
            mu_v, sd_v = chl_scale(b, va)
            np.savez_compressed(
                os.path.join(outdir, "preds_val.npz"),
                prob=pv["prob"], reg_raw=pv["reg"] * sd_v[:, None] + mu_v[:, None],
                y_cls=b.y_cls[va], m_cls=b.m_cls[va], y_reg_raw=b.y_reg_raw[va],
                m_reg=b.m_reg[va], station=b.station[va],
                event_adj_h=_event_adj_h(b, H)[va],
                date=b.date[va].astype("datetime64[D]").astype(np.int64))
        if args.perm_importance > 0:
            metrics["permutation_importance"] = permutation_importance(
                model, b, te, horizons, repeats=args.perm_importance, seed=args.seed,
                device=dev, batch_size=args.batch_size)

    mu, sd = chl_scale(b, te)
    np.savez_compressed(
        os.path.join(outdir, "preds_test.npz"),
        prob=pt["prob"], reg_raw=pt["reg"] * sd[:, None] + mu[:, None],
        quant_raw=pt["quant"] * sd[:, None, None] + mu[:, None, None],
        y_cls=b.y_cls[te], m_cls=b.m_cls[te], y_reg_raw=b.y_reg_raw[te],
        m_reg=b.m_reg[te], station=b.station[te], event_adj=b.event_adj[te],
        event_adj_h=_event_adj_h(b, H)[te],
        date=b.date[te].astype("datetime64[D]").astype(np.int64), thr=np.array(thr))
    json.dump(metrics, open(os.path.join(outdir, "metrics.json"), "w"), indent=1)
    json.dump(history, open(os.path.join(outdir, "history.json"), "w"), indent=1)

    d1f = metrics["full_year"]["D+1"]
    d1e = metrics["event_adjacent"]["D+1"]
    print(f"[{tag}] D+1  RMSE={d1f['RMSE']:.3f}  PeakRMSE={d1f['PeakRMSE']:.3f}  "
          f"R2={d1f['R2']:.3f}  TA1={d1f['TA1']*100:.2f}%  acc={d1f['accuracy']*100:.2f}%  "
          f"| event-adj F1={d1e['F1']:.3f} AUPRC={d1e['AUPRC']:.3f} "
          f"| {metrics['runtime_s']}s", flush=True)
    return metrics


# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", default="outputs/bundle_ffill.npz")
    p.add_argument("--outdir", default="outputs/runs")
    p.add_argument("--tag", default="")
    p.add_argument("--model", default="mthab",
                   help="mthab|lstm|gru|tcn|transformer|mlp|stgcn|persistence|climatology|histgb")
    p.add_argument("--graph", default="corr", choices=["corr", "adaptive", "hybrid",
                                                       "identity", "complete"])
    p.add_argument("--no_g3d", action="store_true")
    p.add_argument("--no_factorized", action="store_true")
    p.add_argument("--channels", default="96,192,384")
    p.add_argument("--topk", type=int, default=4)
    p.add_argument("--cls_loss", default="bce", choices=["bce", "wbce", "focal"])
    p.add_argument("--reg_loss", default="huber",
                   choices=["huber", "mse", "peak_huber", "quantile"])
    p.add_argument("--weighting", default="fixed", choices=["fixed", "uncertainty", "dwa"])
    p.add_argument("--lam", type=float, default=0.8)
    p.add_argument("--delta", type=float, default=1.5)
    p.add_argument("--peak_gamma", type=float, default=1.0)
    p.add_argument("--graph_l1", type=float, default=0.0)
    p.add_argument("--n_quantiles", type=int, default=1)
    p.add_argument("--quantiles", default="0.1,0.5,0.9")
    p.add_argument("--mc_dropout", type=int, default=0)
    p.add_argument("--horizons", type=int, default=-1)
    p.add_argument("--horizon_weight_decay", type=float, default=0.0)
    p.add_argument("--pos_sampler", action="store_true")
    p.add_argument("--pos_weight_cap", type=float, default=0.0,
                   help="upper bound on the positive-oversampling weight "
                        "(0 = uncapped ~156x, which biases the regression head)")
    p.add_argument("--residual", action="store_true",
                   help="predict the increment w.r.t. the last observed Chl-a")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--clip", type=float, default=5.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--f1_weight", type=float, default=0.5,
                   help="weight of the classification term in the model-selection "
                        "score; the rest goes to mean R^2")
    p.add_argument("--select_metric", default="auprc", choices=["auprc", "f1"],
                   help="validation metric driving early stopping. auprc is "
                        "threshold-free and ~5x more stable than f1 on this "
                        "dataset (54 validation positives)")
    p.add_argument("--select_legacy", action="store_true",
                   help="reproduce the v5 selection rule: D+1 only, F1-based")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--station_emb", type=int, default=0,
                   help="width of a learned per-station embedding, appended to the "
                        "input channels. 0 = off. The 17 sites differ strongly in "
                        "bloom frequency (0.000-0.049) and Chl-a completeness "
                        "(0.00-0.95), which per-station standardisation cannot convey.")
    p.add_argument("--threads", type=int, default=0,
                   help="torch intra-op threads. Set this when several runs share "
                        "one machine: N parallel workers x all-cores each is "
                        "oversubscription and is slower than N x (cores/N).")
    p.add_argument("--device", default="cpu",
                   help="cpu | cuda | cuda:0 | auto  (default cpu, so previously "
                        "published runs stay bit-reproducible)")
    p.add_argument("--nondeterministic", action="store_true",
                   help="opt out of deterministic kernels. Only for the case "
                        "where an op has no deterministic CUDA implementation "
                        "and warn_only is not enough; results then stop being "
                        "reproducible run-to-run.")
    p.add_argument("--task", default="both", choices=["both", "cls", "reg"],
                   help="revision R2-M2: 'cls' drops the regression loss (lambda = 0), 'reg' drops "
                        "the classification loss; backbone, heads, budget and selection are unchanged")
    p.add_argument("--save_state", action="store_true",
                   help="write model_state.pt (needed for post-hoc interpretation)")
    p.add_argument("--save_val_preds", action="store_true",
                   help="write preds_val.npz (needed for threshold-rule studies)")
    p.add_argument("--perm_importance", type=int, default=0,
                   help="permutation-importance repeats over the variable nodes on the test "
                        "split (0 = off; 5 is enough)")
    p.add_argument("--verbose", action="store_true")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
