"""
mthab_g3d.py
============
MT-HAB-G3D (reproduction of the manuscript) + the extended variants required by
the additional experiments.

Everything is controlled by `ModelConfig` so that a single class covers:

  * the published model                       (graph="corr", g3d=True, fact=True)
  * ablation  w/o G3D pathway                 (g3d=False)          [P3]
  * ablation  w/o factorised pathway          (fact=False)         [P3]
  * ablation  graph = identity / complete     (graph="identity")   [P1]
  * adaptive / hybrid graph learning          (graph="adaptive"|"hybrid")  [P1]
  * quantile (pinball) regression head        (n_quantiles > 1)    [P7]
  * MC-dropout predictive uncertainty         (mc_dropout=True)    [P7]
  * multi-horizon direct heads D+1..D+H       (H > 1)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from .layers import GraphLearner, STGCBlock
from ..data.dataset import k_adjacency, normalize_adj


@dataclass
class ModelConfig:
    num_nodes: int = 9
    in_channels: int = 1
    channels: tuple = (96, 192, 384)
    strides: tuple = (1, 2, 2)
    taus: tuple = (3, 5)
    scales: int = 3                 # k-adjacency scales k = 0..scales-1
    graph: str = "corr"             # corr | adaptive | hybrid | identity | complete
    emb_dim: int = 16
    topk: int = 4
    use_g3d: bool = True
    use_factorized: bool = True
    dropout: float = 0.1
    head_hidden: int = 128
    horizons: int = 7
    n_quantiles: int = 1            # >1 -> pinball / quantile regression head
    quantiles: tuple = (0.1, 0.5, 0.9)
    mc_dropout: bool = False


class MTHABG3D(nn.Module):
    def __init__(self, cfg: ModelConfig, A_prior: np.ndarray):
        super().__init__()
        self.cfg = cfg
        N = cfg.num_nodes

        # ---- the graph mode has to determine the FIXED supports as well as ---
        # ---- the learned component; otherwise "identity" and "complete" -----
        # ---- would silently reduce to the correlation baseline --------------
        A_prior = np.asarray(A_prior, dtype=np.float32).copy()
        if cfg.graph == "identity":             # no cross-variable edges at all
            A_prior = np.zeros((N, N), dtype=np.float32)
        elif cfg.graph == "complete":           # every variable pair connected
            A_prior = np.ones((N, N), dtype=np.float32) - np.eye(N, dtype=np.float32)
        self.prior_edges = int(A_prior.sum())

        A_bin = [k_adjacency(A_prior, k) for k in range(cfg.scales)]
        A_norm = [normalize_adj(a) if k > 0 else np.eye(N, dtype=np.float32)
                  for k, a in enumerate(A_bin)]

        self.graph_learner = GraphLearner(N, cfg.emb_dim, topk=cfg.topk,
                                          mode=cfg.graph, prior=A_prior)

        self.input_bn = nn.BatchNorm1d(cfg.in_channels * N)

        blocks, c_prev = [], cfg.in_channels
        for c_out, s in zip(cfg.channels, cfg.strides):
            blocks.append(STGCBlock(c_prev, c_out, A_norm, A_bin, stride=s,
                                    taus=cfg.taus, use_g3d=cfg.use_g3d,
                                    use_factorized=cfg.use_factorized,
                                    dropout=cfg.dropout))
            c_prev = c_out
        self.blocks = nn.ModuleList(blocks)

        H, Q = cfg.horizons, cfg.n_quantiles
        self.cls_head = nn.Sequential(
            nn.Linear(c_prev, cfg.head_hidden), nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout), nn.Linear(cfg.head_hidden, H))
        self.reg_head = nn.Sequential(
            nn.Linear(c_prev, cfg.head_hidden), nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout), nn.Linear(cfg.head_hidden, H * Q))

    # ------------------------------------------------------------------ #
    def features(self, x):
        """x: (B, T, N, C) -> pooled feature (B, C_last)"""
        B, T, N, C = x.shape
        h = x.permute(0, 3, 1, 2).contiguous()                # (B, C, T, N)
        h = self.input_bn(h.permute(0, 1, 3, 2).reshape(B, C * N, T)).reshape(B, C, N, T)
        h = h.permute(0, 1, 3, 2).contiguous()                # (B, C, T, N)
        A_dyn = None
        if self.cfg.graph in {"adaptive", "hybrid"}:
            A_dyn = self.graph_learner()
        for blk in self.blocks:
            h = blk(h, A_dyn)
        return h.mean(dim=(2, 3))                             # global average pooling

    def forward(self, x):
        f = self.features(x)
        logit = self.cls_head(f)                              # (B, H)
        q = self.reg_head(f).view(-1, self.cfg.horizons, self.cfg.n_quantiles)
        return {"logit": logit, "reg": q}

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def mc_predict(self, x, n_samples: int = 30):
        """Monte-Carlo dropout predictive mean / std (epistemic uncertainty)."""
        was_training = self.training
        self.train()                                          # keep dropout active
        for m in self.modules():                              # but freeze BN statistics
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.eval()
        P, R = [], []
        for _ in range(n_samples):
            o = self.forward(x)
            P.append(torch.sigmoid(o["logit"]))
            R.append(o["reg"][..., o["reg"].shape[-1] // 2])
        self.train(was_training)
        P, R = torch.stack(P), torch.stack(R)
        return {"p_mean": P.mean(0), "p_std": P.std(0),
                "r_mean": R.mean(0), "r_std": R.std(0)}

    def graph_regularizer(self):
        return self.graph_learner.regularizer()

    def learned_adjacency(self) -> np.ndarray:
        with torch.no_grad():
            return self.graph_learner().detach().cpu().numpy()

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
