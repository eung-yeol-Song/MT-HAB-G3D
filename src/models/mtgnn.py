"""
mtgnn.py
========
MTGNN / AM-MTGNN baselines for the extended comparison.

Provenance
----------
Re-implementation of the architecture family used by

    Ding, F. et al. (2025) "An Improved Graph Neural Network Integrating
    Indicator Attention and Spatio-Temporal Correlation for Dissolved Oxygen
    Prediction", Ecological Informatics 87, 103126
    code: github.com/haoshilong-0920/Dissolved-oxygen-prediction

which itself builds on

    Wu, Z. et al. (2020) "Connecting the Dots: Multivariate Time Series
    Forecasting with Graph Neural Networks", KDD 2020.

Both papers are already cited in the present manuscript, which makes this the
most defensible graph baseline available: same journal, same domain (water
quality), same "variables-as-graph-nodes" formulation.

Why a re-implementation instead of the released code
----------------------------------------------------
The released AM-MTGNN/BO-AM-MTGNN sources have their key dimensions redacted
(`nn.MultiheadAttention(embed_dim="", ...)`, `input.view(b, c, n, "", "")`,
`window_size = ""` in create_data.py), so they cannot be executed as published.
The modules below therefore reproduce the *described* design:

    graph_constructor   node-embedding adjacency, top-k sparsified   (verbatim logic)
    dilated_inception   multi-kernel dilated temporal convolution    (verbatim logic)
    mixprop             mix-hop graph propagation                    (verbatim logic)
    IndicatorAttention  multi-head self-attention across the variable
                        ("indicator") axis, applied to the input tensor
                        before the spatio-temporal stack               (AM block)

Two further deviations from the released pipeline are deliberate and are
documented in the paper:
  1. their `create_data.py` fits the StandardScaler on the *whole* series before
     the chronological split (train/test leakage); we keep our train-only
     per-station statistics;
  2. their kernel set [2,3,6,7] requires >= 7 input steps; with the manuscript's
     T = 5 day window we use [2,3,4,5] and left-pad, which is the minimal change
     that preserves the inception structure.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
class GraphConstructor(nn.Module):
    """Wu et al. (2020) uni-directional node-embedding graph, top-k sparsified."""

    def __init__(self, nnodes: int, k: int = 4, dim: int = 40, alpha: float = 3.0):
        super().__init__()
        self.emb1 = nn.Embedding(nnodes, dim)
        self.emb2 = nn.Embedding(nnodes, dim)
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.k, self.alpha, self.nnodes = min(k, nnodes), alpha, nnodes

    def full(self, idx):
        v1 = torch.tanh(self.alpha * self.lin1(self.emb1(idx)))
        v2 = torch.tanh(self.alpha * self.lin2(self.emb2(idx)))
        a = v1 @ v2.t() - v2 @ v1.t()
        return F.relu(torch.tanh(self.alpha * a))

    def forward(self, idx):
        adj = self.full(idx)
        mask = torch.zeros_like(adj)
        _, t1 = (adj + torch.rand_like(adj) * 0.01).topk(self.k, 1)
        mask.scatter_(1, t1, 1.0)
        return adj * mask


class MixProp(nn.Module):
    """Mix-hop propagation with a retain ratio alpha."""

    def __init__(self, c_in, c_out, gdep=2, alpha=0.05):
        super().__init__()
        self.mlp = nn.Conv2d((gdep + 1) * c_in, c_out, 1)
        self.gdep, self.alpha = gdep, alpha

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0), device=x.device)
        a = adj / adj.sum(1).clamp(min=1e-6).view(-1, 1)
        h, out = x, [x]
        for _ in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * torch.einsum("ncwl,vw->ncvl", h, a)
            out.append(h)
        return self.mlp(torch.cat(out, 1))


class DilatedInception(nn.Module):
    def __init__(self, cin, cout, dilation=1, kernel_set=(2, 3, 4, 5)):
        super().__init__()
        self.kernel_set = kernel_set
        c = max(cout // len(kernel_set), 1)
        self.tconv = nn.ModuleList(
            [nn.Conv2d(cin, c, (1, k), dilation=(1, dilation)) for k in kernel_set])

    def forward(self, x):
        xs = [c(x) for c in self.tconv]
        L = min(t.size(3) for t in xs)
        return torch.cat([t[..., -L:] for t in xs], dim=1)


class IndicatorAttention(nn.Module):
    """
    "AM" block: multi-head self-attention over the indicator (variable) axis.

    Input  (B, C, N, T) -> tokens are the N indicators, each embedded from its
    length-T history -> attention mixes indicators -> projected back to (B, C, N, T).
    A residual connection keeps the block a strict refinement of the raw input.
    """

    def __init__(self, in_dim: int, seq_len: int, d_model: int = 64, heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Linear(in_dim * seq_len, d_model)
        self.att = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, in_dim * seq_len)
        self.in_dim, self.seq_len = in_dim, seq_len
        self.gate = nn.Parameter(torch.tensor(0.0))
        self.last_attn = None

    def forward(self, x):                      # (B, C, N, T)
        B, C, N, T = x.shape
        tok = x.permute(0, 2, 1, 3).reshape(B, N, C * T)
        h = self.norm(self.inp(tok))
        h, w = self.att(h, h, h, need_weights=True, average_attn_weights=True)
        self.last_attn = w.detach()
        d = self.out(h).view(B, N, C, T).permute(0, 2, 1, 3)
        return x + torch.tanh(self.gate) * d


# --------------------------------------------------------------------------- #
class MTGNNMulti(nn.Module):
    """
    MTGNN backbone with the multi-task, multi-horizon head used everywhere in
    this study.  `use_attention=True` gives the AM-MTGNN variant.
    """

    def __init__(self, num_nodes=9, in_dim=1, seq_len=5, H=7, Q=1,
                 conv_channels=32, residual_channels=32, skip_channels=64,
                 end_channels=128, layers=2, gcn_depth=2, subgraph_size=4,
                 node_dim=40, propalpha=0.05, tanhalpha=3.0, dropout=0.3,
                 use_attention=False, predefined_A=None, use_graph=True):
        super().__init__()
        self.seq_len, self.layers, self.use_graph = seq_len, layers, use_graph
        self.dropout = dropout
        self.attn = IndicatorAttention(in_dim, seq_len, d_model=64,
                                       dropout=dropout) if use_attention else None
        self.gc = GraphConstructor(num_nodes, subgraph_size, node_dim, tanhalpha)
        self.register_buffer("idx", torch.arange(num_nodes))
        if predefined_A is not None:
            self.register_buffer(
                "predefined_A",
                torch.from_numpy(np.ascontiguousarray(predefined_A, dtype=np.float32)))
        else:
            self.predefined_A = None

        self.start = nn.Conv2d(in_dim, residual_channels, 1)
        self.filt, self.gate, self.res, self.skipc = (nn.ModuleList() for _ in range(4))
        self.g1, self.g2, self.norm = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for _ in range(layers):
            self.filt.append(DilatedInception(residual_channels, conv_channels))
            self.gate.append(DilatedInception(residual_channels, conv_channels))
            self.res.append(nn.Conv2d(conv_channels, residual_channels, 1))
            self.skipc.append(nn.Conv2d(conv_channels, skip_channels, 1))
            self.g1.append(MixProp(conv_channels, residual_channels, gcn_depth, propalpha))
            self.g2.append(MixProp(conv_channels, residual_channels, gcn_depth, propalpha))
            self.norm.append(nn.BatchNorm2d(residual_channels))
        self.skip0 = nn.Conv2d(in_dim, skip_channels, 1)
        self.end1 = nn.Conv2d(skip_channels, end_channels, 1)
        self.H, self.Q = H, Q
        self.cls = nn.Sequential(nn.Linear(end_channels, end_channels), nn.ReLU(True),
                                 nn.Dropout(dropout), nn.Linear(end_channels, H))
        self.reg = nn.Sequential(nn.Linear(end_channels, end_channels), nn.ReLU(True),
                                 nn.Dropout(dropout), nn.Linear(end_channels, H * Q))

    def forward(self, x):                       # x: (B, T, N, C)
        x = x.permute(0, 3, 2, 1).contiguous()  # (B, C, N, T)
        if self.attn is not None:
            x = self.attn(x)
        pad = max(self.attn_pad(), 0)
        if pad:
            x = F.pad(x, (pad, 0))
        adj = None
        if self.use_graph:
            adj = self.gc(self.idx) if self.predefined_A is None else self.predefined_A

        h = self.start(x)
        skip = self.skip0(x).mean(3, keepdim=True)
        for i in range(self.layers):
            residual = h
            f = torch.tanh(self.filt[i](h))
            g = torch.sigmoid(self.gate[i](h))
            h = F.dropout(f * g, self.dropout, training=self.training)
            skip = skip + self.skipc[i](h).mean(3, keepdim=True)
            if self.use_graph:
                h = self.g1[i](h, adj) + self.g2[i](h, adj.transpose(1, 0))
            else:
                h = self.res[i](h)
            h = h + residual[..., -h.size(3):]
            h = self.norm[i](h)
        z = F.relu(self.end1(F.relu(skip))).mean(dim=(2, 3))
        return {"logit": self.cls(z), "reg": self.reg(z).view(-1, self.H, self.Q)}

    def attn_pad(self):
        """Left padding so that `layers` inception blocks keep a positive length."""
        need = self.layers * (max((2, 3, 4, 5)) - 1) + 1
        return need - self.seq_len

    def learned_adjacency(self):
        with torch.no_grad():
            return self.gc.full(self.idx).cpu().numpy()

    def indicator_attention(self):
        return None if self.attn is None else self.attn.last_attn.mean(0).cpu().numpy()

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
