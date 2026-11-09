"""
baselines.py
============
Reference models for the extended comparison table.

The manuscript only contrasts MT-HAB-G3D with a single LSTM, which is the
weakest point of the experimental section.  This module supplies the family of
baselines a reviewer of *Ecological Informatics* will expect:

  naive        Persistence (D+h = last observed value / last label)
  climatology  Day-of-year climatology from the training years
  histgb       Gradient-boosted trees on flattened windows (strong tabular ref.)
  mlp          Fully connected network on flattened windows
  lstm         LSTM encoder  (the manuscript baseline)
  gru          GRU encoder
  cnn_lstm     CNN-LSTM hybrid: temporal Conv2d stack -> LSTM  (see cnn_lstm.py)
  tcn          Dilated temporal CNN (no graph)
  transformer  Transformer encoder with learned positional embeddings
  stgcn        ST-GCN-lite: graph conv + temporal conv, no G3D pathway

All neural baselines share the multi-task, multi-horizon output interface of
MT-HAB-G3D so that training / evaluation code is identical:
    forward(x) -> {"logit": (B, H), "reg": (B, H, Q)}
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class _MTHead(nn.Module):
    def __init__(self, c_in, hidden, H, Q, dropout=0.1):
        super().__init__()
        self.H, self.Q = H, Q
        self.cls = nn.Sequential(nn.Linear(c_in, hidden), nn.ReLU(inplace=True),
                                 nn.Dropout(dropout), nn.Linear(hidden, H))
        self.reg = nn.Sequential(nn.Linear(c_in, hidden), nn.ReLU(inplace=True),
                                 nn.Dropout(dropout), nn.Linear(hidden, H * Q))

    def forward(self, f):
        return {"logit": self.cls(f), "reg": self.reg(f).view(-1, self.H, self.Q)}


class SeqBaseline(nn.Module):
    """LSTM / GRU / Transformer / TCN / MLP over the flattened variable axis."""

    def __init__(self, kind="lstm", num_nodes=9, in_channels=1, hidden=128, layers=2,
                 H=7, Q=1, dropout=0.1, T=5):
        super().__init__()
        self.kind = kind
        d_in = num_nodes * in_channels
        if kind in {"lstm", "gru"}:
            rnn = nn.LSTM if kind == "lstm" else nn.GRU
            self.enc = rnn(d_in, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
            c_out = hidden
        elif kind == "tcn":
            ch = []
            c = d_in
            for i, d in enumerate((1, 2, 4)):
                ch += [nn.Conv1d(c, hidden, 3, padding=d, dilation=d),
                       nn.BatchNorm1d(hidden), nn.ReLU(inplace=True), nn.Dropout(dropout)]
                c = hidden
            self.enc = nn.Sequential(*ch)
            c_out = hidden
        elif kind == "transformer":
            self.inp = nn.Linear(d_in, hidden)
            self.pos = nn.Parameter(torch.randn(1, T, hidden) * 0.02)
            layer = nn.TransformerEncoderLayer(hidden, nhead=4, dim_feedforward=hidden * 2,
                                               dropout=dropout, batch_first=True)
            self.enc = nn.TransformerEncoder(layer, num_layers=layers)
            c_out = hidden
        elif kind == "mlp":
            self.enc = nn.Sequential(nn.Linear(d_in * T, hidden), nn.ReLU(inplace=True),
                                     nn.Dropout(dropout), nn.Linear(hidden, hidden),
                                     nn.ReLU(inplace=True))
            c_out = hidden
        else:
            raise ValueError(kind)
        self.head = _MTHead(c_out, hidden, H, Q, dropout)

    def forward(self, x):
        B, T, N, C = x.shape
        s = x.reshape(B, T, N * C)
        if self.kind in {"lstm", "gru"}:
            o, _ = self.enc(s)
            f = o[:, -1]
        elif self.kind == "tcn":
            f = self.enc(s.transpose(1, 2))[:, :, -1]
        elif self.kind == "transformer":
            f = self.enc(self.inp(s) + self.pos[:, :T]).mean(1)
        else:
            f = self.enc(s.reshape(B, -1))
        return self.head(f)


class STGCNLite(nn.Module):
    """Graph conv + temporal conv, i.e. MT-HAB-G3D without the G3D pathway and
    without the multi-scale TCN aggregation - a classic ST-GCN reference."""

    def __init__(self, A_norm: np.ndarray, num_nodes=9, in_channels=1, hidden=64,
                 H=7, Q=1, dropout=0.1):
        super().__init__()
        self.register_buffer("A", torch.from_numpy(np.ascontiguousarray(A_norm, dtype=np.float32)))
        self.g1 = nn.Conv2d(in_channels, hidden, 1)
        self.t1 = nn.Conv2d(hidden, hidden, (3, 1), padding=(1, 0))
        self.g2 = nn.Conv2d(hidden, hidden * 2, 1)
        self.t2 = nn.Conv2d(hidden * 2, hidden * 2, (3, 1), padding=(1, 0))
        self.bn = nn.BatchNorm2d(hidden * 2)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        self.head = _MTHead(hidden * 2, hidden * 2, H, Q, dropout)

    def _gc(self, conv, x):
        return conv(torch.einsum("bctv,vw->bctw", x, self.A))

    def forward(self, x):
        h = x.permute(0, 3, 1, 2)                       # (B, C, T, N)
        h = self.act(self.t1(self.act(self._gc(self.g1, h))))
        h = self.drop(self.act(self.bn(self.t2(self.act(self._gc(self.g2, h))))))
        return self.head(h.mean(dim=(2, 3)))


# --------------------------------------------------------------------------- #
# Non-learned / classical references live in reference.py (torch-free); re-exported here
# so that `from .models.baselines import persistence, climatology, histgb` keeps working.
# --------------------------------------------------------------------------- #
from .reference import persistence, climatology, histgb, persistence_valid_anchor  # noqa: E402,F401


def build_model(name: str, A_prior: np.ndarray, in_channels=1, num_nodes=9,
                H=7, Q=1, T=5, dropout=0.1, model_cfg_kwargs=None):
    """Factory shared by train.py."""
    from .mthab_g3d import MTHABG3D, ModelConfig
    from ..data.dataset import normalize_adj

    kw = dict(num_nodes=num_nodes, in_channels=in_channels, horizons=H,
              n_quantiles=Q, dropout=dropout)
    kw.update(model_cfg_kwargs or {})

    if name in {"lstm", "gru", "tcn", "transformer", "mlp"}:
        return SeqBaseline(name, num_nodes, in_channels, hidden=128, layers=2,
                           H=H, Q=Q, dropout=dropout, T=T)
    if name == "cnn_lstm":
        from .cnn_lstm import CNNLSTM
        return CNNLSTM(num_nodes=num_nodes, in_channels=in_channels,
                       conv_channels=(32, 64), hidden=128, layers=2, kernel=3,
                       H=H, Q=Q, dropout=dropout, T=T)
    if name == "stgcn":
        return STGCNLite(normalize_adj(A_prior), num_nodes, in_channels,
                         hidden=64, H=H, Q=Q, dropout=dropout)
    if name in {"mtgnn", "am_mtgnn", "mtgnn_nograph"}:
        from .mtgnn import MTGNNMulti
        return MTGNNMulti(num_nodes=num_nodes, in_dim=in_channels, seq_len=T, H=H, Q=Q,
                          dropout=dropout, use_attention=(name == "am_mtgnn"),
                          use_graph=(name != "mtgnn_nograph"),
                          subgraph_size=min(4, num_nodes))
    if name.startswith("mthab"):
        return MTHABG3D(ModelConfig(**kw), A_prior)
    raise ValueError(f"unknown model {name}")
