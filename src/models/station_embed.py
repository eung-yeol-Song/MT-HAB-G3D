"""
station_embed.py
================
Learned per-station vector, injected as extra INPUT CHANNELS.

Why
---
The 17 sites are far from exchangeable.  Red-tide frequency spans 0.000
(Geumgang, Saemangeum, Sihwa, Incheon) to 0.049 (Gwangyang-Jeokryang), and
Chl-a completeness spans 0.00 (Gwangyang-Chonam, Masan-Bongam, Masan-Yangdeok)
to 0.95.  Per-station standardisation removes the level differences but tells
the network nothing about *which* site it is looking at, so it cannot learn a
site-specific bloom propensity or a site-specific missingness pattern.

How
---
An `nn.Embedding` of width d is broadcast over (T, N) and concatenated to the
input channels.  The backbone is untouched -- it simply receives C + d channels
-- so this works identically for every architecture in the comparison and can be
switched off by setting d = 0.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class StationEmbed(nn.Module):
    """Wraps any model in this repo; `dim = 0` is a transparent pass-through."""

    def __init__(self, base: nn.Module, n_stations: int, dim: int = 0):
        super().__init__()
        self.base = base
        self.dim = int(dim)
        self.emb = nn.Embedding(n_stations, self.dim) if self.dim > 0 else None
        if self.emb is not None:
            nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, x: torch.Tensor, st: torch.Tensor | None = None):
        if self.emb is not None and st is not None:
            B, T, N, _C = x.shape
            e = self.emb(st.long())                       # (B, d)
            e = e[:, None, None, :].expand(B, T, N, self.dim)
            x = torch.cat([x, e], dim=-1)
        return self.base(x)

    # n_params is defined here on purpose: it must include the embedding.
    # learned_adjacency is deliberately NOT defined - it is reached through
    # __getattr__ so that hasattr() stays False for backbones that lack it,
    # which is what train.py checks before recording the learned graph.
    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def mc_predict(self, x, n, st=None):
        if self.emb is not None and st is not None:
            B, T, N, _C = x.shape
            e = self.emb(st.long())[:, None, None, :].expand(B, T, N, self.dim)
            x = torch.cat([x, e], dim=-1)
        return self.base.mc_predict(x, n)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._modules["base"], name)
