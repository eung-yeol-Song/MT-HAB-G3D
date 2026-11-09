"""
cnn_lstm.py
===========
CNN-LSTM baseline — the one additional comparison model requested for Table 3.

Design rationale
----------------
CNN-LSTM is the single most frequently used hybrid architecture in the water
quality / algal-bloom forecasting literature, and it is the obvious omission
from the manuscript's baseline set: the table already contains a pure recurrent
encoder (LSTM, GRU), a pure convolutional encoder (dilated TCN) and an attention
encoder (Transformer), but not the convolution-then-recurrence hybrid.  Adding
it isolates the question "does local temporal feature extraction *before* the
recurrent encoder help?" without changing anything else in the protocol.

Architecture
------------
    x                       (B, T, N, C)      T = 5 days, N = 9 nodes, C = 1 or 2
      |
      | permute             (B, C, T, N)
      v
    Conv2d(k=(3,1))         temporal convolution applied independently to every
    BatchNorm2d + ReLU      variable (kernel spans time only, weights shared
    Dropout                 across the N variables)  ->  (B, c1, T, N)
      |
    Conv2d(k=(3,1))                                  ->  (B, c2, T, N)
    BatchNorm2d + ReLU
    Dropout
      |
      | permute + flatten   (B, T, c2 * N)
      v
    LSTM(num_layers=2)      sequence model over the CNN feature maps
      |
      | last hidden state   (B, hidden)
      v
    _MTHead                 {"logit": (B, H), "reg": (B, H, Q)}

Notes
-----
* `padding=(k//2, 0)` keeps the temporal length at T, which matters because the
  input window is only 5 days long — a valid-padding stack would collapse it.
* The convolution deliberately does NOT mix variables (kernel width 1 on the
  node axis).  Mixing variables is exactly what the graph models (ST-GCN,
  MTGNN, MT-HAB-G3D) do, so keeping CNN-LSTM variable-independent preserves the
  interpretation of the graph ablation.
* The multi-task head, the loss, the optimiser, the early-stopping criterion,
  the validation-selected decision threshold and the evaluation protocol are all
  inherited unchanged from `train.py`, so the row is directly comparable with
  every other row of Table 3.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .baselines import _MTHead


class CNNLSTM(nn.Module):
    """Convolution-then-recurrence hybrid with the shared multi-task interface."""

    def __init__(self, num_nodes: int = 9, in_channels: int = 1,
                 conv_channels: tuple = (32, 64), hidden: int = 128,
                 layers: int = 2, kernel: int = 3, H: int = 7, Q: int = 1,
                 dropout: float = 0.1, T: int = 5):
        super().__init__()
        self.num_nodes = num_nodes
        self.T = T
        c1, c2 = conv_channels
        pad = (kernel // 2, 0)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, c1, (kernel, 1), padding=pad),
            nn.BatchNorm2d(c1), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Conv2d(c1, c2, (kernel, 1), padding=pad),
            nn.BatchNorm2d(c2), nn.ReLU(inplace=True), nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(c2 * num_nodes, hidden, num_layers=layers,
                            batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = _MTHead(hidden, hidden, H, Q, dropout)

    def forward(self, x: torch.Tensor) -> dict:
        # x: (B, T, N, C)
        B, T, N, C = x.shape
        h = x.permute(0, 3, 1, 2)                 # (B, C, T, N)
        h = self.conv(h)                          # (B, c2, T, N)
        h = h.permute(0, 2, 1, 3).reshape(B, T, -1)   # (B, T, c2 * N)
        o, _ = self.lstm(h)
        return self.head(o[:, -1])                # last time step

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
