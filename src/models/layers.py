"""
layers.py
=========
Building blocks of MT-HAB-G3D and of its extended variants.

Tensor convention (ST-GCN style):  X : (B, C, T, N)
    B batch, C channels, T time steps, N nodes (= water-quality variables)

Contents
--------
GraphLearner   adaptive (data-driven) adjacency, MTGNN-style       [EXTENSION]
MultiScaleGCN  k-adjacency graph convolution with learnable residual mask
MultiScaleTCN  dilated multi-branch temporal convolution
MultiScaleG3D  joint spatio-temporal graph convolution over a sliding window
STGCBlock      factorised pathway + G3D pathway + aggregation TCN
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
def normalize_adj_t(A: torch.Tensor, add_self_loop: bool = True) -> torch.Tensor:
    if add_self_loop:
        A = A + torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    d = A.sum(-1).clamp(min=1e-6)
    dm = d.pow(-0.5)
    return dm.unsqueeze(-1) * A * dm.unsqueeze(-2)


# --------------------------------------------------------------------------- #
class GraphLearner(nn.Module):
    """
    EXTENSION (Priority 1): learn the variable graph instead of fixing it to a
    thresholded Pearson-correlation graph.

    A_adp = softmax( relu( tanh(alpha * E1 E2^T) - tanh(alpha * E2 E1^T) ) )
    (MTGNN, Wu et al. KDD'20) with optional top-k sparsification and an
    interpolation with the correlation prior:

        A = (1 - beta) * A_prior + beta * A_adp ,  beta = sigmoid(b) learnable
    """

    def __init__(self, num_nodes: int, emb_dim: int = 16, alpha: float = 3.0,
                 topk: int | None = 4, mode: str = "hybrid", prior: np.ndarray | None = None):
        super().__init__()
        assert mode in {"corr", "adaptive", "hybrid", "identity", "complete"}
        self.mode, self.alpha, self.topk, self.N = mode, alpha, topk, num_nodes
        self.E1 = nn.Parameter(torch.randn(num_nodes, emb_dim) * 0.1)
        self.E2 = nn.Parameter(torch.randn(num_nodes, emb_dim) * 0.1)
        self.beta = nn.Parameter(torch.tensor(0.0))          # sigmoid(0) = 0.5
        if prior is None:
            prior = np.eye(num_nodes, dtype=np.float32)
        self.register_buffer("prior", torch.from_numpy(np.ascontiguousarray(prior, dtype=np.float32)))

    def adaptive(self) -> torch.Tensor:
        M1 = torch.tanh(self.alpha * self.E1)
        M2 = torch.tanh(self.alpha * self.E2)
        A = F.relu(torch.tanh(self.alpha * (M1 @ M2.t() - M2 @ M1.t())))
        if self.topk is not None and self.topk < self.N:
            mask = torch.zeros_like(A)
            _, idx = A.topk(self.topk, dim=-1)
            mask.scatter_(-1, idx, 1.0)
            A = A * mask
        return A

    def forward(self) -> torch.Tensor:
        if self.mode == "corr":
            return self.prior
        if self.mode == "identity":
            return torch.zeros_like(self.prior)
        if self.mode == "complete":
            return torch.ones_like(self.prior) - torch.eye(self.N, device=self.prior.device)
        Aa = self.adaptive()
        Aa = Aa / Aa.sum(-1, keepdim=True).clamp(min=1e-6)
        if self.mode == "adaptive":
            return Aa
        b = torch.sigmoid(self.beta)
        return (1 - b) * self.prior + b * Aa

    def regularizer(self) -> torch.Tensor:
        """L1 sparsity on the learned part (keeps the graph interpretable)."""
        if self.mode in {"adaptive", "hybrid"}:
            return self.adaptive().abs().mean()
        return torch.zeros((), device=self.prior.device)


# --------------------------------------------------------------------------- #
class MultiScaleGCN(nn.Module):
    """
    Multi-scale graph convolution over k-hop adjacencies (Eq. 5-6) with the
    MS-G3D learnable residual mask A^{res}_(k).
    """

    def __init__(self, c_in: int, c_out: int, A_scales: list[np.ndarray], dropout: float = 0.0):
        super().__init__()
        self.K = len(A_scales)
        A = torch.from_numpy(np.stack(A_scales).astype(np.float32))   # (K, N, N)
        self.register_buffer("A", A)
        self.res_mask = nn.Parameter(torch.zeros_like(A))
        self.proj = nn.Conv2d(c_in * self.K, c_out, kernel_size=1)
        self.bn = nn.BatchNorm2d(c_out)
        self.drop = nn.Dropout(dropout)
        self.residual = (nn.Identity() if c_in == c_out else nn.Conv2d(c_in, c_out, 1))

    def forward(self, x, A_dyn: torch.Tensor | None = None):
        # x: (B, C, T, N)
        A = self.A + self.res_mask
        if A_dyn is not None:                                  # dynamic graph injection
            A = A + A_dyn.unsqueeze(0)
        outs = [torch.einsum("bctv,vw->bctw", x, A[k]) for k in range(self.K)]
        y = self.proj(torch.cat(outs, dim=1))
        y = self.bn(y) + self.residual(x)
        return self.drop(F.relu(y))


# --------------------------------------------------------------------------- #
class TemporalBranch(nn.Module):
    def __init__(self, c_in, c_mid, c_out, kernel=3, dilation=1, stride=1):
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_mid, 1), nn.BatchNorm2d(c_mid), nn.ReLU(inplace=True),
            nn.Conv2d(c_mid, c_out, (kernel, 1), stride=(stride, 1),
                      padding=(pad, 0), dilation=(dilation, 1)),
            nn.BatchNorm2d(c_out),
        )

    def forward(self, x):
        return self.net(x)


class MultiScaleTCN(nn.Module):
    """Dilated multi-branch temporal convolution (Fig. 2, MS-TCN block)."""

    def __init__(self, c_in: int, c_out: int, dilations=(1, 2, 3, 4), stride: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        n_branch = len(dilations) + 2
        c_b = max(c_out // n_branch, 8)
        self.branches = nn.ModuleList(
            [TemporalBranch(c_in, c_b, c_b, 3, d, stride) for d in dilations])
        self.maxpool = nn.Sequential(
            nn.Conv2d(c_in, c_b, 1), nn.BatchNorm2d(c_b), nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 1), stride=(stride, 1), padding=(1, 0)), nn.BatchNorm2d(c_b))
        self.one = nn.Sequential(
            nn.Conv2d(c_in, c_b, 1, stride=(stride, 1)), nn.BatchNorm2d(c_b))
        self.out = nn.Conv2d(c_b * n_branch, c_out, 1)
        self.bn = nn.BatchNorm2d(c_out)
        self.drop = nn.Dropout(dropout)
        if c_in == c_out and stride == 1:
            self.res = nn.Identity()
        else:
            self.res = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride=(stride, 1)),
                                     nn.BatchNorm2d(c_out))

    def forward(self, x):
        feats = [b(x) for b in self.branches] + [self.maxpool(x), self.one(x)]
        T = min(f.shape[2] for f in feats)
        feats = [f[:, :, :T] for f in feats]
        y = self.bn(self.out(torch.cat(feats, 1)))
        r = self.res(x)[:, :, :T]
        return self.drop(F.relu(y + r))


# --------------------------------------------------------------------------- #
class MultiScaleG3D(nn.Module):
    """
    Joint spatio-temporal graph convolution (Eq. 7-10).

    A sliding window of length tau is unfolded, the Kronecker joint adjacency
    A_(tau) = I_tau (x) A~ + A_temp (x) I_N is applied, then the window is
    collapsed back to a single frame (reshape + FC).
    """

    def __init__(self, c_in: int, c_out: int, A_scales: list[np.ndarray], tau: int = 3,
                 stride: int = 1, K_t: int = 1, learn_temporal: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        self.tau, self.stride, self.N = tau, stride, A_scales[0].shape[0]
        self.K_t = K_t
        Araw = torch.from_numpy(np.stack(A_scales).astype(np.float32))  # (K, N, N) binary
        self.register_buffer("Ak", Araw)
        self.res_mask = nn.Parameter(torch.zeros_like(Araw))
        self.learn_temporal = learn_temporal
        if learn_temporal:                                    # Eq. (11)-(12)
            self.alpha_t = nn.Parameter(torch.zeros(K_t))
        self.proj = nn.Conv2d(c_in * len(A_scales), c_out, 1)
        self.bn = nn.BatchNorm2d(c_out)
        self.collapse = nn.Conv2d(c_out * tau, c_out, 1)      # window collapse (FC)
        self.drop = nn.Dropout(dropout)

    def _joint_adj(self, device):
        N, tau = self.N, self.tau
        S = torch.zeros(tau, tau, device=device)
        for k in range(1, self.K_t + 1):
            g = torch.sigmoid(self.alpha_t[k - 1]) if self.learn_temporal else 1.0
            band = (torch.abs(torch.arange(tau, device=device)[:, None]
                              - torch.arange(tau, device=device)[None, :]) == k).float()
            S = S + g * band
        I_tau = torch.eye(tau, device=device)
        I_N = torch.eye(N, device=device)
        As = []
        for k in range(self.Ak.shape[0]):
            Ak = self.Ak[k] + self.res_mask[k]
            Aj = torch.kron(I_tau, Ak + I_N) + torch.kron(S, I_N)
            As.append(normalize_adj_t(Aj, add_self_loop=False))
        return torch.stack(As)                                 # (K, tau*N, tau*N)

    def forward(self, x):
        B, C, T, N = x.shape
        tau, s = self.tau, self.stride
        if T < tau:                                            # pad short sequences
            x = F.pad(x, (0, 0, 0, tau - T))
            T = tau
        # unfold: (B, C, T', tau, N) -> (B, C, T', tau*N)
        w = x.unfold(2, tau, s)                                # (B, C, T', N, tau)
        w = w.permute(0, 1, 2, 4, 3).contiguous()              # (B, C, T', tau, N)
        Tp = w.shape[2]
        w = w.reshape(B, C, Tp, tau * N)

        Aj = self._joint_adj(x.device)
        outs = [torch.einsum("bctv,vw->bctw", w, Aj[k]) for k in range(Aj.shape[0])]
        y = F.relu(self.bn(self.proj(torch.cat(outs, 1))))     # (B, C_out, T', tau*N)
        Co = y.shape[1]
        y = y.reshape(B, Co, Tp, tau, N).permute(0, 1, 3, 2, 4).reshape(B, Co * tau, Tp, N)
        y = self.collapse(y)                                   # (B, C_out, T', N)
        return self.drop(F.relu(y))


# --------------------------------------------------------------------------- #
class STGCBlock(nn.Module):
    """One spatio-temporal graph convolution block (Fig. 2)."""

    def __init__(self, c_in, c_out, A_norm_scales, A_bin_scales, stride=1,
                 taus=(3, 5), use_g3d=True, use_factorized=True, dropout=0.1):
        super().__init__()
        assert use_g3d or use_factorized, "at least one pathway must stay active"
        self.use_g3d, self.use_fact = use_g3d, use_factorized

        if use_factorized:
            self.gcn = MultiScaleGCN(c_in, c_out, A_norm_scales, dropout)
            self.tcn1 = MultiScaleTCN(c_out, c_out, stride=stride, dropout=dropout)
        if use_g3d:
            self.g3d = nn.ModuleList([
                MultiScaleG3D(c_in, c_out, A_bin_scales, tau=t, stride=stride, dropout=dropout)
                for t in taus])
        self.agg = MultiScaleTCN(c_out, c_out, stride=1, dropout=dropout)

    def forward(self, x, A_dyn=None):
        outs = []
        if self.use_fact:
            outs.append(self.tcn1(self.gcn(x, A_dyn)))
        if self.use_g3d:
            outs += [g(x) for g in self.g3d]
        T = min(o.shape[2] for o in outs)
        y = sum(o[:, :, :T] for o in outs) / len(outs)
        return self.agg(y)
