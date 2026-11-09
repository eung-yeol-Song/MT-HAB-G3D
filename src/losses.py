"""
losses.py
=========
Objectives for the multi-task HAB forecaster.

Classification (red tide)
    bce            plain binary cross-entropy                     (manuscript)
    wbce           class-weighted BCE (pos_weight = N-/N+)        [P4]
    focal          focal loss, gamma = 2, alpha balanced          [P4]

Regression (Chl-a)
    mse            manuscript ablation
    huber          Huber, delta = 1.5                             (manuscript)
    peak_huber     Huber re-weighted by target magnitude          [P8]
    quantile       pinball loss over several quantiles            [P7]

Multi-task weighting  L = L_cls + lambda * L_reg
    fixed          lambda = 0.8                                   (manuscript)
    uncertainty    homoscedastic uncertainty weighting
                   (Kendall, Gal & Cipolla, CVPR 2018)            [P5]
    dwa            dynamic weight averaging (Liu et al., 2019)    [P5]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
def masked_mean(v: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    m = m.float()
    s = m.sum()
    if s < 1:
        return v.sum() * 0.0
    return (v * m).sum() / s


class ClassificationLoss(nn.Module):
    def __init__(self, kind="bce", pos_weight: float | None = None, gamma: float = 2.0,
                 alpha: float = 0.25, label_smoothing: float = 0.0):
        super().__init__()
        self.kind, self.gamma, self.alpha, self.ls = kind, gamma, alpha, label_smoothing
        self.pos_weight = pos_weight

    def forward(self, logit, y, mask):
        y = y.clamp(0, 1)
        if self.ls > 0:
            y = y * (1 - self.ls) + 0.5 * self.ls
        if self.kind == "focal":
            p = torch.sigmoid(logit)
            ce = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
            pt = p * y + (1 - p) * (1 - y)
            a = self.alpha * y + (1 - self.alpha) * (1 - y)
            loss = a * (1 - pt).pow(self.gamma) * ce
        else:
            pw = None
            if self.kind == "wbce" and self.pos_weight is not None:
                pw = torch.tensor(self.pos_weight, device=logit.device)
            loss = F.binary_cross_entropy_with_logits(logit, y, reduction="none",
                                                      pos_weight=pw)
        return masked_mean(loss, mask)


class RegressionLoss(nn.Module):
    def __init__(self, kind="huber", delta: float = 1.5, quantiles=(0.1, 0.5, 0.9),
                 peak_gamma: float = 1.0, peak_thr: float = 1.0):
        super().__init__()
        self.kind, self.delta = kind, delta
        self.register_buffer("q", torch.tensor(quantiles, dtype=torch.float32))
        self.peak_gamma, self.peak_thr = peak_gamma, peak_thr

    def forward(self, pred, y, mask):
        """pred: (B, H, Q) ; y, mask: (B, H)"""
        if self.kind == "quantile":
            e = y.unsqueeze(-1) - pred                              # (B, H, Q)
            q = self.q.view(1, 1, -1)
            loss = torch.maximum(q * e, (q - 1) * e).mean(-1)
        else:
            p = pred[..., pred.shape[-1] // 2]
            e = y - p
            if self.kind == "mse":
                loss = e.pow(2)
            else:
                a = e.abs()
                loss = torch.where(a <= self.delta, 0.5 * e.pow(2),
                                   self.delta * (a - 0.5 * self.delta))
            if self.kind == "peak_huber":
                # up-weight high-concentration days (standardised target scale)
                w = 1.0 + self.peak_gamma * F.relu(y - self.peak_thr)
                loss = loss * w
        return masked_mean(loss, mask)


# --------------------------------------------------------------------------- #
class MultiTaskObjective(nn.Module):
    """
    Combines the two task losses.

    weighting = "fixed"        L = L_cls + lambda L_reg
                "uncertainty"  L = exp(-s1) L_cls + s1/2 + exp(-s2) L_reg + s2/2
                "dwa"          lambda_t from the relative descent rate of losses
    """

    def __init__(self, cls_loss: ClassificationLoss, reg_loss: RegressionLoss,
                 weighting="fixed", lam: float = 0.8, temperature: float = 2.0,
                 graph_l1: float = 0.0, horizon_weights=None, cls_weight: float = 1.0):
        super().__init__()
        self.cls_loss, self.reg_loss = cls_loss, reg_loss
        self.weighting, self.lam, self.temp = weighting, lam, temperature
        # cls_weight = 0 -> regression-only single-task control; lam = 0 -> classification-only
        # (revision, R2-M2). Both heads stay in the graph so the architecture is identical.
        self.cls_weight = cls_weight
        self.graph_l1 = graph_l1
        self.log_var = nn.Parameter(torch.zeros(2))
        self.register_buffer("hist", torch.ones(2, 2))          # DWA loss history
        self.horizon_weights = horizon_weights

    def forward(self, out, batch, model=None):
        y_cls, m_cls = batch["y_cls"], batch["m_cls"]
        y_reg, m_reg = batch["y_reg"], batch["m_reg"]
        if self.horizon_weights is not None:
            w = self.horizon_weights.to(y_cls.device).view(1, -1)
            m_cls = m_cls.float() * w
            m_reg = m_reg.float() * w

        l_c = self.cls_loss(out["logit"], y_cls, m_cls)
        l_r = self.reg_loss(out["reg"], y_reg, m_reg)

        if self.weighting == "uncertainty":
            total = (torch.exp(-self.log_var[0]) * l_c + 0.5 * self.log_var[0]
                     + torch.exp(-self.log_var[1]) * l_r + 0.5 * self.log_var[1])
            lam_eff = float(torch.exp(-self.log_var[1]) / torch.exp(-self.log_var[0]))
        elif self.weighting == "dwa":
            r = self.hist[0] / self.hist[1].clamp(min=1e-8)
            w = 2 * torch.softmax(r / self.temp, dim=0)
            total = w[0] * l_c + w[1] * l_r
            lam_eff = float(w[1] / w[0].clamp(min=1e-8))
        else:
            total = self.cls_weight * l_c + self.lam * l_r
            lam_eff = self.lam

        if self.graph_l1 > 0 and model is not None and hasattr(model, "graph_regularizer"):
            total = total + self.graph_l1 * model.graph_regularizer()

        return total, {"cls": float(l_c.detach()), "reg": float(l_r.detach()),
                       "lambda": lam_eff}

    @torch.no_grad()
    def update_dwa(self, l_cls: float, l_reg: float):
        self.hist[1] = self.hist[0].clone()
        self.hist[0] = torch.tensor([l_cls, l_reg], device=self.hist.device)


def build_objective(cls_kind="bce", reg_kind="huber", weighting="fixed", lam=0.8,
                    delta=1.5, pos_weight=None, quantiles=(0.1, 0.5, 0.9),
                    peak_gamma=1.0, graph_l1=0.0, horizon_weights=None, cls_weight=1.0):
    return MultiTaskObjective(
        ClassificationLoss(cls_kind, pos_weight=pos_weight),
        RegressionLoss(reg_kind, delta=delta, quantiles=quantiles, peak_gamma=peak_gamma),
        weighting=weighting, lam=lam, graph_l1=graph_l1,
        horizon_weights=horizon_weights, cls_weight=cls_weight)
