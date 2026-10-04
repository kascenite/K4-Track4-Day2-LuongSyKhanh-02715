"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

Giao diện:
    build_criterion(kind, **kw)                 -> callable(logits, target) -> loss scalar
    class_weights(counts, beta)                 -> tensor trọng số lớp
    mix_batch(x, y, alpha, mode)                -> (x_mixed, (y_a, y_b, lam))
    mixed_loss(criterion, logits, targets)      -> loss scalar
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCE(nn.Module):
    """CE với label smoothing q'(k) = (1-eps)*1[k==y] + eps/K, tự cài đặt. eps=0 -> đúng CE."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.eps = smoothing

    def forward(self, logits, target):
        logp = F.log_softmax(logits.float(), dim=-1)
        nll = -logp.gather(1, target[:, None]).squeeze(1)
        smooth = -logp.mean(dim=-1)
        return ((1 - self.eps) * nll + self.eps * smooth).mean()


class FocalLoss(nn.Module):
    """FL(p_t) = -alpha_t (1-p_t)^gamma log p_t, trung bình batch. gamma=0 (alpha=None) -> đúng CE."""

    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", None if alpha is None else torch.as_tensor(alpha, dtype=torch.float32))

    def forward(self, logits, target):
        logp = F.log_softmax(logits.float(), dim=-1).gather(1, target[:, None]).squeeze(1)
        loss = -((1 - logp.exp()).clamp(min=0) ** self.gamma) * logp
        if self.alpha is not None:
            loss = self.alpha[target] * loss
        return loss.mean()


def class_weights(counts, beta: float = 0.0):
    """Trọng số lớp từ số ảnh mỗi lớp của TRAIN.

    beta=0: w_c ∝ 1/n_c, chuẩn hoá về trung bình 1.
    beta>0: class-balanced (Cui et al.) w_c = (1-beta)/(1-beta^n_c), chuẩn hoá tổng = số lớp.
    """
    n = torch.as_tensor(np.asarray(counts), dtype=torch.float64)
    w = 1.0 / n if not beta else (1 - beta) / (1 - torch.pow(torch.tensor(float(beta), dtype=torch.float64), n))
    return (w / w.sum() * len(n)).float()


def build_criterion(kind: str = "ce", **kw):
    """kind: "ce" | "ls" (smoothing=) | "focal" (gamma=, alpha=) | "ce_weighted" (weight=)."""
    if kind == "ce":
        return nn.CrossEntropyLoss()
    if kind == "ls":
        return LabelSmoothingCE(kw.get("smoothing", 0.1))
    if kind == "focal":
        return FocalLoss(kw.get("gamma", 2.0), kw.get("alpha"))
    if kind == "ce_weighted":
        return nn.CrossEntropyLoss(weight=torch.as_tensor(kw["weight"], dtype=torch.float32))
    raise ValueError(f"kind không hợp lệ: {kind!r}")


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix"):
    """Trộn batch. Trả về (x_mix, (y_a, y_b, lam)); lam của cutmix tính lại theo diện tích hộp thực."""
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    if mode == "mixup":
        return lam * x + (1 - lam) * x[perm], (y, y[perm], lam)
    if mode != "cutmix":
        raise ValueError(f"mode phải là mixup|cutmix, nhận {mode!r}")
    h, w = x.shape[-2:]
    cut = np.sqrt(1 - lam)
    ch, cw = int(h * cut), int(w * cut)
    cy, cx = np.random.randint(h), np.random.randint(w)
    y1, y2 = max(cy - ch // 2, 0), min(cy + ch // 2, h)
    x1, x2 = max(cx - cw // 2, 0), min(cx + cw // 2, w)
    x = x.clone()
    x[:, :, y1:y2, x1:x2] = x[perm][:, :, y1:y2, x1:x2]
    lam = 1 - (y2 - y1) * (x2 - x1) / (h * w)
    return x, (y, y[perm], lam)


def mixed_loss(criterion, logits, targets):
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
