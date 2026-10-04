"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

Giao diện:
    build_model(name, pretrained, num_classes, drop_rate, init) -> nn.Module
    freeze_backbone(model)                                        -> None
    param_groups(model, lr_backbone, lr_head, weight_decay)       -> list[dict] cho optimizer
    count_params(model) -> float (triệu)     count_gmacs(model, img_size) -> float
    weight_tag(model) -> dict (tag trọng số timm thực sự được tải, để ghi vào results.xlsx)
"""
from __future__ import annotations

import timm
import torch
from torch.utils.flop_counter import FlopCounterMode

SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",
    "mobilenetv3": "mobilenetv3_large_100",
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune"):
    """Model phân loại 9 lớp. init: scratch | frozen | finetune (trục A)."""
    if init not in ("scratch", "frozen", "finetune"):
        raise ValueError(f"init phải là scratch|frozen|finetune, nhận {init!r}")
    model = timm.create_model(name, pretrained=(init != "scratch" and pretrained),
                              num_classes=num_classes, drop_rate=drop_rate)
    if init == "frozen":
        freeze_backbone(model)
    return model


def weight_tag(model) -> dict:
    """Thông tin trọng số tiền huấn luyện timm đã tải (rỗng nếu train từ đầu)."""
    c = getattr(model, "pretrained_cfg", None) or {}
    return {k: c.get(k) for k in ("architecture", "tag", "hf_hub_id", "mean", "std", "input_size")}


def _head_params(model):
    return list(model.get_classifier().parameters())


def freeze_backbone(model) -> None:
    """requires_grad=False cho mọi tham số trừ head. BN của backbone phải ở eval khi train
    (train.train_one_epoch xử lý: model.eval() rồi chỉ bật train cho head)."""
    head = {id(p) for p in _head_params(model)}
    for p in model.parameters():
        p.requires_grad = id(p) in head


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    """3 nhóm tham số (slide trang 52): backbone ndim>1 (có wd), norm/bias backbone (wd=0), head (có wd)."""
    head = {id(p) for p in _head_params(model)}
    bb_w, bb_nb, hd = [], [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if id(p) in head:
            hd.append(p)
        elif p.ndim > 1:
            bb_w.append(p)
        else:
            bb_nb.append(p)
    groups = [
        {"params": bb_w, "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": bb_nb, "lr": lr_backbone, "weight_decay": 0.0},
        {"params": hd, "lr": lr_head, "weight_decay": weight_decay},
    ]
    return [g for g in groups if g["params"]]


def count_params(model) -> float:
    """Số tham số (triệu), gồm cả tham số đóng băng."""
    return sum(p.numel() for p in model.parameters()) / 1e6


@torch.no_grad()
def count_gmacs(model, img_size: int = 224) -> float:
    """GMAC/ảnh bằng torch.utils.flop_counter (đếm conv, matmul, attention; MAC = FLOPs / 2).
    Không đếm các phép phần tử (BN, activation). Có thể lệch vài % so với fvcore/thop."""
    was_training = model.training
    model.eval()
    dev = next(model.parameters()).device
    x = torch.zeros(1, 3, img_size, img_size, device=dev)
    with FlopCounterMode(display=False) as fc:
        model(x)
    model.train(was_training)
    return fc.get_total_flops() / 2 / 1e9
