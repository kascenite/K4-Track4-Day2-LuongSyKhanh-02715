"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

Mọi hàm chạy ở eval, không gradient. Chọn phương pháp CHỈ dựa trên val; T khớp trên VAL.
    predict_logits(model, loader, device, view=None, amp=True) -> (filenames, y_true, logits[N, 9])
    predict_views(model, loader, device, views_fn, amp=True)   -> (filenames, y_true, [logits[N, 9] cho từng view])
    aggregate_views(list_of_logits, space)                     -> probs[N, 9]
    fit_temperature(val_logits, val_labels)                    -> float T
    apply_temperature(logits, T)                               -> probs
    ensemble_probs(list_of_probs)                              -> probs
    fuse_conv_bn(model)                                        -> model (BN đã gộp vào conv)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _softmax(z, axis=-1):
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


@torch.inference_mode()
def predict_views(model, loader, device, views_fn, amp: bool = True):
    """views_fn(x) -> list các batch (mỗi view một batch). Trả về logit từng view, cùng thứ tự file."""
    model.eval()
    names, ys, outs = [], [], None
    for x, y, f in loader:
        views = views_fn(x.to(device, non_blocking=True))
        if outs is None:
            outs = [[] for _ in views]
        for k, v in enumerate(views):
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                outs[k].append(model(v).float().cpu())
        ys.append(y)
        names += list(f)
    return names, torch.cat(ys).numpy(), [torch.cat(o).numpy() for o in outs]


def predict_logits(model, loader, device, view=None, amp: bool = True):
    fn = (lambda x: [x]) if view is None else (lambda x: [view(x)])
    names, y, outs = predict_views(model, loader, device, fn, amp)
    return names, y, outs[0]


def view_identity(x):
    return x


def view_hflip(x):
    return torch.flip(x, dims=[-1])


def views_multicrop(x, crop: int, flip: bool = False):
    """5 crop (4 góc + giữa) cỡ `crop` từ batch (N,C,H,W); flip=True thêm bản lật của từng crop."""
    h, w = x.shape[-2:]
    t, l = h - crop, w - crop
    c = [x[..., :crop, :crop], x[..., :crop, l:], x[..., t:, :crop], x[..., t:, l:],
         x[..., t // 2:t // 2 + crop, l // 2:l // 2 + crop]]
    return c + [view_hflip(v) for v in c] if flip else c


def views_multiscale(x, sizes):
    """Resize batch về từng size (bilinear). Chỉ hợp với CNN có global pooling."""
    return [x if x.shape[-1] == s else F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False)
            for s in sizes]


def aggregate_views(logits_per_view, space: str = "prob"):
    """space="prob": trung bình softmax; "logit": trung bình logit rồi softmax."""
    if space == "prob":
        return np.mean([_softmax(l) for l in logits_per_view], axis=0)
    if space == "logit":
        return _softmax(np.mean(logits_per_view, axis=0))
    raise ValueError("space phải là prob|logit")


def ensemble_probs(list_of_probs):
    return np.mean(list_of_probs, axis=0)


def fit_temperature(val_logits, val_labels) -> float:
    """T > 0 cực tiểu NLL trên VAL (lưới thô trên log T rồi tinh hai lần)."""
    z = np.asarray(val_logits, dtype=np.float64)
    y = np.asarray(val_labels)

    def nll(T):
        lp = np.log(np.clip(_softmax(z / T), 1e-12, None))
        return -lp[np.arange(len(y)), y].mean()

    lo, hi = np.log(0.05), np.log(20.0)
    for _ in range(3):
        grid = np.linspace(lo, hi, 81)
        best = grid[int(np.argmin([nll(np.exp(g)) for g in grid]))]
        step = (hi - lo) / 80
        lo, hi = best - 2 * step, best + 2 * step
    return float(np.exp(best))


def apply_temperature(logits, T: float):
    return _softmax(np.asarray(logits, dtype=np.float64) / T)


def fuse_conv_bn(model):
    """Gộp BN vào Conv liền trước bằng torch.fx (chính xác lúc suy luận). Trả về GraphModule mới.

    Chỉ áp dụng cho kiến trúc dùng BatchNorm và trace được (ResNet/ResNeXt). timm dùng BatchNormAct2d
    (BN + activation); sau khi gộp, BN được thay bằng activation của nó. ViT/Swin/ConvNeXt dùng LayerNorm
    nên không áp dụng.
    """
    import torch.fx as fx
    from torch.nn.utils.fusion import fuse_conv_bn_eval

    model = model.eval()

    class Tr(fx.Tracer):
        def is_leaf_module(self, m, name):
            return isinstance(m, nn.BatchNorm2d) or super().is_leaf_module(m, name)

    graph = Tr().trace(model)
    gm = fx.GraphModule(model, graph)
    mods = dict(gm.named_modules())

    def set_mod(target, new):
        parent, _, leaf = target.rpartition(".")
        setattr(gm.get_submodule(parent) if parent else gm, leaf, new)

    n_fused = 0
    for node in list(gm.graph.nodes):
        if node.op != "call_module" or not isinstance(mods[node.target], nn.BatchNorm2d):
            continue
        prev = node.args[0]
        if (prev.op == "call_module" and isinstance(mods[prev.target], nn.Conv2d) and len(prev.users) == 1):
            bn = mods[node.target]
            set_mod(prev.target, fuse_conv_bn_eval(mods[prev.target], bn))
            act = getattr(bn, "act", None)
            set_mod(node.target, act if act is not None else nn.Identity())
            n_fused += 1
    gm.recompile()
    gm.n_fused = n_fused
    return gm
