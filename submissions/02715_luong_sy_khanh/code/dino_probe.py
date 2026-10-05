"""dino_probe.py - điểm thưởng: linear probe trên đặc trưng DINOv2 đóng băng, so với tinh chỉnh ConvNeXt (chỉ dùng train và VAL).

    python dino_probe.py IMAGES_DIR LABELS_DIR OUT_JSON
Trích đặc trưng (CLS) bằng timm `vit_small_patch14_dinov2.lvd142m` ở 224x224 (CenterCrop từ ảnh 256), huấn luyện
LogisticRegression trên train, chọn C bằng macro-F1 val. KHÔNG dùng test.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import timm
import torch
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

import dataset
from eval import compute_metrics


@torch.inference_mode()
def feats(net, loader, dev):
    X, Y = [], []
    for x, y, _ in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
            X.append(net(x.to(dev)).float().cpu())
        Y.append(y)
    return torch.cat(X).numpy(), torch.cat(Y).numpy()


def main(images_dir, labels_dir, out_json):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224).to(dev).eval()
    cfg = timm.data.resolve_data_config({}, model=net)
    tf = dataset.build_transforms(False, 224, "basic", cfg["mean"], cfg["std"])
    tr, va, _ = dataset.load_split(labels_dir)
    mk = lambda df: dataset.make_loader(df, images_dir, tf, 128, False, None, 2, True)  # noqa: E731
    Xtr, ytr = feats(net, mk(tr), dev)
    Xva, yva = feats(net, mk(va), dev)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
    best = None
    for C in (0.01, 0.1, 1.0, 10.0):
        clf = LogisticRegression(C=C, max_iter=3000).fit(Xtr, ytr)
        p = clf.predict_proba(Xva)
        m = compute_metrics(yva, p.argmax(1), p)
        print(f"C={C}: val macro-F1={m['macro_f1']:.4f} acc={m['top1']:.4f} ECE={m['ece']:.4f}", flush=True)
        if best is None or m["macro_f1"] > best["macro_f1"]:
            best = {"C": C, "macro_f1": m["macro_f1"], "top1": m["top1"], "ece": m["ece"], "f1_per_class": [float(v) for v in m["f1"]]}
    best.update(model="vit_small_patch14_dinov2.lvd142m", feature_dim=int(Xtr.shape[1]), note="đặc trưng đóng băng + LogisticRegression; chỉ val")
    json.dump(best, open(out_json, "w"), indent=1)
    print("TỐT NHẤT", best)


if __name__ == "__main__":
    main(*sys.argv[1:4])
