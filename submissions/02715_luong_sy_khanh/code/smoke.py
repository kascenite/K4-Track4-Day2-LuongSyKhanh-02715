"""smoke.py - kiểm tra pipeline trước khi chạy thật (GUIDE.md mục 1.3). Chạy trên Colab (cần GPU).

    python smoke.py IMAGES_DIR LABELS_DIR
Không đụng tới test set: chỉ dùng train/val.
"""
from __future__ import annotations

import sys

import numpy as np
import torch
import torch.nn.functional as F

import dataset
import inference
import losses
import model as M


def main(images_dir, labels_dir):
    dev = torch.device("cuda")
    torch.manual_seed(0)
    train_df, val_df, test_df = dataset.load_split(labels_dir)
    dataset.check_split(train_df, val_df, test_df, images_dir)

    # 1) loss ban đầu ~ ln 9 với head mới (finetune, backbone resnet50)
    net = M.build_model("resnet50", True, 9, 0.0, "finetune").to(dev)
    sub = train_df.sample(256, random_state=0)
    tf = dataset.build_transforms(True, 224, "basic")
    loader = dataset.make_loader(sub, images_dir, tf, 64, True, num_workers=2)
    net.eval()
    with torch.inference_mode():
        x, y, _ = next(iter(loader))
        l0 = F.cross_entropy(net(x.to(dev)).float(), y.to(dev)).item()
    print(f"loss khởi tạo = {l0:.3f} (kỳ vọng ~2.197)")

    # 2) focal gamma=0 == CE
    lg, t = torch.randn(64, 9), torch.randint(0, 9, (64,))
    d = abs(losses.FocalLoss(0.0)(lg, t) - F.cross_entropy(lg, t)).item()
    d2 = abs(losses.LabelSmoothingCE(0.0)(lg, t) - F.cross_entropy(lg, t)).item()
    print(f"|focal(γ=0) - CE| = {d:.2e}; |LS(ε=0) - CE| = {d2:.2e}")
    assert d < 1e-6 and d2 < 1e-6

    # 3) overfit một batch nhỏ
    xb, yb = x[:16].to(dev), y[:16].to(dev)
    net.train()
    opt = torch.optim.AdamW(M.param_groups(net, 1e-4, 1e-3, 0.0))
    for i in range(60):
        opt.zero_grad()
        loss = F.cross_entropy(net(xb).float(), yb)
        loss.backward()
        opt.step()
    print(f"loss sau 60 bước trên 16 ảnh = {loss.item():.4f} (kỳ vọng < 0.05)")

    # 4) mix_batch: lam hợp lệ, hình dạng giữ nguyên
    xm, (ya, yb2, lam) = losses.mix_batch(x, y, 1.0, "cutmix")
    assert xm.shape == x.shape and 0 <= lam <= 1
    print(f"cutmix lam = {lam:.3f}")

    # 5) temperature scaling trên logits ngẫu nhiên có độ tự tin thừa
    z = np.random.randn(500, 9) * 5
    yy = np.random.randint(0, 9, 500)
    print("T =", inference.fit_temperature(z, yy))

    # 6) gộp BN: sai số so với model gốc
    fused = inference.fuse_conv_bn(net.eval())
    with torch.inference_mode():
        err = (fused(xb) - net(xb)).abs().max().item()
    print(f"fuse_conv_bn: {fused.n_fused} cặp, sai số lớn nhất = {err:.2e}")

    print("params", M.count_params(net), "GMAC", M.count_gmacs(net, 224))
    torch.save(x[:16].cpu(), "/tmp/aug_batch.pt")
    print("SMOKE OK")


if __name__ == "__main__":
    main(*sys.argv[1:3])
