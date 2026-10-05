"""analysis3c.py - phân tích lỗi và lệch phân phối SAU khi đã chạy test (RUBRIC G và điểm thưởng).

    python analysis3c.py --exp T03 --seed 0 --ckpt_dir /tmp/ckpt --out_dir OUT/runs --pred_dir OUT/predictions \
        --images /tmp/data --labels /tmp/data/labels --out OUT/analysis
Chỉ ĐỌC các file dự đoán TEST đã có để mô tả lỗi (không chọn/đổi gì dựa trên chúng). Lệch phân phối đo trên VAL.
Ghi vào --out: confusion_{F01,T00}.png, errors_F01.csv, errors_grid_gradcam.png, shift_analysis.csv, shift_analysis.png
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

import dataset
import inference as I
import model as M
from eval import compute_metrics, confusion_matrix, read_pred

NAMES = ["Chinee apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly acacia", "Rubber vine", "Siam weed", "Snake weed", "Negative"]


def sum_conf(pattern):
    cm = 0
    for p in sorted(glob.glob(pattern)):
        r = read_pred(p)
        cm = cm + confusion_matrix(r.y_true, r.y_pred)
    return cm


def plot_cm(cm, title, path):
    norm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(9), NAMES, rotation=40, ha="right")
    ax.set_yticks(range(9), NAMES)
    for i in range(9):
        for j in range(9):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=8, color="white" if norm[i, j] > .5 else "black")
    ax.set(xlabel="dự đoán", ylabel="nhãn thật", title=title)
    fig.colorbar(im, label="tỉ lệ theo hàng")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def load_net(a, dev):
    rd = Path(a.out_dir) / a.exp / f"seed{a.seed}"
    conf, res = json.load(open(rd / "config.json")), json.load(open(rd / "result.json"))
    net = M.build_model(conf["backbone"], False, 9, 0.0, "scratch")
    net.load_state_dict(torch.load(Path(a.ckpt_dir) / a.exp / f"seed{a.seed}.pt", map_location="cpu"))
    return net.to(dev).eval(), res["weights"]["mean"], res["weights"]["std"]


def gradcam(net, x, cls):
    """Grad-CAM trên đầu ra stage cuối của ConvNeXt (x: 1x3xHxW đã chuẩn hoá). Trả về bản đồ HxW trong [0,1]."""
    store = {}

    def hook(_, __, out):
        store["a"] = out
        out.register_hook(lambda g: store.__setitem__("g", g))

    h = net.stages[-1].register_forward_hook(hook)
    net.zero_grad()
    out = net(x)
    out[0, cls].backward()
    h.remove()
    a, g = store["a"][0].detach(), store["g"][0]
    cam = F.relu((g.mean((1, 2))[:, None, None] * a).sum(0))
    cam = cam / cam.max().clamp(min=1e-8)
    return F.interpolate(cam[None, None], size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0].cpu().numpy()


def error_analysis(a, net, mean, std, dev, out):
    P = sorted(glob.glob(f"{a.pred_dir}/F01_seed*_test.csv"))
    preds = [read_pred(p) for p in P]
    base = preds[0]
    df = pd.DataFrame({"Filename": base.filenames, "y_true": base.y_true})
    for p in preds:
        df[f"pred_seed{p.seed}"] = p.y_pred
        df[f"conf_seed{p.seed}"] = p.probs.max(1)
    pc = [c for c in df.columns if c.startswith("pred_seed")]
    df["n_seeds_wrong"] = sum((df[c] != df.y_true).astype(int) for c in pc)
    err = df[df.n_seeds_wrong > 0].copy()
    err["true_name"] = err.y_true.map(dict(enumerate(NAMES)))
    err["pred_seed0_name"] = err[pc[0]].map(dict(enumerate(NAMES)))
    err.sort_values(["n_seeds_wrong", f"conf_{pc[0][5:]}"], ascending=False).to_csv(out / "errors_F01.csv", index=False)
    print(f"lỗi: {len(err)} ảnh sai ở ít nhất 1 seed; {(err.n_seeds_wrong == len(pc)).sum()} ảnh sai ở cả {len(pc)} seed")
    pair = err[((err.y_true == 0) & (err[pc[0]] == 7)) | ((err.y_true == 7) & (err[pc[0]] == 0))]
    print(f"nhầm Chinee<->Snake (seed 0): {len(pair)}")

    s0 = err[err[pc[0]] != err.y_true]                                          # sai ở seed 0 (mô hình có checkpoint T03)
    s0 = pd.concat([s0[s0.index.isin(pair.index)], s0[~s0.index.isin(pair.index)].sort_values(f"conf_{pc[0][5:]}", ascending=False)]).head(12)
    tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    fig, axes = plt.subplots(2, 6, figsize=(16, 6.4))
    for k, (_, r) in enumerate(s0.iterrows()):
        im = Image.open(Path(a.images) / r.Filename).convert("RGB")
        x = tf(im)[None].to(dev)
        cls = int(r[pc[0]])
        cam = gradcam(net, x, cls)
        ax = axes[(k // 6), k % 6]
        ax.imshow(im)
        ax.imshow(cam, cmap="jet", alpha=0.45)
        ax.set_title(f"thật: {NAMES[int(r.y_true)]}\ndự đoán: {NAMES[cls]} ({r[f'conf_{pc[0][5:]}']:.2f})", fontsize=8)
        ax.axis("off")
    for k in range(len(s0), 12):
        axes[k // 6, k % 6].axis("off")
    fig.suptitle("Ảnh test bị đoán sai (F01 seed 0) kèm Grad-CAM (vùng mô hình dựa vào để đoán nhãn sai)")
    fig.tight_layout()
    fig.savefig(out / "errors_grid_gradcam.png", dpi=110)
    plt.close(fig)


class Perturb:
    def __init__(self, kind, level):
        self.kind, self.level = kind, level

    def __call__(self, x):                                                    # x: CHW trong [0, 1]
        if self.kind == "blur":
            k = 2 * int(np.ceil(3 * self.level)) + 1
            return TF.gaussian_blur(x, [k, k], [self.level, self.level])
        if self.kind == "dark":
            return x * self.level
        if self.kind == "noise":
            return (x + torch.randn_like(x) * self.level).clamp(0, 1)
        return x


def shift_analysis(a, net, mean, std, dev, out):
    _, val_df, _ = dataset.load_split(a.labels)
    conds = [("sạch", "none", 0), ("làm mờ σ=2", "blur", 2.0), ("làm mờ σ=4", "blur", 4.0), ("tối ×0.5", "dark", 0.5),
             ("tối ×0.25", "dark", 0.25), ("nhiễu σ=0.05", "noise", 0.05), ("nhiễu σ=0.15", "noise", 0.15)]
    rows, T_fit = [], None
    for name, kind, lv in conds:
        torch.manual_seed(0)
        tf = T.Compose([T.ToTensor(), Perturb(kind, lv), T.Normalize(mean, std)])
        loader = dataset.make_loader(val_df, a.images, tf, 128, False, None, 0, True)
        _, y, (lg,) = I.predict_views(net, loader, dev, lambda x: [x])
        if T_fit is None:
            T_fit = I.fit_temperature(lg, y)                                  # T chỉ khớp trên VAL SẠCH
        p_raw, p_ts = I.apply_temperature(lg, 1.0), I.apply_temperature(lg, T_fit)
        m0, m1 = compute_metrics(y, p_raw.argmax(1), p_raw), compute_metrics(y, p_ts.argmax(1), p_ts)
        rows.append({"điều kiện": name, "macro-F1 val": m0["macro_f1"], "top-1 val": m0["top1"],
                     "ECE trước TS": m0["ece"], "ECE sau TS (T val sạch)": m1["ece"], "T": T_fit})
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "shift_analysis.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].bar(df["điều kiện"], df["macro-F1 val"], color="C0")
    ax[0].set(ylabel="macro-F1 val", title="Độ chính xác dưới lệch phân phối")
    w = 0.38
    xs = np.arange(len(df))
    ax[1].bar(xs - w / 2, df["ECE trước TS"], w, label="trước TS")
    ax[1].bar(xs + w / 2, df["ECE sau TS (T val sạch)"], w, label="sau TS")
    ax[1].set_xticks(xs, df["điều kiện"], rotation=30, ha="right")
    ax[1].set(ylabel="ECE", title="Hiệu chuẩn dưới lệch phân phối")
    ax[1].legend()
    ax[0].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out / "shift_analysis.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="T03")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plot_cm(sum_conf(f"{a.pred_dir}/F01_seed*_test.csv"), "F01 (tổng 3 seed), test", out / "confusion_F01.png")
    plot_cm(sum_conf(f"{a.pred_dir}/T00_seed*_test.csv"), "T00 mốc (tổng 3 seed), test", out / "confusion_T00.png")
    net, mean, std = load_net(a, dev)
    error_analysis(a, net, mean, std, dev, out)
    shift_analysis(a, net, mean, std, dev, out)


if __name__ == "__main__":
    main()
