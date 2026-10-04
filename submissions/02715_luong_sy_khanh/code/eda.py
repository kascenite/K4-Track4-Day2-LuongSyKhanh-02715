"""eda.py - EDA và bằng chứng kiểm tra pipeline (RUBRIC A). Chỉ dùng train/val/test để ĐẾM và XEM ẢNH, không huấn luyện.

    python eda.py IMAGES_DIR LABELS_DIR OUT_DIR
Ghi vào OUT_DIR: class_counts.csv (kèm đối chiếu Table 1), class_distribution.png, samples_per_class.png,
augmented_samples.png (ảnh sau augmentation đã giải chuẩn hoá cùng nhãn), image_stats.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import dataset

TABLE1 = {0: 1125, 1: 1064, 2: 1031, 3: 1022, 4: 1062, 5: 1009, 6: 1074, 7: 1016, 8: 9106}  # Olsen et al. 2019, Table 1


def main(images_dir, labels_dir, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tr, va, te = dataset.load_split(labels_dir)
    stats = dataset.check_split(tr, va, te, images_dir)

    # 1) đếm theo lớp, đối chiếu Table 1
    cols = {k: pd.Series(v) for k, v in stats["per_class"].items()}
    df = pd.DataFrame(cols)
    df.index = dataset.CLASS_NAMES
    df["total"] = df.sum(axis=1)
    df["table1"] = [TABLE1[i] for i in range(9)]
    df["khớp_table1"] = df["total"] == df["table1"]
    df["tỉ_lệ_train/val/test"] = [f"{r.train / r.total:.3f}/{r.val / r.total:.3f}/{r.test / r.total:.3f}" for r in df.itertuples()]
    df.to_csv(out / "class_counts.csv")
    big, small = df["total"].max(), df["total"][:8].min()
    print(df.to_string())
    print(f"Tỉ lệ lớp lớn nhất (Negatives) / lớp loài nhỏ nhất = {big / small:.2f}; Negatives = {big / df['total'].sum():.1%} tổng")

    fig, ax = plt.subplots(figsize=(9, 4))
    w = 0.27
    x = np.arange(9)
    for k, (s, c) in enumerate(zip(("train", "val", "test"), ("C0", "C1", "C2"))):
        ax.bar(x + (k - 1) * w, df[s], w, label=s, color=c)
    ax.set_xticks(x, dataset.CLASS_NAMES, rotation=30, ha="right")
    ax.set(ylabel="số ảnh", title="DeepWeeds fold 0: số ảnh mỗi lớp trong train/val/test")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "class_distribution.png", dpi=130)
    plt.close(fig)

    # 2) 3 ảnh mỗi lớp (từ train)
    fig, axes = plt.subplots(9, 3, figsize=(6, 18))
    rng = np.random.RandomState(0)
    for c in range(9):
        files = tr[tr.Label == c].Filename.to_numpy()
        for j, f in enumerate(rng.choice(files, 3, replace=False)):
            axes[c, j].imshow(Image.open(Path(images_dir) / f).convert("RGB"))
            axes[c, j].axis("off")
        axes[c, 0].set_title(dataset.CLASS_NAMES[c], fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(out / "samples_per_class.png", dpi=90)
    plt.close(fig)

    # 3) ảnh sau augmentation (giải chuẩn hoá) + nhãn: kiểm tra ảnh và nhãn khớp nhau
    tf = dataset.build_transforms(True, 224, "basic")
    ds = dataset.DeepWeedsDataset(tr.sample(12, random_state=1), images_dir, tf, cache=False)
    mean, std = np.array(dataset.IMAGENET_MEAN)[:, None, None], np.array(dataset.IMAGENET_STD)[:, None, None]
    fig, axes = plt.subplots(2, 6, figsize=(14, 5))
    for a, i in zip(axes.ravel(), range(12)):
        img, y, f = ds[i]
        a.imshow(np.clip(img.numpy() * std + mean, 0, 1).transpose(1, 2, 0))
        a.set_title(f"{dataset.CLASS_NAMES[y]} ({y})", fontsize=8)
        a.axis("off")
    fig.suptitle("Sau augmentation (RandomResizedCrop + flip), đã giải chuẩn hoá")
    fig.tight_layout()
    fig.savefig(out / "augmented_samples.png", dpi=110)
    plt.close(fig)

    # 4) thống kê ảnh
    sizes, modes = {}, {}
    for f in pd.concat([tr, va, te]).Filename.sample(500, random_state=0):
        with Image.open(Path(images_dir) / f) as im:
            sizes[im.size] = sizes.get(im.size, 0) + 1
            modes[im.mode] = modes.get(im.mode, 0) + 1
    (out / "image_stats.txt").write_text(f"kích thước (mẫu 500 ảnh): {sizes}\nchế độ màu: {modes}\n")
    print("kích thước:", sizes, "| chế độ:", modes)


if __name__ == "__main__":
    main(*sys.argv[1:4])
