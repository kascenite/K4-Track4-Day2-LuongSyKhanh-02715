"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Quy tắc chia dữ liệu bắt buộc (S1-S6) nằm ở README.md, mục 2.1. Đọc trước khi viết.

Giao diện bạn phải giữ (để notebook, train.py và eval.py ghép được với nhau):
    load_split(labels_dir, fold=0)            -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict  (số liệu để ghi báo cáo)
    build_transforms(train, img_size, aug)    -> torchvision transform
    DeepWeedsDataset[i]                       -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)
"""
from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)  # đổi nếu trọng số timm bạn dùng yêu cầu mean/std khác
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(labels_dir: str | Path, fold: int = 0):
    """Đọc train/val/test_subset{fold}.csv nguyên bản (S1). Trả về (train_df, val_df, test_df)."""
    d = Path(labels_dir)
    return tuple(pd.read_csv(d / f"{s}_subset{fold}.csv") for s in ("train", "val", "test"))


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> dict:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1). In ra và trả về dict số liệu."""
    dfs = {"train": train_df, "val": val_df, "test": test_df}
    n = {k: len(v) for k, v in dfs.items()}
    total = sum(n.values())
    per_class = {k: v["Label"].value_counts().reindex(range(NUM_CLASSES), fill_value=0).to_dict()
                 for k, v in dfs.items()}
    names = {k: set(v["Filename"]) for k, v in dfs.items()}
    overlap = {f"{a}&{b}": len(names[a] & names[b])
               for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    union = len(names["train"] | names["val"] | names["test"])
    images_dir = Path(images_dir)
    missing = [f for f in names["train"] | names["val"] | names["test"]
               if not (images_dir / f).exists()]

    print("n:", n, "| tổng:", total)
    print("tỉ lệ:", {k: round(v / total, 4) for k, v in n.items()})
    print("per_class:", pd.DataFrame(per_class).rename(index=dict(enumerate(CLASS_NAMES))), sep="\n")
    print("overlap:", overlap, "| union:", union, "| thiếu file:", len(missing))

    assert all(v == 0 for v in overlap.values()), f"train/val/test giao nhau: {overlap}"
    assert total == union == 17509, f"tổng={total}, hợp={union}, kỳ vọng 17509"
    assert all(abs(n[k] / total - r) <= 0.01 for k, r in (("train", .6), ("val", .2), ("test", .2))), \
        f"tỉ lệ lệch 60/20/20 quá 1 điểm %: {n}"
    assert not missing, f"{len(missing)} file không tồn tại, ví dụ {missing[:3]}"
    return {"n": n, "per_class": per_class, "overlap": overlap, "union": union}


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic",
                     mean=IMAGENET_MEAN, std=IMAGENET_STD, crop_min: float = 0.08):
    """Tạo transform. aug: "basic" | "color" | "trivial" | "randaug" (chỉ dùng khi train=True).

    Val/test: ảnh gốc 256x256 -> CenterCrop(img_size) nếu img_size < 256, ngược lại Resize(img_size)
    (giữ nguyên ở 256; lớn hơn dùng cho thí nghiệm độ phân giải I04). Không augmentation ngẫu nhiên.
    Chỉ lật ngang; lật dọc là một thí nghiệm riêng nếu muốn kiểm tra (GUIDE trục B).
    """
    norm = [T.ToTensor(), T.Normalize(mean, std)]
    if not train:
        resize = T.CenterCrop(img_size) if img_size < 256 else T.Resize((img_size, img_size))
        return T.Compose([resize, *norm])
    extra = {
        "basic": [],
        "color": [T.ColorJitter(0.3, 0.3, 0.3, 0.05)],
        "trivial": [T.TrivialAugmentWide()],
        "randaug": [T.RandAugment()],
    }
    if aug not in extra:
        raise ValueError(f"aug phải thuộc {list(extra)}, nhận {aug!r}")
    return T.Compose([T.RandomResizedCrop(img_size, scale=(crop_min, 1.0)), T.RandomHorizontalFlip(), *extra[aug], *norm])


_IMG_CACHE: dict = {}  # (images_dir, filename) -> uint8 array HxWx3; filled in the main process, shared by fork


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


class DeepWeedsDataset(Dataset):
    """Dataset đọc ảnh từ `images_dir` theo DataFrame (Filename, Label).

    __getitem__(i) -> (ảnh đã transform, nhãn int, tên file str).
    cache=True: giải mã toàn bộ ảnh một lần vào RAM (~3,4 GB cho 17,5k ảnh 256x256) vì Colab ít CPU.
    """

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None, cache: bool = True):
        self.files = df["Filename"].tolist()
        self.labels = df["Label"].astype(int).tolist()
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.cache = cache
        if cache:
            todo = [f for f in self.files if (str(self.images_dir), f) not in _IMG_CACHE]
            with ThreadPoolExecutor(4) as ex:
                for f, arr in zip(todo, ex.map(lambda f: _read_rgb(self.images_dir / f), todo)):
                    _IMG_CACHE[(str(self.images_dir), f)] = arr

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int):
        f = self.files[i]
        if self.cache:
            img = Image.fromarray(_IMG_CACHE[(str(self.images_dir), f)])
        else:
            img = Image.open(self.images_dir / f).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[i], f


def _seed_worker(_):
    s = torch.initial_seed() % 2**32
    random.seed(s)
    np.random.seed(s)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2, cache: bool = True):
    """DataLoader. Val/test: không shuffle, giữ thứ tự df. sampler: None | "balanced"."""
    ds = DeepWeedsDataset(df, images_dir, transform, cache)
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True,
              worker_init_fn=_seed_worker, persistent_workers=num_workers > 0)
    if not train:
        return DataLoader(ds, shuffle=False, **kw)
    if sampler is None:
        return DataLoader(ds, shuffle=True, drop_last=True, **kw)
    if sampler == "balanced":
        counts = df["Label"].value_counts()
        w = torch.tensor(1.0 / df["Label"].map(counts).to_numpy(), dtype=torch.double)
        return DataLoader(ds, sampler=WeightedRandomSampler(w, num_samples=len(ds), replacement=True),
                          drop_last=True, **kw)
    raise ValueError(f"sampler phải là None hoặc 'balanced', nhận {sampler!r}")
