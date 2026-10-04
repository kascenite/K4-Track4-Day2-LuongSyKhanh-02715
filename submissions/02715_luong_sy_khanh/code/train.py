"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F). Một hàm `run(cfg)` duy nhất.

Chạy từ dòng lệnh:
    python train.py --set exp_id=B01 backbone=resnet50 seed=0 images_dir=... labels_dir=...
Macro-F1 chọn checkpoint tính bằng eval.compute_metrics (cùng định nghĩa lúc chấm).
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE.parent, HERE.parent.parent, HERE.parent.parent.parent):  # tìm eval.py của repo gốc
    if (_p / "eval.py").exists():
        sys.path.insert(0, str(_p))
        break
from eval import compute_metrics, save_predictions  # noqa: E402

import dataset  # noqa: E402
import losses  # noqa: E402
import model as M  # noqa: E402


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    desc: str = ""                    # mô tả ngắn cho tên ảnh curves/<exp_id>_<desc>.png
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug
    crop_min: float = 0.08            # RandomResizedCrop scale=(crop_min, 1); 0.08 = mặc định torchvision
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    grad_clip: float = 1.0
    num_workers: int = 2
    cache: bool = True
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"             # config.json, history.csv, logit của từng lần chạy
    pred_dir: str = "predictions"     # file dự đoán đúng định dạng eval.py
    curves_dir: str = "curves"
    ckpt_dir: str = "/content/ckpt"   # checkpoint tốt nhất (ổ cục bộ, không đẩy lên Drive)
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    """random, numpy, torch CPU+CUDA. cudnn.benchmark=True để nhanh: KHÔNG bit-for-bit tái lập
    (sai khác cỡ nhiễu số học); seed vẫn cố định khởi tạo head, thứ tự batch và augmentation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_optimizer(model, cfg: Config):
    return torch.optim.AdamW(M.param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay))


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Theo bước: warmup tuyến tính (từ 1% LR) rồi cosine về 0."""
    total = cfg.epochs * steps_per_epoch
    warm = max(1, int(cfg.warmup_epochs * steps_per_epoch))

    def f(step):
        if step < warm:
            return 0.01 + 0.99 * step / warm
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total - warm))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


class EMA:
    """W_ema <- d*W_ema + (1-d)*W (d tăng dần từ đầu: min(decay, (1+n)/(10+n))). Buffer BN được sao chép."""

    def __init__(self, model, decay: float):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay, self.n = decay, 0

    @torch.no_grad()
    def update(self, model) -> None:
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        for e, p in zip(self.module.parameters(), model.parameters()):
            e.mul_(d).add_(p.detach(), alpha=1 - d)
        for e, b in zip(self.module.buffers(), model.buffers()):
            e.copy_(b)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None) -> dict:
    model.train()
    if cfg.init == "frozen":                 # BN của backbone phải ở eval; chỉ head ở train
        model.eval()
        model.get_classifier().train()
    tot, n, lrs = 0.0, 0, []
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        targets = None
        if cfg.mix:
            x, targets = losses.mix_batch(x, y, cfg.mix_alpha, cfg.mix)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=cfg.amp):
            out = model(x)
        out = out.float()
        loss = losses.mixed_loss(criterion, out, targets) if targets else criterion(out, y)
        scaler.scale(loss).backward()
        if cfg.grad_clip:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if ema is not None:
            ema.update(model)
        tot += loss.item() * len(y)
        n += len(y)
        lrs.append(optimizer.param_groups[0]["lr"])
    return {"train_loss": tot / n, "lr_steps": lrs}


@torch.inference_mode()
def evaluate(model, loader, criterion, device, amp: bool = True):
    """(filenames, y_true[N], logits[N,9] float32, loss) theo đúng thứ tự loader."""
    model.eval()
    names, ys, outs = [], [], []
    for x, y, f in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            o = model(x.to(device, non_blocking=True))
        outs.append(o.float().cpu())
        ys.append(y)
        names += list(f)
    logits = torch.cat(outs)
    y_true = torch.cat(ys)
    return names, y_true.numpy(), logits.numpy(), criterion(logits, y_true).item()


def plot_curves(history: list[dict], path: str | Path, title: str, lr_steps=None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = pd.DataFrame(history)
    ncol = 3 if lr_steps is not None else 2
    fig, ax = plt.subplots(1, ncol, figsize=(5 * ncol, 3.8))
    ax[0].plot(h.epoch, h.train_loss, "o-", label="train loss")
    ax[0].plot(h.epoch, h.val_loss, "s-", label="val loss (CE)")
    ax[0].set(xlabel="epoch", ylabel="loss", title="Loss")
    ax[1].plot(h.epoch, h.val_macro_f1, "s-", c="C2", label="val macro-F1")
    ax[1].plot(h.epoch, h.val_top1, "^--", c="C1", label="val top-1")
    ax[1].set(xlabel="epoch", ylabel="score", title="Val metrics")
    if lr_steps is not None:
        ax[2].plot(lr_steps)
        ax[2].set(xlabel="step", ylabel="LR (nhóm backbone)", title="LR schedule")
    for a in ax[:2]:
        a.grid(alpha=.3)
        a.legend()
    fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run(cfg: Config) -> dict:
    """Huấn luyện một cấu hình, lưu config/history/logit/dự đoán val (+test nếu bật), trả về tóm tắt."""
    dev = torch.device("cuda")
    set_seed(cfg.seed)
    rd = run_dir(cfg)
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=1))

    train_df, val_df, test_df = dataset.load_split(cfg.labels_dir, cfg.fold)
    dataset.check_split(train_df, val_df, test_df, cfg.images_dir)  # dừng nếu vi phạm S1-S4

    net = M.build_model(cfg.backbone, True, dataset.NUM_CLASSES, cfg.drop_rate, cfg.init).to(dev)
    tag = M.weight_tag(net)
    mean, std = (tag["mean"], tag["std"]) if tag.get("mean") else (dataset.IMAGENET_MEAN, dataset.IMAGENET_STD)
    ttf = dataset.build_transforms(True, cfg.img_size, cfg.aug, mean, std, cfg.crop_min)
    vtf = dataset.build_transforms(False, cfg.img_size, "basic", mean, std)
    train_loader = dataset.make_loader(train_df, cfg.images_dir, ttf, cfg.batch_size, True,
                                       cfg.sampler, cfg.num_workers, cfg.cache)
    val_loader = dataset.make_loader(val_df, cfg.images_dir, vtf, cfg.batch_size * 2, False,
                                     None, cfg.num_workers, cfg.cache)

    counts = train_df["Label"].value_counts().reindex(range(dataset.NUM_CLASSES), fill_value=0).to_numpy()
    if cfg.loss == "ce_weighted":
        crit = losses.build_criterion("ce_weighted", weight=losses.class_weights(counts, cfg.class_weight_beta or 0.0))
    elif cfg.loss == "ls":
        crit = losses.build_criterion("ls", smoothing=cfg.label_smoothing or 0.1)
    elif cfg.loss == "focal":
        crit = losses.build_criterion("focal", gamma=cfg.focal_gamma)
    else:
        crit = losses.build_criterion("ce")
    crit = crit.to(dev)
    val_crit = nn.CrossEntropyLoss()                       # val loss luôn là CE thường, so sánh được giữa các loss

    opt = build_optimizer(net, cfg)
    sched = build_scheduler(opt, cfg, len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)
    ema = EMA(net, cfg.ema_decay) if cfg.ema_decay else None

    history, best_f1, best_state, best_ep, lr_steps = [], -1.0, None, -1, []
    t_train = 0.0
    for ep in range(1, cfg.epochs + 1):
        t0 = time.perf_counter()
        tr = train_one_epoch(net, train_loader, crit, opt, sched, scaler, cfg, dev, ema)
        torch.cuda.synchronize()
        t_train += time.perf_counter() - t0
        lr_steps += tr.pop("lr_steps")
        ev_model = ema.module if ema else net
        _, y, lg, vloss = evaluate(ev_model, val_loader, val_crit, dev, cfg.amp)
        pr = torch.softmax(torch.from_numpy(lg), 1).numpy()
        m = compute_metrics(y, pr.argmax(1), pr)
        history.append({"epoch": ep, **tr, "val_loss": vloss, "val_macro_f1": m["macro_f1"],
                        "val_top1": m["top1"], "lr_end": lr_steps[-1], "epoch_time_s": time.perf_counter() - t0})
        print(f"[{cfg.exp_id} s{cfg.seed}] ep{ep} train_loss={tr['train_loss']:.4f} val_loss={vloss:.4f} "
              f"F1={m['macro_f1']:.4f} acc={m['top1']:.4f} t={history[-1]['epoch_time_s']:.0f}s", flush=True)
        if m["macro_f1"] > best_f1:                        # hòa -> giữ epoch sớm hơn
            best_f1, best_ep = m["macro_f1"], ep
            best_state = {k: v.detach().cpu().clone() for k, v in ev_model.state_dict().items()}

    final = ema.module if ema else net
    final.load_state_dict(best_state)
    ck = Path(cfg.ckpt_dir) / cfg.exp_id
    ck.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, ck / f"seed{cfg.seed}.pt")

    names, y, lg, _ = evaluate(final, val_loader, val_crit, dev, cfg.amp)
    pr = torch.softmax(torch.from_numpy(lg), 1).numpy()
    np.save(rd / "val_logits.npy", lg)
    save_predictions(pred_path(cfg, "val"), names, y, pr)
    mv = compute_metrics(y, pr.argmax(1), pr)

    if cfg.save_test_predictions:                          # chỉ Bước 4: đúng MỘT lần
        test_loader = dataset.make_loader(test_df, cfg.images_dir, vtf, cfg.batch_size * 2, False,
                                          None, cfg.num_workers, cfg.cache)
        tn, ty, tl, _ = evaluate(final, test_loader, val_crit, dev, cfg.amp)
        tp = torch.softmax(torch.from_numpy(tl), 1).numpy()
        np.save(rd / "test_logits.npy", tl)
        save_predictions(pred_path(cfg, "test"), tn, ty, tp)

    pd.DataFrame(history).to_csv(rd / "history.csv", index=False)
    np.save(rd / "lr_steps.npy", np.asarray(lr_steps, dtype=np.float32))
    suffix = "" if cfg.seed == 0 else f"_seed{cfg.seed}"
    plot_curves(history, Path(cfg.curves_dir) / f"{cfg.exp_id}_{cfg.desc or cfg.backbone}{suffix}.png",
                f"{cfg.exp_id} - {cfg.backbone} (seed {cfg.seed})", lr_steps)

    summary = {"exp_id": cfg.exp_id, "backbone": cfg.backbone, "seed": cfg.seed, "best_epoch": best_ep,
               "val_macro_f1": mv["macro_f1"], "val_top1": mv["top1"], "val_ece": mv["ece"],
               "val_f1_per_class": [float(v) for v in mv["f1"]],
               "train_time_per_epoch_s": t_train / cfg.epochs, "params_m": M.count_params(net),
               "gmac": M.count_gmacs(net, cfg.img_size), "weights": tag,
               "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
    (rd / "result.json").write_text(json.dumps(summary, indent=1))     # result.json = lần chạy đã xong
    del net, ema, final, opt
    torch.cuda.empty_cache()
    return summary


def parse_overrides(pairs: list[str]) -> dict:
    """['seed=1', 'loss=focal', 'ema_decay=none'] -> dict ép kiểu theo field của Config."""
    types = {f.name: str(f.type) for f in dataclasses.fields(Config)}
    out = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"Cần dạng KEY=VALUE, nhận {p!r}")
        k, v = p.split("=", 1)
        if k not in types:
            raise KeyError(f"{k!r} không có trong Config; các key hợp lệ: {sorted(types)}")
        t = types[k]
        if v.lower() in ("none", "null") and "None" in t:
            out[k] = None
        elif "bool" in t:
            out[k] = v.lower() in ("1", "true", "yes")
        elif "int" in t and "float" not in t:
            out[k] = int(v)
        elif "float" in t:
            out[k] = float(v)
        else:
            out[k] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    res = run(Config(**parse_overrides(args.set)))
    print(json.dumps({k: v for k, v in res.items() if k != "val_f1_per_class"}, indent=1))


if __name__ == "__main__":
    main()
