"""final_test.py - Bước 4: chạy TEST đúng MỘT lần cho mỗi seed, với phương pháp suy luận đã chốt trên VAL.

    python final_test.py --train_exp F01 --name F01 --seeds 0 1 2 --method flip --space prob --temp 1 --common ...
Ghi (pred_dir):
    <name>_seed<k>_val.csv      xác suất VAL cuối cùng (đã áp T nếu --temp 1)
    <name>_seed<k>_test.csv     xác suất TEST cuối cùng
    <name>uncal_seed<k>_test.csv  TEST khi CHƯA temperature scaling (cho eval.py grade --uncal)
Từ chối ghi đè file test đã có (kỷ luật S4: đã mở test thì không chạy lại) trừ khi --force.
T luôn khớp trên VAL của cùng seed rồi áp sang test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import dataset
import inference as I
import model as M
import train
from eval import save_predictions


def view_fn(method):
    c = lambda x: x[..., 16:240, 16:240]  # noqa: E731  (= CenterCrop 224 từ ảnh 256)
    if method == "1view":
        return lambda x: [c(x)]
    if method == "flip":
        return lambda x: [c(x), I.view_hflip(c(x))]
    if method == "5crop":
        return lambda x: I.views_multicrop(x, 224)
    if method == "5cropflip":
        return lambda x: I.views_multicrop(x, 224, flip=True)
    if method.startswith("res"):
        s = int(method[3:])
        return lambda x: [x if s == 256 else F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False)]
    raise ValueError(method)


def to_z(L, space):
    """Biểu diễn 'logit' của dự đoán gộp, để áp nhiệt độ: logit trung bình, hoặc log của xác suất trung bình."""
    if space == "logit":
        return np.mean(L, axis=0)
    return np.log(np.clip(I.aggregate_views(L, "prob"), 1e-12, None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_exp", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--method", default="1view")
    ap.add_argument("--space", default="prob")
    ap.add_argument("--temp", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry_run", action="store_true", help="chỉ chạy VAL để kiểm tra code, KHÔNG đọc test và không ghi file")
    ap.add_argument("--common", nargs="*", default=[])
    a = ap.parse_args()

    base = train.parse_overrides(a.common)
    dev = torch.device("cuda")
    info = {}
    for seed in a.seeds:
        cfg = train.Config(**{**base, "exp_id": a.name, "seed": seed})
        tp = train.pred_path(cfg, "test")
        if tp.exists() and not a.force and not a.dry_run:
            raise SystemExit(f"{tp} đã tồn tại: test đã được mở cho {a.name} seed{seed}. Không chạy lại (dùng --force nếu thật sự cần).")
        tcfg = train.Config(**{**base, "exp_id": a.train_exp, "seed": seed})
        rd = train.run_dir(tcfg)
        conf, res = json.load(open(rd / "config.json")), json.load(open(rd / "result.json"))
        mean, std = res["weights"]["mean"], res["weights"]["std"]
        net = M.build_model(conf["backbone"], False, 9, 0.0, "scratch")
        net.load_state_dict(torch.load(Path(cfg.ckpt_dir) / a.train_exp / f"seed{seed}.pt"))
        net.to(dev).eval()
        _, val_df, test_df = dataset.load_split(cfg.labels_dir)
        tf = dataset.build_transforms(False, 256, "basic", mean, std)
        fn = view_fn(a.method)

        vn, vy, VL = I.predict_views(net, dataset.make_loader(val_df, cfg.images_dir, tf, 128, False, None, 2, True), dev, fn)
        zv = to_z(VL, a.space)
        T = I.fit_temperature(zv, vy) if a.temp else 1.0
        pv = I.apply_temperature(zv, T)

        if a.dry_run:
            from eval import compute_metrics
            m = compute_metrics(vy, pv.argmax(1), pv)
            print(f"[dry_run] {a.train_exp} seed{seed} method={a.method}/{a.space} T={T:.3f} val F1={m['macro_f1']:.4f} acc={m['top1']:.4f} ECE={m['ece']:.4f}", flush=True)
            continue
        tn, ty, TL = I.predict_views(net, dataset.make_loader(test_df, cfg.images_dir, tf, 128, False, None, 2, True), dev, fn)
        zt = to_z(TL, a.space)
        pt, pt_uncal = I.apply_temperature(zt, T), I.apply_temperature(zt, 1.0)

        save_predictions(train.pred_path(cfg, "val"), vn, vy, pv)
        save_predictions(tp, tn, ty, pt)
        ucfg = train.Config(**{**base, "exp_id": a.name + "uncal", "seed": seed})
        save_predictions(train.pred_path(ucfg, "test"), tn, ty, pt_uncal)
        info[seed] = {"T": T, "method": a.method, "space": a.space}
        print(f"{a.name} seed{seed}: method={a.method}/{a.space} T={T:.3f} -> {tp}", flush=True)
        del net
        torch.cuda.empty_cache()
    (Path(base.get("out_dir", "runs")) / f"final_{a.name}_info.json").write_text(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
