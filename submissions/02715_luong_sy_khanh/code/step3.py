"""step3.py - Bước 3: so sánh phương pháp suy luận trên VAL. Không huấn luyện lại, KHÔNG đụng test.

    python step3.py --exp B01 --seed 0 [--ensemble B01 B02 B03] [--ema_exp T09] [--cnn 1] --common images_dir=... ...
Cần: checkpoint <ckpt_dir>/<exp>/seed<k>.pt và <out_dir>/<exp>/seed<k>/{config,result}.json từ train.run.
Đo độ trễ nên chạy khi GPU rảnh (không có hàng đợi huấn luyện đang chạy).
Ghi: <out_dir>/step3_<exp>/inference.csv, latency.csv, logits.npz
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import benchmark as Bm
import dataset
import inference as I
import model as M
import train
from eval import compute_metrics


def mets(probs, y):
    m = compute_metrics(y, probs.argmax(1), probs)
    return m["macro_f1"], m["top1"], m["ece"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ensemble", nargs="*", default=[], help="exp_id các mô hình cho I05 (dùng val_logits.npy)")
    ap.add_argument("--ema_exp", default=None, help="exp_id của lần chạy có EMA (I06)")
    ap.add_argument("--cnn", type=int, default=1, help="1 nếu backbone là CNN (cho phép đổi độ phân giải, gộp BN)")
    ap.add_argument("--common", nargs="*", default=[])
    a = ap.parse_args()

    cfg = train.Config(**{**train.parse_overrides(a.common), "exp_id": a.exp, "seed": a.seed})
    rd = train.run_dir(cfg)
    conf = json.load(open(rd / "config.json"))
    res = json.load(open(rd / "result.json"))
    mean, std = res["weights"]["mean"], res["weights"]["std"]
    out = Path(cfg.out_dir) / f"step3_{a.exp}"
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")

    net = M.build_model(conf["backbone"], False, 9, 0.0, "scratch")
    net.load_state_dict(torch.load(Path(cfg.ckpt_dir) / a.exp / f"seed{a.seed}.pt"))
    net.to(dev).eval()
    _, val_df, _ = dataset.load_split(cfg.labels_dir)
    loader = dataset.make_loader(val_df, cfg.images_dir, dataset.build_transforms(False, 256, "basic", mean, std),
                                 128, False, None, 2, True)

    c224 = lambda x: x[..., 16:240, 16:240]          # = CenterCrop(224) từ ảnh 256, cùng tiền xử lý lúc train/val  # noqa: E731
    rows, lat_rows, store = [], [], {}

    def lat(model, size, dtype="fp32", tag="", fused=False):
        for b in (1, 32):
            r = Bm.latency_report(model, b, size, dtype, fused_bn=fused)
            r["config"] = tag
            lat_rows.append(r)
            if b == 1:
                b1 = r
        return b1

    def add(mid, method, probs, K, y, l, note=""):
        f1, acc, ece = mets(probs, y)
        rows.append({"exp_id": mid, "method": method, "model": a.exp, "K": K, "val_macro_f1": f1,
                     "val_top1": acc, "val_ece": ece, "p50_ms": l["p50"], "p95_ms": l["p95"], "p99_ms": l["p99"],
                     "note": note})
        print(f"{mid} {method}: F1={f1:.4f} acc={acc:.4f} ECE={ece:.4f} p50={l['p50']:.2f}ms", flush=True)

    # --- I00: 1 view (mốc) ---
    names, y, (l0,) = I.predict_views(net, loader, dev, lambda x: [c224(x)])
    p0 = I.aggregate_views([l0], "prob")
    base = lat(net, 224, "fp32", "I00 fp32 224")
    add("I00", "1 view (center crop 224)", p0, 1, y, base)
    store["I00"] = l0

    # --- I01 / I03: TTA lật ngang, gộp prob vs logit ---
    _, _, L1 = I.predict_views(net, loader, dev, lambda x: [c224(x), I.view_hflip(c224(x))])
    t2 = Bm.tta_latency(net, 2, batch_size=1, img_size=224)
    for sp in ("prob", "logit"):
        add("I01" if sp == "prob" else "I03a", f"TTA flip K=2 (gộp {sp})", I.aggregate_views(L1, sp), 2, y, t2)
    store["I01"] = np.stack(L1)

    # --- I02: 5 crop, 5 crop + lật ---
    _, _, L5 = I.predict_views(net, loader, dev, lambda x: I.views_multicrop(x, 224))
    _, _, L10 = I.predict_views(net, loader, dev, lambda x: I.views_multicrop(x, 224, flip=True))
    for K, L, mid in ((5, L5, "I02a"), (10, L10, "I02b")):
        tk = Bm.tta_latency(net, K, batch_size=1, img_size=224)
        for sp in ("prob", "logit"):
            add(mid if sp == "prob" else mid + "-logit", f"TTA {'5crop' if K == 5 else '5crop+flip'} K={K} (gộp {sp})",
                I.aggregate_views(L, sp), K, y, tk)
    store["I02a"], store["I02b"] = np.stack(L5), np.stack(L10)

    # --- I04: độ phân giải kiểm tra (chỉ CNN): resize cả ảnh 256 về s, không crop ---
    if a.cnn:
        for s in (256, 288, 320):
            fn = (lambda x: [x]) if s == 256 else (lambda x, s=s: [F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False)])
            _, _, (ls,) = I.predict_views(net, loader, dev, fn)
            add(f"I04-{s}", f"Độ phân giải kiểm tra {s} (resize toàn ảnh)", I.aggregate_views([ls], "prob"), 1, y,
                lat(net, s, "fp32", f"I04 fp32 {s}"))
            store[f"I04-{s}"] = ls

    # --- I05: ensemble (dùng val_logits.npy của các lần chạy khác nhau, cùng thứ tự file) ---
    if a.ensemble:
        P, cost = [], {"p50": 0, "p95": 0, "p99": 0}
        for e in a.ensemble:
            c2 = train.Config(**{**train.parse_overrides(a.common), "exp_id": e, "seed": a.seed})
            P.append(I.aggregate_views([np.load(train.run_dir(c2) / "val_logits.npy")], "prob"))
            cf = json.load(open(train.run_dir(c2) / "config.json"))
            m2 = M.build_model(cf["backbone"], False, 9, 0.0, "scratch").to(dev)
            r = Bm.latency_report(m2, 1, 224, "fp32")
            for k in cost:
                cost[k] += r[k]
            del m2
        add("I05", f"Ensemble {'+'.join(a.ensemble)} (TB xác suất)", I.ensemble_probs(P), len(a.ensemble), y, cost,
            "độ trễ = tổng độ trễ từng mô hình")

    # --- I06: trọng số EMA (lần chạy huấn luyện riêng) ---
    if a.ema_exp:
        c2 = train.Config(**{**train.parse_overrides(a.common), "exp_id": a.ema_exp, "seed": a.seed})
        add("I06", f"Trọng số EMA ({a.ema_exp})", I.aggregate_views([np.load(train.run_dir(c2) / "val_logits.npy")], "prob"),
            1, y, base, "không tốn thêm khi suy luận")

    # --- I07: temperature scaling. T khớp trên VAL (cả val, và kiểm chứng 2-fold) ---
    T = I.fit_temperature(l0, y)
    add("I07", f"Temperature scaling T={T:.3f} (khớp trên val)", I.apply_temperature(l0, T), 1, y, base,
        "in-sample trên val; xem I07-cv cho ước lượng không thiên lệch")
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(y))
    h = len(y) // 2
    ece_cv = []
    for tr_i, te_i in ((idx[:h], idx[h:]), (idx[h:], idx[:h])):
        Tf = I.fit_temperature(l0[tr_i], y[tr_i])
        ece_cv.append((mets(I.apply_temperature(l0[te_i], Tf), y[te_i])[2], mets(I.aggregate_views([l0[te_i]]), y[te_i])[2], Tf))
    rows.append({"exp_id": "I07-cv", "method": "Temperature scaling, 2-fold trên val", "model": a.exp, "K": 1,
                 "val_ece": float(np.mean([e[0] for e in ece_cv])), "note":
                 f"ECE trước={np.mean([e[1] for e in ece_cv]):.4f}, T các fold={[round(e[2], 3) for e in ece_cv]}"})
    print("I07-cv", rows[-1], flush=True)

    # --- I08: gộp BN và FP16 ---
    _, _, (lf32,) = I.predict_views(net, loader, dev, lambda x: [c224(x)], amp=False)
    add("I08-fp32", "FP32 thuần (không autocast)", I.aggregate_views([lf32], "prob"), 1, y, base)
    lamp = lat(net, 224, "amp", "I08 amp 224")
    add("I08-amp", "AMP (autocast fp16)", I.aggregate_views([l0], "prob"), 1, y, lamp)
    half = copy.deepcopy(net).half()
    _, _, (lh,) = I.predict_views(half, loader, dev, lambda x: [c224(x).half()], amp=False)
    add("I08-fp16", "FP16 (model.half())", I.aggregate_views([lh], "prob"), 1, y, lat(net, 224, "fp16", "I08 fp16 224"))
    del half
    if a.cnn:
        try:
            fused = I.fuse_conv_bn(copy.deepcopy(net).eval()).to(dev)
            with torch.inference_mode():
                x0 = torch.randn(4, 3, 224, 224, device=dev)
                err = (fused(x0) - net(x0)).abs().max().item()
            if fused.n_fused == 0:
                print("I08-fuse: không có cặp Conv-BN (mạng dùng LayerNorm) -> không áp dụng", flush=True)
            else:
                _, _, (lfu,) = I.predict_views(fused, loader, dev, lambda x: [c224(x)], amp=False)
                add("I08-fuse", f"Gộp BN vào conv ({fused.n_fused} cặp, sai số {err:.1e})", I.aggregate_views([lfu], "prob"), 1, y,
                    lat(fused, 224, "fp32", "I08 fused-BN fp32 224", fused=True))
        except Exception as e:  # noqa: BLE001
            print("fuse_conv_bn lỗi:", repr(e), flush=True)

    df = pd.DataFrame(rows)
    p50_0 = base["p50"]
    df["cost_vs_I00"] = df["p50_ms"] / p50_0
    df.to_csv(out / "inference.csv", index=False)
    pd.DataFrame(lat_rows).to_csv(out / "latency.csv", index=False)
    np.savez_compressed(out / "logits.npz", y=y, **store)
    print(df.to_string())


if __name__ == "__main__":
    main()
