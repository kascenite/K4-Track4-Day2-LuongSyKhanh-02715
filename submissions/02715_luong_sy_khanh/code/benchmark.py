"""benchmark.py - đo độ trễ suy luận đúng cách (warmup, synchronize, >= 50 lần, p50/p95/p99).

Điều kiện đo: chỉ forward của model trên tensor ngẫu nhiên đã nằm sẵn trên GPU (KHÔNG tính tiền xử lý
và chép host->device). Ghi rõ điều này trong báo cáo.
"""
from __future__ import annotations

import time

import numpy as np
import torch


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Thời gian (ms) của fn(): bỏ `warmup` lần đầu; sync() trước và sau mỗi lần đo."""
    sync = sync or (lambda: None)
    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(iters):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append((time.perf_counter() - t0) * 1000)
    p = np.percentile(ts, [50, 95, 99])
    return {"p50": float(p[0]), "p95": float(p[1]), "p99": float(p[2]), "mean": float(np.mean(ts)), "n": iters}


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100, fused_bn: bool = False) -> dict:
    """dtype: fp32 | amp (autocast fp16) | fp16 (model.half()). Model được deepcopy, không đổi model gốc."""
    import copy
    m = copy.deepcopy(model).to(device).eval()
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    if dtype == "fp16":
        m, x = m.half(), x.half()

    @torch.inference_mode()
    def fn():
        with torch.autocast("cuda", dtype=torch.float16, enabled=(dtype == "amp")):
            m(x)

    sync = torch.cuda.synchronize if device == "cuda" else None
    r = bench(fn, warmup, iters, sync)
    return {"gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu", "dtype": dtype,
            "batch": batch_size, "img_size": img_size, "fused_bn": fused_bn, "p50": r["p50"], "p95": r["p95"],
            "p99": r["p99"], "images_per_s": batch_size / (r["p50"] / 1000), "torch": torch.__version__}


def tta_latency(model, k_views: int, **kw) -> dict:
    """Đo thật TTA K view bằng cách chạy batch K ảnh (mô phỏng K lượt, batch-1 mỗi view chạy tuần tự)."""
    single = latency_report(model, **kw)
    import copy
    m = copy.deepcopy(model).to(kw.get("device", "cuda")).eval()
    x = torch.randn(1, 3, kw["img_size"], kw["img_size"], device=kw.get("device", "cuda"))

    @torch.inference_mode()
    def fn():
        for _ in range(k_views):
            m(x)

    r = bench(fn, kw.get("warmup", 10), kw.get("iters", 100), torch.cuda.synchronize)
    return {"k": k_views, "p50": r["p50"], "p95": r["p95"], "p99": r["p99"],
            "single_p50": single["p50"], "ratio_vs_single": r["p50"] / single["p50"]}
