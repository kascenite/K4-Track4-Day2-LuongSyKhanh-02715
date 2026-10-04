"""run_queue.py - chạy tuần tự nhiều thí nghiệm trong MỘT tiến trình (cache ảnh chỉ nạp một lần).

    python run_queue.py --jobs jobs_B.txt --common images_dir=... labels_dir=... out_dir=... pred_dir=...
Mỗi dòng của jobs file: KEY=VALUE ... (ví dụ `exp_id=B01 backbone=resnet50 seed=0`). Dòng trống/`#` bị bỏ qua.
Lần chạy đã có result.json thì BỎ QUA (chạy lại được sau khi bị ngắt). Một job lỗi không dừng cả hàng đợi.
"""
from __future__ import annotations

import argparse
import shlex
import traceback

import train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--common", nargs="*", default=[])
    a = ap.parse_args()
    common = train.parse_overrides(a.common)
    for line in open(a.jobs):
        line = line.split("#")[0].strip()
        if not line:
            continue
        cfg = train.Config(**{**common, **train.parse_overrides(shlex.split(line))})
        if (train.run_dir(cfg) / "result.json").exists():
            print(f"SKIP {cfg.exp_id} seed{cfg.seed} (đã xong)", flush=True)
            continue
        try:
            r = train.run(cfg)
            print(f"DONE {cfg.exp_id} seed{cfg.seed} F1={r['val_macro_f1']:.4f}", flush=True)
        except Exception:
            print(f"FAIL {cfg.exp_id} seed{cfg.seed}\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    main()
