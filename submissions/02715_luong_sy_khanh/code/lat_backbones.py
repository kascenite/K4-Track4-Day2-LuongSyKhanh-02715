"""lat_backbones.py - độ trễ forward của từng backbone (kiến trúc, không cần trọng số), batch 1 và 32.

    python lat_backbones.py OUT.csv resnet50 resnext50_32x4d ...
Chạy khi GPU rảnh. Không tính tiền xử lý; đầu vào ngẫu nhiên đã nằm trên GPU (xem benchmark.py).
"""
from __future__ import annotations

import sys

import pandas as pd

import benchmark as Bm
import model as M


def main(out, names):
    rows = []
    for n in names:
        m = M.build_model(n, False, 9, 0.0, "scratch").cuda().eval()
        for dtype in ("fp32", "amp"):
            for b in (1, 32):
                r = Bm.latency_report(m, b, 224, dtype)
                r.update(backbone=n, params_m=M.count_params(m))
                rows.append(r)
                print(n, dtype, b, f"p50={r['p50']:.2f} p95={r['p95']:.2f} p99={r['p99']:.2f}", flush=True)
        del m
    pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
