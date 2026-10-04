"""make_results.py - gộp mọi lần chạy thành results.xlsx (GUIDE 6.1). Mọi số đọc từ file thật, không gõ tay.

    python make_results.py --root /content/drive/MyDrive/lab_day2 --out results.xlsx \
        --final F01 --baseline T00 --seeds 0 1 2 [--test-csv .../test_subset0.csv --labels .../labels.csv]
Đọc: <root>/runs/<exp>/seed<k>/{config,result}.json, <root>/predictions/*.csv (định dạng eval.py),
     <root>/latency_backbones.csv, <root>/runs/step3_*/{inference,latency}.csv
Số val của chung kết/mốc lấy từ <name>_seed<k>_val.csv, số test từ <name>_seed<k>_test.csv (tính bằng eval.compute_metrics).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE.parent, HERE.parent.parent, HERE.parent.parent.parent):
    if (_p / "eval.py").exists():
        sys.path.insert(0, str(_p))
        break
import eval as ev  # noqa: E402

CLASSES = ["Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia", "Rubber Vine",
           "Siam Weed", "Snake Weed", "Negatives"]
AXIS = {"init": "A. Khởi tạo", "aug": "B. Augmentation", "mix": "B. Augmentation", "crop_min": "B. Augmentation",
        "loss": "C. Loss", "label_smoothing": "C. Loss", "focal_gamma": "C. Loss", "class_weight_beta": "C. Loss",
        "sampler": "D. Cân bằng mẫu", "lr_backbone": "E. LR/optimizer", "lr_head": "E. LR/optimizer",
        "warmup_epochs": "E. LR/optimizer", "ema_decay": "F. Chính quy hoá", "weight_decay": "F. Chính quy hoá",
        "drop_rate": "F. Chính quy hoá", "img_size": "G. Độ phân giải/epoch", "epochs": "G. Độ phân giải/epoch",
        "backbone": "Backbone"}
IGNORE = {"exp_id", "seed", "desc", "images_dir", "labels_dir", "out_dir", "pred_dir", "curves_dir", "ckpt_dir",
          "save_test_predictions", "num_workers", "cache", "fold"}


def load_runs(root):
    rows = []
    for rj in sorted(glob.glob(f"{root}/runs/*/seed*/result.json")):
        r = json.load(open(rj))
        c = json.load(open(Path(rj).with_name("config.json")))
        w = r.get("weights") or {}
        rows.append({**r, "cfg": c, "tag": f"{w.get('architecture')}.{w.get('tag')}" if w.get("tag") else str(w.get("architecture")),
                     "hub": w.get("hf_hub_id")})
    return rows


def diff_vs(cfg, base):
    d = {k: v for k, v in cfg.items() if k not in IGNORE and base.get(k) != v}
    return ", ".join(f"{k}={v}" for k, v in d.items()) or "(chính là nền)", sorted({AXIS.get(k, k) for k in d})


def metrics_from_pred(path, test_csv=None, k=9):
    p = ev.read_pred(str(path))
    return ev.compute_metrics(p.y_true, p.y_pred, p.probs)


def msd(v):
    v = np.asarray(v, dtype=float)
    return (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else float("nan"))


def fmt(m, s):
    return f"{m:.4f} ± {s:.4f}" if not np.isnan(s) else f"{m:.4f} (1 seed)"


def build(a):
    root = Path(a.root)
    runs = load_runs(root)
    by = lambda prefix: [r for r in runs if r["exp_id"].startswith(prefix)]  # noqa: E731
    sheets = {}

    # --- Backbones ---
    lat = pd.read_csv(root / "latency_backbones.csv") if (root / "latency_backbones.csv").exists() else None
    rows = []
    for r in by("B"):
        l1 = None
        if lat is not None:
            q = lat[(lat.backbone == r["backbone"]) & (lat.dtype == "fp32") & (lat.batch == 1)]
            l1 = float(q.p50.iloc[0]) if len(q) else None
        c = r["cfg"]
        rows.append({"exp_id": r["exp_id"], "backbone": r["backbone"], "tag trọng số": r["tag"], "#tham số (M)": r["params_m"],
                     "GMAC": r["gmac"], "độ phân giải": c["img_size"], "epoch": c["epochs"], "seed": r["seed"],
                     "macro-F1 val": r["val_macro_f1"], "top-1 val": r["val_top1"], "epoch tốt nhất": r["best_epoch"],
                     "thời gian train/epoch (s)": r["train_time_per_epoch_s"], "độ trễ batch-1 p50 FP32 (ms)": l1,
                     "ghi chú": f"GPU {r['gpu']}, torch {r['torch']}"})
    sheets["Backbones"] = pd.DataFrame(rows)

    # --- Training ---
    base_rows = [r for r in runs if r["exp_id"] == a.baseline]
    base_cfg = base_rows[0]["cfg"] if base_rows else {}
    base0 = next((r for r in base_rows if r["seed"] == 0), None)
    rows = []
    for r in by("T"):
        d, ax = diff_vs(r["cfg"], base_cfg) if base_cfg else ("?", [])
        f = r["val_f1_per_class"]
        rows.append({"exp_id": r["exp_id"], "backbone": r["backbone"], "trục thay đổi": "; ".join(ax), "khác T00 ở điểm nào": d,
                     "seed": r["seed"], "macro-F1 val": r["val_macro_f1"], "top-1 val": r["val_top1"],
                     "Δ macro-F1 so với T00 (seed 0)": r["val_macro_f1"] - base0["val_macro_f1"] if base0 else None,
                     "F1 val Chinee apple": f[0], "F1 val Snake weed": f[7], "ECE val": r["val_ece"],
                     "ghi chú": "1 seed" if r["seed"] is not None else ""})
    sheets["Training"] = pd.DataFrame(rows)

    # --- Inference (step3) ---
    inf, latr = [], []
    for p in sorted(glob.glob(f"{root}/runs/step3_*/inference.csv")):
        inf.append(pd.read_csv(p))
    for p in sorted(glob.glob(f"{root}/runs/step3_*/latency.csv")):
        latr.append(pd.read_csv(p))
    sheets["Inference"] = pd.concat(inf, ignore_index=True) if inf else pd.DataFrame()
    lat_all = ([lat.assign(config=lat.backbone + " (kiến trúc)")] if lat is not None else []) + latr
    lt = pd.concat(lat_all, ignore_index=True) if lat_all else pd.DataFrame()
    if len(lt):
        lt = lt.rename(columns={"config": "cấu hình", "gpu": "GPU", "dtype": "dtype", "batch": "batch",
                                "fused_bn": "gộp BN", "images_per_s": "ảnh/s"})
        keep = [c for c in ["cấu hình", "GPU", "dtype", "batch", "img_size", "gộp BN", "p50", "p95", "p99", "ảnh/s", "torch"] if c in lt]
        lt = lt[keep]
    sheets["Latency"] = lt

    # --- Final / PerClass từ predictions ---
    pdir = root / "predictions"
    final_rows, pc_rows, M = [], [], {}
    for name in (a.baseline, a.final):
        ms, vs = [], []
        for s in a.seeds:
            tp, vp = pdir / f"{name}_seed{s}_test.csv", pdir / f"{name}_seed{s}_val.csv"
            if not tp.exists():
                continue
            mt = metrics_from_pred(tp)
            mv = metrics_from_pred(vp) if vp.exists() else None
            ms.append(mt)
            vs.append(mv)
            final_rows.append({"exp_id": name, "cấu hình": "mốc T00 + I00" if name == a.baseline else "chung kết", "seed": s,
                               "macro-F1 val": mv["macro_f1"] if mv else None, "macro-F1 test": mt["macro_f1"],
                               "top-1 test": mt["top1"], "balanced acc test": mt["balanced_acc"], "ECE test": mt["ece"],
                               "recall Chinee apple (test)": mt["recall"][0], "recall Snake weed (test)": mt["recall"][7]})
        if ms:
            M[name] = ms
            rec = {}
            for key, label in (("macro_f1", "macro-F1 test"), ("top1", "top-1 test"), ("ece", "ECE test")):
                rec[label] = fmt(*msd([m[key] for m in ms]))
            if all(v for v in vs):
                rec["macro-F1 val"] = fmt(*msd([v["macro_f1"] for v in vs]))
            rec["recall Chinee apple (test)"] = fmt(*msd([m["recall"][0] for m in ms]))
            rec["recall Snake weed (test)"] = fmt(*msd([m["recall"][7] for m in ms]))
            final_rows.append({"exp_id": name, "cấu hình": "TỔNG HỢP mean ± std", "seed": f"{len(ms)} seed", **rec})
            for c in range(9):
                pc_rows.append({"cấu hình": name, "lớp": CLASSES[c], "số ảnh test": int(ms[0]["support"][c]),
                                "precision": float(np.mean([m["precision"][c] for m in ms])),
                                "recall": float(np.mean([m["recall"][c] for m in ms])),
                                "F1": float(np.mean([m["f1"][c] for m in ms]))})
    sheets["Final"] = pd.DataFrame(final_rows)
    sheets["PerClass"] = pd.DataFrame(pc_rows)

    # --- Summary: top 10 theo macro-F1 val (mọi lần chạy seed 0 của B/T + chung kết/mốc theo mean val) ---
    sm = []
    for r in runs:
        if r["exp_id"][0] in "BT" and r["seed"] == 0:
            sm.append({"exp_id": r["exp_id"], "nhóm": "backbone" if r["exp_id"][0] == "B" else "huấn luyện", "backbone": r["backbone"],
                       "macro-F1 val": r["val_macro_f1"], "top-1 val": r["val_top1"], "#tham số (M)": r["params_m"], "GMAC": r["gmac"],
                       "ghi chú": diff_vs(r["cfg"], base_cfg)[0] if r["exp_id"][0] == "T" and base_cfg else ""})
    s = pd.DataFrame(sm).sort_values("macro-F1 val", ascending=False).head(10) if sm else pd.DataFrame()
    sheets["Summary"] = s
    return sheets, M


def write(sheets, out):
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        order = ["Summary", "Backbones", "Training", "Inference", "Final", "PerClass", "Latency"]
        for name in order:
            df = sheets.get(name, pd.DataFrame())
            df.to_excel(xw, sheet_name=name, index=False)
            ws = xw.sheets[name]
            ws.freeze_panes = "A2"
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(60, max(10, width + 2))
            for row in ws.iter_rows(min_row=2):
                for c in row:
                    if isinstance(c.value, float):
                        c.number_format = "0.0000"
            f1col = next((i for i, c in enumerate(ws[1], 1) if c.value in ("macro-F1 val", "macro-F1 test")), None)
            if f1col and ws.max_row > 1:
                vals = [(ws.cell(r, f1col).value, r) for r in range(2, ws.max_row + 1) if isinstance(ws.cell(r, f1col).value, float)]
                if vals:
                    from openpyxl.styles import PatternFill
                    best = max(vals)[1]
                    for c in ws[best]:
                        c.fill = PatternFill("solid", fgColor="C6EFCE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="results.xlsx")
    ap.add_argument("--final", default="F01")
    ap.add_argument("--baseline", default="T00")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    sheets, _ = build(a)
    write(sheets, a.out)
    for k, v in sheets.items():
        print(f"{k}: {len(v)} dòng")


if __name__ == "__main__":
    main()
