# Lab Day 2: DeepWeeds, bài làm của Lương Sỹ Khánh (02715)

Backbone, công thức huấn luyện và suy luận trên DeepWeeds (fold 0). Kết luận và số liệu ở [`report.md`](report.md); bảng so sánh ở [`results.xlsx`](results.xlsx).

**Kết quả chính (test, 3 seed, 3.507 ảnh):** cấu hình F01 (ConvNeXt-T + TrivialAugment, suy luận ở độ phân giải 256 + temperature scaling) đạt top-1 **98,00 ± 0,08%**, macro-F1 **0,9739 ± 0,0009**, ECE 0,0052; mốc T00 + I00: top-1 97,31 ± 0,37%, macro-F1 0,9660 ± 0,0042. `eval.py grade` tự chấm phần I: 19/20.

## Link chạy lại

- Mã nguồn: https://github.com/kascenite/K4-Track4-Day2-LuongSyKhanh-02715 (thư mục `submissions/02715_luong_sy_khanh/code/`).
- Notebook Kaggle đã dùng cho bước chạy test, bảng và phân tích:
  - https://www.kaggle.com/code/kascnte/notebook-3b (test một lần mỗi seed, `eval.py`, `results.xlsx`)
  - https://www.kaggle.com/code/kascnte/notebookc-3c (phân tích lỗi, lệch phân phối, DINOv2)
  
  (Hai notebook cần ở chế độ Public để mở được; phần huấn luyện B01–B06, T00–T11, F01 và Bước 3 chạy trong các notebook khác, không công khai.)
- Notebook gom toàn bộ lệnh theo thứ tự chạy: [`code/lab_day2_kaggle.ipynb`](code/lab_day2_kaggle.ipynb) (Kaggle, GPU T4, Internet bật).

## Cấu trúc thư mục

```
README.md            file này
report.md            báo cáo kết luận
results.xlsx         7 sheet: Summary, Backbones, Training, Inference, Final, PerClass, Latency
curves/              ảnh training của từng thí nghiệm: <exp_id>_<mô tả>.png (B01-B06, T00-T11, F01; F01 seed 0 là bản sao của T03)
predictions/         F01 và T00 (mốc): *_seed{0,1,2}_test.csv, *_val.csv, và *uncal_seed*_test.csv (chưa temperature scaling)
runs/                config.json, result.json, history.csv, val_logits.npy của từng lần chạy; step3_T03/ (suy luận, độ trễ); latency_backbones.csv
analysis/            EDA, ma trận nhầm lẫn, ảnh lỗi + Grad-CAM, lệch phân phối, DINOv2, eval_out/ (kết quả eval.py)
logs/                log của mọi bước (queue, smoke test, unit test, final test, eval.py, ...)
code/                toàn bộ mã (xem bên dưới)
```

Không nộp: dataset, checkpoint (khoảng 110 MB mỗi cái, lưu trên Kaggle/Drive của mình).

## Mã nguồn (`code/`)

| file | vai trò |
|---|---|
| `dataset.py`, `model.py`, `losses.py`, `train.py` | bộ khung `starter/` đã hoàn thiện; một hàm `train.run(Config)` duy nhất cho mọi thí nghiệm |
| `inference.py`, `benchmark.py` | TTA, ensemble, temperature scaling, gộp BN; đo độ trễ (warmup, synchronize, p50/p95/p99) |
| `run_queue.py`, `jobs_B.txt` | chạy tuần tự nhiều cấu hình, bỏ qua lần chạy đã xong |
| `step3.py`, `lat_backbones.py` | Bước 3 trên val; độ trễ các backbone |
| `final_test.py` | chạy test một lần mỗi seed (từ chối ghi đè); ghi `predictions/` đúng định dạng của `eval.py` |
| `make_results.py` | gộp mọi `result.json` và `predictions/` thành `results.xlsx` |
| `eda.py`, `smoke.py`, `test_components.py` | EDA, kiểm tra pipeline, 14 unit test (focal γ=0, CutMix, weight decay, EMA, BN fusion, temperature scaling) |
| `analysis3c.py`, `dino_probe.py` | phân tích lỗi + Grad-CAM + lệch phân phối; linear probe DINOv2 |
| `lab_day2_kaggle.ipynb` | notebook chạy lại toàn bộ |

`eval.py` là bản gốc của giảng viên, không sửa.

## Thứ tự chạy lại

1. Cài đặt và tải dữ liệu (ô 0 của notebook): `pip install timm openpyxl gdown`, tải `images.zip` (MD5 `b7b30f96d466fba86016aa5a26606e0f`) và 4 file CSV của fold 0 từ GitHub của tác giả; đặt ảnh phẳng trong một thư mục.
2. `python eda.py`, `python smoke.py`, `python -m unittest test_components`.
3. Bước 1: `python run_queue.py --jobs jobs_B.txt --common images_dir=... labels_dir=... out_dir=... pred_dir=... curves_dir=... ckpt_dir=...`.
4. Bước 2: job file T00–T11 (ô 3 của notebook), cùng lệnh `run_queue.py`.
5. Huấn luyện F01 seed 1 và 2 (ô 4).
6. Bước 3: `python step3.py --exp T03 --seed 0 --ensemble T03 T02 T05 --cnn 1 --common ...` (cần GPU rảnh để đo độ trễ), `python lat_backbones.py ...`.
7. Test một lần mỗi seed: `final_test.py` cho T00 và F01 (ô 6), rồi `eval.py score` và `eval.py grade`.
8. `make_results.py`, `analysis3c.py`, `dino_probe.py`.

## Seed, phiên bản, tái lập

- Seed: 0 cho B01–B06 và T01–T11; 0, 1, 2 cho T00 và F01. Seed chỉ đổi khởi tạo head, thứ tự batch và augmentation, không đổi cách chia dữ liệu. Seed 0 của F01 là checkpoint của lần chạy T03 seed 0.
- Thư viện: B01–B06, T01 trên Colab với torch 2.11.0+cu130, timm 1.0.29; các lần chạy còn lại trên Kaggle với torch 2.11.0+cu128 và timm cài mới nhất khi chạy (không ghi lại số phiên bản). Phần cứng: Tesla T4. `scikit-learn` dùng cho `dino_probe.py`; `eval.py` cần numpy, pandas.
- `cudnn.benchmark=True` nên kết quả không trùng từng bit khi chạy lại: cùng cấu hình T00 seed 0 cho macro-F1 val 0,9646 (Kaggle) và 0,9671 (Colab). Chạy lại sẽ ra kết quả cùng mức (chênh lệch cỡ ±0,003), không giống hệt.
- Test chỉ mở trong `final_test.py`, đúng một lần mỗi seed; chọn mọi thứ trên val.

## Chưa làm (ghi rõ trong báo cáo, mục 8)

Kết hợp các yếu tố tốt (T12/T13), khởi tạo từ đầu, nhiều seed cho các ablation (T01–T11 chỉ 1 seed), nhiều fold, chưng cất, ONNX.
