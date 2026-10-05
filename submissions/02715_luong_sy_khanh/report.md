# Báo cáo Lab Day 2: backbone, công thức huấn luyện và suy luận trên DeepWeeds

Lương Sỹ Khánh, mã số 02715. Mọi con số trong báo cáo đến từ các file trong thư mục này và truy được về `exp_id`: `results.xlsx`, `runs/<exp_id>/seed<k>/{config,result}.json`, `predictions/`, `analysis/eval_out/` (do `eval.py` gốc tính), `logs/`.

## 1. Tóm tắt

- **Bài toán:** phân loại 9 lớp ảnh cỏ dại DeepWeeds (17.509 ảnh 256×256), fold 0 chia sẵn, chia train/val/test = 10.501 / 3.501 / 3.507.
- **Đã làm:** 6 backbone (B01–B06), 11 ablation công thức huấn luyện theo 7 trục (T01–T11) cộng nền T00 với 3 seed, 8 nhóm phương pháp suy luận (I00–I08) kèm độ trễ, chung kết F01 với 3 seed, và test đúng một lần mỗi seed.
- **Cấu hình tốt nhất (F01):** ConvNeXt-T (`in12k_ft_in1k`), tinh chỉnh toàn bộ 12 epoch, TrivialAugment, suy luận ở độ phân giải 256 (không crop) rồi temperature scaling. **Test (3 seed, mean ± std, 3.507 ảnh): top-1 98,00 ± 0,08%, macro-F1 0,9739 ± 0,0009, ECE 0,0052 ± 0,0008.**
- **So với mốc T00 + I00 (cùng 3 seed):** macro-F1 0,9660 ± 0,0042, top-1 97,31 ± 0,37%, ECE 0,0157. Chung kết tốt hơn +0,0079 macro-F1 (lớn hơn std của mốc 0,0042).
- **Kết luận chính:** (1) chọn backbone và trọng số tiền huấn luyện quyết định nhiều nhất (ConvNeXt-T 0,965 so với ResNet-50 0,805 val), (2) công thức huấn luyện cộng thêm khoảng +0,005 và (3) suy luận ở độ phân giải 256 cộng thêm khoảng +0,004 gần như không tốn thời gian. TTA và ensemble không thắng phương pháp rẻ này.
- **Giới hạn quan trọng:** cấu hình chung kết có seed 0 là checkpoint của lần chạy T03, hai seed còn lại huấn luyện lại. Các ablation chỉ có 1 seed. Không chạy kết hợp các yếu tố tốt (T12/T13) và không chạy huấn luyện từ đầu. Xem mục 8.

## 2. Dữ liệu và thiết lập

**Dữ liệu.** DeepWeeds, fold 0 tải nguyên bản từ GitHub của tác giả (`train/val/test_subset0.csv`), không sửa, không chia lại. `images.zip` có MD5 `b7b30f96d466fba86016aa5a26606e0f`. Kiểm tra bắt buộc đã chạy (`logs/eda.log`, `logs/smoke.log`): số ảnh 10.501 / 3.501 / 3.507 (59,97% / 20,00% / 20,03%), giao từng cặp tập rỗng, hợp đúng 17.509 ảnh, mọi file tồn tại. Phân bố lớp ở `analysis/eda/class_distribution.png`, ảnh mẫu ở `analysis/eda/samples_per_class.png`.

- `Negative` chiếm 52,0% (9.106 ảnh), gấp 9,02 lần lớp loài nhỏ nhất (Rubber vine 1.009 ảnh). Top-1 accuracy vì vậy bị lớp này kéo cao, nên chỉ số chính là macro-F1.
- Đối chiếu Table 1 của bài báo: 7 lớp khớp đúng; **Chinee apple có 1.126 ảnh (bài báo 1.125) và Lantana có 1.063 (bài báo 1.064)**, tổng vẫn 17.509. Lệch 1 ảnh giữa hai lớp, mình không rõ nguyên nhân (có thể một ảnh bị gán nhãn khác nhau giữa bản CSV và bảng).
- Mọi ảnh đều 256×256 RGB (mẫu 500 ảnh, `analysis/eda/image_stats.txt`).

**Kiểm tra pipeline trước khi chạy** (`logs/smoke.log`, `logs/unittests.log`): loss khởi tạo 2,177 (kỳ vọng ln 9 = 2,197); focal(γ=0) và label-smoothing(ε=0) khớp CE với sai số 2,4e-7; overfit 16 ảnh xuống loss 0,0047 sau 60 bước; gộp BN khớp với sai số 5e-5; `test_components.py` chạy 14/14 test (focal γ=0, CutMix trộn nhãn đúng diện tích, không weight decay cho norm/bias, EMA, lịch LR, temperature scaling, BN fusion). Ảnh sau augmentation đã giải chuẩn hoá cùng nhãn ở `analysis/eda/augmented_samples.png`.

**Công thức nền T00** (GUIDE 1.4): AdamW, LR backbone 1e-4 và head 1e-3, weight decay 0,05 (không áp cho norm/bias), warmup 1 epoch rồi cosine, CE, batch 64, 12 epoch, AMP, RandomResizedCrop(224) + lật ngang, chuẩn hoá theo trọng số, chọn checkpoint theo macro-F1 val (hòa lấy epoch sớm), clip gradient 1,0.

**Chỉ số** (theo `eval.py`): macro-F1 9 lớp là chỉ số chính, kèm top-1, balanced accuracy, precision/recall/F1 từng lớp, ECE 15 bin, mean ± std qua seed (`ddof=1`).

**Phần cứng và phần mềm.** Tất cả huấn luyện trên GPU Tesla T4: B01–B06 và T01 trên Google Colab (torch 2.11.0+cu130, timm 1.0.29), T00 và T02–T11, F01 và mọi suy luận/độ trễ trên Kaggle (torch 2.11.0+cu128; timm cài bằng `pip` mới nhất, mình không ghi lại phiên bản). Seed: 0 cho B và các ablation; 0, 1, 2 cho T00 và F01. `cudnn.benchmark=True` nên không tái lập từng bit: cùng cấu hình T00 seed 0 cho 0,9646 trên Kaggle và 0,9671 trên Colab (`runs_extra` không nộp, số này ghi ở đây như một mẫu nhiễu giữa hai máy).

## 3. So sánh backbone (B01–B06)

Cùng công thức nền, cùng split, cùng seed 0, 12 epoch. Số liệu từ `runs/B0*/seed0/result.json` (`results.xlsx`, sheet `Backbones`).

| exp | backbone | tag trọng số timm | tham số (M) | GMAC | macro-F1 val | top-1 val | s/epoch | batch-1 p50 FP32 (ms) |
|---|---|---|---|---|---|---|---|---|
| B03 | convnext_tiny | in12k_ft_in1k | 27,8 | 4,45 | **0,9645** | **0,9729** | 54 | 5,54 |
| B04 | deit_small_patch16_224 | fb_in1k | 21,7 | 4,60 | 0,9507 | 0,9637 | 38 | 5,05 |
| B01 | resnet50 | a1_in1k | 23,5 | 4,09 | 0,8048 | 0,8583 | 47 | 5,79 |
| B05 | efficientnet_b0 | ra_in1k | 4,0 | 0,38 | 0,8036 | 0,8569 | 36 | 7,43 |
| B02 | resnext50_32x4d | a1h_in1k | 23,0 | 4,23 | 0,7657 | 0,8326 | 62 | 7,73 |
| B06 | mobilenetv3_large_100 | ra_in1k | 4,2 | 0,22 | 0,7260 | 0,7966 | 34 | 6,06 |

Đủ ràng buộc: có ResNet (B01), ResNeXt (B02), ConvNeXt (B03), transformer (B04), mạng nhẹ (B05, B06).

**Nhận xét.**
- **Khoảng cách rất lớn giữa hai nhóm** (ConvNeXt-T, DeiT-S ≈ 0,95–0,96 so với 0,73–0,80). Đây là một quan sát dưới đúng một công thức (LR 1e-4, CE, 12 epoch), **không** kết luận được rằng ResNet hay EfficientNet kém hơn nói chung. Ba mạng yếu dùng các tag `a1`, `a1h`, `ra` (công thức huấn luyện ImageNet với BCE hoặc RandAugment mạnh); giả thuyết của mình là loại trọng số này tinh chỉnh chậm với CE và LR thấp, nhưng mình **không kiểm chứng** (không thử tag khác như `tv_in1k`, không tăng LR riêng cho từng mạng). Đường cong của B01, B02, B05, B06 cho thấy chưa hội tụ (xem mục 4.1).
- **ConvNeXt-T dùng trọng số `in12k_ft_in1k`:** tiền huấn luyện trên ImageNet-12k rồi tinh chỉnh ImageNet-1k, tức nhiều dữ liệu hơn các tag còn lại. Vì vậy phần "hơn" của ConvNeXt đến từ kiến trúc lẫn dữ liệu tiền huấn luyện, không tách được trong thí nghiệm này (GUIDE câu hỏi 1).
- **Thứ hạng so với ImageNet:** mình không có số ImageNet của các tag này để đối chiếu, nên không kết luận thứ hạng ở đây có giống ImageNet hay không.
- **FLOPs không dự đoán độ trễ:** ở batch 1 mọi backbone đều 5–8 ms. MobileNetV3 (0,22 GMAC) mất 6,06 ms, ResNet-50 (4,09 GMAC) mất 5,79 ms, DeiT-S (4,60 GMAC) nhanh nhất với 5,05 ms. Batch 1 trên T4 bị giới hạn bởi chi phí khởi chạy kernel; ở batch 32 FLOPs mới thể hiện (FP32: MobileNetV3 1.468 ảnh/s, ConvNeXt-T 253 ảnh/s).
- **Chọn backbone đi tiếp: ConvNeXt-T.** Lý do dựa trên số liệu: macro-F1 val cao nhất (0,9645), top-1 cao nhất, độ trễ batch-1 5,5 ms (không đắt hơn mạng khác), và F1 epoch 1 đã 0,87 nên tinh chỉnh ổn định. DeiT-S kém 0,014 macro-F1 nhưng nhanh hơn khoảng 9% ở batch 1 (5,05 so với 5,54 ms) và khoảng 1,7 lần ở batch 32 với AMP (1.109 so với 660 ảnh/s); mình chọn theo chất lượng vì chênh lệch độ trễ batch-1 chỉ khoảng 0,5 ms.

## 4. Công thức huấn luyện (T00–T11)

Mỗi ablation (ConvNeXt-T, seed 0) chỉ khác nền T00 một yếu tố (cột "khác T00" tự sinh từ `config.json`, xem sheet `Training`). Nền T00 chạy 3 seed: macro-F1 val **0,9654 ± 0,0009** (0,9646 / 0,9664 / 0,9651), top-1 0,9736 ± 0,0010. Mình dùng σ = 0,0009 làm thước đo nhiễu, và coi chênh lệch dưới khoảng 0,003 (~3σ) là "không phân biệt được". σ chỉ ước lượng từ 3 seed nên chưa chắc chắn, và mỗi ablation chỉ có 1 seed.

| exp | trục | khác T00 | macro-F1 val | Δ so với mean T00 | Δ/σ | top-1 val | ECE val | F1 Chinee / Snake (val) | kết luận |
|---|---|---|---|---|---|---|---|---|---|
| T03 | B. augmentation | TrivialAugment | **0,9700** | +0,0046 | +4,8 | 0,9780 | 0,013 | 0,947 / 0,930 | giúp |
| T02 | B. augmentation | RandomResizedCrop scale tối thiểu 0,35 | 0,9695 | +0,0041 | +4,3 | 0,9777 | 0,017 | 0,945 / 0,925 | giúp |
| T11 | G. thời gian | 20 epoch | 0,9684 | +0,0031 | +3,2 | 0,9760 | 0,017 | 0,940 / 0,941 | giúp, tốn 1,7× thời gian |
| T05 | C. loss | label smoothing 0,1 | 0,9684 | +0,0030 | +3,2 | 0,9754 | **0,080** | 0,942 / 0,931 | giúp F1, hại hiệu chuẩn |
| T07 | C. loss | CE có trọng số 1/nₖ | 0,9655 | +0,0002 | +0,2 | 0,9740 | 0,015 | 0,936 / 0,917 | không phân biệt được |
| T10 | F. chính quy hoá | EMA 0,998 | 0,9646 | −0,0007 | −0,8 | 0,9726 | 0,014 | 0,941 / 0,918 | không phân biệt được |
| T04 | B. augmentation | CutMix (α=1) | 0,9642 | −0,0012 | −1,3 | 0,9712 | 0,009 | 0,944 / 0,915 | không phân biệt được |
| T08 | D. cân bằng mẫu | sampler cân bằng lớp | 0,9641 | −0,0013 | −1,4 | 0,9712 | 0,018 | 0,943 / 0,918 | không phân biệt được |
| T09 | E. LR | LR gấp đôi (2e-4 / 2e-3) | 0,9605 | −0,0048 | −5,1 | 0,9700 | 0,018 | 0,926 / 0,914 | hại |
| T06 | C. loss | focal loss γ=2 | 0,9594 | −0,0060 | −6,3 | 0,9686 | 0,015 | 0,923 / 0,913 | hại |
| T01 | A. khởi tạo | đóng băng backbone, chỉ train head | 0,8537 | −0,1117 | −118 | 0,8840 | 0,027 | 0,816 / 0,785 | hại rất nhiều |

Bảy trục (A–G) đã đo, có cả loss (T05, T06, T07) và augmentation (T02, T03, T04).

**Phân tích.**
- **Augmentation giúp, cả hai cách đều làm ảnh khó hơn.** TrivialAugment và crop 0,35 đều +0,004 đến +0,005 (4–5σ) và giảm khoảng cách train/val: T03 có val loss thấp nhất (0,096, train loss 0,074), trong khi nền T00 có train loss 0,038 và val loss 0,115. Crop `scale=0,08` mặc định thường cắt mất cây cỏ (nhãn là loài cỏ nhưng vùng crop chỉ có nền), nâng lên 0,35 giảm nhiễu nhãn; đây là giải thích hợp lý nhưng chưa kiểm chứng riêng.
- **CutMix không giúp** (−1,3σ) và làm F1 Snake weed giảm (0,915 so với 0,924–0,933) dù F1 Chinee apple tăng (0,944): với ảnh mà cỏ chỉ chiếm vùng nhỏ, dán đè một hộp có thể xoá mất đối tượng nhưng vẫn giữ nhãn trộn; với 1 seed mình chỉ dùng làm giả thuyết.
- **Loss:** focal làm hại (−6σ), vì mất cân bằng chỉ ở lớp `Negative` mà các loài cỏ đã cân bằng với nhau, nên không có lớp hiếm cần ưu tiên. Class-weight và sampler cân bằng không giúp (cùng lý do). Label smoothing tăng F1 nhưng làm mô hình kém tự tin (ECE 0,080, so với 0,016): temperature scaling (mục 5) sẽ sửa phần này.
- **Hiệu ứng cộng dồn hay triệt tiêu:** mình **chưa** chạy kết hợp (xem mục 8). Cấu hình chung kết chỉ dùng TrivialAugment (T03) cộng một lựa chọn suy luận; không có bằng chứng về việc crop 0,35, 20 epoch hay label smoothing cộng dồn với TrivialAugment.
- **EMA không giúp:** với 12 epoch và cosine về 0, trọng số cuối đã gần như ổn định; EMA chỉ cho F1 epoch 1 cao hơn (0,900 so với 0,77).
- **LR gấp đôi hại (−5σ), 20 epoch giúp (+3σ):** mạng tiền huấn luyện nhạy với LR cao và còn cải thiện ở epoch cuối (T11 đạt tốt nhất ở epoch 20).
- **Đóng băng backbone (T01)** kém xa tinh chỉnh (0,854 so với 0,965): đặc trưng ImageNet chưa đủ cho bài toán này mà không điều chỉnh.
- **Không có trục "huấn luyện từ đầu":** mình bỏ T01-scratch để tiết kiệm GPU.

### 4.1 Nhận xét đường cong training (`curves/`)

- **B01, B02, B05, B06 chưa hội tụ:** train loss cuối 0,18–0,38 và val loss 0,43–0,68, F1 epoch 3 chỉ 0,57–0,69. B05/B06 dao động mạnh (F1 epoch cuối 0,7785 và 0,6978 thấp hơn đỉnh 0,8036 và 0,7260). B03 và B04 hội tụ nhanh (F1 epoch 1 là 0,87 và 0,80) với train loss 0,04.
- **Quá khớp nhẹ ở nền:** T00 train loss 0,038, val loss 0,115. F01 (T03 công thức) có val loss thấp nhất ở epoch 7 rồi tăng nhẹ; macro-F1 val vẫn tăng đến epoch 11.
- **T04 (CutMix) và T05 (label smoothing):** train loss 0,478 và 0,519 không so sánh được với các run khác vì nhãn đã bị trộn hoặc làm mềm, nên chỉ dựa vào val.
- **T02:** train loss cuối 0,008, thấp nhất nhưng val loss 0,129 cao hơn T03, tức quá khớp hơn dù F1 tương đương.
- **T11:** F1 val còn tăng đến epoch 20, gợi ý thêm epoch có thể còn cải thiện.

## 5. Suy luận (I00–I08) và độ trễ

Mô hình: checkpoint T03 seed 0 (công thức của F01). Chỉ dùng val (3.501 ảnh) để so sánh và chọn (`runs/step3_T03/inference.csv`, sheet `Inference`). Độ trễ đo trên Tesla T4, FP32 nếu không ghi khác, **không tính tiền xử lý**, warmup 10 lần, `cuda.synchronize` trước và sau, 100 lần đo, báo p50/p95/p99 (`benchmark.py`).

| mã | phương pháp | K | macro-F1 val | top-1 val | ECE val | p50 / p95 (ms) | chi phí vs I00 |
|---|---|---|---|---|---|---|---|
| I00 | 1 view, center-crop 224 (mốc) | 1 | 0,9700 | 0,9780 | 0,0127 | 5,50 / 9,16 | 1,0× |
| **I04-256** | **resize toàn ảnh 256, không crop** | 1 | **0,9744** | **0,9809** | 0,0107 | 6,25 / 10,14 | 1,1× |
| I04-288 | độ phân giải 288 | 1 | 0,9735 | 0,9797 | 0,0110 | 7,68 / 12,45 | 1,4× |
| I04-320 | độ phân giải 320 | 1 | 0,9727 | 0,9786 | 0,0104 | 9,43 / 9,68 | 1,7× |
| I01 | TTA lật ngang (gộp xác suất) | 2 | 0,9715 | 0,9789 | 0,0107 | 11,37 / 11,89 | 2,1× |
| I03a | TTA lật, gộp logit | 2 | 0,9715 | 0,9789 | 0,0124 | 11,37 / 11,89 | 2,1× |
| I02a | TTA 5 crop (xác suất / logit) | 5 | 0,9709 / 0,9706 | 0,9786 | 0,0105 / 0,0127 | 27,6 / 29,9 | 5,0× |
| I02b | TTA 5 crop + lật (xác suất / logit) | 10 | 0,9740 / 0,9740 | 0,9809 | 0,0074 / 0,0113 | 55,1 / 56,9 | 10,0× |
| I05 | ensemble T03 + T02 + T05 (TB xác suất) | 3 | 0,9738 | 0,9809 | 0,0286 | 16,8 / 18,1 | 3,1× |
| I07 | temperature scaling (T=1,67, khớp trên val) | 1 | 0,9700 | 0,9780 | **0,0037** | 5,50 / 9,16 | 1,0× |
| I08 | FP32 / AMP / FP16 | 1 | 0,9700 / 0,9700 / 0,9700 | 0,9780 | 0,0127 | 5,50 / 7,67 / 5,58 | 1,0× / 1,4× / 1,0× |

Không áp dụng cho ConvNeXt: gộp BN (dùng LayerNorm nên `fuse_conv_bn` báo 0 cặp; hàm đã kiểm tra trên ResNet-50 trong `smoke.log` với sai số 5e-5 và trên ResNet-18 trong `test_components.py`). Không làm: model soup.

**Nhận xét.**
- **Phương pháp rẻ nhất lại tốt nhất:** dùng cả ảnh 256 thay vì crop 224 nâng macro-F1 val +0,0044 mà chỉ tốn thêm 0,7 ms. Hiệu ứng giống FixRes: khi huấn luyện, `RandomResizedCrop` làm vật thể to hơn lúc kiểm tra nếu chỉ crop trung tâm. Tăng lên 288 và 320 không tốt thêm (0,9735, 0,9727), tức đỉnh quanh 256.
- **TTA và ensemble không đáng:** TTA lật (+0,0015) tốn 2×, 5 crop + lật (0,9740) tốn 10× và vẫn thấp hơn độ phân giải 256, ensemble ba mô hình (0,9738) tốn 3× và làm ECE xấu đi 0,0286. Gộp xác suất và gộp logit cho macro-F1 gần như bằng nhau (chênh nhiều nhất 0,0003), còn gộp xác suất cho ECE thấp hơn (0,0107 so với 0,0124 với TTA lật; 0,0074 so với 0,0113 với 5 crop + lật). Vì vậy TTA và ensemble hợp xử lý ngoại tuyến (không có ràng buộc thời gian) còn trên robot nên dùng độ phân giải 256, FP32 hoặc FP16 và temperature scaling (không tốn thêm).
- **Độ chính xác không đổi giữa FP32, AMP, FP16**, nhưng **AMP ở batch 1 chậm hơn FP32** (7,67 so với 5,50 ms); FP16 thuần bằng FP32. Ở batch 32 AMP nhanh hơn rõ (ConvNeXt-T: 660 ảnh/s so với 253 ảnh/s FP32, sheet `Latency`). Đúng cảnh báo của slide: không giả định FP16/AMP luôn nhanh hơn.
- **Hiệu chuẩn:** temperature scaling hạ ECE val từ 0,0127 xuống 0,0037 (T=1,67, khớp trên cùng val nên là ước lượng lạc quan); ước lượng không thiên lệch (khớp T trên một nửa val, đo trên nửa còn lại, 2 chiều) là 0,0137 → 0,0065. Accuracy không đổi vì thứ hạng lớp không đổi.
- **Quy tắc chọn đặt trước (trước khi mở test), chỉ trên val:** chọn phương pháp rẻ nhất, trừ khi phương pháp đắt hơn vượt ≥ 0,001 macro-F1 val. Kết quả là I04-256; không phương pháp nào đắt hơn vượt nó. Chọn bằng một mô hình, một seed, nên cũng có rủi ro chọn theo nhiễu (độ lớn +0,0044 so với σ = 0,0009).

## 6. Cấu hình tốt nhất và kết quả test

**F01:** ConvNeXt-T (`convnext_tiny.in12k_ft_in1k`), tinh chỉnh toàn bộ, 12 epoch, TrivialAugment (T03), các siêu tham số khác như nền; suy luận: resize cả ảnh 256, softmax, temperature scaling với T khớp trên val của **từng seed** (T = 1,563 / 1,424 / 1,428 cho seed 0 / 1 / 2). **Seed 0 là checkpoint của lần chạy T03 seed 0** (cùng công thức, cùng seed); seed 1 và 2 huấn luyện lại bằng `run_queue.py` (`exp_id=F01`). **Mốc:** T00 + I00 (ConvNeXt-T nền, 1 view, không temperature scaling), 3 seed.

Test chạy đúng một lần mỗi seed bằng `final_test.py` (từ chối ghi đè), toàn bộ 3.507 ảnh; số bên dưới do `eval.py score` tính từ `predictions/`.

| | mốc T00 + I00 | chung kết F01 |
|---|---|---|
| top-1 | 0,9731 ± 0,0037 | **0,9800 ± 0,0008** |
| macro-F1 | 0,9660 ± 0,0042 | **0,9739 ± 0,0009** |
| balanced accuracy | 0,9692 ± 0,0032 | 0,9718 ± 0,0014 |
| ECE (15 bin) | 0,0157 ± 0,0013 | **0,0052 ± 0,0008** |
| NLL | 0,1006 ± 0,0142 | 0,0610 ± 0,0052 |
| macro-F1 val (cùng cấu hình) | 0,9654 ± 0,0009 | 0,9721 ± 0,0020 |
| recall Chinee apple | 0,944 ± 0,010 | 0,932 ± 0,003 |
| recall Snake weed | 0,943 ± 0,010 | 0,954 ± 0,016 |

Tự chấm `eval.py grade` (phần I của RUBRIC, đề xuất): I1 7/7, I2 4/5 (Δ = +0,0079 > s = 0,0042 nhưng < 0,01), I3 4/4, I4 2/2 (ECE test không temperature scaling 0,0090, sau 0,0052; chênh macro-F1 val/test 0,0018), I5 2/2 (p95 = 10,1 ms ≪ 100 ms). Tổng 19/20 (`analysis/eval_out/grade_I.json`).

**Đọc kết quả.**
- Chung kết cao hơn mốc +0,0079 macro-F1 và +0,69 điểm top-1, ECE giảm ba lần. Std mốc trên test (0,0042) lớn hơn nhiều so với trên val (0,0009), nên nhiễu thật trên test lớn hơn mình ước lượng ở mục 4. Khi so sánh với mốc, chung kết hơn mốc 1,9 lần std mốc, nhưng chỉ có 3 seed mỗi bên.
- **Không tách được trên test** phần đóng góp của công thức huấn luyện (TrivialAugment) và của suy luận (độ phân giải 256, temperature scaling): để tách cần chạy thêm test cho T03 ở chế độ 1-view, điều mình không làm để không mở test lần nữa. Trên val: công thức +0,0046, độ phân giải 256 +0,0044.
- **Recall Chinee apple của F01 (0,932) thấp hơn mốc (0,944)** dù cao hơn mốc bài báo (88,5%); Snake weed cao hơn (0,954 so với 0,943). Hai lớp này vẫn là điểm yếu, xem 6.1. Hơn mốc bài báo (95,7% / 88,5% / 88,8%) cần lưu ý điều kiện khác: bài báo huấn luyện khoảng 100 epoch trên GTX 1080Ti, định nghĩa "weighted average accuracy", và dùng cùng kiểu chia ngẫu nhiên.

### 6.1 Phân tích lỗi (`analysis/`)

Ma trận nhầm lẫn tổng 3 seed của F01 (`analysis/confusion_F01.png`; mốc: `confusion_T00.png`), tổng 3 × 3.507 = 10.521 dự đoán:
- **Lỗi chính không phải cặp Chinee ↔ Snake mà là "loài cỏ ↔ Negative":** Chinee apple → Negative 22 lần và → Snake weed 21 lần (trên 678 lượt); Snake weed → Negative 20 lần và → Chinee apple 3 lần (trên 612 lượt); Negative → Prickly acacia 22 lần, → Rubber vine 9, → Lantana 8 (trên 5.466 lượt). Giữa hai loài cỏ, Chinee → Snake weed (21 trên 678 lượt, 3,1%) vẫn là nhầm lẫn lớn nhất, nhưng chiều ngược lại chỉ 3 trên 612 lượt (0,5%), khác với bài báo (3,4% và 4,1%).
- **Lỗi lặp lại giữa các seed:** 111 ảnh test bị đoán sai ở ít nhất một seed, **38 ảnh sai ở cả 3 seed** (`analysis/errors_F01.csv`). Đây là các ca khó hoặc nhãn nghi ngờ chứ không phải nhiễu huấn luyện.
- **Ảnh và Grad-CAM** (`analysis/errors_grid_gradcam.png`, seed 0): các ca Chinee apple ↔ Snake weed thường có độ tin cậy trung bình (0,45–0,89) và Grad-CAM tập trung vào một vài mảng lá cụ thể; các ca đoán nhầm sang `Negative` với độ tin cậy rất cao (0,98–1,00) cũng có Grad-CAM tập trung vào một vài mảng lá. Mình không thể xác nhận từ ảnh liệu các mảng đó có phải cây cỏ mục tiêu hay không. Giả thuyết của mình (chưa kiểm chứng): ảnh được gán nhãn theo toàn khung hình nên một số ảnh chứa cỏ mục tiêu chỉ ở mép hoặc rất nhỏ, và một số ảnh `Negative` chứa cây giống loài mục tiêu. Mình chưa xem xét từng ảnh để kết luận có ảnh gắn nhãn sai hay không.

### 6.2 Lệch phân phối (điểm thưởng; val, F01 seed 0, T chỉ khớp trên val sạch, `analysis/shift_analysis.csv`)

| điều kiện | macro-F1 | top-1 | ECE trước TS | ECE sau TS |
|---|---|---|---|---|
| sạch | 0,9744 | 0,9809 | 0,0107 | 0,0040 |
| tối ×0,5 | 0,9698 | 0,9769 | 0,0124 | 0,0038 |
| tối ×0,25 | 0,9672 | 0,9743 | 0,0132 | 0,0038 |
| nhiễu Gauss σ=0,05 | 0,9620 | 0,9709 | 0,0146 | 0,0055 |
| nhiễu Gauss σ=0,15 | 0,7711 | 0,8340 | 0,0996 | 0,0488 |
| làm mờ σ=2 | 0,6347 | 0,7475 | 0,1367 | 0,0487 |
| làm mờ σ=4 | 0,2914 | 0,5870 | 0,2043 | 0,0623 |

- **Chịu được thay đổi ánh sáng** (giảm 0,005–0,007 F1 khi tối đi 4 lần), có thể nhờ TrivialAugment có các phép đổi màu/độ sáng.
- **Rất nhạy với làm mờ và nhiễu mạnh:** làm mờ σ=2 hạ macro-F1 từ 0,974 xuống 0,635. Đây là rủi ro lớn nếu robot di chuyển nhanh (ảnh nhoè do chuyển động).
- **Temperature scaling (T khớp trên val sạch) giảm ECE khoảng 2–3,3 lần ở mọi điều kiện nhưng không đủ** (ECE còn 0,049–0,062 khi bị mờ hoặc nhiễu mạnh): mô hình vẫn quá tự tin khi dự đoán sai. Nếu miền triển khai khác miền val, T khớp trên val không còn đáng tin.

### 6.3 Linear probe DINOv2 (điểm thưởng; chỉ val, `analysis/dino_probe.json`)

Đặc trưng CLS đóng băng của `vit_small_patch14_dinov2.lvd142m` + hồi quy logistic (C=0,1 chọn trên val): macro-F1 val **0,891**, top-1 0,911. So sánh: cao hơn ConvNeXt-T đóng băng chỉ train head (T01: 0,854) nhưng thấp xa ConvNeXt-T tinh chỉnh (0,965–0,970) và DeiT-S tinh chỉnh (0,951). Với ~10k ảnh có nhãn, tinh chỉnh vẫn hơn đáng kể so với đặc trưng đóng băng.

## 7. Kết luận và khuyến nghị

- **Cấu hình tốt nhất** là F01 (mục 6): test top-1 98,00 ± 0,08%, macro-F1 0,9739 ± 0,0009, tốt hơn mốc +0,0079 macro-F1 (hơn std mốc 1,9 lần, trên 3 seed).
- **Yếu tố đóng góp nhiều nhất:** backbone cộng trọng số tiền huấn luyện (ConvNeXt-T so với ResNet-50: +0,16 macro-F1 val), rồi công thức huấn luyện (+0,005, TrivialAugment), rồi suy luận (+0,004, độ phân giải 256). Hai yếu tố sau nhỏ hơn rất nhiều và gần với nhiễu test (σ mốc 0,0042), nên đáng tin ở mức "có xu hướng", riêng backbone thì rõ ràng.
- **Triển khai trên robot (ngân sách 30–100 ms/khung):** dùng F01 một mô hình với độ phân giải 256, FP32 hoặc FP16, temperature scaling: p50 6,2 ms, p95 10,1 ms, p99 10,2 ms ở batch 1 trên T4, macro-F1 test 0,9739 (đây chính là cấu hình đã chạy test). Không dùng AMP ở batch 1 (chậm hơn). TTA, ensemble chỉ hợp ngoại tuyến (ví dụ gán nhãn hàng loạt) và ở đây không đem lại lợi ích so với độ phân giải 256.
- **Cần cảnh báo trước khi triển khai:** mô hình mất độ chính xác nghiêm trọng khi ảnh mờ hoặc nhiễu (mục 6.2) và có nhiều lỗi "cỏ → Negative" với độ tin cậy cao. Nên huấn luyện thêm với augmentation làm mờ/nhiễu, và đặt ngưỡng từ chối theo độ tin cậy sau hiệu chuẩn.

## 8. Hạn chế và việc tiếp theo

- **Một fold, chia ngẫu nhiên, không theo địa điểm:** điểm test có thể lạc quan so với khi gặp địa điểm hoặc mùa mới. Mọi lệch phân phối thật (khác ánh sáng, mờ) chưa được kiểm tra bằng dữ liệu thật, chỉ mô phỏng (mục 6.2).
- **Số seed:** nền T00 và chung kết F01 mỗi bên 3 seed; các ablation (T01–T11) và so sánh backbone chỉ 1 seed, nên các kết luận "giúp/hại" mới là vượt nhiễu nền ước lượng từ 3 seed của T00, chưa được xác nhận bằng nhiều seed. Std của T00 trên test (0,0042) lớn gấp 4,7 lần trên val (0,0009).
- **Seed 0 của F01 là checkpoint của T03 seed 0**, tức không phải một lần huấn luyện riêng; seed 1 và 2 huấn luyện lại (kết quả val 0,9690 và 0,9719 so với 0,9700). Số F01 val và test dùng cùng quy tắc suy luận chọn trên val của T03 seed 0, nên có rủi ro chọn theo nhiễu của đúng mô hình đó.
- **Thí nghiệm không thực hiện:** kết hợp các yếu tố tốt (T12: crop 0,35 + TrivialAugment, T13: thêm 20 epoch và label smoothing) và khởi tạo từ đầu. Vì vậy chưa trả lời được các yếu tố có cộng dồn hay triệt tiêu, và cấu hình chung kết có thể chưa phải tốt nhất có thể. Ngân sách GPU giảm theo GUIDE mục 7: ablation chỉ trên một backbone, một seed.
- **Hai nền tảng:** B01–B06 và T01 chạy trên Colab, các run còn lại trên Kaggle (cùng GPU T4 nhưng khác phiên bản torch và có thể khác timm); T00 seed 0 cho 0,9671 trên Colab và 0,9646 trên Kaggle (cùng cấu hình, cùng seed), cho thấy mức nhiễu giữa hai môi trường.
- **Độ trễ:** đo trên T4 ở batch 1 và 32 cho chính mô hình, không tính tiền xử lý và chép dữ liệu CPU→GPU; thiết bị nhúng trên robot (ví dụ Jetson) sẽ khác. Các số batch 1 (5–8 ms) bị chi phí khởi chạy kernel chi phối, không phản ánh FLOPs.
- **Không tách được** đóng góp công thức và suy luận trên test (mục 6).
- **Việc tiếp theo nếu có thêm thời gian:** chạy T12/T13 và nhiều seed cho các ablation hứa hẹn; thử các tag trọng số khác cho ResNet/EfficientNet để tách ảnh hưởng của công thức tiền huấn luyện; huấn luyện với augmentation làm mờ/nhiễu; xem lại từng ảnh trong 38 ca sai ở cả 3 seed để loại trừ nhãn sai; chạy các fold 1–4.

## 9. Phụ lục

### 9.1 Quy ước và danh sách thí nghiệm

`B` = backbone, `T` = công thức huấn luyện, `I` = suy luận, `F` = chung kết. Mọi lần chạy có `config.json` đầy đủ trong `runs/`; ảnh training ở `curves/<exp_id>_<mô tả>.png` (T00 và F01 mỗi seed một ảnh; ảnh seed 0 của F01 `F01_final_trivialaug.png` là bản sao của `T03_trivialaug.png` vì cùng một lần chạy).

| exp_id | backbone | khác nền | seed | nền tảng |
|---|---|---|---|---|
| B01–B06 | resnet50, resnext50_32x4d, convnext_tiny, deit_small_patch16_224, efficientnet_b0, mobilenetv3_large_100 | (công thức nền) | 0 | Colab T4 |
| T00 | convnext_tiny | (nền) | 0, 1, 2 | Kaggle T4 |
| T01 | convnext_tiny | `init=frozen` | 0 | Colab T4 |
| T02 | convnext_tiny | `crop_min=0.35` | 0 | Kaggle T4 |
| T03 | convnext_tiny | `aug=trivial` | 0 | Kaggle T4 |
| T04 | convnext_tiny | `mix=cutmix` | 0 | Kaggle T4 |
| T05 | convnext_tiny | `loss=ls, label_smoothing=0.1` | 0 | Kaggle T4 |
| T06 | convnext_tiny | `loss=focal, focal_gamma=2` | 0 | Kaggle T4 |
| T07 | convnext_tiny | `loss=ce_weighted, class_weight_beta=0` | 0 | Kaggle T4 |
| T08 | convnext_tiny | `sampler=balanced` | 0 | Kaggle T4 |
| T09 | convnext_tiny | `lr_backbone=2e-4, lr_head=2e-3` | 0 | Kaggle T4 |
| T10 | convnext_tiny | `ema_decay=0.998` | 0 | Kaggle T4 |
| T11 | convnext_tiny | `epochs=20` | 0 | Kaggle T4 |
| F01 | convnext_tiny | `aug=trivial` + suy luận res256 + TS | 0 (= T03 seed 0), 1, 2 | Kaggle T4 |

### 9.2 Điểm thưởng đã làm

Linear probe DINOv2 (6.3), phân tích lệch phân phối với ECE trước/sau temperature scaling (6.2), Grad-CAM để giải thích lỗi (6.1). Không làm: nhiều fold, chưng cất, test-time adaptation (ConvNeXt dùng LayerNorm nên chuẩn hoá lại thống kê BN không áp dụng), xuất ONNX.

### 9.3 Quy trình test (tuân thủ S1–S6)

Chọn mọi thứ (backbone, công thức, phương pháp suy luận, nhiệt độ T) trên val. Test chỉ được mở trong `final_test.py`, mỗi seed một lần, sau khi cấu hình và quy tắc chọn suy luận đã chốt; trước đó một lần chạy thử `--dry_run` chỉ đọc val. Log: `logs/final_T00.log`, `logs/final_F01_s0.log`, `logs/final_F01_s12.log`. Không có số test nào dùng để đổi quyết định.
