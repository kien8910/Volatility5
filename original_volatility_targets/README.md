# Original volatility targets

Pipeline này kiểm tra liệu biểu diễn/prototype tin tức FinTexTS có bổ sung tín
hiệu ngoài dữ liệu giá cho **log-volatility gốc ngày kế tiếp** hay không. Đây là
một project độc lập với `fintexts_semiconductor_prototype`: mọi target, model,
checkpoint, log, bảng và hình mới chỉ được ghi dưới
`original_volatility_targets/`.

Chế độ `--quick` là smoke test kỹ thuật. Nó luôn bị chặn khỏi quyết định GO.
Chỉ `--full`, sau khi vượt các cổng validation, locked test, placebo và độ ổn
định qua fold/seed, mới có thể sinh quyết định GO.

## Dữ liệu dùng chung và ranh giới an toàn

Pipeline đọc các artifact sau từ project residual, theo thứ tự ưu tiên
`runs/light/...` rồi thư mục mặc định:

- `market_supervised.parquet`: OHLC/volatility đã xử lý, feature ngày `t`,
  target ngày `t+1`, ticker và chronological split;
- `representation_manifest.csv` và các feature R0–R11;
- `canonical_events.parquet` và `prototype_assignments.parquet` nếu có, để
  kiểm tra hợp đồng cache;
- `chronological_folds.csv`;
- các bảng kết quả residual nếu có, chỉ để đối chiếu cuối cùng.

Không có file nào được ghi vào `fintexts_semiconductor_prototype`. Embedding và
prototype không được tính lại. Đường dẫn tuyệt đối của server lưu trong manifest
được tự động ánh xạ lại bằng tên file khi project chạy ở máy khác.

Các file residual đã được kiểm tra để định hướng/port logic:

- `src/preprocess_market.py`: định nghĩa Garman–Klass, shift `t -> t+1` và price
  features;
- `src/aggregate_features.py`: hợp đồng R0–R11;
- `src/train_targets.py`: mô hình nhỏ và cách tạo target train-only;
- `src/evaluate_targets.py`: metric và khóa cấu hình bằng validation;
- `src/placebo_tests.py`: placebo R9–R11;
- `src/utils.py`: cấu hình, logging, atomic output và kiểm tra chronology;
- `run_pipeline.py`: cách điều phối stage.

Không file nào được copy nguyên trạng và không module residual nào được import.
Các phần phù hợp được viết lại trong `src/utils.py`, `src/modeling.py`,
`src/progress_tracker.py` và `src/compare_representations.py`, với namespace và
output path mới. Điều này tránh side effect và tránh thay đổi hành vi pipeline
residual.

Khác biệt chính:

- target mới lấy trực tiếp từ `target_log_variance`, không lấy từ residual;
- q90/q95, q50/q90 và q33/q67 đều fit lại trên train của đúng task/fold;
- có target sector mean, breadth, spike và regime trên đủ 11 ticker;
- uncertainty dùng mean model và scale target từ expanding OOF prediction trên
  train, không dùng fitted in-sample residual;
- kết quả được so sánh hậu nghiệm với residual nhưng residual không đi vào
  target hoặc model.

## Cài đặt

Yêu cầu Python 3.10 trở lên. Từ root repository:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -r original_volatility_targets/requirements.txt
```

Linux:

```bash
source .venv/bin/activate
pip install -r original_volatility_targets/requirements.txt
```

Mặc định config tìm artifacts trong
`fintexts_semiconductor_prototype/runs/light`. Có thể sửa
`shared.residual_project_root` và danh sách `*_candidates` trong
`config/config_original_volatility.yaml` nếu server dùng vị trí khác.

## Chạy bản nhẹ

Chạy end-to-end:

```bash
python original_volatility_targets/run_original_volatility_pipeline.py --stage all --quick
```

`--quick` dùng một seed, holdout chính, cửa sổ ngày rút gọn, model tuyến tính và
một tập representation đại diện. Nó vẫn tạo target, level, q90/q95 spike,
regime, Gaussian fixed-mean uncertainty, sector targets, evaluation, placebo,
hình và báo cáo cuối.

Chạy từng stage:

```bash
python original_volatility_targets/run_original_volatility_pipeline.py --stage prepare --quick
python original_volatility_targets/run_original_volatility_pipeline.py --stage level --quick
python original_volatility_targets/run_original_volatility_pipeline.py --stage spike --quick
python original_volatility_targets/run_original_volatility_pipeline.py --stage regime --quick
python original_volatility_targets/run_original_volatility_pipeline.py --stage uncertainty --quick
python original_volatility_targets/run_original_volatility_pipeline.py --stage sector --quick
python original_volatility_targets/run_original_volatility_pipeline.py --stage evaluate --quick
```

Stage riêng giả định các dependency trước đó đã hoàn thành. `evaluate` chỉ đọc
checkpoint model hiện có; không chọn lại model bằng test.

## Chạy bản đầy đủ và lọc grid

## Xác nhận R6 trên 3 folds x 5 seeds

Trước tiên tạo các representation theo fold trong project residual:

```bash
python fintexts_semiconductor_prototype/run_pipeline.py --stage r6-confirmatory --config fintexts_semiconductor_prototype/config/config_r6_confirmatory.yaml
```

Sau đó chạy grid mức volatility đã khóa:

```bash
python original_volatility_targets/run_original_volatility_pipeline.py --stage level --r6-confirmatory --resume
python original_volatility_targets/run_original_volatility_pipeline.py --stage evaluate --r6-confirmatory --resume
```

Profile này tạo đúng `9 representations x 3 folds x 5 paired seeds = 135`
task: R6 được so sánh cố định với R0, R3, R4, R9, R10, R11, P_LAGGED và
P_PERMUTED. Model duy nhất là Ridge `alpha=10`, input là `price_plus_text`
(R0 dùng price-only), family duy nhất là `grid_k4_pca64_tau0p1`.

Các bảng riêng được ghi với tiền tố `r6_confirmatory_` trong
`original_volatility_targets/outputs/tables`. Báo cáo chỉ phát
`CONFIRMATORY-PASS` hoặc `CONFIRMATORY-FAIL`; nó không đọc test holdout và
không tự nâng kết luận thành GO. Chỉ khi PASS mới khóa cấu hình và thực hiện
một lần đánh giá holdout cuối.

### Audit hậu nghiệm khi R6 confirmatory thất bại

Sau khi stage `level` và `evaluate` của profile R6 đã hoàn thành, chạy:

```bash
python original_volatility_targets/run_original_volatility_pipeline.py --stage audit --r6-confirmatory --resume
```

Audit chỉ đọc validation của ba chronological folds, model/feature/prototype
đã fit trên train của từng fold và embedding cache đóng băng. Nó không đọc
locked holdout, không train lại model và không thay đổi
`CONFIRMATORY-FAIL`. Kết quả gồm:

```text
r6_ticker_diagnostics.csv
r6_news_day_diagnostics.csv
r6_news_level_diagnostics.csv
r6_fold_distribution_shift.csv
r6_prototype_drift.csv
r6_failure_audit_summary.csv
```

`r6_prototype_drift.csv` so centroid bằng Hungarian matching trong không gian
embedding gốc, không so trực tiếp tọa độ PCA khác nhau giữa các fold. Dòng
`final_recommendation` chỉ là hướng nghiên cứu hậu nghiệm như
`TRY-NEWS-GATING-EXPLORATORY`, `TRY-LEVEL-SPECIFIC-EXPLORATORY`,
`MOVE-TO-SPIKE-OR-MAGNITUDE` hoặc `STOP-DIRECT`. Mọi cấu hình phát hiện từ
audit phải được xác nhận trên một giai đoạn thời gian mới.

## Xác nhận volatility level chỉ với target-company news

Đây là thử nghiệm mới, tách biệt khỏi `r6_confirmatory`, được khóa trước khi
chạy theo kết quả audit:

- chỉ giữ các cột text có token `__target__`; macro, sector và related không
  được đưa vào model;
- train Ridge `alpha=10` trên toàn bộ ngày train của từng fold để giữ cùng
  price baseline và cỡ mẫu;
- metric quyết định là QLIKE trên các ngày validation thực sự có target-company
  news; `all_days` và `no_target_news_days` chỉ là chẩn đoán phụ;
- dùng cùng true-news mask cho R0, R3, R4, R6 và mọi placebo;
- giữ 3 chronological folds × 5 paired prototype/model seeds;
- không đọc hoặc đánh giá locked holdout.

Không cần tạo lại embedding/prototype vì codebook đã được xây riêng theo news
level. Cần có các fold representation của bước `r6-confirmatory` trước đó.

Chạy training và evaluation:

```bash
python original_volatility_targets/run_original_volatility_pipeline.py --stage level --target-news-only --resume
python original_volatility_targets/run_original_volatility_pipeline.py --stage evaluate --target-news-only --resume
```

Hoặc chạy cả profile:

```bash
python original_volatility_targets/run_original_volatility_pipeline.py --stage all --target-news-only --resume
```

Kết quả riêng:

```text
target_news_only_fold_results.csv
target_news_only_comparisons.csv
target_news_only_summary.csv
target_news_only_cohort_summary.csv
target_news_only_decision.csv
target_news_only_report.json
```

Chỉ
`TARGET-NEWS-ONLY-PASS` mới cho phép khóa target-only R6 và mở locked holdout
một lần; `TARGET-NEWS-ONLY-FAIL` giữ holdout đóng.

```bash
python original_volatility_targets/run_original_volatility_pipeline.py --stage all --full
```

Bản full dùng 5 seed, holdout và 3 expanding chronological fold, hai định nghĩa
regime, q90/q95, threshold ticker/pooled-standardized, model tuyến tính và MLP,
cùng các representation/cấu hình trong manifest. Grid có thể lớn; nên chạy theo
stage và resume trên server.

Ví dụ lọc:

```bash
python original_volatility_targets/run_original_volatility_pipeline.py --stage spike --full --fold 1 --seed 11 --representation R7 --model weighted_logistic
python original_volatility_targets/run_original_volatility_pipeline.py --stage level --full --target volatility_level --representation R0 --representation R7
```

Các lựa chọn hỗ trợ:

```text
--quick
--full
--r6-confirmatory
--target-news-only
--target-mechanism-audit
--target-component-audit
--fold
--seed
--target
--representation
--model
--force
--no-cache
```

`--force` và `--no-cache` chỉ bỏ qua cache của project mới. Chúng không xóa hoặc
ghi đè cache residual.

## Progress, log và resume

Trước khi chạy, toàn bộ task được lập thành
`outputs/logs/task_manifest.json`. Terminal hiển thị stage/task hiện tại,
fold/seed/target/representation/model, phần trăm tổng, số task còn lại, elapsed
và ETA. Cùng dữ liệu được ghi vào:

```text
outputs/logs/pipeline.log
outputs/logs/progress_history.csv
outputs/checkpoints/progress_state.json
```

Tiếp tục lần chạy bị ngắt:

```bash
python original_volatility_targets/run_original_volatility_pipeline.py --resume
```

Lệnh này mặc định resume grid quick/all. Khi resume một grid full hoặc stage đã
lọc, phải đưa lại đúng `--full`, `--stage` và các bộ lọc ban đầu. Task chỉ được
bỏ qua khi trạng thái là `COMPLETED` và toàn bộ output khai báo còn tồn tại,
khác rỗng.

Task model độc lập có thể thất bại và các task khác vẫn tiếp tục; stack trace và
cấu hình được lưu đầy đủ. Lỗi ở prepare/evaluate là lỗi toàn vẹn nên pipeline
dừng. Cuối lượt chạy, log báo số completed, failed, skipped và danh sách lỗi.

## Leakage contract

- Dữ liệu feature/news ngày `t` dự báo target đúng trading day `t+1`.
- `feature_date < target_date` được assertion trên mọi hàng.
- Giữ nguyên train/validation/test của residual; không random split.
- Scaler, imputer, one-hot, model và threshold chỉ fit trên task-train.
- Validation chọn cấu hình; test chỉ được đọc sau khi task đã cố định.
- Scale target uncertainty được tạo bằng expanding OOF prediction trong train.
- Prototype main-train hợp lệ cho holdout chính. Với expanding folds, pipeline
  tự tìm `fold_representation_manifest.csv` và chỉ dùng R5–R7 có
  `fit_scope=fold_train_only` đúng fold/family/seed. Các representation phụ
  thuộc train khác không có artifact refit (ví dụ PCA/prototype placebo) bị
  loại khỏi fold. Artifact chỉ fit main-train không bao giờ được tính là bằng
  chứng ổn định cho fold sớm.
- Placebo permutation chỉ thực hiện trong từng split; lagged placebo dịch theo
  lịch ticker và không trộn split.

## Targets và representation

Stock targets:

- `volatility_level`;
- `volatility_spike_q90_*`, `volatility_spike_q95_*`;
- `volatility_regime_q50_q90`, `volatility_regime_q33_q67`;
- `volatility_uncertainty` Gaussian/Student-t, fixed-price-mean hoặc full.

Sector targets:

- mean standardized volatility;
- q90 breadth;
- spike tối thiểu 3/11, 5/11 hoặc 6/11;
- sector regime q50/q90 và q33/q67.

R0–R11 giữ đúng artifact từ residual. Ngoài ra:

- `P_LAGGED`: R7 dịch lùi 30 ticker-trading-days;
- `P_PERMUTED`: R7 hoán vị hàng độc lập trong mỗi split.

Hai placebo bổ sung chỉ được tạo từ cached features, không sửa prototype.

## Kết quả

Tất cả file bắt buộc nằm trong `outputs/tables` và `outputs/figures`. Prediction
từng task nằm ở `outputs/checkpoints/tasks`; model/preprocessor ở
`outputs/models`.

Quy tắc kết luận ưu tiên validation QLIKE, PR-AUC, macro-F1 hoặc NLL theo target.
Prototype phải tốt hơn R0, raw embedding, PCA, random projection và các
placebo; gain test phải cùng chiều; bản full phải ổn định qua ít nhất 3 fold và
5 seed. Nếu thiếu comparator hoặc artifact fold-safe, điều kiện không đạt. Do đó
silhouette, train loss hoặc một ticker riêng không thể tạo kết luận GO.

Các kết luận có thể là:

```text
GO-VOLATILITY-LEVEL
GO-VOLATILITY-SPIKE
GO-VOLATILITY-UNCERTAINTY
GO-VOLATILITY-REGIME
GO-SECTOR-REGIME
WEAK-GO
NO-GO
```

## Audit cơ chế target-news

Audit này chỉ dùng volatility level trên các ngày validation có target-company
news. Nó kiểm tra riêng:

- `R7` dùng toàn bộ `meta__target__*`: metadata cơ bản cùng
  entropy/novelty/distance phụ thuộc prototype, nhưng không dùng soft assignment;
- `R6` semantic prototype đã khóa từ thí nghiệm target-news-only;
- 30 representation `R9_NULL_<seed>` tạo từ assignment prototype ngẫu nhiên,
  độc lập cho từng fold và prototype seed;
- `R0` price-only làm mốc chung.

Trước tiên tạo artifact ở project residual:

```bash
python fintexts_semiconductor_prototype/run_pipeline.py --stage target-mechanism-artifacts --config fintexts_semiconductor_prototype/config/config_r6_confirmatory.yaml
```

Sau đó train 465 task Ridge (31 representation × 3 fold × 5 paired seeds) và
chạy audit:

```bash
python original_volatility_targets/run_original_volatility_pipeline.py --stage level --target-mechanism-audit --resume
python original_volatility_targets/run_original_volatility_pipeline.py --stage audit --target-mechanism-audit --resume
```

Hoặc:

```bash
python original_volatility_targets/run_original_volatility_pipeline.py --stage all --target-mechanism-audit --resume
```

### Component ablation sau kết quả METADATA-ONLY

Follow-up này tách khối `meta__target__*` thành hai phần không chồng lặp:

- `META_BASIC`: news count, canonical count, event lag, no-news mask,
  days-since-last-news và has-prior-news;
- `PROTO_DIAG`: assignment entropy, novelty và nearest-prototype distance.

Cả hai alias đọc cùng artifact R7 đã fit riêng trong từng fold. Không cần tạo
lại embedding, PCA, K-means hoặc random prototypes. Chỉ có 30 task Ridge mới:
2 component × 3 folds × 5 paired seeds.

```bash
python original_volatility_targets/run_original_volatility_pipeline.py --stage level --target-component-audit --resume
python original_volatility_targets/run_original_volatility_pipeline.py --stage audit --target-component-audit --resume
```

Hoặc:

```bash
python original_volatility_targets/run_original_volatility_pipeline.py --stage all --target-component-audit --resume
```

Audit tái sử dụng R0/R6 từ `target_news_only`, R7 và 30 random seeds từ
`target_mechanism_audit`. Kết quả có tiền tố `target_component_`; locked test
vẫn đóng và quyết định mechanism trước đó không bị ghi đè.

Các file kết quả có tiền tố `target_mechanism_`. Audit báo empirical percentile
và one-sided p-value của gain R6 so với phân phối 30 random-prototype seeds.
Đây là phân tích hậu nghiệm trên validation: không đọc locked test và không tự
động thay đổi `TARGET-NEWS-ONLY-FAIL`.

Hiện implementation cố ý dùng `NO-GO` khi cổng đầy đủ không đạt; `WEAK-GO` chỉ
nên được thêm sau một quy tắc định lượng được khóa trước cho subgroup, tránh
diễn giải hậu nghiệm.
