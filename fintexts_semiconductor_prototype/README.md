# FinTexTS Semiconductor News Prototypes

Project Python độc lập để trả lời một câu hỏi thực nghiệm hẹp: **gom embedding
tin tức thành semantic prototype có tạo tín hiệu ngoài mẫu, và tín hiệu đó hữu
ích nhất cho mục tiêu dự báo biến động nào?**

Pipeline được viết từ đầu cho `EXAONE-BI/FinTexTS`, chỉ dùng 11 ticker:
`ADI, AMAT, AMD, AVGO, INTC, KLAC, LRCX, MU, NVDA, QCOM, TXN`. Mọi kết luận
được tạo sau khi chạy trên dữ liệu thật; repository mới tạo **không chứa sẵn
một tuyên bố GO/NO-GO**.

## Nguyên tắc thực nghiệm

- News khả dụng đến ngày `t` và price features tại `t` chỉ dự báo target ngày
  giao dịch chung kế tiếp `t+1`.
- Train, validation và test là các block ngày chung 60/20/20; không random
  split. Assertion kiểm tra `train_end < validation_start` và
  `validation_end < test_start`.
- Scaler, PCA, random projection, clustering, threshold và model chỉ fit trên
  train tương ứng. Validation chọn cấu hình; test chỉ được mở sau khi cấu hình
  đã khóa.
- Residual train là dự báo expanding-window OOF. Không dùng fitted in-sample
  residual để huấn luyện nhánh text.
- Prototype được fit riêng cho `macro`, `sector`, `related`, `target`. Mọi event
  validation/test chỉ được gán bằng PCA/centroid đã fit trên train.
- Placebo R9/R10/R11, nhiều seed và chronological fold là điều kiện của quyết
  định cuối; silhouette không phải tiêu chí xác nhận khả năng forecasting.
  Fold refit đầy đủ chỉ chạy sau khi validation đã khóa một family, tránh dựng
  trước toàn bộ tích Descartes fold × grid không được dùng. Các fold xác nhận
  chỉ nằm trong original train và không chồng lên main validation đã dùng để
  khóa cấu hình; chúng đo độ bền lịch sử của cấu hình đã khóa, không được trình
  bày như một nested-CV estimate độc lập.

## Cài đặt

Yêu cầu Python 3.10 trở lên. Tạo môi trường riêng từ thư mục project:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu server cần bản PyTorch/CUDA riêng, cài wheel phù hợp từ PyTorch trước rồi
cài phần còn lại. Encoder và các MLP nhỏ tự dùng CUDA khi `torch.cuda.is_available()`.
Hugging Face token chỉ cần khi model/dataset yêu cầu; đặt biến môi trường
`HF_TOKEN`, không ghi token vào YAML.

Encoder mặc định là
`FinLang/finance-embeddings-investopedia` (BERT nhỏ hơn, phù hợp T4/RTX 4070).
FinTexTS cũng phát hành encoder chính thức
`EXAONE-BI/FinTexTS-Embedding`; bật
`embedding.use_official_fintexts_model: true` để dùng model 4096 chiều này nếu
server đủ VRAM. Có thể đổi `embedding.model_name` trong `config/config.yaml`.
Encoder luôn frozen, chạy batch, L2-normalize và cache theo `event_id`,
`text_hash`, model.

## Chạy pipeline

Tất cả lệnh sau chạy tại thư mục `fintexts_semiconductor_prototype`:

```bash
python run_pipeline.py --stage all
python run_pipeline.py --stage download
python run_pipeline.py --stage baseline
python run_pipeline.py --stage prototypes
python run_pipeline.py --stage targets
python run_pipeline.py --stage evaluate
```

Xem thứ tự module mà không tải dữ liệu hoặc chạy model:

```bash
python run_pipeline.py --stage all --dry-run
```

### Pilot nhẹ trước khi chạy full grid

`config/config_light.yaml` kế thừa toàn bộ quy tắc dữ liệu và leakage từ
`config.yaml`, nhưng chỉ chạy `K=[4,8]`, PCA 64 chiều, temperature 0.1, mean
pooling, một model nhỏ cho mỗi target và một seed robustness. Pilot vẫn giữ
R0–R11, toàn bộ nhóm target, ba chronological fold và các placebo bắt buộc.
Response-aware prototype được để dành cho full run.

Chạy pilot end-to-end:

```bash
python run_pipeline.py --stage all --config config/config_light.yaml
```

Nếu chạy từng stage, phải truyền cùng cấu hình ở mọi lệnh:

```bash
python run_pipeline.py --stage download --config config/config_light.yaml
python run_pipeline.py --stage baseline --config config/config_light.yaml
python run_pipeline.py --stage prototypes --config config/config_light.yaml
python run_pipeline.py --stage targets --config config/config_light.yaml
python run_pipeline.py --stage evaluate --config config/config_light.yaml
```

Artifact pilot được cách ly tại `runs/light/`; raw snapshot và cache embedding
vẫn dùng chung để lần chạy full không phải tải hoặc encode lại dữ liệu giống
nhau. Kết luận của pilot là kiểm tra vận hành và tín hiệu sơ bộ, không thay thế
kết luận robustness của full grid.

Luồng khuyến nghị khi chạy từng phần:

1. `download`: tải snapshot Parquet và inspect schema.
2. `baseline`: tạo market panel, split/fold, chọn B0/B1/B2 bằng validation
   QLIKE và sinh residual OOF.
3. `prototypes`: chuẩn hóa/deduplicate event, embedding, train-only prototype,
   aggregation và R0–R11.
4. `targets`: chấm mọi family K/PCA/temperature/pooling bằng mô hình tuyến tính
   rẻ trên phần đuôi của **train**, giữ hai family tốt nhất cho mỗi
   target/representation, rồi mới chạy full model grid trên validation và khóa
   cấu hình trước khi dự báo test.
5. `evaluate`: phân tích prototype/cross-stock, metrics, placebo, robustness và
   quyết định tự động.

Các stage hẹp bổ sung là `market`, `news` và `embeddings`. Một stage không âm
thầm chạy dependency nằm trước nó: nếu artifact đầu vào thiếu, module dừng với
thông báo cụ thể. `--stage all` là cách tái lập đầy đủ.

## Schema và dữ liệu

`inspect_schema.py` in mọi tên cột/dtype, sau đó ánh xạ dựa trên override,
aliases và pattern. Mapping thực tế được lưu tại `config/schema_mapping.yaml`.
Nếu auto-detection mơ hồ, đặt rõ:

```yaml
schema:
  overrides:
    date: actual_date_column
    ticker: actual_ticker_column
    industry: actual_industry_column
    open: actual_open_column
    high: actual_high_column
    low: actual_low_column
    close: actual_close_column
    macro_news: [actual_macro_column]
    sector_news: [actual_sector_column]
    related_news: [actual_related_column]
    target_news: [actual_target_column]
```

Nếu snapshot FinTexTS không có cột `Industry-level`, pipeline không giả vờ rằng
cột tồn tại. Nó kiểm tra đủ 11 ticker, gán nhãn universe cấu hình
`semiconductor`, và ghi `industry_verification_source=config` vào audit.

Raw snapshot không bị chỉnh sửa. Dữ liệu sinh ra và model/cache bị `.gitignore`
để tránh commit file lớn.

## Volatility và price features

Garman–Klass variance mặc định:

```text
v_t = 0.5 * log(H_t/L_t)^2
      - (2*log(2)-1) * log(C_t/O_t)^2
y_t = log(clip(v_t, epsilon) + epsilon)
```

Pipeline cũng tính Parkinson để kiểm tra độ bền, audit giá trị không hợp lệ,
non-positive clipping, extreme quantiles và phân phối theo ticker. Features tại
`t` gồm `y_t`, lag 1–22, HAR daily/weekly/monthly, rolling
mean/std/min/max, log return, absolute return và rolling return volatility.
Target là `y_{t+1}`; hàng không nối đúng ngày giao dịch chung kế tiếp bị loại.

## Baseline và residual

- **B0:** historical mean theo ticker, fallback pooled.
- **B1:** HAR Ridge riêng từng ticker.
- **B2:** pooled HAR Ridge + ticker one-hot.

Alpha và baseline được chọn bằng validation QLIKE. Validation và test đều chỉ
được dự báo bằng baseline/scaler/model đã fit trên train; pipeline không refit
trên validation. Để tạo target text, phần train bắt đầu sau lịch sử tối thiểu,
dự báo block kế tiếp, mở rộng lịch sử rồi lặp. Threshold q50/q90/q95 chỉ được
fit trên residual train OOF, cả phiên bản riêng ticker và pooled sau
standardization. Cả hai họ nhãn spike/regime đều được đưa vào target search,
không chỉ được xuất để audit.

## Event, embedding và prototype

Event canonical có:

```text
event_id, date, news_level, text, target_ticker,
related_tickers, available_to_tickers
```

Text rỗng/placeholder được loại; whitespace/Unicode được chuẩn hóa; exact hash
dedup luôn được báo cáo. Macro/sector giống nhau trong cùng ngày được canonical
hóa cho toàn universe. Target news không bị gộp qua hai ticker chỉ vì cùng text.
Near-duplicate merge là tùy chọn và có audit trước/sau.

`build_events` luôn giữ song song bản không dedup (`raw`), exact-dedup
(`exact`) và canonical; `events.variant` chọn biến thể đi tiếp qua embedding,
prototype và aggregation. Khi bật near-duplicate, canonical là bản sau merge và
summary vẫn ghi cả số lượng trước/sau. Artifact mỗi biến thể có suffix riêng để
không ghi đè; manifest active luôn ghi rõ biến thể đang được đánh giá.

Mỗi news level có grid:

```yaml
k_values: [4, 8, 12, 16, 24, 32]
pca_dims: [null, 32, 64, 128]
temperatures: [0.05, 0.1, 0.2, 0.5]
```

Candidate không đủ event, quá nhiều cluster nhỏ/chết hoặc effective number quá
thấp bị loại. Pipeline ghi hard ID, soft assignment, entropy, nearest distance,
novelty `1-max cosine` và model/PCA train-only. Candidate cuối cho forecasting
phải do validation target metric chọn; clustering diagnostics chỉ là filter/tie
audit.

Aggregation ticker–date hỗ trợ mean, normalized sum, max và exponential decay,
kèm count, no-news mask, days-since-news, entropy, novelty và distance theo từng
level. Các representation:

- R0 price-only;
- R1 metadata;
- R2 raw embedding pooling;
- R3 PCA embedding;
- R4 random projection có cùng chiều;
- R5/R6 hard/soft prototype;
- R7 soft prototype + uncertainty metadata;
- R8 embedding + prototype;
- R9 random prototype placebo;
- R10 shuffled-date placebo (train-only permutation);
- R11 shuffled-ticker placebo cho related/target.

## Target và metric

Mô hình chủ ý nhỏ: Ridge, Elastic Net, logistic/class-weighted logistic và MLP
tối đa hai hidden layers với chronological early stopping. Uncertainty giữ
`mu=baseline_prediction`, chỉ học scale (và degrees of freedom cho Student-t).

Metric chính:

- signed residual: final QLIKE, MAE/RMSE/correlation/direction;
- magnitude/squared: MAE/RMSE/Spearman, top-10% recall, NDCG;
- spike q90/q95: PR-AUC, ROC-AUC, macro F1, Brier, ECE, recall tại precision;
- regime: macro F1, balanced accuracy, per-class metrics, multiclass Brier;
- uncertainty: NLL, CRPS, PIT KS, coverage/width, VaR95/99, Kupiec và
  Christoffersen.

Với volatility residual, VaR được hiểu là one-sided predictive residual tail;
định nghĩa và tail direction được ghi trong bảng metric.

## Response-aware và cross-stock

Response-aware grouping chỉ dùng response vector train. Sau khi cluster
`[sqrt(lambda_z)u; sqrt(lambda_q)q]`, validation/test được gán bằng text centroid
hoặc classifier học từ text. Nếu validation không dự đoán được impact group,
báo cáo kết luận đây chỉ là grouping hậu nghiệm. Shuffled-response là control.
Các family response-aware được khóa/đánh giá riêng và không được phép thay thế
semantic prototype trong quyết định GO chính.

Cross-stock report dùng vector residual 11 ticker theo ngày để tính breadth,
same-sign share, mean absolute response, correlation, q90 breadth,
concentration, common component và firm-specific component. Target-news report
so target ticker với trung bình 10 ticker còn lại và hiệu ứng lan truyền.

## Artifact chính

CSV bắt buộc nằm trong `outputs/tables`:

```text
dataset_summary.csv
ticker_summary.csv
news_summary.csv
event_deduplication_summary.csv
canonical_events.csv
baseline_results.csv
residual_summary.csv
prototype_summary.csv
prototype_examples.csv
prototype_stability.csv
target_screening_manifest.csv
target_comparison_validation.csv
target_comparison_test.csv
ticker_level_results.csv
news_level_results.csv
cross_stock_results.csv
placebo_results.csv
robustness_results.csv
final_decision.csv
```

Figures nằm trong `outputs/figures`; model/PCA/centroid/scaler train-only trong
`outputs/models`; nhật ký và run manifest trong `outputs/logs`. File
`final_decision.csv` lưu bằng chứng so sánh và một dòng final với một trong:

```text
GO-DIRECT, GO-MAGNITUDE, GO-SPIKE, GO-UNCERTAINTY,
GO-REGIME, WEAK-GO, NO-GO
```

Quyết định GO yêu cầu gain validation và test đúng chiều, tốt hơn price-only,
raw/PCA/random projection, tốt hơn placebo shuffled/random, và đủ ổn định qua
seed/fold. Kết quả chỉ ở một ticker, một level hay một regime được hạ xuống
`WEAK-GO`; train loss hoặc silhouette đơn độc không thể tạo GO.

## Kiểm soát chi phí

Grid đầy đủ (4 level × K × PCA × temperature × 5 seed × target/model) vẫn có
thể tốn nhiều giờ và dung lượng. `models.validation_screening` giữ nguyên việc
so sánh mọi K/PCA/temperature/pooling. Representation của vòng này đã được fit
không giám sát trên **main train**; riêng scaler và model screening chỉ fit trên
train-prefix rồi được chấm trên train-tail. Không có validation/test tham gia
vòng screening. Nếu một family được materialize bằng nhiều prototype seed, seed
được lấy trung bình như replicate chứ không phải hyperparameter. Mặc định
holdout chỉ materialize primary prototype seed để hạn chế dung lượng; bảng
prototype-stability và cổng chronological full-refit bắt buộc vẫn fit lại toàn
bộ 6 seed (seed chính + 5 robustness seed). Chỉ hai family đứng đầu đi vào full
model grid trên validation. Mọi full-grid candidate score nằm trong
`target_training_manifest.csv`; audit screening nằm trong
`target_screening_manifest.csv`. Row-level predictions chỉ lưu cho cấu hình đã
khóa để tránh artifact hàng chục triệu dòng. RTX 4070 Super/T4 phù hợp với
encoder frozen và MLP nhỏ. Kết luận chính thức phải giữ 5 seed, tối thiểu 3 fold
và các placebo. Không dùng test để quyết định cấu hình.
