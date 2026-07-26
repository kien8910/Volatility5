# T1 sector-relative five-day volatility

This is an independent, leakage-controlled experiment for testing whether
target-company semantic news predicts a stock's volatility relative to the
other semiconductor stocks over the next five trading days.

It does not write to or modify:

- `fintexts_semiconductor_prototype`
- `original_volatility_targets`

The experiment reads the validated artifacts under:

```text
fintexts_semiconductor_prototype/runs/light/data/processed/
```

and writes only under:

```text
experiments/t1_sector_relative_volatility/outputs/
```

## Target

For ticker `i` and anchor date `t`:

```text
forward_mean_5d(i,t)
  = mean(log_volatility(i,t+1), ..., log_volatility(i,t+5))
```

The peer benchmark excludes ticker `i`:

```text
peer_mean_loo(i,t)
  = mean(forward_mean_5d(j,t) for all valid j != i)
```

The target is:

```text
T1(i,t) = forward_mean_5d(i,t) - peer_mean_loo(i,t)
```

The main experiment requires at least 8 valid tickers on an anchor date.

## Discovered source mapping

The implementation verifies the real schema before training:

| Component | Artifact | Columns used |
|---|---|---|
| Market | `market_supervised.parquet` | `feature_date`, `ticker`, `log_variance`, HAR/lag features, `split` |
| Metadata | `features_R1_mean.parquet` | `meta__target__*` |
| Semantics | `features_R3_mean.parquet` | `pca__target__*` |

`R3` is the default because it is a train-fitted, eight-dimensional target-news
semantic representation. Use `--semantic-representation R2` to run the
768-dimensional raw target-news embedding control.

The existing split is retained and then purged for the five-day outcome:

```text
Train:      2019-01-31 → 2022-01-10
Validation: 2022-01-11 → 2023-01-03
Test:       2023-01-04 → 2023-12-28
```

The exact post-purge dates are written to `split_summary.json`.

## Models

- `M0_PRICE`: historical price/HAR features only.
- `M1_PRICE_METADATA`: price plus target-company news metadata.
- `M2_PRICE_SEMANTIC`: price plus target-company semantic representation.
- `M3_PRICE_METADATA_SEMANTIC`: price, metadata and target semantics.

The semantic increments are:

```text
M2 − M0
M3 − M1
```

All models are pooled Ridge models with ticker one-hot indicators. Numeric
imputation and scaling are fit only on train. Ridge alpha is selected using the
mean validation MAE across five non-overlapping offsets.

## Installation

From the workspace root:

```bash
python -m pip install -r experiments/t1_sector_relative_volatility/requirements.txt
```

Python 3.10 or newer is required.

## Tests

```bash
python -m pytest experiments/t1_sector_relative_volatility/tests -q
```

Tests cover:

- exact `t+1` through `t+5` target construction;
- leave-one-out peer mean;
- no shared outcome dates after purging;
- embargo handling;
- all five non-overlapping offsets.

## Debug run

From the workspace root:

```bash
python experiments/t1_sector_relative_volatility/run_debug.py
```

Debug mode:

- uses the most recent 126 train dates;
- uses 50 validation and 50 test dates;
- trains `M0` and `M2`;
- runs 100 block-bootstrap repetitions;
- retains all target and leakage assertions;
- writes to `outputs/debug`.

## Full run

```bash
python experiments/t1_sector_relative_volatility/run_experiment.py \
  --config experiments/t1_sector_relative_volatility/config.py
```

PowerShell single-line form:

```powershell
python experiments\t1_sector_relative_volatility\run_experiment.py --config experiments\t1_sector_relative_volatility\config.py
```

Useful overrides:

```bash
python experiments/t1_sector_relative_volatility/run_experiment.py \
  --embargo-days 5 \
  --semantic-representation R3 \
  --bootstrap 2000 \
  --seeds 42 123 2026
```

Robustness runs:

```bash
python experiments/t1_sector_relative_volatility/run_experiment.py \
  --embargo-days 5 \
  --output-directory experiments/t1_sector_relative_volatility/outputs/embargo_5

python experiments/t1_sector_relative_volatility/run_experiment.py \
  --embargo-days 10 \
  --output-directory experiments/t1_sector_relative_volatility/outputs/embargo_10
```

## Progress

The CLI prints eight top-level stages. Long loops use `tqdm`, including target
construction, model/seed training, offset evaluation, HAC tests and block
bootstrap. Progress bars show completed work, percentage, elapsed time and ETA.

Block bootstrap is vectorized on the date dimension. Per-date ticker-preserving
loss and IC statistics are computed once; bootstrap repetitions use NumPy index
matrices instead of rebuilding pandas DataFrames or recalculating Spearman
correlations. This preserves cross-sectional dependence while avoiding the
Python/GIL bottleneck.

## Required outputs

Each run produces:

```text
data_audit.json
target_validation.csv
target_validation_sample.csv
split_summary.json
config_used.yaml
model_metrics_overlapping.csv
model_metrics_non_overlapping.csv
metrics_by_offset.csv
metrics_by_ticker.csv
daily_cross_sectional_ic.csv
paired_loss_difference.csv
hac_dm_results.csv
bootstrap_results.csv
effective_sample_size.csv
ranking_portfolio_results.csv
predictions/validation_predictions.csv
predictions/test_predictions.csv
figures/
final_report.md
```

Additional audit tables contain the target distributions, ACF, correlation
matrices, model selection and ranking-portfolio bootstrap.

## Interpretation

Positive paired loss difference means the semantic model is better. A full
`GO` requires:

- improvement on non-overlapping test offsets;
- consistency across most offsets and tickers;
- positive block-bootstrap evidence;
- compatible HAC/DM evidence;
- no leakage failure.

An improvement only in the overlapping daily evaluation is not sufficient.
