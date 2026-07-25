# Prototype cross-attention without structured-event extraction

This experiment reads the existing daily panels directly:

- `R1`: pure metadata such as news count, canonical count, lag, no-news mask
  and days-since-last-news;
- `R6`: semantic soft prototypes;
- `R9`, `R10`, `R11`: random-prototype, shuffled-date and shuffled-ticker
  placebos.

It does not read the LLM cache, `structured_events.parquet`, the
structured-event feature panel, or any output from
`run_structured_event_pilot.py`.

## Architecture

```text
HAR/price features at t -> market query
4 news levels x K prototype activations -> separate prototype tokens
market query cross-attends to prototype tokens
price prediction + learned gate * news correction -> target at t+1
```

`META_BASIC_MLP` and `PRICE_META_RIDGE` are metadata controls. They use price
features and news-arrival metadata, but deliberately exclude entropy, novelty
and centroid distance so that metadata-only does not contain prototype-derived
information.

## Required old-pipeline artifacts

The server must already contain fold-safe R6 and placebo panels:

```bash
python fintexts_semiconductor_prototype/run_pipeline.py \
  --stage r6-confirmatory \
  --config fintexts_semiconductor_prototype/config/config_r6_confirmatory.yaml
```

No embedding or prototype needs to be rebuilt when these fold artifacts already
exist.

## Smoke run

One fold, one seed, five models and at most 20 epochs:

```bash
python original_volatility_targets/run_prototype_cross_attention.py \
  --stage plan --quick
python original_volatility_targets/run_prototype_cross_attention.py \
  --stage train --quick --resume
python original_volatility_targets/run_prototype_cross_attention.py \
  --stage evaluate --quick
```

The smoke evaluation is technical only. It cannot pass the full stability
gate because it does not contain all three folds, all five seeds and all fixed
placebos.

## Full chronological validation

Volatility level, 3 folds x 5 paired seeds:

```bash
python original_volatility_targets/run_prototype_cross_attention.py \
  --stage all --resume
```

Run q90 volatility spike separately:

```bash
python original_volatility_targets/run_prototype_cross_attention.py \
  --stage all --target spike_q90 --resume
```

`--model`, `--fold` and `--seed` can be repeated to run selected cells:

```bash
python original_volatility_targets/run_prototype_cross_attention.py \
  --stage train \
  --target volatility_level \
  --model R6_META_XATTN \
  --fold 1 \
  --seed 11 \
  --resume
```

## Live progress

The terminal progress bar shows task count, current fold/seed/model, epoch and
validation loss. From another terminal:

```bash
tail -f original_volatility_targets/outputs/prototype_cross_attention/logs/prototype_cross_attention.log
watch -n 5 cat original_volatility_targets/outputs/prototype_cross_attention/logs/progress_state.json
```

## Outputs

```text
outputs/prototype_cross_attention/
├── checkpoints/
├── models/
├── logs/
│   ├── prototype_cross_attention.log
│   └── progress_state.json
└── tables/
    ├── cross_attention_task_plan.csv
    ├── cross_attention_results.csv
    ├── cross_attention_aggregate.csv
    ├── cross_attention_comparisons.csv
    ├── cross_attention_attention_audit.csv
    └── cross_attention_decision.csv
```

`R6_META_XATTN` receives `XATTN-PASS` only if it beats price-only,
metadata-only, R6 concatenation and all three fixed placebos under the locked
mean-gain, cell-win-rate and all-fold stability gates. This runner evaluates
chronological validation folds only and never opens the locked holdout.
