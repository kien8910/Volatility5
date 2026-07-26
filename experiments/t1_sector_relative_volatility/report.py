from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ExperimentConfig


def ticker_comparison_table(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    test = predictions[predictions["split"].eq("test")]
    comparisons = {
        "M2_MINUS_M0": ("M0_PRICE", "M2_PRICE_SEMANTIC"),
        "M3_MINUS_M1": ("M1_PRICE_METADATA", "M3_PRICE_METADATA_SEMANTIC"),
    }
    rows = []
    for comparison, (baseline, text_model) in comparisons.items():
        if not {baseline, text_model}.issubset(set(test["model_name"])):
            continue
        for (seed, ticker), group in test.groupby(["seed", "ticker"], sort=True):
            base = group[group["model_name"].eq(baseline)]
            text = group[group["model_name"].eq(text_model)]
            merged = base.merge(
                text,
                on=["date", "ticker", "seed", "split", "offset"],
                suffixes=("_baseline", "_text"),
                validate="one_to_one",
            )
            actual = merged["actual_t1_baseline"].to_numpy(dtype=float)
            base_prediction = merged["prediction_baseline"].to_numpy(dtype=float)
            text_prediction = merged["prediction_text"].to_numpy(dtype=float)
            rows.append(
                {
                    "comparison": comparison,
                    "ticker": ticker,
                    "seed": int(seed),
                    "n_samples": len(merged),
                    "MAE_baseline": float(np.mean(np.abs(actual - base_prediction))),
                    "MAE_text": float(np.mean(np.abs(actual - text_prediction))),
                    "delta_MAE": float(
                        np.mean(np.abs(actual - base_prediction))
                        - np.mean(np.abs(actual - text_prediction))
                    ),
                    "RMSE_baseline": float(
                        np.sqrt(np.mean((actual - base_prediction) ** 2))
                    ),
                    "RMSE_text": float(
                        np.sqrt(np.mean((actual - text_prediction) ** 2))
                    ),
                    "delta_RMSE": float(
                        np.sqrt(np.mean((actual - base_prediction) ** 2))
                        - np.sqrt(np.mean((actual - text_prediction) ** 2))
                    ),
                    "Pearson": float(
                        pd.Series(actual).corr(pd.Series(text_prediction), method="pearson")
                    ),
                    "Spearman": float(
                        pd.Series(actual).corr(pd.Series(text_prediction), method="spearman")
                    ),
                    "sign_accuracy": float(
                        ((actual > 0) == (text_prediction > 0)).mean()
                    ),
                    "mean_target": float(np.mean(actual)),
                    "std_target": float(np.std(actual, ddof=1)),
                }
            )
    return pd.DataFrame(rows)


def _decision(
    config: ExperimentConfig,
    offsets: pd.DataFrame,
    bootstrap: pd.DataFrame,
    hac: pd.DataFrame,
    ticker_table: pd.DataFrame,
) -> tuple[str, list[str]]:
    if config.debug:
        return "DEBUG-NOT-ELIGIBLE", [
            "Debug mode verifies execution and leakage controls only."
        ]
    reasons = []
    passes = []
    for comparison in sorted(offsets.get("comparison", pd.Series(dtype=str)).unique()):
        passes.append(False)
    comparison_map = {
        "M2_MINUS_M0": ("M0_PRICE", "M2_PRICE_SEMANTIC"),
        "M3_MINUS_M1": ("M1_PRICE_METADATA", "M3_PRICE_METADATA_SEMANTIC"),
    }
    conditional = False
    for comparison, (baseline, text_model) in comparison_map.items():
        base = offsets[
            offsets["split"].eq("test") & offsets["model_name"].eq(baseline)
        ]
        text = offsets[
            offsets["split"].eq("test") & offsets["model_name"].eq(text_model)
        ]
        if base.empty or text.empty:
            continue
        paired = base.merge(
            text,
            on=["split", "seed", "offset"],
            suffixes=("_base", "_text"),
        )
        offset_win_rate = float((paired["mae_base"] > paired["mae_text"]).mean())
        daily_ic_pass = bool(
            paired["mean_daily_ic_text"].mean()
            > paired["mean_daily_ic_base"].mean()
        )
        boot = bootstrap[
            bootstrap["comparison"].eq(comparison)
            & bootstrap["metric"].eq("delta_mae")
        ]
        bootstrap_pass = bool(
            not boot.empty
            and (boot["ci_lower"] > 0).mean() >= 0.5
            and boot["probability_improvement_gt_zero"].mean() >= 0.95
        )
        hac_subset = hac[
            hac["comparison"].eq(comparison)
            & hac["split"].eq("test")
            & hac["loss"].eq("absolute_loss_difference")
        ]
        hac_pass = bool(
            not hac_subset.empty
            and (hac_subset["mean"] > 0).all()
            and (hac_subset["p_value"] < 0.05).mean() >= 0.5
        )
        tickers = ticker_table[ticker_table["comparison"].eq(comparison)]
        ticker_win_rate = float((tickers["delta_MAE"] > 0).mean())
        full_pass = (
            offset_win_rate >= 0.6
            and bootstrap_pass
            and hac_pass
            and daily_ic_pass
            and ticker_win_rate >= 0.6
        )
        conditional |= (
            offset_win_rate >= 0.6
            and ticker_win_rate >= 0.5
            and daily_ic_pass
        )
        passes.append(full_pass)
        reasons.append(
            f"{comparison}: offset_win_rate={offset_win_rate:.3f}, "
            f"bootstrap_pass={bootstrap_pass}, HAC_pass={hac_pass}, "
            f"daily_IC_pass={daily_ic_pass}, "
            f"ticker_win_rate={ticker_win_rate:.3f}."
        )
    if any(passes):
        return "GO", reasons
    if conditional:
        return "CONDITIONAL-GO", reasons
    return "NO-GO", reasons


def generate_final_report(
    *,
    config: ExperimentConfig,
    data_audit: dict[str, Any],
    target_audit: dict[str, Any],
    split_summary: dict[str, Any],
    target_distribution: pd.DataFrame,
    effective_sample: pd.DataFrame,
    overlapping: pd.DataFrame,
    non_overlapping: pd.DataFrame,
    offsets: pd.DataFrame,
    bootstrap: pd.DataFrame,
    hac: pd.DataFrame,
    ticker_table: pd.DataFrame,
    output_path: Path,
) -> tuple[str, list[str]]:
    decision, reasons = _decision(config, offsets, bootstrap, hac, ticker_table)
    test_non_overlap = non_overlapping[non_overlapping["split"].eq("test")]
    report = f"""# T1 sector-relative volatility experiment

## A. Data audit

- Source market artifact: `{data_audit['paths']['market']}`
- Date column: `{data_audit['date_column']}`
- Ticker column: `{data_audit['ticker_column']}`
- Base volatility: `{data_audit['base_volatility_column']}`
- Initial dates: {target_audit['initial_dates']}
- Dates after five-day forward target: {target_audit['dates_after_complete_forward_target']}
- Dates after cross-sectional filter: {target_audit['dates_after_minimum_cross_section']}
- Final target rows: {target_audit['rows_after_target']}
- Mean valid tickers per date: {target_audit['mean_valid_tickers_per_date']:.3f}

## B. Target diagnostics

The target is the five-day forward average log volatility of ticker `i` minus
the leave-one-out mean of the other available semiconductor tickers. It can be
negative; QLIKE is not used.

Target distribution:

{target_distribution.to_markdown(index=False)}

Cross-sectional dependence:

{effective_sample.to_markdown(index=False)}

## C. Purged chronological splits

{pd.DataFrame(split_summary['splits']).to_markdown(index=False)}

Outcome-date sets are disjoint across train, validation and test:
`{split_summary['outcome_sets_are_disjoint']}`.

## D. Predictive results

Overlapping test evaluation is descriptive:

{overlapping[overlapping['split'].eq('test')].to_markdown(index=False)}

The confirmatory result is the mean across five non-overlapping offsets:

{test_non_overlap.to_markdown(index=False)}

## E. Statistical significance

Block-bootstrap results use contiguous date blocks and preserve all tickers
whenever a date is selected.

{bootstrap.to_markdown(index=False)}

HAC/DM results:

{hac[hac['split'].eq('test')].to_markdown(index=False)}

## F. Semantic increment

The two estimands are:

- `M2 − M0`: target-company semantic representation beyond historical price.
- `M3 − M1`: target-company semantic representation beyond price and target-news metadata.

Positive loss differences mean the semantic model is better.

## G. GO/NO-GO decision

**{decision}**

{chr(10).join(f'- {reason}' for reason in reasons)}

The locked interpretation must prioritize non-overlapping offsets, block
bootstrap, HAC/DM, and cross-ticker consistency. An improvement found only in
daily overlapping rows is not sufficient for GO.
"""
    output_path.write_text(report, encoding="utf-8")
    return decision, reasons
