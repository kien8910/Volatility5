from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from .config import ExperimentConfig
from .evaluation_common import huber_values
from .utils import progress


COMPARISONS = {
    "M2_MINUS_M0": ("M0_PRICE", "M2_PRICE_SEMANTIC"),
    "M3_MINUS_M1": ("M1_PRICE_METADATA", "M3_PRICE_METADATA_SEMANTIC"),
}


def paired_loss_differences(
    predictions: pd.DataFrame,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    available = set(predictions["model_name"].unique())
    row_parts: list[pd.DataFrame] = []
    for comparison, (baseline, text_model) in COMPARISONS.items():
        if baseline not in available or text_model not in available:
            continue
        keys = ["date", "ticker", "split", "offset", "seed"]
        base = predictions[predictions["model_name"].eq(baseline)].copy()
        text = predictions[predictions["model_name"].eq(text_model)].copy()
        merged = base.merge(
            text,
            on=keys,
            how="inner",
            suffixes=("_base", "_text"),
            validate="one_to_one",
        )
        actual = merged["actual_t1_base"].to_numpy(dtype=float)
        if not np.allclose(actual, merged["actual_t1_text"].to_numpy(dtype=float)):
            raise AssertionError(f"Actual targets differ in {comparison}")
        base_prediction = merged["prediction_base"].to_numpy(dtype=float)
        text_prediction = merged["prediction_text"].to_numpy(dtype=float)
        paired = merged[keys].copy()
        paired["comparison"] = comparison
        paired["baseline_model"] = baseline
        paired["text_model"] = text_model
        paired["actual_t1"] = actual
        paired["baseline_prediction"] = base_prediction
        paired["text_prediction"] = text_prediction
        paired["absolute_loss_difference"] = np.abs(
            actual - base_prediction
        ) - np.abs(actual - text_prediction)
        paired["squared_loss_difference"] = (
            actual - base_prediction
        ) ** 2 - (actual - text_prediction) ** 2
        paired["huber_loss_difference"] = huber_values(
            actual, base_prediction, config.huber_delta
        ) - huber_values(actual, text_prediction, config.huber_delta)
        row_parts.append(paired)
    if not row_parts:
        raise AssertionError("No paired semantic comparisons can be constructed")
    row_level = pd.concat(row_parts, ignore_index=True)
    daily = (
        row_level.groupby(
            ["comparison", "baseline_model", "text_model", "split", "seed", "date"],
            sort=True,
        )
        .agg(
            absolute_loss_difference=("absolute_loss_difference", "mean"),
            squared_loss_difference=("squared_loss_difference", "mean"),
            huber_loss_difference=("huber_loss_difference", "mean"),
            n_tickers=("ticker", "nunique"),
        )
        .reset_index()
    )
    return row_level, daily


def newey_west_test(values: np.ndarray, lag: int) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n <= lag + 2:
        return {
            "mean": np.nan,
            "hac_standard_error": np.nan,
            "t_statistic": np.nan,
            "p_value": np.nan,
        }
    centered = values - values.mean()
    gamma_zero = float(np.dot(centered, centered) / n)
    long_run_variance = gamma_zero
    for order in range(1, lag + 1):
        covariance = float(
            np.dot(centered[order:], centered[:-order]) / n
        )
        weight = 1.0 - order / (lag + 1.0)
        long_run_variance += 2.0 * weight * covariance
    long_run_variance = max(long_run_variance, 0.0)
    standard_error = math.sqrt(long_run_variance / n)
    statistic = values.mean() / standard_error if standard_error > 0 else np.nan
    p_value = 2.0 * (1.0 - norm.cdf(abs(statistic))) if np.isfinite(statistic) else np.nan
    return {
        "mean": float(values.mean()),
        "hac_standard_error": float(standard_error),
        "t_statistic": float(statistic),
        "p_value": float(p_value),
    }


def hac_dm_results(
    daily_differences: pd.DataFrame,
    config: ExperimentConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    loss_columns = [
        "absolute_loss_difference",
        "squared_loss_difference",
        "huber_loss_difference",
    ]
    groups = list(
        daily_differences.groupby(["comparison", "split", "seed"], sort=True)
    )
    tasks = [
        (keys, group, loss, lag)
        for keys, group in groups
        for loss in loss_columns
        for lag in config.hac_lags
    ]
    for ((comparison, split, seed), group, loss, lag) in progress(
        tasks,
        total=len(tasks),
        description="[HAC/DM] tests",
    ):
        result = newey_west_test(group[loss].to_numpy(), lag)
        rows.append(
            {
                "comparison": comparison,
                "split": split,
                "seed": int(seed),
                "loss": loss,
                "lag": int(lag),
                "n_dates": int(group["date"].nunique()),
                **result,
                "positive_means_text_better": True,
                "test": "DM_with_Newey_West_long_run_variance",
            }
        )
    return pd.DataFrame(rows)
