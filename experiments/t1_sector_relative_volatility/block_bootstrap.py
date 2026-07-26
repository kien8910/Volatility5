from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .evaluation_common import huber_values, safe_correlation
from .statistical_tests import COMPARISONS
from .utils import progress


def moving_block_indices(
    n_dates: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if n_dates <= 0:
        raise ValueError("n_dates must be positive")
    block_length = min(block_length, n_dates)
    block_count = int(np.ceil(n_dates / block_length))
    starts = rng.integers(0, n_dates, size=block_count)
    indices = np.concatenate(
        [
            (start + np.arange(block_length, dtype=int)) % n_dates
            for start in starts
        ]
    )
    return indices[:n_dates]


def moving_block_index_matrix(
    n_dates: int,
    block_length: int,
    repetitions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate all moving-block draws as a compact date-index matrix."""

    if n_dates <= 0:
        raise ValueError("n_dates must be positive")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    block_length = min(block_length, n_dates)
    block_count = int(np.ceil(n_dates / block_length))
    starts = rng.integers(
        0,
        n_dates,
        size=(repetitions, block_count),
        dtype=np.int64,
    )
    offsets = np.arange(block_length, dtype=np.int64)
    indices = (starts[:, :, None] + offsets[None, None, :]) % n_dates
    return indices.reshape(repetitions, -1)[:, :n_dates]


def _daily_sufficient_statistics(
    group: pd.DataFrame,
    huber_delta: float,
) -> dict[str, np.ndarray]:
    """Collapse ticker rows once; bootstrap then operates only on date arrays."""

    rows: list[dict[str, float]] = []
    for _, day in group.groupby("date", sort=True):
        actual = day["actual_t1"].to_numpy(dtype=float)
        baseline = day["baseline_prediction"].to_numpy(dtype=float)
        text = day["text_prediction"].to_numpy(dtype=float)
        rows.append(
            {
                "n": float(len(day)),
                "absolute_base_sum": float(np.abs(actual - baseline).sum()),
                "absolute_text_sum": float(np.abs(actual - text).sum()),
                "squared_base_sum": float(((actual - baseline) ** 2).sum()),
                "squared_text_sum": float(((actual - text) ** 2).sum()),
                "huber_base_sum": float(
                    huber_values(actual, baseline, huber_delta).sum()
                ),
                "huber_text_sum": float(
                    huber_values(actual, text, huber_delta).sum()
                ),
                "daily_ic_base": safe_correlation(
                    actual, baseline, method="spearman"
                ),
                "daily_ic_text": safe_correlation(
                    actual, text, method="spearman"
                ),
            }
        )
    daily = pd.DataFrame(rows)
    return {
        column: daily[column].to_numpy(dtype=float)
        for column in daily.columns
    }


def _vectorized_statistics(
    daily: dict[str, np.ndarray],
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    observation_count = daily["n"][indices].sum(axis=1)
    absolute_base = daily["absolute_base_sum"][indices].sum(axis=1)
    absolute_text = daily["absolute_text_sum"][indices].sum(axis=1)
    squared_base = daily["squared_base_sum"][indices].sum(axis=1)
    squared_text = daily["squared_text_sum"][indices].sum(axis=1)
    huber_base = daily["huber_base_sum"][indices].sum(axis=1)
    huber_text = daily["huber_text_sum"][indices].sum(axis=1)
    return {
        "delta_mae": absolute_base / observation_count
        - absolute_text / observation_count,
        "delta_rmse": np.sqrt(squared_base / observation_count)
        - np.sqrt(squared_text / observation_count),
        "mean_paired_huber_difference": huber_base / observation_count
        - huber_text / observation_count,
        "delta_mean_daily_ic": np.nanmean(
            daily["daily_ic_text"][indices], axis=1
        )
        - np.nanmean(daily["daily_ic_base"][indices], axis=1),
    }


def block_bootstrap(
    paired_rows: pd.DataFrame,
    config: ExperimentConfig,
) -> pd.DataFrame:
    test = paired_rows[paired_rows["split"].eq("test")].copy()
    rows: list[dict[str, Any]] = []
    tasks = [
        (comparison, int(seed), block_length)
        for comparison in sorted(test["comparison"].unique())
        for seed in sorted(test["seed"].unique())
        for block_length in config.bootstrap_block_lengths
    ]
    for comparison, seed, block_length in progress(
        tasks,
        total=len(tasks),
        description="[Block bootstrap] comparison/seed/block",
    ):
        group = test[
            test["comparison"].eq(comparison) & test["seed"].eq(seed)
        ].sort_values(["date", "ticker"])
        daily = _daily_sufficient_statistics(group, config.huber_delta)
        n_dates = len(daily["n"])
        rng = np.random.default_rng(seed + 1009 * block_length)
        indices = moving_block_index_matrix(
            n_dates,
            block_length,
            config.bootstrap_repetitions,
            rng,
        )
        draws = _vectorized_statistics(daily, indices)
        alpha = (1.0 - config.bootstrap_confidence) / 2.0
        for metric, values in draws.items():
            array = np.asarray(values, dtype=float)
            rows.append(
                {
                    "comparison": comparison,
                    "split": "test",
                    "seed": seed,
                    "block_length": block_length,
                    "bootstrap_repetitions": len(array),
                    "metric": metric,
                    "bootstrap_mean": float(np.nanmean(array)),
                    "bootstrap_standard_error": float(np.nanstd(array, ddof=1)),
                    "ci_lower": float(np.nanquantile(array, alpha)),
                    "ci_upper": float(np.nanquantile(array, 1.0 - alpha)),
                    "probability_improvement_gt_zero": float(
                        np.nanmean(array > 0)
                    ),
                    "positive_means_text_better": True,
                }
            )
    return pd.DataFrame(rows)
