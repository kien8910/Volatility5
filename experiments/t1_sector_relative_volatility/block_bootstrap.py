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


def _daily_ic(group: pd.DataFrame, prediction_column: str) -> float:
    values = []
    for _, day in group.groupby("date", sort=False):
        values.append(
            safe_correlation(
                day["actual_t1"].to_numpy(),
                day[prediction_column].to_numpy(),
                method="spearman",
            )
        )
    return float(np.nanmean(values))


def _statistics(sample: pd.DataFrame, huber_delta: float) -> dict[str, float]:
    actual = sample["actual_t1"].to_numpy(dtype=float)
    baseline = sample["baseline_prediction"].to_numpy(dtype=float)
    text = sample["text_prediction"].to_numpy(dtype=float)
    return {
        "delta_mae": float(
            np.mean(np.abs(actual - baseline)) - np.mean(np.abs(actual - text))
        ),
        "delta_rmse": float(
            np.sqrt(np.mean((actual - baseline) ** 2))
            - np.sqrt(np.mean((actual - text) ** 2))
        ),
        "mean_paired_huber_difference": float(
            huber_values(actual, baseline, huber_delta).mean()
            - huber_values(actual, text, huber_delta).mean()
        ),
        "delta_mean_daily_ic": float(
            _daily_ic(sample, "text_prediction")
            - _daily_ic(sample, "baseline_prediction")
        ),
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
        ].copy()
        dates = np.array(sorted(group["date"].unique()))
        by_date = {date: part for date, part in group.groupby("date", sort=False)}
        rng = np.random.default_rng(seed + 1009 * block_length)
        draws: dict[str, list[float]] = {
            "delta_mae": [],
            "delta_rmse": [],
            "mean_paired_huber_difference": [],
            "delta_mean_daily_ic": [],
        }
        for _ in range(config.bootstrap_repetitions):
            selected = dates[moving_block_indices(len(dates), block_length, rng)]
            pieces = []
            for sample_index, date in enumerate(selected):
                piece = by_date[date].copy()
                piece["date"] = sample_index
                pieces.append(piece)
            sample = pd.concat(pieces, ignore_index=True)
            statistics = _statistics(sample, config.huber_delta)
            for metric, value in statistics.items():
                draws[metric].append(value)
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
