from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .evaluation_common import (
    daily_ic_frame,
    regression_metrics,
    summarize_daily_ic,
)
from .utils import progress


def validate_offsets(predictions: pd.DataFrame, stride: int, horizon: int) -> None:
    for (split, model, seed, offset), group in predictions.groupby(
        ["split", "model_name", "seed", "offset"], sort=True
    ):
        dates = pd.DatetimeIndex(sorted(group["date"].unique()))
        positions = {
            date: index
            for index, date in enumerate(
                sorted(predictions.loc[predictions["split"].eq(split), "date"].unique())
            )
        }
        selected_positions = [positions[date] for date in dates]
        if len(selected_positions) > 1:
            gaps = np.diff(selected_positions)
            if (gaps < horizon).any():
                raise AssertionError(
                    f"Overlapping dates in {split}/{model}/{seed}/offset={offset}"
                )
            if not (gaps % stride == 0).all():
                raise AssertionError(
                    f"Invalid stride in {split}/{model}/{seed}/offset={offset}"
                )


def evaluate_non_overlapping(
    predictions: pd.DataFrame,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_offsets(predictions, config.eval_stride, config.horizon)
    ic = daily_ic_frame(predictions)
    offset_rows: list[dict[str, Any]] = []
    groups = list(
        predictions.groupby(["split", "model_name", "seed", "offset"], sort=True)
    )
    for (split, model, seed, offset), group in progress(
        groups,
        total=len(groups),
        description="[Non-overlap evaluation] offsets",
    ):
        ic_subset = ic[
            ic["split"].eq(split)
            & ic["model_name"].eq(model)
            & ic["seed"].eq(seed)
            & ic["offset"].eq(offset)
        ]
        offset_rows.append(
            {
                "split": split,
                "model_name": model,
                "seed": int(seed),
                "offset": int(offset),
                **regression_metrics(group, huber_delta=config.huber_delta),
                **summarize_daily_ic(ic_subset),
                "top3_accuracy": float(ic_subset["top3_accuracy"].mean()),
                "bottom3_accuracy": float(ic_subset["bottom3_accuracy"].mean()),
            }
        )
    offsets = pd.DataFrame(offset_rows)
    aggregate_rows = []
    metric_columns = [
        "mae",
        "rmse",
        "huber",
        "r2_oos",
        "pearson",
        "spearman",
        "sign_accuracy",
        "balanced_sign_accuracy",
        "mean_daily_ic",
        "top3_accuracy",
        "bottom3_accuracy",
    ]
    for (split, model, seed), group in offsets.groupby(
        ["split", "model_name", "seed"], sort=True
    ):
        row: dict[str, Any] = {
            "split": split,
            "model_name": model,
            "seed": int(seed),
            "offset_count": int(group["offset"].nunique()),
        }
        for column in metric_columns:
            row[f"mean_{column}"] = float(group[column].mean())
            row[f"std_{column}"] = float(group[column].std(ddof=1))
        row["worst_offset_mae"] = int(group.loc[group["mae"].idxmax(), "offset"])
        row["best_offset_mae"] = int(group.loc[group["mae"].idxmin(), "offset"])
        aggregate_rows.append(row)
    return offsets, pd.DataFrame(aggregate_rows)
