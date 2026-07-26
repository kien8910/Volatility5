from __future__ import annotations

from typing import Any

import pandas as pd

from .config import ExperimentConfig
from .evaluation_common import (
    daily_ic_frame,
    regression_metrics,
    summarize_daily_ic,
)
from .utils import progress


def evaluate_overlapping(
    predictions: pd.DataFrame,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ic = daily_ic_frame(predictions)
    metric_rows: list[dict[str, Any]] = []
    ticker_rows: list[dict[str, Any]] = []
    groups = list(
        predictions.groupby(["split", "model_name", "seed"], sort=True)
    )
    for (split, model, seed), group in progress(
        groups,
        total=len(groups),
        description="[Daily evaluation] models",
    ):
        row = {
            "split": split,
            "model_name": model,
            "seed": int(seed),
            **regression_metrics(group, huber_delta=config.huber_delta),
            **summarize_daily_ic(
                ic[
                    ic["split"].eq(split)
                    & ic["model_name"].eq(model)
                    & ic["seed"].eq(seed)
                ]
            ),
        }
        metric_rows.append(row)
        for ticker, ticker_group in group.groupby("ticker", sort=True):
            ticker_rows.append(
                {
                    "split": split,
                    "model_name": model,
                    "seed": int(seed),
                    "ticker": ticker,
                    "mean_target": float(ticker_group["actual_t1"].mean()),
                    "std_target": float(ticker_group["actual_t1"].std()),
                    **regression_metrics(
                        ticker_group, huber_delta=config.huber_delta
                    ),
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(ticker_rows), ic
