from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .utils import assert_unique, parse_dates_mixed, progress, require_columns


def build_t1_target(
    market: pd.DataFrame,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build a five-day forward, leave-one-out sector-relative target.

    Shifts are performed on the common trading-date index. A ticker must have
    a valid observation on every one of the next ``horizon`` common dates.
    """

    require_columns(
        market,
        ["ticker", "feature_date", "log_variance", "split"],
        "market panel",
    )
    source = market[["ticker", "feature_date", "log_variance", "split"]].copy()
    source["feature_date"] = parse_dates_mixed(source["feature_date"])
    source["ticker"] = source["ticker"].astype(str)
    source = source[source["ticker"].isin(config.tickers)]
    assert_unique(source, ["ticker", "feature_date"], "market target source")

    panel = (
        source.pivot(index="feature_date", columns="ticker", values="log_variance")
        .sort_index()
        .reindex(columns=config.tickers)
    )
    calendar = panel.index
    if not calendar.is_monotonic_increasing or calendar.has_duplicates:
        raise AssertionError("Common trading calendar must be sorted and unique")

    parts: list[pd.DataFrame] = []
    for ticker in progress(
        config.tickers,
        total=len(config.tickers),
        description="[Target construction] tickers",
    ):
        part = pd.DataFrame(
            {
                "date": calendar,
                "ticker": ticker,
                "base_log_volatility_t": panel[ticker].to_numpy(),
            }
        )
        for step in range(1, config.horizon + 1):
            part[f"base_target_t_plus_{step}"] = panel[ticker].shift(-step).to_numpy()
            date_values = np.full(len(calendar), np.datetime64("NaT"), dtype="datetime64[ns]")
            if len(calendar) > step:
                date_values[:-step] = calendar[step:].to_numpy()
            part[f"target_date_t_plus_{step}"] = date_values
        parts.append(part)

    target = pd.concat(parts, ignore_index=True)
    target = target.merge(
        source[["ticker", "feature_date", "split"]].rename(
            columns={"feature_date": "date"}
        ),
        on=["ticker", "date"],
        how="left",
        validate="one_to_one",
    )
    future_columns = [
        f"base_target_t_plus_{step}" for step in range(1, config.horizon + 1)
    ]
    target["has_complete_forward_window"] = target[future_columns].notna().all(axis=1)
    target["forward_mean_5d"] = target[future_columns].mean(axis=1)
    target.loc[
        ~target["has_complete_forward_window"], "forward_mean_5d"
    ] = np.nan

    date_stats = (
        target.groupby("date", sort=True)["forward_mean_5d"]
        .agg(valid_ticker_count="count", cross_section_sum="sum")
        .reset_index()
    )
    target = target.merge(date_stats, on="date", how="left", validate="many_to_one")
    target["peer_mean_leave_one_out"] = (
        target["cross_section_sum"] - target["forward_mean_5d"]
    ) / (target["valid_ticker_count"] - 1)
    target["t1_target"] = (
        target["forward_mean_5d"] - target["peer_mean_leave_one_out"]
    )
    target["full_cross_section_mean"] = (
        target["cross_section_sum"] / target["valid_ticker_count"]
    )
    target["full_demeaned_target"] = (
        target["forward_mean_5d"] - target["full_cross_section_mean"]
    )
    target["outcome_start_date"] = target["target_date_t_plus_1"]
    target["outcome_end_date"] = target[f"target_date_t_plus_{config.horizon}"]

    initial_dates = int(source["feature_date"].nunique())
    complete = target[target["has_complete_forward_window"]].copy()
    dates_after_forward = int(complete["date"].nunique())
    eligible_dates = date_stats.loc[
        date_stats["valid_ticker_count"] >= config.min_cross_section_size, "date"
    ]
    final = complete[complete["date"].isin(eligible_dates)].copy()
    final = final[np.isfinite(final["t1_target"])].copy()
    final = final.sort_values(["date", "ticker"]).reset_index(drop=True)

    distribution = (
        date_stats["valid_ticker_count"].value_counts().sort_index().to_dict()
    )
    audit = {
        "initial_dates": initial_dates,
        "dates_after_complete_forward_target": dates_after_forward,
        "dates_after_minimum_cross_section": int(final["date"].nunique()),
        "dates_removed_missing_forward": initial_dates - dates_after_forward,
        "dates_removed_cross_section": dates_after_forward
        - int(final["date"].nunique()),
        "rows_after_target": int(len(final)),
        "samples_by_ticker": {
            str(key): int(value)
            for key, value in final["ticker"].value_counts().sort_index().items()
        },
        "mean_valid_tickers_per_date": float(
            date_stats["valid_ticker_count"].mean()
        ),
        "valid_ticker_count_distribution": {
            str(key): int(value) for key, value in distribution.items()
        },
        "minimum_cross_section_size": config.min_cross_section_size,
        "horizon": config.horizon,
        "common_calendar_start": str(calendar.min()),
        "common_calendar_end": str(calendar.max()),
    }
    return final, audit


def manual_validation_sample(target: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    columns = [
        "date",
        "ticker",
        "base_target_t_plus_1",
        "base_target_t_plus_2",
        "base_target_t_plus_3",
        "base_target_t_plus_4",
        "base_target_t_plus_5",
        "forward_mean_5d",
        "peer_mean_leave_one_out",
        "t1_target",
    ]
    if len(target) <= n:
        return target[columns].copy()
    indices = np.linspace(0, len(target) - 1, n, dtype=int)
    return target.iloc[indices][columns].reset_index(drop=True)
