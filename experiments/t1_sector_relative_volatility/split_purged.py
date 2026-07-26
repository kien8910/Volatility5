from __future__ import annotations

from typing import Any

import pandas as pd

from .config import ExperimentConfig
from .utils import parse_dates_mixed, require_columns


SPLIT_ORDER = ["train", "validation", "test"]


def build_purged_splits(
    frame: pd.DataFrame,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    require_columns(
        frame,
        ["date", "split", "outcome_start_date", "outcome_end_date"],
        "prepared T1 frame",
    )
    result = frame.copy()
    for column in ["date", "outcome_start_date", "outcome_end_date"]:
        result[column] = parse_dates_mixed(result[column])
    result["split"] = result["split"].astype(str)

    raw_starts = {
        split: result.loc[result["split"].eq(split), "date"].min()
        for split in SPLIT_ORDER
    }
    if any(pd.isna(value) for value in raw_starts.values()):
        raise AssertionError(f"Missing split dates: {raw_starts}")
    if not raw_starts["train"] < raw_starts["validation"] < raw_starts["test"]:
        raise AssertionError(f"Chronological split order is invalid: {raw_starts}")

    train_ok = ~result["split"].eq("train") | result["outcome_end_date"].lt(
        raw_starts["validation"]
    )
    validation_ok = ~result["split"].eq(
        "validation"
    ) | result["outcome_end_date"].lt(raw_starts["test"])
    result = result[train_ok & validation_ok].copy()

    embargo_removed: dict[str, list[str]] = {}
    if config.embargo_days > 0:
        for split in ["validation", "test"]:
            split_dates = sorted(result.loc[result["split"].eq(split), "date"].unique())
            remove_dates = split_dates[: config.embargo_days]
            embargo_removed[split] = [str(pd.Timestamp(date)) for date in remove_dates]
            result = result[
                ~(result["split"].eq(split) & result["date"].isin(remove_dates))
            ]

    if config.debug:
        limits = {
            "train": config.debug_train_dates,
            "validation": config.debug_validation_dates,
            "test": config.debug_test_dates,
        }
        keep_parts = []
        for split, maximum in limits.items():
            split_frame = result[result["split"].eq(split)]
            dates = sorted(split_frame["date"].unique())
            selected = dates[-maximum:] if split == "train" else dates[:maximum]
            keep_parts.append(split_frame[split_frame["date"].isin(selected)])
        result = pd.concat(keep_parts, ignore_index=True)

    result = result.sort_values(["date", "ticker"]).reset_index(drop=True)
    assert_no_shared_outcomes(result, config)
    summary_rows = []
    for split in SPLIT_ORDER:
        subset = result[result["split"].eq(split)]
        if subset.empty:
            raise AssertionError(f"Purging produced an empty {split} split")
        summary_rows.append(
            {
                "split": split,
                "start_date": str(subset["date"].min()),
                "end_date": str(subset["date"].max()),
                "outcome_start": str(subset["outcome_start_date"].min()),
                "outcome_end": str(subset["outcome_end_date"].max()),
                "dates": int(subset["date"].nunique()),
                "rows": int(len(subset)),
                "tickers": int(subset["ticker"].nunique()),
            }
        )
    summary = {
        "horizon": config.horizon,
        "purge_anchor_days": config.purge_anchor_days,
        "embargo_days": config.embargo_days,
        "raw_split_starts": {key: str(value) for key, value in raw_starts.items()},
        "embargo_removed_dates": embargo_removed,
        "splits": summary_rows,
        "outcome_sets_are_disjoint": True,
    }
    return result, summary


def assert_no_shared_outcomes(frame: pd.DataFrame, config: ExperimentConfig) -> None:
    outcome_columns = [
        f"target_date_t_plus_{step}" for step in range(1, config.horizon + 1)
    ]
    require_columns(frame, outcome_columns, "purged split")
    sets: dict[str, set[pd.Timestamp]] = {}
    for split in SPLIT_ORDER:
        subset = frame[frame["split"].eq(split)]
        values: set[pd.Timestamp] = set()
        for column in outcome_columns:
            values.update(pd.to_datetime(subset[column]).dropna().tolist())
        sets[split] = values
    pairs = [("train", "validation"), ("validation", "test"), ("train", "test")]
    overlaps = {
        f"{left}_{right}": sorted(sets[left].intersection(sets[right]))
        for left, right in pairs
    }
    bad = {key: value[:10] for key, value in overlaps.items() if value}
    if bad:
        raise AssertionError(f"Outcome dates overlap across splits: {bad}")


def assign_non_overlapping_offsets(
    frame: pd.DataFrame,
    *,
    stride: int,
) -> pd.DataFrame:
    result = frame.copy()
    result["offset"] = -1
    for split in ["validation", "test"]:
        dates = sorted(result.loc[result["split"].eq(split), "date"].unique())
        mapping = {pd.Timestamp(date): index % stride for index, date in enumerate(dates)}
        mask = result["split"].eq(split)
        result.loc[mask, "offset"] = (
            result.loc[mask, "date"].map(mapping).astype(int)
        )
    return result
