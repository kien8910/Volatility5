from __future__ import annotations

import pandas as pd

from experiments.t1_sector_relative_volatility.config import ExperimentConfig, TICKERS
from experiments.t1_sector_relative_volatility.split_purged import build_purged_splits


def frame_with_boundaries() -> pd.DataFrame:
    dates = pd.bdate_range("2021-01-01", periods=50)
    rows = []
    for index, date in enumerate(dates[:-5]):
        split = "train" if index < 25 else ("validation" if index < 35 else "test")
        for ticker in TICKERS:
            row = {
                "date": date,
                "ticker": ticker,
                "split": split,
                "outcome_start_date": dates[index + 1],
                "outcome_end_date": dates[index + 5],
                "t1_target": float(index),
            }
            for step in range(1, 6):
                row[f"target_date_t_plus_{step}"] = dates[index + step]
            rows.append(row)
    return pd.DataFrame(rows)


def test_purged_split_is_chronological_and_supports_embargo(tmp_path) -> None:
    config = ExperimentConfig(
        workspace_root=tmp_path,
        output_directory=tmp_path / "out",
        embargo_days=2,
    )
    result, summary = build_purged_splits(frame_with_boundaries(), config)
    rows = {item["split"]: item for item in summary["splits"]}
    assert pd.Timestamp(rows["train"]["end_date"]) < pd.Timestamp(
        rows["validation"]["start_date"]
    )
    assert pd.Timestamp(rows["validation"]["end_date"]) < pd.Timestamp(
        rows["test"]["start_date"]
    )
    assert len(summary["embargo_removed_dates"]["validation"]) == 2
    assert len(summary["embargo_removed_dates"]["test"]) == 2
    assert set(result["ticker"]) == set(TICKERS)
