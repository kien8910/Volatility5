from __future__ import annotations

import pandas as pd

from experiments.t1_sector_relative_volatility.config import ExperimentConfig, TICKERS
from experiments.t1_sector_relative_volatility.split_purged import (
    assert_no_shared_outcomes,
    build_purged_splits,
)


def synthetic_target_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=40)
    rows = []
    for index, date in enumerate(dates[:-5]):
        split = "train" if index < 20 else ("validation" if index < 28 else "test")
        for ticker in TICKERS:
            row = {
                "date": date,
                "ticker": ticker,
                "split": split,
                "outcome_start_date": dates[index + 1],
                "outcome_end_date": dates[index + 5],
                "t1_target": 0.0,
            }
            for step in range(1, 6):
                row[f"target_date_t_plus_{step}"] = dates[index + step]
            rows.append(row)
    return pd.DataFrame(rows)


def test_purge_removes_shared_outcome_dates(tmp_path) -> None:
    config = ExperimentConfig(
        workspace_root=tmp_path,
        output_directory=tmp_path / "out",
    )
    purged, summary = build_purged_splits(synthetic_target_frame(), config)
    assert summary["outcome_sets_are_disjoint"]
    assert_no_shared_outcomes(purged, config)
    validation_start = purged.loc[purged["split"].eq("validation"), "date"].min()
    assert (
        purged.loc[purged["split"].eq("train"), "outcome_end_date"]
        < validation_start
    ).all()
