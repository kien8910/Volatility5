from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.t1_sector_relative_volatility.build_t1_target import build_t1_target
from experiments.t1_sector_relative_volatility.config import ExperimentConfig, TICKERS
from experiments.t1_sector_relative_volatility.validate_target import validate_t1_target


def synthetic_market(n_dates: int = 14) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    rows = []
    for ticker_index, ticker in enumerate(TICKERS):
        for date_index, date in enumerate(dates):
            split = "train" if date_index < 7 else "validation"
            rows.append(
                {
                    "ticker": ticker,
                    "feature_date": date,
                    "log_variance": ticker_index + date_index / 10.0,
                    "split": split,
                }
            )
    return pd.DataFrame(rows)


def test_target_uses_exact_next_five_common_dates(tmp_path) -> None:
    config = ExperimentConfig(
        workspace_root=tmp_path,
        output_directory=tmp_path / "out",
        min_cross_section_size=8,
    )
    target, audit = build_t1_target(synthetic_market(), config)
    row = target[
        target["ticker"].eq("ADI")
        & target["date"].eq(pd.Timestamp("2020-01-01"))
    ].iloc[0]
    expected = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    observed = row[
        [f"base_target_t_plus_{step}" for step in range(1, 6)]
    ].to_numpy(dtype=float)
    assert np.allclose(observed, expected)
    assert np.isclose(row["forward_mean_5d"], expected.mean())
    assert audit["dates_after_complete_forward_target"] == 9


def test_leave_one_out_peer_excludes_current_ticker(tmp_path) -> None:
    config = ExperimentConfig(
        workspace_root=tmp_path,
        output_directory=tmp_path / "out",
    )
    target, _ = build_t1_target(synthetic_market(), config)
    date = target["date"].min()
    group = target[target["date"].eq(date)]
    row = group[group["ticker"].eq("AMD")].iloc[0]
    expected_peer = group.loc[~group["ticker"].eq("AMD"), "forward_mean_5d"].mean()
    assert np.isclose(row["peer_mean_leave_one_out"], expected_peer)
    assert np.isclose(row["t1_target"], row["forward_mean_5d"] - expected_peer)
    checks, summary = validate_t1_target(target, config)
    assert summary["passed"]
    assert checks["passed"].all()
