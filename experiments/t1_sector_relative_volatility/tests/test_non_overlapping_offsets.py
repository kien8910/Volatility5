from __future__ import annotations

import pandas as pd
import numpy as np

from experiments.t1_sector_relative_volatility.block_bootstrap import (
    moving_block_index_matrix,
)
from experiments.t1_sector_relative_volatility.config import TICKERS
from experiments.t1_sector_relative_volatility.split_purged import (
    assign_non_overlapping_offsets,
)


def test_five_offsets_partition_dates_without_overlap() -> None:
    dates = pd.bdate_range("2022-01-03", periods=30)
    frame = pd.DataFrame(
        [
            {"date": date, "ticker": ticker, "split": "validation"}
            for date in dates
            for ticker in TICKERS
        ]
    )
    assigned = assign_non_overlapping_offsets(frame, stride=5)
    assert set(assigned["offset"].unique()) == {0, 1, 2, 3, 4}
    date_positions = {date: index for index, date in enumerate(dates)}
    for offset, group in assigned.groupby("offset"):
        selected = sorted(group["date"].unique())
        positions = [date_positions[pd.Timestamp(date)] for date in selected]
        assert all(right - left == 5 for left, right in zip(positions, positions[1:]))
        assert all(position % 5 == offset for position in positions)


def test_vectorized_moving_blocks_are_deterministic_and_contiguous() -> None:
    first = moving_block_index_matrix(
        n_dates=20,
        block_length=5,
        repetitions=25,
        rng=np.random.default_rng(42),
    )
    second = moving_block_index_matrix(
        n_dates=20,
        block_length=5,
        repetitions=25,
        rng=np.random.default_rng(42),
    )
    assert first.shape == (25, 20)
    assert np.array_equal(first, second)
    assert ((np.diff(first.reshape(25, 4, 5), axis=2) % 20) == 1).all()
