from __future__ import annotations

import pandas as pd

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
