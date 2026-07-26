from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .utils import assert_unique, require_columns


def validate_t1_target(
    target: pd.DataFrame,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    future_columns = [
        f"base_target_t_plus_{step}" for step in range(1, config.horizon + 1)
    ]
    date_columns = [
        f"target_date_t_plus_{step}" for step in range(1, config.horizon + 1)
    ]
    required = [
        "date",
        "ticker",
        *future_columns,
        *date_columns,
        "forward_mean_5d",
        "peer_mean_leave_one_out",
        "t1_target",
        "valid_ticker_count",
        "cross_section_sum",
        "full_demeaned_target",
    ]
    require_columns(target, required, "T1 target")
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})
        if not passed:
            raise AssertionError(f"{name}: {detail}")

    assert_unique(target, ["date", "ticker"], "T1 target")
    record("unique_date_ticker", True, f"{len(target)} unique rows")
    record(
        "sorted_date_ticker",
        target.index.equals(
            target.sort_values(["date", "ticker"]).reset_index(drop=True).index
        )
        and target[["date", "ticker"]]
        .equals(target.sort_values(["date", "ticker"])[["date", "ticker"]]),
        "Rows must be sorted by date then ticker",
    )
    finite_columns = [*future_columns, "forward_mean_5d", "peer_mean_leave_one_out", "t1_target"]
    finite = np.isfinite(target[finite_columns].to_numpy(dtype=float)).all()
    record("finite_final_target", finite, f"columns={finite_columns}")
    correct_mean = np.allclose(
        target["forward_mean_5d"].to_numpy(),
        target[future_columns].mean(axis=1).to_numpy(),
        rtol=1e-10,
        atol=1e-12,
    )
    record("forward_mean_uses_exactly_t_plus_1_to_5", correct_mean, "Mean mismatch")

    dates_strict = np.ones(len(target), dtype=bool)
    previous = target["date"]
    for step, column in enumerate(date_columns, start=1):
        dates_strict &= target[column].gt(previous).to_numpy()
        previous = target[column]
    record(
        "future_dates_strictly_after_anchor",
        bool(dates_strict.all()),
        "Every target date must be later than its anchor and ordered",
    )

    recomputed_peer = (
        target["cross_section_sum"] - target["forward_mean_5d"]
    ) / (target["valid_ticker_count"] - 1)
    record(
        "leave_one_out_excludes_current_ticker",
        np.allclose(
            recomputed_peer,
            target["peer_mean_leave_one_out"],
            rtol=1e-10,
            atol=1e-12,
        ),
        "Leave-one-out peer mean mismatch",
    )
    algebra = (
        target["valid_ticker_count"]
        / (target["valid_ticker_count"] - 1)
        * (
            target["forward_mean_5d"]
            - target["cross_section_sum"] / target["valid_ticker_count"]
        )
    )
    record(
        "leave_one_out_algebra",
        np.allclose(algebra, target["t1_target"], rtol=1e-10, atol=1e-12),
        "LOO target is not n/(n-1) times the full-demeaned target",
    )
    daily_full_sum = target.groupby("date")["full_demeaned_target"].sum()
    record(
        "full_demeaned_daily_sum_near_zero",
        bool(np.allclose(daily_full_sum, 0.0, atol=1e-10)),
        f"max_abs_sum={daily_full_sum.abs().max():.3e}",
    )
    record(
        "minimum_cross_section_size",
        bool((target["valid_ticker_count"] >= config.min_cross_section_size).all()),
        f"minimum={target['valid_ticker_count'].min()}",
    )

    summary = {
        "passed": True,
        "check_count": len(checks),
        "maximum_absolute_full_demeaned_daily_sum": float(daily_full_sum.abs().max()),
        "note": (
            "For a fixed equal-weight cross-section the LOO targets also sum to "
            "approximately zero algebraically; floating-point and later filtering "
            "can prevent exact equality."
        ),
    }
    return pd.DataFrame(checks), summary
