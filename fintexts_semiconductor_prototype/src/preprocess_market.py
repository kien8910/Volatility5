"""Build leakage-safe market features and next-day volatility targets.

All features in this module are observable by the end of input date ``t``.
Splits are assigned using ``target_date`` (the next trading observation for a
ticker), so a train sample can never consume a validation target at a split
boundary.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.utils import (
    atomic_write_csv,
    get_logger,
    load_config,
    project_path,
    safe_read_table,
    set_global_seed,
    validate_required_columns,
    write_table,
)


DEFAULT_TICKERS = [
    "ADI",
    "AMAT",
    "AMD",
    "AVGO",
    "INTC",
    "KLAC",
    "LRCX",
    "MU",
    "NVDA",
    "QCOM",
    "TXN",
]
OHLC = ["open", "high", "low", "close"]
CORE_COLUMN_KEYS = ["date", "ticker", "industry", *OHLC]


def _first_config_value(
    config: Mapping[str, Any],
    paths: Sequence[Sequence[str]],
    default: Any = None,
) -> Any:
    for keys in paths:
        current: Any = config
        found = True
        for key in keys:
            if not isinstance(current, Mapping) or key not in current:
                found = False
                break
            current = current[key]
        if found and current is not None:
            return current
    return default


def _resolve_project_path(config: Mapping[str, Any], value: str | Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return project_path(config, *candidate.parts)


def _load_schema_mapping(path: Path) -> dict[str, str | None]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Schema mapping not found at {path}. Run inspect_schema first."
        )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"Schema mapping must be a YAML mapping: {path}")

    columns: Any = payload.get("columns", payload.get("schema_mapping", payload))
    if not isinstance(columns, Mapping):
        raise TypeError("schema_mapping.yaml must contain a 'columns' mapping.")

    mapping: dict[str, str | None] = {}
    for key in ["date", "ticker", *OHLC]:
        value = columns.get(key)
        if not isinstance(value, str) or not value.strip():
            raise KeyError(
                f"schema_mapping.yaml has no scalar mapping for required key {key!r}."
            )
        mapping[key] = value
    industry_value = columns.get("industry")
    if industry_value is not None and (
        not isinstance(industry_value, str) or not industry_value.strip()
    ):
        raise TypeError(
            "columns.industry must be a source-column name or null when the "
            "dataset does not expose industry metadata."
        )
    mapping["industry"] = industry_value
    mapped_values = [value for value in mapping.values() if value is not None]
    if len(set(mapped_values)) != len(mapped_values):
        raise ValueError(
            "Date, ticker, industry, and OHLC fields must map to distinct columns: "
            f"{mapping}"
        )
    return mapping


def _normalise_industry(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.normalize("NFKC")
        .str.strip()
        .str.casefold()
        .str.replace(r"[\s_\-/]+", "", regex=True)
    )


def _parse_dates(values: pd.Series, column_name: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    invalid = int(parsed.isna().sum())
    if invalid:
        examples = values.loc[parsed.isna()].astype("string").dropna().head(5).tolist()
        raise ValueError(
            f"Column {column_name!r} has {invalid:,} unparseable dates; "
            f"examples={examples}"
        )
    return parsed.dt.tz_convert(None).dt.normalize()


def _market_rows(
    raw: pd.DataFrame,
    mapping: Mapping[str, str | None],
    tickers: Sequence[str],
    industry_label: str,
    allow_conflicting_duplicates: bool,
    logger: Any,
) -> tuple[pd.DataFrame, dict[str, int]]:
    source_columns = [
        mapping[key] for key in CORE_COLUMN_KEYS if mapping.get(key) is not None
    ]
    validate_required_columns(raw, source_columns, context="raw FinTexTS market data")
    market = raw.loc[:, source_columns].rename(
        columns={
            source: canonical
            for canonical, source in mapping.items()
            if source is not None
        }
    )
    market["ticker"] = market["ticker"].astype("string").str.strip().str.upper()

    requested = list(dict.fromkeys(str(ticker).strip().upper() for ticker in tickers))
    market = market.loc[market["ticker"].isin(requested)].copy()
    observed_tickers = set(market["ticker"].dropna().astype(str))
    missing_tickers = sorted(set(requested).difference(observed_tickers))
    if missing_tickers:
        raise ValueError(
            "The raw dataset has no rows for requested ticker(s): "
            f"{missing_tickers}"
        )
    market["date"] = _parse_dates(market["date"], mapping["date"])
    if mapping.get("industry") is None:
        market["industry"] = industry_label
        market["industry_source"] = "configured_ticker_universe"
        logger.warning(
            "FinTexTS exposes no industry column; assigning industry=%r from the "
            "explicit 11-ticker experiment universe.",
            industry_label,
        )
    else:
        market["industry"] = market["industry"].astype("string").str.strip()
        expected_industry = _normalise_industry(pd.Series([industry_label])).iloc[0]
        industry_normalised = _normalise_industry(market["industry"])
        belongs = industry_normalised.eq(expected_industry)
        # FinTexTS revisions have used small spelling variants such as
        # "Semiconductors"; accept only an unambiguous semiconductor stem.
        belongs |= industry_normalised.str.contains("semiconductor", na=False)
        wrong_industry = market.loc[
            ~belongs, ["ticker", "industry"]
        ].drop_duplicates()
        if not wrong_industry.empty:
            raise ValueError(
                "Requested ticker rows outside the semiconductor industry were "
                f"found: {wrong_industry.to_dict(orient='records')}"
            )
        market = market.loc[belongs].copy()
        market["industry_source"] = f"dataset_column:{mapping['industry']}"

    missing_ohlc = int(market[OHLC].isna().any(axis=1).sum())
    for column in OHLC:
        market[column] = pd.to_numeric(market[column], errors="coerce")
    non_numeric_ohlc = int(market[OHLC].isna().any(axis=1).sum())
    finite = np.isfinite(market[OHLC].to_numpy(dtype=float)).all(axis=1)
    positive = market[OHLC].gt(0).all(axis=1)
    consistent_range = (
        market["high"].ge(market[["open", "close", "low"]].max(axis=1))
        & market["low"].le(market[["open", "close", "high"]].min(axis=1))
    )
    valid = finite & positive.to_numpy() & consistent_range.to_numpy()
    invalid_ohlc = int((~valid).sum())
    if invalid_ohlc:
        logger.warning(
            "Dropping %,d rows with missing, non-finite, non-positive, or "
            "range-inconsistent OHLC values.",
            invalid_ohlc,
        )
    market = market.loc[valid].copy()

    duplicate_mask = market.duplicated(["ticker", "date"], keep=False)
    duplicate_rows = int(duplicate_mask.sum())
    duplicate_keys = int(
        market.loc[duplicate_mask, ["ticker", "date"]].drop_duplicates().shape[0]
    )
    conflicting_keys = 0
    if duplicate_rows:
        variation = (
            market.loc[duplicate_mask]
            .groupby(["ticker", "date"], observed=True)[OHLC]
            .nunique(dropna=False)
        )
        conflicting = variation.gt(1).any(axis=1)
        conflicting_keys = int(conflicting.sum())
        if conflicting_keys and not allow_conflicting_duplicates:
            examples = [
                {"ticker": ticker, "date": str(date)}
                for ticker, date in conflicting.loc[conflicting].index[:10]
            ]
            raise ValueError(
                f"Found {conflicting_keys:,} ticker-date keys with conflicting OHLC "
                f"values. Examples: {examples}. Set "
                "market.allow_conflicting_ohlc_duplicates=true only after auditing."
            )
        if conflicting_keys:
            logger.warning(
                "Averaging OHLC for %,d conflicting duplicate ticker-date keys.",
                conflicting_keys,
            )
            market = (
                market.groupby(["ticker", "date"], as_index=False, observed=True)
                .agg(
                    {
                        "industry": "first",
                        "industry_source": "first",
                        "open": "mean",
                        "high": "max",
                        "low": "min",
                        "close": "mean",
                    }
                )
            )
        else:
            market = market.drop_duplicates(["ticker", "date"], keep="first")

    market = market.sort_values(["ticker", "date"], kind="stable").reset_index(
        drop=True
    )
    counts = {
        "input_rows_for_requested_tickers": int(len(valid)),
        "missing_ohlc_rows_before_numeric_conversion": missing_ohlc,
        "missing_or_non_numeric_ohlc_rows": non_numeric_ohlc,
        "invalid_ohlc_rows_dropped": invalid_ohlc,
        "duplicate_rows": duplicate_rows,
        "duplicate_ticker_date_keys": duplicate_keys,
        "conflicting_duplicate_keys": conflicting_keys,
        "valid_unique_market_rows": int(len(market)),
        "industry_from_config": int(mapping.get("industry") is None),
    }
    return market, counts


def _add_volatility(
    market: pd.DataFrame,
    epsilon: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if epsilon <= 0:
        raise ValueError(f"market.variance_epsilon must be positive, got {epsilon}.")

    log_hl = np.log(market["high"] / market["low"])
    log_co = np.log(market["close"] / market["open"])
    gk_raw = 0.5 * np.square(log_hl) - (2.0 * np.log(2.0) - 1.0) * np.square(
        log_co
    )
    parkinson_raw = np.square(log_hl) / (4.0 * np.log(2.0))
    finite_gk = np.isfinite(gk_raw)
    finite_parkinson = np.isfinite(parkinson_raw)
    invalid = ~(finite_gk & finite_parkinson)
    invalid_count = int(invalid.sum())
    if invalid_count:
        market = market.loc[~invalid].copy()
        gk_raw = gk_raw.loc[~invalid]
        parkinson_raw = parkinson_raw.loc[~invalid]

    nonpositive_gk = int(gk_raw.le(0).sum())
    nonpositive_parkinson = int(parkinson_raw.le(0).sum())
    market["gk_variance_raw"] = gk_raw.to_numpy(dtype=float)
    market["gk_variance"] = np.clip(
        market["gk_variance_raw"].to_numpy(dtype=float), epsilon, None
    )
    market["log_variance"] = np.log(market["gk_variance"] + epsilon)
    market["parkinson_variance"] = np.clip(
        parkinson_raw.to_numpy(dtype=float), epsilon, None
    )
    market["log_parkinson_variance"] = np.log(
        market["parkinson_variance"] + epsilon
    )
    if not np.isfinite(
        market[
            [
                "gk_variance",
                "log_variance",
                "parkinson_variance",
                "log_parkinson_variance",
            ]
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError("Non-finite volatility values remain after clipping.")
    if not market["gk_variance"].gt(0).all():
        raise AssertionError("Garman–Klass variance must be strictly positive.")

    return market, {
        "nonfinite_volatility_rows_dropped": invalid_count,
        "nonpositive_gk_values_clipped": nonpositive_gk,
        "nonpositive_parkinson_values_clipped": nonpositive_parkinson,
    }


def _rolling_std(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).std(ddof=0)


def _add_time_series_features(
    market: pd.DataFrame,
    lag_days: int,
    rolling_windows: Sequence[int],
    return_volatility_windows: Sequence[int],
) -> tuple[pd.DataFrame, list[str]]:
    if lag_days < 22:
        raise ValueError(
            "market.max_lag must be at least 22 to construct HAR monthly features."
        )
    if not rolling_windows or min(rolling_windows) < 2:
        raise ValueError("market.rolling_windows must contain integers >= 2.")
    if not return_volatility_windows or min(return_volatility_windows) < 2:
        raise ValueError(
            "market.return_volatility_windows must contain integers >= 2."
        )

    market = market.sort_values(["ticker", "date"], kind="stable").copy()
    groups = market.groupby("ticker", sort=False, observed=True)
    feature_columns: list[str] = []

    # At input date t, y_t is already observable from that day's OHLC.
    market["har_daily"] = market["log_variance"]
    feature_columns.append("har_daily")
    for lag in range(1, lag_days + 1):
        column = f"log_variance_lag_{lag}"
        market[column] = groups["log_variance"].shift(lag)
        feature_columns.append(column)

    market["har_weekly"] = groups["log_variance"].transform(
        lambda values: values.rolling(5, min_periods=5).mean()
    )
    market["har_monthly"] = groups["log_variance"].transform(
        lambda values: values.rolling(22, min_periods=22).mean()
    )
    feature_columns.extend(["har_weekly", "har_monthly"])

    market["historical_mean_log_variance"] = groups["log_variance"].transform(
        lambda values: values.expanding(min_periods=1).mean()
    )
    feature_columns.append("historical_mean_log_variance")
    for window in sorted(set(int(value) for value in rolling_windows)):
        prefix = f"log_variance_roll_{window}"
        market[f"{prefix}_mean"] = groups["log_variance"].transform(
            lambda values, w=window: values.rolling(w, min_periods=w).mean()
        )
        market[f"{prefix}_std"] = groups["log_variance"].transform(
            lambda values, w=window: _rolling_std(values, w)
        )
        market[f"{prefix}_min"] = groups["log_variance"].transform(
            lambda values, w=window: values.rolling(w, min_periods=w).min()
        )
        market[f"{prefix}_max"] = groups["log_variance"].transform(
            lambda values, w=window: values.rolling(w, min_periods=w).max()
        )
        feature_columns.extend(
            [
                f"{prefix}_mean",
                f"{prefix}_std",
                f"{prefix}_min",
                f"{prefix}_max",
            ]
        )

    previous_close = groups["close"].shift(1)
    market["log_return"] = np.log(market["close"] / previous_close)
    market["absolute_log_return"] = market["log_return"].abs()
    feature_columns.extend(["log_return", "absolute_log_return"])
    return_groups = market.groupby("ticker", sort=False, observed=True)
    for window in sorted(set(int(value) for value in return_volatility_windows)):
        column = f"log_return_volatility_{window}"
        market[column] = return_groups["log_return"].transform(
            lambda values, w=window: _rolling_std(values, w)
        )
        feature_columns.append(column)

    target_groups = market.groupby("ticker", sort=False, observed=True)
    market["target_date"] = target_groups["date"].shift(-1)
    market["target_gk_variance"] = target_groups["gk_variance"].shift(-1)
    market["target_log_variance"] = target_groups["log_variance"].shift(-1)
    market["target_log_parkinson_variance"] = target_groups[
        "log_parkinson_variance"
    ].shift(-1)
    common_dates = pd.DatetimeIndex(market["date"].dropna().unique()).sort_values()
    next_common_date = {
        current: following
        for current, following in zip(common_dates[:-1], common_dates[1:])
    }
    market["expected_next_trading_date"] = market["date"].map(next_common_date)
    market["next_trading_day_aligned"] = market["target_date"].eq(
        market["expected_next_trading_date"]
    )
    market["input_date"] = market["date"]
    market["feature_date"] = market["date"]
    market["y_t"] = market["log_variance"]
    market["y_t_plus_1"] = market["target_log_variance"]

    numeric_required = [
        *feature_columns,
        "target_gk_variance",
        "target_log_variance",
    ]
    finite = np.isfinite(market[numeric_required].to_numpy(dtype=float)).all(axis=1)
    market["model_ready"] = (
        finite
        & market["target_date"].notna()
        & market["next_trading_day_aligned"]
    )
    return market, feature_columns


def _split_date_blocks(
    dates: pd.Series,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> dict[str, pd.DatetimeIndex]:
    ratios = np.array([train_ratio, validation_ratio, test_ratio], dtype=float)
    if not np.isfinite(ratios).all() or (ratios <= 0).any():
        raise ValueError(f"Split ratios must be finite and positive, got {ratios}.")
    if not np.isclose(ratios.sum(), 1.0, atol=1e-8):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratios.sum():.8f}.")

    unique_dates = pd.DatetimeIndex(pd.Series(dates).dropna().unique()).sort_values()
    if len(unique_dates) < 3:
        raise ValueError("At least three target trading dates are required.")
    train_end_index = int(np.floor(len(unique_dates) * train_ratio))
    validation_end_index = int(
        np.floor(len(unique_dates) * (train_ratio + validation_ratio))
    )
    train_end_index = min(max(train_end_index, 1), len(unique_dates) - 2)
    validation_end_index = min(
        max(validation_end_index, train_end_index + 1), len(unique_dates) - 1
    )
    blocks = {
        "train": unique_dates[:train_end_index],
        "validation": unique_dates[train_end_index:validation_end_index],
        "test": unique_dates[validation_end_index:],
    }
    if any(len(block) == 0 for block in blocks.values()):
        raise AssertionError(f"Chronological split produced an empty date block: {blocks}")

    train_end = blocks["train"].max()
    validation_start = blocks["validation"].min()
    validation_end = blocks["validation"].max()
    test_start = blocks["test"].min()
    assert train_end < validation_start
    assert validation_end < test_start
    return blocks


def _assign_split(
    market: pd.DataFrame,
    date_blocks: Mapping[str, pd.DatetimeIndex],
) -> pd.DataFrame:
    date_to_split = {
        date: split
        for split, dates in date_blocks.items()
        for date in dates
    }
    market = market.copy()
    market["split"] = market["target_date"].map(date_to_split)
    if market["split"].isna().any():
        missing = market.loc[market["split"].isna(), "target_date"].drop_duplicates()
        raise AssertionError(
            "Some model-ready target dates were not assigned to a split: "
            f"{missing.head(10).tolist()}"
        )
    market["split"] = pd.Categorical(
        market["split"], categories=["train", "validation", "test"], ordered=True
    )
    return market


def _split_summary(
    market: pd.DataFrame,
    tickers: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    requested = set(tickers)
    for split in ["train", "validation", "test"]:
        subset = market.loc[market["split"].eq(split)]
        if subset.empty:
            raise AssertionError(f"Split {split!r} contains no samples.")
        observed = set(subset["ticker"].astype(str))
        missing = sorted(requested.difference(observed))
        if missing:
            raise ValueError(f"Split {split!r} has no samples for ticker(s): {missing}")
        rows.append(
            {
                "split": split,
                "input_start": subset["input_date"].min(),
                "input_end": subset["input_date"].max(),
                "target_start": subset["target_date"].min(),
                "target_end": subset["target_date"].max(),
                "n_target_dates": subset["target_date"].nunique(),
                "n_samples": len(subset),
                "n_tickers": subset["ticker"].nunique(),
            }
        )
    summary = pd.DataFrame(rows)
    train = summary.loc[summary["split"].eq("train")].iloc[0]
    validation = summary.loc[summary["split"].eq("validation")].iloc[0]
    test = summary.loc[summary["split"].eq("test")].iloc[0]
    assert train["target_end"] < validation["target_start"]
    assert validation["target_end"] < test["target_start"]
    return summary


def _expanding_folds(
    market: pd.DataFrame,
    n_folds: int,
    minimum_train_dates: int,
) -> pd.DataFrame:
    if n_folds < 3:
        raise ValueError("At least three expanding chronological folds are required.")
    development = market.loc[market["split"].isin(["train", "validation"])]
    dates = pd.DatetimeIndex(development["target_date"].unique()).sort_values()
    if len(dates) < minimum_train_dates + n_folds:
        raise ValueError(
            f"Only {len(dates)} development dates are available; expanding folds "
            f"need at least {minimum_train_dates + n_folds}."
        )

    validation_size = max(1, len(dates) // (n_folds + 2))
    initial_train_size = len(dates) - n_folds * validation_size
    if initial_train_size < minimum_train_dates:
        validation_size = max(
            1, (len(dates) - minimum_train_dates) // n_folds
        )
        initial_train_size = len(dates) - n_folds * validation_size
    if initial_train_size < minimum_train_dates or validation_size < 1:
        raise ValueError(
            "Unable to construct expanding folds with the requested minimum "
            f"training history ({minimum_train_dates} dates)."
        )

    rows: list[dict[str, Any]] = []
    for fold in range(n_folds):
        validation_start_index = initial_train_size + fold * validation_size
        validation_end_index = (
            len(dates)
            if fold == n_folds - 1
            else validation_start_index + validation_size
        )
        train_dates = dates[:validation_start_index]
        validation_dates = dates[validation_start_index:validation_end_index]
        if len(validation_dates) == 0:
            raise AssertionError(f"Expanding fold {fold + 1} has no validation dates.")
        train_end = train_dates.max()
        validation_start = validation_dates.min()
        assert train_end < validation_start
        train_samples = market["target_date"].isin(train_dates)
        validation_samples = market["target_date"].isin(validation_dates)
        rows.append(
            {
                "fold": fold + 1,
                "train_start": train_dates.min(),
                "train_end": train_end,
                "validation_start": validation_start,
                "validation_end": validation_dates.max(),
                "n_train_dates": len(train_dates),
                "n_validation_dates": len(validation_dates),
                "n_train_samples": int(train_samples.sum()),
                "n_validation_samples": int(validation_samples.sum()),
            }
        )
    return pd.DataFrame(rows)


def _distribution_table(market: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "gk_variance_raw",
        "gk_variance",
        "log_variance",
        "parkinson_variance",
        "log_parkinson_variance",
    ]
    quantiles = [0.0, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 1.0]
    rows: list[dict[str, Any]] = []
    for ticker_key, subset in [
        ("__ALL__", market),
        *[(ticker, frame) for ticker, frame in market.groupby("ticker", observed=True)],
    ]:
        for metric in metrics:
            values = subset[metric].dropna()
            row: dict[str, Any] = {
                "ticker": ticker_key,
                "metric": metric,
                "count": len(values),
                "mean": values.mean(),
                "std": values.std(ddof=1),
            }
            for quantile, value in values.quantile(quantiles).items():
                row[f"q{quantile:g}"] = value
            rows.append(row)
    return pd.DataFrame(rows)


def _quality_summary(
    counts: Mapping[str, int],
    volatility_counts: Mapping[str, int],
    rows_before_model_filter: int,
    model_rows: int,
    nonconsecutive_target_rows: int,
    epsilon: float,
) -> pd.DataFrame:
    payload: dict[str, Any] = {
        **counts,
        **volatility_counts,
        "rows_before_model_ready_filter": rows_before_model_filter,
        "nonconsecutive_next_trading_day_rows_dropped": nonconsecutive_target_rows,
        "warmup_or_missing_target_rows_dropped": rows_before_model_filter
        - model_rows,
        "model_ready_rows": model_rows,
        "variance_epsilon": epsilon,
    }
    return pd.DataFrame(
        [{"metric": key, "value": value} for key, value in payload.items()]
    )


def _plot_volatility(market: pd.DataFrame, path: Path) -> None:
    tickers = sorted(market["ticker"].astype(str).unique())
    colors = plt.get_cmap("tab20")
    figure, axis = plt.subplots(figsize=(16, 9))
    for index, ticker in enumerate(tickers):
        subset = market.loc[market["ticker"].eq(ticker)]
        axis.plot(
            subset["date"],
            subset["log_variance"],
            linewidth=0.75,
            alpha=0.8,
            label=ticker,
            color=colors(index % 20),
        )
    axis.set_title("Garman–Klass log variance by semiconductor ticker")
    axis.set_xlabel("Trading date")
    axis.set_ylabel("log(GK variance + epsilon)")
    axis.grid(alpha=0.2)
    axis.legend(ncol=4, frameon=False)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run(config: dict) -> dict[str, Path]:
    """Create volatility targets, causal price features, and time splits."""

    logger = get_logger(__name__, config)
    seed = int(
        _first_config_value(
            config, (("seed",), ("project", "seed"), ("runtime", "seed")), 42
        )
    )
    deterministic = bool(
        _first_config_value(
            config,
            (
                ("deterministic",),
                ("project", "deterministic"),
                ("runtime", "deterministic"),
            ),
            True,
        )
    )
    set_global_seed(seed, deterministic=deterministic)

    tickers = list(
        _first_config_value(
            config,
            (("tickers",), ("data", "tickers"), ("universe", "tickers")),
            DEFAULT_TICKERS,
        )
    )
    if set(map(str.upper, tickers)) != set(DEFAULT_TICKERS):
        logger.warning(
            "Configured ticker universe differs from the 11-ticker default: %s",
            tickers,
        )
    industry_label = str(
        _first_config_value(
            config,
            (
                ("industry",),
                ("data", "industry"),
                ("data", "industry_value"),
                ("universe", "industry"),
            ),
            "semiconductor",
        )
    )

    raw_filename = str(
        _first_config_value(
            config,
            (
                ("paths", "raw_filename"),
                ("dataset", "raw_filename"),
                ("data", "raw_filename"),
            ),
            "fintexts_raw.parquet",
        )
    )
    raw_path = project_path(config, "data", "raw", raw_filename)
    mapping_value = _first_config_value(
        config,
        (
            ("paths", "schema_mapping"),
            ("schema", "mapping_path"),
            ("schema", "mapping_file"),
        ),
        "config/schema_mapping.yaml",
    )
    mapping_path = _resolve_project_path(config, str(mapping_value))
    processed_value = _first_config_value(
        config,
        (("paths", "market_features"), ("market", "output_file")),
        "data/processed/market_supervised.parquet",
    )
    processed_path = _resolve_project_path(config, str(processed_value))
    table_dir = project_path(config, "outputs", "tables")
    figure_dir = project_path(config, "outputs", "figures")
    split_summary_path = table_dir / "market_split_summary.csv"
    folds_path = table_dir / "chronological_folds.csv"
    distribution_path = table_dir / "market_volatility_distribution.csv"
    quality_path = table_dir / "market_preprocessing_summary.csv"
    figure_path = figure_dir / "volatility_timeseries.png"
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    if not raw_path.is_file():
        raise FileNotFoundError(
            f"Raw FinTexTS Parquet not found at {raw_path}. Run download first."
        )
    mapping = _load_schema_mapping(mapping_path)
    raw = safe_read_table(raw_path)
    allow_conflicts = bool(
        _first_config_value(
            config,
            (("market", "allow_conflicting_ohlc_duplicates"),),
            False,
        )
    )
    market, market_counts = _market_rows(
        raw=raw,
        mapping=mapping,
        tickers=tickers,
        industry_label=industry_label,
        allow_conflicting_duplicates=allow_conflicts,
        logger=logger,
    )

    epsilon = float(
        _first_config_value(
            config,
            (
                ("market", "variance_epsilon"),
                ("market", "epsilon"),
                ("targets", "epsilon"),
            ),
            1e-12,
        )
    )
    market, volatility_counts = _add_volatility(market, epsilon)
    lag_days = int(
        _first_config_value(
            config, (("market", "max_lag"), ("market", "lag_days")), 22
        )
    )
    rolling_windows = list(
        _first_config_value(
            config, (("market", "rolling_windows"),), [5, 10, 22]
        )
    )
    return_volatility_windows = list(
        _first_config_value(
            config,
            (("market", "return_volatility_windows"),),
            [5, 10, 22],
        )
    )
    market, feature_columns = _add_time_series_features(
        market,
        lag_days=lag_days,
        rolling_windows=rolling_windows,
        return_volatility_windows=return_volatility_windows,
    )
    rows_before_filter = len(market)
    model_market = market.loc[market["model_ready"]].copy()
    if model_market.empty:
        raise ValueError(
            "No model-ready rows remain after lag construction and target alignment."
        )
    if model_market[feature_columns].isna().any().any():
        missing = model_market[feature_columns].isna().sum()
        raise AssertionError(
            "Model-ready feature matrix contains missing values: "
            f"{missing.loc[missing.gt(0)].to_dict()}"
        )

    train_ratio = float(
        _first_config_value(
            config,
            (
                ("splits", "train_ratio"),
                ("split", "train_ratio"),
                ("split", "train"),
            ),
            0.60,
        )
    )
    validation_ratio = float(
        _first_config_value(
            config,
            (
                ("splits", "validation_ratio"),
                ("split", "validation_ratio"),
                ("split", "validation"),
            ),
            0.20,
        )
    )
    test_ratio = float(
        _first_config_value(
            config,
            (
                ("splits", "test_ratio"),
                ("split", "test_ratio"),
                ("split", "test"),
            ),
            0.20,
        )
    )
    blocks = _split_date_blocks(
        model_market["target_date"],
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
    )
    model_market = _assign_split(model_market, blocks)
    split_summary = _split_summary(model_market, tickers)

    n_folds = int(
        _first_config_value(
            config,
            (
                ("splits", "n_expanding_folds"),
                ("split", "n_expanding_folds"),
                ("validation", "n_folds"),
            ),
            3,
        )
    )
    minimum_train_dates = int(
        _first_config_value(
            config,
            (
                ("splits", "minimum_fold_train_dates"),
                ("split", "min_train_days"),
            ),
            max(44, lag_days * 2),
        )
    )
    folds = _expanding_folds(
        model_market,
        n_folds=n_folds,
        minimum_train_dates=minimum_train_dates,
    )

    distribution = _distribution_table(market)
    quality = _quality_summary(
        market_counts,
        volatility_counts,
        rows_before_model_filter=rows_before_filter,
        model_rows=len(model_market),
        nonconsecutive_target_rows=int(
            (
                market["target_date"].notna()
                & ~market["next_trading_day_aligned"]
            ).sum()
        ),
        epsilon=epsilon,
    )

    output_order = [
        "ticker",
        "date",
        "feature_date",
        "input_date",
        "target_date",
        "expected_next_trading_date",
        "next_trading_day_aligned",
        "split",
        "industry",
        "industry_source",
        *OHLC,
        "gk_variance_raw",
        "gk_variance",
        "log_variance",
        "parkinson_variance",
        "log_parkinson_variance",
        *feature_columns,
        "target_gk_variance",
        "target_log_variance",
        "target_log_parkinson_variance",
        "y_t",
        "y_t_plus_1",
        "model_ready",
    ]
    output_order = list(dict.fromkeys(output_order))
    write_table(model_market.loc[:, output_order], processed_path, index=False)
    atomic_write_csv(split_summary, split_summary_path, index=False)
    atomic_write_csv(folds, folds_path, index=False)
    atomic_write_csv(distribution, distribution_path, index=False)
    atomic_write_csv(quality, quality_path, index=False)
    _plot_volatility(market, figure_path)

    for row in split_summary.to_dict(orient="records"):
        logger.info(
            "%s: target dates %s to %s; %,d samples across %,d dates",
            str(row["split"]).upper(),
            row["target_start"],
            row["target_end"],
            row["n_samples"],
            row["n_target_dates"],
        )
    logger.info(
        "Saved %,d leakage-safe market samples with %d price features to %s",
        len(model_market),
        len(feature_columns),
        processed_path,
    )
    return {
        "market_supervised": processed_path,
        "market_features": processed_path,
        "market_split_summary": split_summary_path,
        "chronological_folds": folds_path,
        "market_volatility_distribution": distribution_path,
        "market_preprocessing_summary": quality_path,
        "volatility_figure": figure_path,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create GK volatility, causal market features, and time splits."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Path to the project YAML configuration.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    outputs = run(config)
    print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
