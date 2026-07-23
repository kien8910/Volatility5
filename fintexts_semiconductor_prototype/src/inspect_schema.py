"""Inspect and validate the raw FinTexTS schema.

This stage is deliberately read-only with respect to the raw dataset.  It
discovers the market/news columns, validates the semiconductor universe, writes
the resolved mapping for downstream stages, and emits compact audit tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import tempfile
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.utils import (
    atomic_write_csv,
    get_logger,
    load_config,
    project_path,
    safe_read_table,
)


LOGGER = logging.getLogger(__name__)

TICKERS: tuple[str, ...] = (
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
)
EXPECTED_INDUSTRY = "semiconductor"
NEWS_LEVELS: tuple[str, ...] = ("macro", "sector", "related", "target")
CORE_FIELDS: tuple[str, ...] = (
    "date",
    "ticker",
    "industry",
    "open",
    "high",
    "low",
    "close",
)
REQUIRED_CORE_FIELDS: tuple[str, ...] = (
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
)
SUPPORTED_TABLE_SUFFIXES = frozenset(
    {".parquet", ".pq", ".csv", ".json", ".jsonl", ".ndjson"}
)
DEFAULT_PLACEHOLDERS = frozenset(
    {
        "",
        "-",
        "--",
        "n/a",
        "na",
        "nan",
        "nil",
        "no news",
        "none",
        "not available",
        "null",
        "unknown",
    }
)

_CORE_ALIASES: dict[str, tuple[str, ...]] = {
    "date": (
        "date",
        "datetime",
        "timestamp",
        "trading_date",
        "trade_date",
        "market_date",
    ),
    "ticker": (
        "ticker",
        "symbol",
        "stock_ticker",
        "target_ticker",
        "tic",
    ),
    "industry": (
        "Industry-level",
        "industry_level",
        "industry-level",
        "industry",
        "industry_name",
        "gics_industry",
    ),
    "open": ("open", "open_price", "price_open", "adj_open", "adjusted_open"),
    "high": ("high", "high_price", "price_high", "adj_high", "adjusted_high"),
    "low": ("low", "low_price", "price_low", "adj_low", "adjusted_low"),
    "close": (
        "close",
        "close_price",
        "price_close",
        "adj_close",
        "adjusted_close",
    ),
}

_CORE_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "date": (
        re.compile(r"^(?:trading|trade|market)?date$"),
        re.compile(r"^(?:date)?timestamp$"),
    ),
    "ticker": (
        re.compile(r"^(?:stock|target)?ticker$"),
        re.compile(r"^(?:stock)?symbol$"),
    ),
    "industry": (
        re.compile(r"^(?:gics)?industry(?:level|name)?$"),
    ),
    "open": (re.compile(r"^(?:adjusted|adj|price)?open(?:price)?$"),),
    "high": (re.compile(r"^(?:adjusted|adj|price)?high(?:price)?$"),),
    "low": (re.compile(r"^(?:adjusted|adj|price)?low(?:price)?$"),),
    "close": (re.compile(r"^(?:adjusted|adj|price)?close(?:price)?$"),),
}

_NEWS_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "macro": (
        re.compile(
            r"^macro(?:economic)?(?:news|category|text|headline|article|summary)?\d*$"
        ),
    ),
    "sector": (
        re.compile(
            r"^(?:industry|sector)(?:news|category|text|headline|article|summary)?\d*$"
        ),
    ),
    "related": (
        re.compile(
            r"^related(?:company|companies|firm)?"
            r"(?:news|category|text|headline|article|summary)?\d*$"
        ),
    ),
    "target": (
        re.compile(
            r"^(?:target|targetcompany|companytarget)"
            r"(?:company|news|category|text|headline|article|summary)*\d*$"
        ),
    ),
}


def _deep_get(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    value: Any = config
    for key in dotted_key.split("."):
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def _first_config_value(config: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = _deep_get(config, key)
        if value is not None:
            return value
    return None


def _as_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    if isinstance(value, (int, np.integer)):
        return bool(value)
    raise TypeError(f"Expected a boolean-like value, received {value!r}.")


def _normalize_column_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _natural_column_key(value: str) -> tuple[str, int, str]:
    normalized = _normalize_column_name(value)
    match = re.search(r"(\d+)$", normalized)
    if match is None:
        return normalized, -1, value
    return normalized[: match.start()], int(match.group(1)), value


def _normalize_ticker(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def _industry_matches(value: Any, expected: str = EXPECTED_INDUSTRY) -> bool:
    if pd.isna(value):
        return False
    normalized = re.sub(r"[^a-z]+", " ", str(value).casefold()).strip()
    expected_normalized = re.sub(r"[^a-z]+", " ", expected.casefold()).strip()
    return expected_normalized in normalized


def _schema_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    value = config.get("schema", {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("config['schema'] must be a mapping when provided.")
    return value


def _column_overrides(config: Mapping[str, Any]) -> dict[str, Any]:
    candidates = (
        _deep_get(config, "schema.column_overrides"),
        _deep_get(config, "schema.overrides"),
        _deep_get(config, "schema.columns"),
        _deep_get(config, "dataset.column_mapping"),
        config.get("column_mapping"),
    )
    for candidate in candidates:
        if candidate is None:
            continue
        if not isinstance(candidate, Mapping):
            raise TypeError("Schema column overrides must be a mapping.")
        overrides = dict(candidate)
        news_overrides = dict(overrides.get("news", {}))
        for level in NEWS_LEVELS:
            legacy_key = f"{level}_news"
            legacy_value = overrides.get(legacy_key)
            if legacy_value:
                news_overrides[level] = legacy_value
        if news_overrides:
            overrides["news"] = news_overrides
        return overrides
    return {}


def _configured_aliases(
    config: Mapping[str, Any],
    field: str,
) -> tuple[str, ...]:
    aliases = _deep_get(config, f"schema.aliases.{field}")
    if aliases is None:
        return ()
    if isinstance(aliases, str):
        return (aliases,)
    if not isinstance(aliases, Sequence):
        raise TypeError(f"schema.aliases.{field} must be a string or list of strings.")
    values = tuple(str(value).strip() for value in aliases)
    if any(not value for value in values):
        raise ValueError(f"schema.aliases.{field} contains an empty alias.")
    return values


def _configured_news_tokens(
    config: Mapping[str, Any],
    level: str,
) -> tuple[str, ...]:
    value = _first_config_value(
        config,
        (
            f"schema.news_patterns.{level}",
            f"schema.news_patterns.{level}_news",
        ),
    )
    if value is None:
        return ()
    if isinstance(value, str):
        candidates = (value,)
    elif isinstance(value, Sequence):
        candidates = tuple(str(item) for item in value)
    else:
        raise TypeError(
            f"schema.news_patterns.{level}_news must be a string or list of strings."
        )
    normalized = tuple(
        _normalize_column_name(candidate.strip()) for candidate in candidates
    )
    if any(not token for token in normalized):
        raise ValueError(
            f"schema.news_patterns.{level}_news contains an empty pattern."
        )
    return normalized


def _select_core_column(
    field: str,
    columns: Sequence[str],
    override: Any,
    configured_aliases: Sequence[str] = (),
) -> str | None:
    if override is not None:
        if not isinstance(override, str) or not override.strip():
            raise TypeError(
                f"Column override for {field!r} must be a non-empty string or null."
            )
        if override not in columns:
            raise ValueError(
                f"Configured {field!r} column {override!r} does not exist. "
                f"Available columns: {list(columns)!r}"
            )
        return override

    normalized_columns: dict[str, list[str]] = {}
    for column in columns:
        normalized_columns.setdefault(_normalize_column_name(column), []).append(column)

    alias_matches: list[str] = []
    for alias in (*configured_aliases, *_CORE_ALIASES[field]):
        alias_matches.extend(normalized_columns.get(_normalize_column_name(alias), []))
    alias_matches = list(dict.fromkeys(alias_matches))
    if len(alias_matches) == 1:
        return alias_matches[0]
    if len(alias_matches) > 1:
        raise ValueError(
            f"Ambiguous schema for {field!r}: alias matches {alias_matches!r}. "
            "Set schema.column_overrides explicitly."
        )

    pattern_matches = [
        column
        for column in columns
        if any(
            pattern.fullmatch(_normalize_column_name(column))
            for pattern in _CORE_PATTERNS[field]
        )
    ]
    if len(pattern_matches) == 1:
        return pattern_matches[0]
    if len(pattern_matches) > 1:
        raise ValueError(
            f"Ambiguous schema for {field!r}: regex matches {pattern_matches!r}. "
            "Set schema.column_overrides explicitly."
        )
    return None


def _coerce_override_columns(
    level: str,
    override: Any,
    columns: Sequence[str],
) -> list[str]:
    if isinstance(override, str):
        values = [override]
    elif isinstance(override, Sequence) and not isinstance(override, (str, bytes)):
        values = list(override)
    else:
        raise TypeError(
            f"News-column override for {level!r} must be a string or a list of strings."
        )
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise TypeError(
            f"News-column override for {level!r} contains a non-string/empty value."
        )
    missing = [value for value in values if value not in columns]
    if missing:
        raise ValueError(
            f"Configured {level!r} news columns are absent: {missing!r}. "
            f"Available columns: {list(columns)!r}"
        )
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate columns in {level!r} news override: {values!r}")
    return sorted(values, key=_natural_column_key)


def _discover_schema(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    columns = [str(column) for column in frame.columns]
    if len(columns) != len(set(columns)):
        duplicates = sorted(
            {column for column in columns if columns.count(column) > 1}
        )
        raise ValueError(f"Raw data contains duplicate column names: {duplicates!r}")

    overrides = _column_overrides(config)
    resolved: dict[str, Any] = {}
    for field in CORE_FIELDS:
        override = overrides.get(field)
        resolved[field] = _select_core_column(
            field,
            columns,
            override,
            configured_aliases=_configured_aliases(config, field),
        )

    missing_core = [field for field in REQUIRED_CORE_FIELDS if resolved[field] is None]
    if missing_core:
        raise ValueError(
            "Could not resolve required FinTexTS fields "
            f"{missing_core!r}. Available columns: {columns!r}. "
            "Provide schema.column_overrides in config/config.yaml."
        )

    news_overrides: Any = overrides.get("news")
    if news_overrides is None:
        news_overrides = _first_config_value(
            config,
            ("schema.news_columns", "dataset.news_columns", "news_columns"),
        )
    if news_overrides is not None and not isinstance(news_overrides, Mapping):
        raise TypeError("News-column overrides must be a mapping keyed by news level.")
    news_overrides = dict(news_overrides or {})

    occupied = {value for value in resolved.values() if isinstance(value, str)}
    news: dict[str, list[str]] = {}
    for level in NEWS_LEVELS:
        if level in news_overrides:
            news[level] = _coerce_override_columns(
                level, news_overrides[level], columns
            )
            continue
        configured_tokens = _configured_news_tokens(config, level)
        news[level] = sorted(
            [
                column
                for column in columns
                if column not in occupied
                and (
                    any(
                        pattern.fullmatch(_normalize_column_name(column))
                        for pattern in _NEWS_PATTERNS[level]
                    )
                    or any(
                        token in _normalize_column_name(column)
                        for token in configured_tokens
                    )
                )
            ],
            key=_natural_column_key,
        )

    assigned_news = [
        (level, column)
        for level, level_columns in news.items()
        for column in level_columns
    ]
    duplicate_news_columns = sorted(
        {
            column
            for _, column in assigned_news
            if sum(candidate == column for _, candidate in assigned_news) > 1
        }
    )
    if duplicate_news_columns:
        raise ValueError(
            "A news column was assigned to more than one level: "
            f"{duplicate_news_columns!r}. Use explicit schema.news_columns overrides."
        )

    allow_missing_levels = _as_bool(
        _deep_get(config, "schema.allow_missing_news_levels"), default=False
    )
    missing_levels = [level for level, level_columns in news.items() if not level_columns]
    if missing_levels and not allow_missing_levels:
        raise ValueError(
            "Could not identify FinTexTS news columns for levels "
            f"{missing_levels!r}. Set schema.news_columns explicitly, or set "
            "schema.allow_missing_news_levels=true only for a deliberate reduced run."
        )

    resolved["news"] = news
    return resolved


def _path_from_config(config: Mapping[str, Any], value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate
    return project_path(config, *candidate.parts)


def _configured_raw_path(config: Mapping[str, Any]) -> Path | None:
    value = _first_config_value(
        config,
        (
            "dataset.raw_path",
            "dataset.local_path",
            "dataset.output_path",
            "paths.raw_dataset",
            "paths.raw_data_file",
            "paths.raw_file",
            "data.raw_path",
            "download.output_path",
        ),
    )
    if value is None:
        filename = _first_config_value(
            config,
            ("data.raw_filename", "dataset.raw_filename", "paths.raw_filename"),
        )
        if filename is None:
            return None
        if not isinstance(filename, (str, Path)):
            raise TypeError("Configured raw filename must be a string or pathlib.Path.")
        filename_path = Path(filename)
        if filename_path.is_absolute() or len(filename_path.parts) > 1:
            return _path_from_config(config, filename_path)
        return project_path(config, "data", "raw", filename_path)
    if not isinstance(value, (str, Path)):
        raise TypeError("Configured raw dataset path must be a string or pathlib.Path.")
    return _path_from_config(config, value)


def _table_suffix(path: Path) -> str:
    return path.suffix.casefold()


def _discover_raw_path(config: Mapping[str, Any]) -> Path:
    configured = _configured_raw_path(config)
    if configured is not None:
        if not configured.exists():
            raise FileNotFoundError(
                f"Configured raw FinTexTS path does not exist: {configured}"
            )
        return configured

    raw_dir = project_path(config, "data", "raw")
    preferred_names = (
        "fintexts.parquet",
        "fintexts_raw.parquet",
        "fintexts_train.parquet",
        "FinTexTS.parquet",
        "train.parquet",
    )
    for name in preferred_names:
        candidate = raw_dir / name
        if candidate.is_file():
            return candidate

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw data directory does not exist: {raw_dir}. "
            "Run `python run_pipeline.py --stage download` first."
        )

    table_candidates = sorted(
        path
        for path in raw_dir.rglob("*")
        if path.is_file() and _table_suffix(path) in SUPPORTED_TABLE_SUFFIXES
    )
    dataset_directories = sorted(
        {
            path.parent
            for marker in ("dataset_dict.json", "dataset_info.json", "state.json")
            for path in raw_dir.rglob(marker)
        }
    )
    if len(table_candidates) == 1 and not dataset_directories:
        return table_candidates[0]
    if not table_candidates and len(dataset_directories) == 1:
        return dataset_directories[0]

    discovered = [str(path) for path in table_candidates + dataset_directories]
    if not discovered:
        raise FileNotFoundError(
            f"No supported raw FinTexTS table was found below {raw_dir}. "
            "Run the download stage first."
        )
    raise ValueError(
        "Multiple possible raw dataset locations were found. Set dataset.raw_path "
        f"explicitly in config/config.yaml. Candidates: {discovered!r}"
    )


def _load_raw_frame(path: Path, config: Mapping[str, Any]) -> pd.DataFrame:
    if path.is_file():
        frame = safe_read_table(path)
    elif path.is_dir():
        from datasets import DatasetDict, load_from_disk

        loaded = load_from_disk(str(path))
        if isinstance(loaded, DatasetDict):
            split = str(
                _first_config_value(
                    config, ("dataset.split", "data.dataset_split")
                )
                or "train"
            )
            if split not in loaded:
                raise KeyError(
                    f"Split {split!r} is not present in dataset at {path}; "
                    f"available splits: {list(loaded.keys())!r}"
                )
            frame = loaded[split].to_pandas()
        else:
            frame = loaded.to_pandas()
    else:
        raise FileNotFoundError(f"Raw dataset path is neither file nor directory: {path}")

    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            f"safe_read_table/load_from_disk returned {type(frame).__name__}, "
            "expected pandas.DataFrame."
        )
    if frame.empty:
        raise ValueError(f"Raw FinTexTS dataset is empty: {path}")
    return frame


def _requested_tickers(config: Mapping[str, Any]) -> list[str]:
    configured = _first_config_value(
        config,
        (
            "universe.tickers",
            "dataset.tickers",
            "data.tickers",
            "experiment.tickers",
            "tickers",
        ),
    )
    if configured is None:
        return list(TICKERS)
    if isinstance(configured, str) or not isinstance(configured, Sequence):
        raise TypeError("Ticker universe must be a list of ticker strings.")
    normalized = [_normalize_ticker(value) for value in configured]
    if any(not ticker for ticker in normalized):
        raise ValueError("Ticker universe contains a missing/empty ticker.")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"Ticker universe contains duplicates: {normalized!r}")

    allow_custom = _as_bool(
        _deep_get(config, "universe.allow_custom_tickers"), default=False
    )
    if not allow_custom and set(normalized) != set(TICKERS):
        raise ValueError(
            "This experiment is fixed to the 11 requested semiconductor tickers. "
            f"Expected {list(TICKERS)!r}, received {normalized!r}. "
            "Set universe.allow_custom_tickers=true only for an explicit extension."
        )
    if set(normalized) == set(TICKERS):
        return [ticker for ticker in TICKERS if ticker in normalized]
    return normalized


def _configured_industry_map(
    config: Mapping[str, Any],
    tickers: Sequence[str],
) -> tuple[dict[str, str], str]:
    configured = _first_config_value(
        config,
        (
            "universe.ticker_industry_map",
            "universe.ticker_industries",
            "dataset.ticker_industry_map",
            "ticker_industry_map",
        ),
    )
    if configured is None:
        return ({ticker: EXPECTED_INDUSTRY for ticker in tickers}, "task_contract")
    if not isinstance(configured, Mapping):
        raise TypeError("Configured ticker-industry map must be a mapping.")
    normalized_map = {
        _normalize_ticker(ticker): str(industry).strip()
        for ticker, industry in configured.items()
    }
    missing = [ticker for ticker in tickers if ticker not in normalized_map]
    if missing:
        raise ValueError(
            f"Configured ticker-industry map is missing requested tickers: {missing!r}"
        )
    return ({ticker: normalized_map[ticker] for ticker in tickers}, "config")


def _verify_industries(
    selected: pd.DataFrame,
    schema: Mapping[str, Any],
    tickers: Sequence[str],
    config: Mapping[str, Any],
    ticker_key: str,
) -> tuple[dict[str, dict[str, Any]], str]:
    industry_column = schema.get("industry")
    expected = str(
        _first_config_value(
            config,
            (
                "universe.industry",
                "universe.industry_level",
                "dataset.industry",
                "data.industry",
                "data.industry_value",
            ),
        )
        or EXPECTED_INDUSTRY
    )
    verification: dict[str, dict[str, Any]] = {}

    if isinstance(industry_column, str):
        for ticker in tickers:
            values = (
                selected.loc[selected[ticker_key].eq(ticker), industry_column]
                .dropna()
                .astype("string")
                .str.strip()
            )
            values = sorted(value for value in values.unique().tolist() if value)
            if not values:
                raise ValueError(
                    f"Ticker {ticker} has no non-missing industry value in "
                    f"{industry_column!r}."
                )
            unexpected = [value for value in values if not _industry_matches(value, expected)]
            if unexpected:
                raise ValueError(
                    f"Ticker {ticker} is not exclusively in {expected!r}; "
                    f"observed industry values: {values!r}"
                )
            verification[ticker] = {
                "industry": " | ".join(values),
                "industry_verified": True,
                "industry_verification_source": f"column:{industry_column}",
            }
        return verification, f"column:{industry_column}"

    configured_map, map_source = _configured_industry_map(config, tickers)
    for ticker in tickers:
        industry = configured_map[ticker]
        if not _industry_matches(industry, expected):
            raise ValueError(
                f"Configured industry for {ticker} is {industry!r}, expected "
                f"a semiconductor label matching {expected!r}."
            )
        verification[ticker] = {
            "industry": industry,
            "industry_verified": True,
            "industry_verification_source": map_source,
        }
    LOGGER.warning(
        "FinTexTS has no Industry-level column. Semiconductor membership is "
        "validated against the %s ticker-industry mapping.",
        map_source,
    )
    return verification, map_source


def _placeholder_values(config: Mapping[str, Any]) -> frozenset[str]:
    configured = _first_config_value(
        config,
        (
            "news.placeholders",
            "events.placeholder_values",
            "preprocessing.news_placeholders",
        ),
    )
    if configured is None:
        return DEFAULT_PLACEHOLDERS
    if isinstance(configured, str) or not isinstance(configured, Sequence):
        raise TypeError("news.placeholders must be a list of strings.")
    return frozenset(
        unicodedata.normalize("NFKC", str(value)).strip().casefold()
        for value in configured
    ) | DEFAULT_PLACEHOLDERS


def _normalized_text_series(
    series: pd.Series,
    placeholders: frozenset[str],
) -> pd.Series:
    normalized = series.astype("string")
    normalized = normalized.str.normalize("NFKC")
    normalized = normalized.str.replace(r"\s+", " ", regex=True).str.strip()
    invalid = normalized.isna() | normalized.str.casefold().isin(placeholders)
    return normalized.mask(invalid)


def _news_masks(
    selected: pd.DataFrame,
    news_columns: Mapping[str, Sequence[str]],
    placeholders: frozenset[str],
) -> tuple[dict[str, pd.Series], dict[str, dict[str, pd.Series]]]:
    level_masks: dict[str, pd.Series] = {}
    normalized_values: dict[str, dict[str, pd.Series]] = {}
    for level in NEWS_LEVELS:
        normalized_values[level] = {}
        column_masks: list[pd.Series] = []
        for column in news_columns[level]:
            normalized = _normalized_text_series(selected[column], placeholders)
            normalized_values[level][column] = normalized
            column_masks.append(normalized.notna())
        if column_masks:
            level_masks[level] = pd.concat(column_masks, axis=1).any(axis=1)
        else:
            level_masks[level] = pd.Series(False, index=selected.index, dtype=bool)
    return level_masks, normalized_values


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _event_index(
    selected: pd.DataFrame,
    normalized_values: Mapping[str, Mapping[str, pd.Series]],
    ticker_key: str,
    date_key: str,
) -> dict[str, pd.DataFrame]:
    events_by_level: dict[str, pd.DataFrame] = {}
    for level in NEWS_LEVELS:
        frames: list[pd.DataFrame] = []
        for column, values in normalized_values[level].items():
            valid = values.notna()
            if not bool(valid.any()):
                continue
            text_hash = values.loc[valid].map(_hash_text)
            frames.append(
                pd.DataFrame(
                    {
                        "date": selected.loc[valid, date_key].to_numpy(),
                        "ticker": selected.loc[valid, ticker_key].to_numpy(),
                        "news_level": level,
                        "source_column": column,
                        "text_hash": text_hash.to_numpy(),
                    }
                )
            )
        if frames:
            events_by_level[level] = pd.concat(frames, ignore_index=True)
        else:
            events_by_level[level] = pd.DataFrame(
                columns=("date", "ticker", "news_level", "source_column", "text_hash")
            )
    return events_by_level


def _event_duplicate_metrics(events: pd.DataFrame) -> dict[str, int]:
    if events.empty:
        return {
            "n_event_cells": 0,
            "n_unique_text_hashes": 0,
            "n_same_ticker_day_duplicate_cells": 0,
            "n_cross_ticker_duplicate_groups_same_day": 0,
            "n_cross_ticker_duplicate_cells_same_day": 0,
            "n_cross_ticker_text_hashes_all_dates": 0,
        }

    unique_ticker_day = events.drop_duplicates(["date", "ticker", "text_hash"])
    same_day_groups = (
        events.groupby(["date", "text_hash"], dropna=False, sort=False)
        .agg(n_event_cells=("ticker", "size"), n_tickers=("ticker", "nunique"))
        .reset_index(drop=True)
    )
    cross_same_day = same_day_groups.loc[same_day_groups["n_tickers"].gt(1)]
    all_date_groups = events.groupby("text_hash", sort=False)["ticker"].nunique()
    return {
        "n_event_cells": int(len(events)),
        "n_unique_text_hashes": int(events["text_hash"].nunique()),
        "n_same_ticker_day_duplicate_cells": int(len(events) - len(unique_ticker_day)),
        "n_cross_ticker_duplicate_groups_same_day": int(len(cross_same_day)),
        "n_cross_ticker_duplicate_cells_same_day": int(
            (cross_same_day["n_event_cells"] - 1).sum()
        ),
        "n_cross_ticker_text_hashes_all_dates": int(all_date_groups.gt(1).sum()),
    }


def _write_yaml_atomic(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            yaml.safe_dump(
                dict(payload),
                handle,
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False,
            )
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _iso_date(value: Any) -> str:
    if pd.isna(value):
        return ""
    return pd.Timestamp(value).date().isoformat()


def _build_reports(
    raw: pd.DataFrame,
    selected: pd.DataFrame,
    schema: Mapping[str, Any],
    tickers: Sequence[str],
    industry_verification: Mapping[str, Mapping[str, Any]],
    industry_source: str,
    source_path: Path,
    generated_at: str,
    level_masks: Mapping[str, pd.Series],
    normalized_values: Mapping[str, Mapping[str, pd.Series]],
    ticker_key: str,
    date_key: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ohlc_columns = [str(schema[field]) for field in ("open", "high", "low", "close")]
    numeric_ohlc = {
        field: pd.to_numeric(selected[str(schema[field])], errors="coerce")
        for field in ("open", "high", "low", "close")
    }
    complete_ohlc = pd.concat(numeric_ohlc, axis=1).notna().all(axis=1)

    presence = selected[[ticker_key, date_key]].copy()
    for level, mask in level_masks.items():
        presence[level] = mask.to_numpy()
    presence["any_news"] = presence[list(NEWS_LEVELS)].any(axis=1)
    daily_presence = (
        presence.dropna(subset=[date_key])
        .groupby([ticker_key, date_key], as_index=False, sort=False)[
            [*NEWS_LEVELS, "any_news"]
        ]
        .max()
    )

    events_by_level = _event_index(
        selected, normalized_values, ticker_key=ticker_key, date_key=date_key
    )
    all_events = pd.concat(
        [events_by_level[level] for level in NEWS_LEVELS], ignore_index=True
    )

    valid_dates = selected[date_key].dropna()
    date_start = valid_dates.min() if not valid_dates.empty else pd.NaT
    date_end = valid_dates.max() if not valid_dates.empty else pd.NaT
    all_date_presence = (
        daily_presence.groupby(date_key, sort=True)["any_news"].max()
        if not daily_presence.empty
        else pd.Series(dtype=bool)
    )

    dataset_records: list[dict[str, Any]] = [
        {"section": "run", "metric": "generated_at_utc", "value": generated_at},
        {"section": "source", "metric": "raw_path", "value": str(source_path)},
        {
            "section": "source",
            "metric": "dataset_id",
            "value": "EXAONE-BI/FinTexTS",
        },
        {"section": "shape", "metric": "n_rows_raw", "value": int(len(raw))},
        {"section": "shape", "metric": "n_columns_raw", "value": int(raw.shape[1])},
        {
            "section": "shape",
            "metric": "n_rows_selected",
            "value": int(len(selected)),
        },
        {
            "section": "shape",
            "metric": "n_tickers_selected",
            "value": int(selected[ticker_key].nunique()),
        },
        {"section": "time", "metric": "date_start", "value": _iso_date(date_start)},
        {"section": "time", "metric": "date_end", "value": _iso_date(date_end)},
        {
            "section": "time",
            "metric": "n_dates_observed",
            "value": int(valid_dates.nunique()),
        },
        {
            "section": "time",
            "metric": "n_trading_dates_with_complete_ohlc",
            "value": int(selected.loc[complete_ohlc, date_key].nunique()),
        },
        {
            "section": "quality",
            "metric": "n_invalid_dates_selected",
            "value": int(selected[date_key].isna().sum()),
        },
        {
            "section": "quality",
            "metric": "n_duplicate_ticker_dates",
            "value": int(
                selected.duplicated([ticker_key, date_key], keep=False).sum()
            ),
        },
        {
            "section": "industry",
            "metric": "industry_verification_source",
            "value": industry_source,
        },
        {
            "section": "news",
            "metric": "n_event_cells_all_levels",
            "value": int(len(all_events)),
        },
        {
            "section": "news",
            "metric": "n_ticker_days_with_news",
            "value": int(daily_presence["any_news"].sum()),
        },
        {
            "section": "news",
            "metric": "n_ticker_days_without_news",
            "value": int((~daily_presence["any_news"]).sum()),
        },
        {
            "section": "news",
            "metric": "ticker_day_news_rate",
            "value": (
                float(daily_presence["any_news"].mean())
                if not daily_presence.empty
                else np.nan
            ),
        },
        {
            "section": "news",
            "metric": "ticker_day_no_news_rate",
            "value": (
                float((~daily_presence["any_news"]).mean())
                if not daily_presence.empty
                else np.nan
            ),
        },
        {
            "section": "news",
            "metric": "n_dates_with_news",
            "value": int(all_date_presence.sum()),
        },
        {
            "section": "news",
            "metric": "n_dates_without_news",
            "value": int((~all_date_presence).sum()),
        },
        {
            "section": "news",
            "metric": "date_news_rate",
            "value": (
                float(all_date_presence.mean())
                if not all_date_presence.empty
                else np.nan
            ),
        },
        {
            "section": "news",
            "metric": "date_no_news_rate",
            "value": (
                float((~all_date_presence).mean())
                if not all_date_presence.empty
                else np.nan
            ),
        },
    ]
    for field, column in zip(("open", "high", "low", "close"), ohlc_columns):
        numeric = numeric_ohlc[field]
        nonnumeric = selected[column].notna() & numeric.isna()
        dataset_records.extend(
            [
                {
                    "section": "ohlc_missing",
                    "metric": f"missing_{field}",
                    "value": int(numeric.isna().sum()),
                },
                {
                    "section": "ohlc_invalid",
                    "metric": f"nonnumeric_{field}",
                    "value": int(nonnumeric.sum()),
                },
            ]
        )
    dataset_records.append(
        {
            "section": "ohlc_missing",
            "metric": "missing_ohlc_total",
            "value": int(
                sum(numeric_ohlc[field].isna().sum() for field in numeric_ohlc)
            ),
        }
    )
    duplicate_metrics_all = _event_duplicate_metrics(all_events)
    for metric, value in duplicate_metrics_all.items():
        dataset_records.append(
            {"section": "news_duplicates", "metric": metric, "value": value}
        )
    for column in raw.columns:
        dataset_records.append(
            {
                "section": "schema",
                "metric": f"dtype::{column}",
                "value": str(raw[column].dtype),
            }
        )
    dataset_summary = pd.DataFrame.from_records(
        dataset_records, columns=("section", "metric", "value")
    )

    ticker_records: list[dict[str, Any]] = []
    for ticker in tickers:
        ticker_rows = selected.loc[selected[ticker_key].eq(ticker)]
        ticker_dates = ticker_rows[date_key].dropna()
        ticker_daily = daily_presence.loc[daily_presence[ticker_key].eq(ticker)]
        ticker_record: dict[str, Any] = {
            "ticker": ticker,
            **industry_verification[ticker],
            "date_start": _iso_date(
                ticker_dates.min() if not ticker_dates.empty else pd.NaT
            ),
            "date_end": _iso_date(
                ticker_dates.max() if not ticker_dates.empty else pd.NaT
            ),
            "n_rows": int(len(ticker_rows)),
            "n_dates_observed": int(ticker_dates.nunique()),
            "n_trading_days_complete_ohlc": int(
                ticker_rows.loc[complete_ohlc.loc[ticker_rows.index], date_key].nunique()
            ),
            "n_duplicate_dates": int(
                ticker_rows.duplicated(date_key, keep=False).sum()
            ),
            "missing_open": int(numeric_ohlc["open"].loc[ticker_rows.index].isna().sum()),
            "missing_high": int(numeric_ohlc["high"].loc[ticker_rows.index].isna().sum()),
            "missing_low": int(numeric_ohlc["low"].loc[ticker_rows.index].isna().sum()),
            "missing_close": int(
                numeric_ohlc["close"].loc[ticker_rows.index].isna().sum()
            ),
            "ticker_days_with_news": int(ticker_daily["any_news"].sum()),
            "ticker_days_without_news": int((~ticker_daily["any_news"]).sum()),
            "ticker_day_news_rate": (
                float(ticker_daily["any_news"].mean())
                if not ticker_daily.empty
                else np.nan
            ),
            "ticker_day_no_news_rate": (
                float((~ticker_daily["any_news"]).mean())
                if not ticker_daily.empty
                else np.nan
            ),
        }
        for level in NEWS_LEVELS:
            ticker_record[f"{level}_news_count"] = int(
                sum(
                    values.loc[ticker_rows.index].notna().sum()
                    for values in normalized_values[level].values()
                )
            )
            ticker_record[f"{level}_days_with_news"] = int(
                ticker_daily[level].sum()
            )
        ticker_records.append(ticker_record)
    ticker_summary = pd.DataFrame.from_records(ticker_records)

    news_records: list[dict[str, Any]] = []
    for level in (*NEWS_LEVELS, "all"):
        if level == "all":
            events = all_events
            relevant_columns = [
                column
                for level_columns in schema["news"].values()
                for column in level_columns
            ]
            presence_column = "any_news"
        else:
            events = events_by_level[level]
            relevant_columns = list(schema["news"][level])
            presence_column = level

        duplicate_metrics = _event_duplicate_metrics(events)
        ticker_days_with_news = int(daily_presence[presence_column].sum())
        date_presence = (
            daily_presence.groupby(date_key, sort=True)[presence_column].max()
            if not daily_presence.empty
            else pd.Series(dtype=bool)
        )
        news_records.append(
            {
                "news_level": level,
                "n_source_columns": len(relevant_columns),
                "source_columns": json.dumps(relevant_columns, ensure_ascii=False),
                **duplicate_metrics,
                "n_ticker_days": int(len(daily_presence)),
                "n_ticker_days_with_news": ticker_days_with_news,
                "n_ticker_days_without_news": int(
                    len(daily_presence) - ticker_days_with_news
                ),
                "ticker_day_news_rate": (
                    float(ticker_days_with_news / len(daily_presence))
                    if len(daily_presence)
                    else np.nan
                ),
                "ticker_day_no_news_rate": (
                    float(1.0 - ticker_days_with_news / len(daily_presence))
                    if len(daily_presence)
                    else np.nan
                ),
                "n_dates": int(len(date_presence)),
                "n_dates_with_news": int(date_presence.sum()),
                "n_dates_without_news": int((~date_presence).sum()),
                "date_news_rate": (
                    float(date_presence.mean()) if not date_presence.empty else np.nan
                ),
                "date_no_news_rate": (
                    float((~date_presence).mean())
                    if not date_presence.empty
                    else np.nan
                ),
            }
        )
    news_summary = pd.DataFrame.from_records(news_records)
    return dataset_summary, ticker_summary, news_summary


def run(config: dict) -> dict[str, Path]:
    """Run schema inspection and write the mapping/audit tables.

    Parameters
    ----------
    config:
        Parsed project configuration returned by :func:`src.utils.load_config`.

    Returns
    -------
    dict[str, pathlib.Path]
        Paths keyed by ``schema_mapping``, ``dataset_summary``,
        ``ticker_summary``, and ``news_summary``.
    """

    if not isinstance(config, dict):
        raise TypeError(f"config must be dict, received {type(config).__name__}.")

    global LOGGER
    LOGGER = get_logger(__name__, config)
    source_path = _discover_raw_path(config)
    LOGGER.info("Reading raw FinTexTS data from %s", source_path)
    raw = _load_raw_frame(source_path, config)

    LOGGER.info("Raw FinTexTS schema (%d columns):", raw.shape[1])
    for column in raw.columns:
        LOGGER.info("  %-40s %s", str(column), str(raw[column].dtype))

    schema = _discover_schema(raw, config)
    LOGGER.info("Resolved core columns: %s", {key: schema[key] for key in CORE_FIELDS})
    for level in NEWS_LEVELS:
        LOGGER.info("Resolved %s news columns: %s", level, schema["news"][level])

    tickers = _requested_tickers(config)
    ticker_column = str(schema["ticker"])
    normalized_tickers = raw[ticker_column].map(_normalize_ticker)
    available_tickers = set(normalized_tickers.loc[normalized_tickers.ne("")].unique())
    missing_tickers = [ticker for ticker in tickers if ticker not in available_tickers]
    if missing_tickers:
        raise ValueError(
            f"Requested FinTexTS tickers are missing: {missing_tickers!r}. "
            f"Available ticker count: {len(available_tickers)}."
        )

    selected = raw.loc[normalized_tickers.isin(tickers)].copy()
    ticker_key = "__inspect_ticker__"
    date_key = "__inspect_date__"
    selected[ticker_key] = normalized_tickers.loc[selected.index]
    selected[date_key] = pd.to_datetime(
        selected[str(schema["date"])], errors="coerce"
    ).dt.normalize()
    invalid_dates = int(selected[date_key].isna().sum())
    strict_dates = _as_bool(_deep_get(config, "schema.strict_dates"), default=True)
    if invalid_dates and strict_dates:
        raise ValueError(
            f"Found {invalid_dates} invalid/missing dates in the selected universe. "
            "Fix the raw data or set schema.strict_dates=false for an explicit audit-only run."
        )

    industry_verification, industry_source = _verify_industries(
        selected,
        schema,
        tickers,
        config,
        ticker_key=ticker_key,
    )
    for ticker in tickers:
        if not bool(industry_verification[ticker]["industry_verified"]):
            raise AssertionError(f"Industry verification unexpectedly failed for {ticker}.")

    placeholders = _placeholder_values(config)
    level_masks, normalized_values = _news_masks(
        selected, schema["news"], placeholders
    )
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    dataset_summary, ticker_summary, news_summary = _build_reports(
        raw=raw,
        selected=selected,
        schema=schema,
        tickers=tickers,
        industry_verification=industry_verification,
        industry_source=industry_source,
        source_path=source_path,
        generated_at=generated_at,
        level_masks=level_masks,
        normalized_values=normalized_values,
        ticker_key=ticker_key,
        date_key=date_key,
    )

    mapping_value = _first_config_value(
        config,
        (
            "schema.mapping_file",
            "schema.mapping_path",
            "paths.schema_mapping",
        ),
    )
    schema_mapping_path = (
        _path_from_config(config, mapping_value)
        if isinstance(mapping_value, (str, Path))
        else project_path(config, "config", "schema_mapping.yaml")
    )
    tables_dir = project_path(config, "outputs", "tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    dataset_summary_path = tables_dir / "dataset_summary.csv"
    ticker_summary_path = tables_dir / "ticker_summary.csv"
    news_summary_path = tables_dir / "news_summary.csv"

    mapping_payload: dict[str, Any] = {
        "version": 1,
        "generated_at_utc": generated_at,
        "dataset_id": str(
            _first_config_value(
                config,
                ("dataset.name", "dataset.id", "data.dataset_id"),
            )
            or "EXAONE-BI/FinTexTS"
        ),
        "raw_path": str(source_path),
        "columns": {
            "date": schema["date"],
            "ticker": schema["ticker"],
            "industry": schema["industry"],
            "open": schema["open"],
            "high": schema["high"],
            "low": schema["low"],
            "close": schema["close"],
            "news": {
                level: list(schema["news"][level]) for level in NEWS_LEVELS
            },
        },
        "industry_resolution": {
            "expected": EXPECTED_INDUSTRY,
            "source": industry_source,
            "tickers": list(tickers),
        },
        "raw_schema": {
            "column_order": [str(column) for column in raw.columns],
            "dtypes": {
                str(column): str(raw[column].dtype) for column in raw.columns
            },
        },
    }
    _write_yaml_atomic(mapping_payload, schema_mapping_path)
    atomic_write_csv(dataset_summary, dataset_summary_path, index=False)
    atomic_write_csv(ticker_summary, ticker_summary_path, index=False)
    atomic_write_csv(news_summary, news_summary_path, index=False)

    LOGGER.info("Schema mapping written to %s", schema_mapping_path)
    LOGGER.info("Dataset summary written to %s", dataset_summary_path)
    LOGGER.info("Ticker summary written to %s", ticker_summary_path)
    LOGGER.info("News summary written to %s", news_summary_path)
    return {
        "schema_mapping": schema_mapping_path,
        "dataset_summary": dataset_summary_path,
        "ticker_summary": ticker_summary_path,
        "news_summary": news_summary_path,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect the raw FinTexTS schema and write audit summaries."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Path to the project YAML configuration.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Console logging level.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    config = load_config(args.config)
    outputs = run(config)
    for name, path in outputs.items():
        LOGGER.info("%s: %s", name, path)


if __name__ == "__main__":
    main()
