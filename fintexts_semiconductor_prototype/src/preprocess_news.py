"""Normalize the wide FinTexTS news fields into an auditable long table.

This module deliberately does *not* deduplicate news.  It preserves one row per
source occurrence so :mod:`src.build_events` can report exactly what was
removed by each deduplication rule.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.utils import (
    atomic_write_csv,
    ensure_directories,
    get_logger,
    load_config,
    project_path,
    safe_read_table,
    set_global_seed,
    validate_required_columns,
    write_table,
)

LOGGER = get_logger(__name__)

NEWS_LEVELS = ("macro", "sector", "related", "target")
DEFAULT_TICKERS = (
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
PLACEHOLDERS = {
    "",
    "-",
    "--",
    "n/a",
    "na",
    "nan",
    "none",
    "null",
    "no news",
    "no_news",
    "not available",
    "[]",
    "{}",
}
TEXT_KEYS = (
    "text",
    "content",
    "article",
    "news",
    "summary",
    "description",
    "body",
    "headline",
    "title",
)
_SPACE_RE = re.compile(r"\s+")
_TICKER_RE = re.compile(r"(?<![A-Z0-9])(?:\$)?([A-Z]{2,5})(?![A-Z0-9])")


def _nested(config: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def _configured_path(
    config: Mapping[str, Any], candidates: Iterable[tuple[str, ...]], default: str
) -> Path:
    value: Any = None
    for keys in candidates:
        value = _nested(config, *keys)
        if value not in (None, ""):
            break
    return project_path(config, str(value or default))


def _load_schema_mapping(config: Mapping[str, Any]) -> dict[str, Any]:
    path = _configured_path(
        config,
        (
            ("schema", "mapping_file"),
            ("schema", "mapping_path"),
            ("paths", "schema_mapping"),
            ("data", "schema_mapping"),
        ),
        "config/schema_mapping.yaml",
    )
    if not path.exists():
        inline = _nested(config, "schema_mapping", default={})
        if isinstance(inline, Mapping) and inline:
            return dict(inline)
        raise FileNotFoundError(
            f"Schema mapping not found at {path}. Run the schema inspection stage first."
        )
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, Mapping):
        raise TypeError(f"Schema mapping at {path} must contain a YAML mapping.")
    if isinstance(loaded.get("columns"), Mapping):
        return dict(loaded["columns"])
    return dict(loaded)


def _column_value(mapping: Mapping[str, Any], key: str) -> Any:
    aliases = {
        "date": ("date", "date_column", "datetime"),
        "ticker": ("ticker", "ticker_column", "symbol"),
        "industry": ("industry", "industry_level", "Industry-level"),
    }
    for alias in aliases.get(key, (key,)):
        if alias in mapping:
            value = mapping[alias]
            if isinstance(value, Mapping):
                return value.get("column") or value.get("name") or value.get("columns")
            return value
    return None


def _as_columns(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        result: list[str] = []
        for nested_value in value.values():
            result.extend(_as_columns(nested_value))
        return list(dict.fromkeys(result))
    if isinstance(value, Iterable):
        return list(dict.fromkeys(str(item) for item in value if item not in (None, "")))
    return [str(value)]


def _news_columns(mapping: Mapping[str, Any], level: str) -> list[str]:
    containers = (
        mapping.get("news"),
        mapping.get("news_columns"),
        mapping.get("text_columns"),
    )
    aliases = {
        "macro": ("macro", "macro_news", "macroeconomic"),
        "sector": ("sector", "sector_news", "industry_news"),
        "related": (
            "related",
            "related_news",
            "related_company",
            "related_company_news",
        ),
        "target": (
            "target",
            "target_news",
            "target_company",
            "target_company_news",
            "company_news",
        ),
    }
    for container in containers:
        if isinstance(container, Mapping):
            for alias in aliases[level]:
                if alias in container:
                    return _as_columns(container[alias])
    for alias in aliases[level]:
        if alias in mapping:
            return _as_columns(mapping[alias])
    return []


def normalize_text(value: str) -> str:
    """Return the canonical text used for hashing and exact comparison."""

    text = unicodedata.normalize("NFKC", value)
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = _SPACE_RE.sub(" ", text).strip()
    return text


def _valid_text(value: str) -> bool:
    return normalize_text(value).casefold() not in PLACEHOLDERS


def _try_parse_container(value: str) -> Any:
    stripped = value.strip()
    if len(stripped) < 2 or stripped[0] not in "[{" or stripped[-1] not in "]}":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            return value


def _flatten_news_value(value: Any) -> list[str]:
    """Extract article-like strings from nested Arrow/Pandas cell values."""

    if value is None:
        return []
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return []
    if isinstance(value, str):
        parsed = _try_parse_container(value)
        if parsed is not value:
            return _flatten_news_value(parsed)
        normalized = normalize_text(value)
        return [normalized] if _valid_text(normalized) else []
    if isinstance(value, Mapping):
        selected: list[str] = []
        for key in TEXT_KEYS:
            if key in value:
                selected.extend(_flatten_news_value(value[key]))
        if selected:
            # A headline and body belonging to one object constitute one event.
            combined = normalize_text(" ".join(dict.fromkeys(selected)))
            return [combined] if _valid_text(combined) else []
        fallback: list[str] = []
        for nested_value in value.values():
            fallback.extend(_flatten_news_value(nested_value))
        return fallback
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
        flattened: list[str] = []
        for item in value:
            flattened.extend(_flatten_news_value(item))
        return flattened
    if pd.isna(value):
        return []
    text = normalize_text(str(value))
    return [text] if _valid_text(text) else []


def _related_tickers(
    text: str, row_ticker: str, configured_tickers: tuple[str, ...]
) -> list[str]:
    allowed = set(configured_tickers)
    found = {
        match.group(1)
        for match in _TICKER_RE.finditer(text.upper())
        if match.group(1) in allowed and match.group(1) != row_ticker
    }
    return sorted(found)


def _source_path(config: Mapping[str, Any]) -> Path:
    raw_filename = str(
        _nested(config, "data", "raw_filename", default="fintexts_raw.parquet")
    )
    configured = _configured_path(
        config,
        (
            ("paths", "raw_dataset"),
            ("paths", "raw_data"),
            ("data", "raw_file"),
            ("data", "raw_dataset"),
        ),
        f"data/raw/{raw_filename}",
    )
    if configured.exists():
        return configured
    raw_dir = project_path(config, "data/raw")
    candidates = sorted(raw_dir.glob("*.parquet"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"No raw parquet file found. Expected {configured} or one parquet in {raw_dir}."
        )
    raise FileNotFoundError(
        f"Raw dataset path {configured} does not exist and {raw_dir} contains "
        f"multiple parquet files; set paths.raw_dataset explicitly."
    )


def _summary(records: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total_days = market[["date", "ticker"]].drop_duplicates()
    linked = records.copy()
    linked["ticker"] = linked["available_to_tickers"].map(
        lambda value: json.loads(value) if isinstance(value, str) else list(value)
    )
    linked = linked.explode("ticker", ignore_index=True)
    linked["ticker"] = linked["ticker"].astype(str)
    event_days = linked[["date", "ticker"]].drop_duplicates()
    day_status = total_days.merge(
        event_days.assign(has_any_news=True), on=["date", "ticker"], how="left"
    )
    day_status["has_any_news"] = day_status["has_any_news"].fillna(False).astype(bool)
    for level in (*NEWS_LEVELS, "all"):
        subset = records if level == "all" else records.loc[records["news_level"] == level]
        linked_subset = (
            linked if level == "all" else linked.loc[linked["news_level"] == level]
        )
        unique_events = subset["text_hash"].nunique()
        rows.append(
            {
                "scope": "overall",
                "ticker": "ALL",
                "news_level": level,
                "source_occurrences": int(len(subset)),
                "unique_text_hashes": int(unique_events),
                "days_with_news": int(subset["date"].nunique()),
                "days_without_news": (
                    int((~day_status["has_any_news"]).sum()) if level == "all" else np.nan
                ),
                "share_days_with_news": (
                    float(day_status["has_any_news"].mean()) if level == "all" else np.nan
                ),
            }
        )
        for ticker, ticker_days in total_days.groupby("ticker", observed=True):
            ticker_subset = linked_subset.loc[linked_subset["ticker"] == ticker]
            n_days = ticker_days["date"].nunique()
            with_news = ticker_subset["date"].nunique()
            rows.append(
                {
                    "scope": "ticker",
                    "ticker": ticker,
                    "news_level": level,
                    "source_occurrences": int(len(ticker_subset)),
                    "unique_text_hashes": int(ticker_subset["text_hash"].nunique()),
                    "days_with_news": int(with_news),
                    "days_without_news": int(max(n_days - with_news, 0)),
                    "share_days_with_news": float(with_news / n_days) if n_days else np.nan,
                }
            )
    return pd.DataFrame(rows)


def run(config: dict) -> dict[str, Path]:
    """Normalize FinTexTS news fields and persist a lossless long table."""

    seed = int(_nested(config, "project", "seed", default=config.get("seed", 42)))
    set_global_seed(seed)
    ensure_directories(config)

    mapping = _load_schema_mapping(config)
    raw_path = _source_path(config)
    raw = safe_read_table(raw_path)
    if not isinstance(raw, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame from {raw_path}, got {type(raw)!r}.")

    date_col = _column_value(mapping, "date")
    ticker_col = _column_value(mapping, "ticker")
    if not isinstance(date_col, str) or not isinstance(ticker_col, str):
        raise KeyError("schema_mapping.yaml must define scalar date and ticker columns.")
    level_columns = {level: _news_columns(mapping, level) for level in NEWS_LEVELS}
    missing_levels = [level for level, columns in level_columns.items() if not columns]
    if missing_levels:
        raise KeyError(
            "No mapped news columns for levels: "
            + ", ".join(missing_levels)
            + ". Update schema_mapping.yaml."
        )
    required = [date_col, ticker_col, *sum(level_columns.values(), [])]
    validate_required_columns(raw, required, "raw FinTexTS news preprocessing")

    tickers = tuple(
        str(ticker).upper()
        for ticker in _nested(config, "data", "tickers", default=DEFAULT_TICKERS)
    )
    frame = raw.loc[
        raw[ticker_col]
        .astype(str)
        .str.strip()
        .str.upper()
        .isin(tickers)
    ].copy()
    frame["__date"] = pd.to_datetime(frame[date_col], errors="coerce").dt.normalize()
    frame["__ticker"] = frame[ticker_col].astype(str).str.upper().str.strip()
    bad_dates = int(frame["__date"].isna().sum())
    if bad_dates:
        raise ValueError(f"Found {bad_dates} rows with invalid dates in {date_col!r}.")

    records: list[dict[str, Any]] = []
    configured_placeholders = {
        normalize_text(str(value)).casefold()
        for value in _nested(
            config,
            "events",
            "placeholder_values",
            default=PLACEHOLDERS,
        )
    } | PLACEHOLDERS
    for row_position, (_, row) in enumerate(frame.iterrows()):
        date = pd.Timestamp(row["__date"])
        ticker = str(row["__ticker"])
        for level, columns in level_columns.items():
            for source_column in columns:
                for item_position, text in enumerate(_flatten_news_value(row[source_column])):
                    normalized = normalize_text(text)
                    if normalized.casefold() in configured_placeholders:
                        continue
                    text_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
                    related = (
                        _related_tickers(normalized, ticker, tickers)
                        if level == "related"
                        else []
                    )
                    if level in {"macro", "sector"}:
                        available = list(tickers)
                        target_ticker: str | None = None
                    elif level == "target":
                        available = [ticker]
                        target_ticker = ticker
                    else:
                        # The row ticker receives its related-company context.  Any
                        # mentioned semiconductor tickers are retained for audit,
                        # but are not silently treated as targets.
                        available = [ticker]
                        target_ticker = ticker
                    records.append(
                        {
                            "source_occurrence_id": hashlib.sha256(
                                (
                                    f"{date.date()}|{ticker}|{row_position}|{source_column}|"
                                    f"{item_position}|{text_hash}"
                                ).encode("utf-8")
                            ).hexdigest()[:24],
                            "source_row_number": row_position,
                            "date": date,
                            "news_level": level,
                            "source_column": source_column,
                            "text": text,
                            "text_normalized": normalized,
                            "text_hash": text_hash,
                            "source_ticker": ticker,
                            "target_ticker": target_ticker,
                            "related_tickers": json.dumps(related, separators=(",", ":")),
                            "available_to_tickers": json.dumps(
                                available, separators=(",", ":")
                            ),
                        }
                    )

    columns = [
        "source_occurrence_id",
        "source_row_number",
        "date",
        "news_level",
        "source_column",
        "text",
        "text_normalized",
        "text_hash",
        "source_ticker",
        "target_ticker",
        "related_tickers",
        "available_to_tickers",
    ]
    news_records = pd.DataFrame.from_records(records, columns=columns)
    if news_records.empty:
        raise ValueError("No valid news text remained after normalization.")
    news_records = news_records.sort_values(
        ["date", "news_level", "source_ticker", "source_occurrence_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    processed_path = _configured_path(
        config,
        (("paths", "news_records"), ("news", "records_path")),
        "data/processed/news_records.parquet",
    )
    # ``inspect_schema`` owns the required news_summary.csv, whose denominators
    # are based on the full raw schema.  This stage writes a separate
    # occurrence-level audit rather than degrading/overwriting that report.
    summary_path = project_path(
        config, "outputs/tables/news_occurrence_summary.csv"
    )
    write_table(news_records, processed_path)
    summary = _summary(
        news_records,
        frame[["__date", "__ticker"]].rename(
            columns={"__date": "date", "__ticker": "ticker"}
        ),
    )
    atomic_write_csv(summary, summary_path, index=False)
    LOGGER.info(
        "Normalized %d source news occurrences from %d semiconductor rows.",
        len(news_records),
        len(frame),
    )
    return {
        "news_records": processed_path,
        "news_occurrence_summary": summary_path,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
