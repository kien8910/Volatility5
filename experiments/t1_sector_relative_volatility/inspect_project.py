from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import ExperimentConfig
from .utils import assert_unique, parse_dates_mixed, require_columns, write_json


KEYS = ["ticker", "feature_date"]
MARKET_REQUIRED = [
    "ticker",
    "feature_date",
    "split",
    "log_variance",
    "target_log_variance",
]


def _schema(frame: pd.DataFrame) -> list[dict[str, str]]:
    return [{"column": column, "dtype": str(frame[column].dtype)} for column in frame]


def _split_summary(market: pd.DataFrame) -> list[dict[str, Any]]:
    summary = (
        market.groupby("split", observed=True)
        .agg(
            start_date=("feature_date", "min"),
            end_date=("feature_date", "max"),
            rows=("ticker", "size"),
            dates=("feature_date", "nunique"),
            tickers=("ticker", "nunique"),
        )
        .reset_index()
    )
    return summary.to_dict("records")


def inspect_artifacts(config: ExperimentConfig) -> tuple[dict[str, Any], pd.DataFrame]:
    paths = {
        "market": config.market_path,
        "metadata": config.metadata_path,
        "semantic": config.semantic_path,
    }
    missing_files = [str(path) for path in paths.values() if not path.exists()]
    if missing_files:
        raise FileNotFoundError(f"Required source artifacts are missing: {missing_files}")

    market = pd.read_parquet(paths["market"])
    metadata = pd.read_parquet(paths["metadata"])
    semantic = pd.read_parquet(paths["semantic"])

    require_columns(market, MARKET_REQUIRED, "market_supervised")
    require_columns(metadata, [*KEYS, "split"], "metadata panel")
    require_columns(semantic, [*KEYS, "split"], "semantic panel")

    for frame in (market, metadata, semantic):
        frame["feature_date"] = parse_dates_mixed(frame["feature_date"])
    assert_unique(market, KEYS, "market_supervised")
    assert_unique(metadata, KEYS, "metadata panel")
    assert_unique(semantic, KEYS, "semantic panel")

    actual_tickers = sorted(market["ticker"].astype(str).unique())
    if actual_tickers != sorted(config.tickers):
        raise AssertionError(
            f"Ticker universe mismatch: expected={sorted(config.tickers)}, "
            f"actual={actual_tickers}"
        )

    price_columns = [
        column
        for column in market.columns
        if any(column.startswith(prefix) for prefix in config.price_prefixes)
        and not any(token in column for token in config.price_exclude_tokens)
        and pd.api.types.is_numeric_dtype(market[column])
    ]
    metadata_columns = [
        column
        for column in metadata.columns
        if any(column.startswith(f"meta__{level}__") for level in config.metadata_levels)
    ]
    semantic_prefix = "raw" if config.semantic_representation == "R2" else "pca"
    semantic_columns = [
        column
        for column in semantic.columns
        if any(
            column.startswith(f"{semantic_prefix}__{level}__")
            for level in config.semantic_levels
        )
    ]
    if not price_columns:
        raise AssertionError("No price/HAR columns were discovered from the real schema")
    if not metadata_columns:
        raise AssertionError("No target-company metadata columns were discovered")
    if not semantic_columns:
        raise AssertionError("No target-company semantic columns were discovered")

    report = {
        "source_profile": config.source_profile,
        "paths": {key: str(value.resolve()) for key, value in paths.items()},
        "market_shape": list(market.shape),
        "metadata_shape": list(metadata.shape),
        "semantic_shape": list(semantic.shape),
        "date_column": "feature_date",
        "ticker_column": "ticker",
        "base_volatility_column": "log_variance",
        "existing_one_day_target_column": "target_log_variance",
        "split_column": "split",
        "ticker_universe": actual_tickers,
        "price_columns": price_columns,
        "metadata_columns": metadata_columns,
        "semantic_columns": semantic_columns,
        "split_summary": _split_summary(market),
        "schemas": {
            "market": _schema(market),
            "metadata": _schema(metadata),
            "semantic": _schema(semantic),
        },
        "mapping": {
            "market_and_har": str(config.market_path.resolve()),
            "target_metadata": str(config.metadata_path.resolve()),
            "target_semantics": str(config.semantic_path.resolve()),
            "new_outputs": str(config.output_directory.resolve()),
            "source_artifacts_are_read_only": True,
        },
    }
    write_json(config.output_directory / "data_audit.json", report)
    return report, market


def write_structure_report(config: ExperimentConfig, report: dict[str, Any]) -> Path:
    path = config.output_directory / "project_mapping.md"
    split_lines = "\n".join(
        f"- `{row['split']}`: {row['start_date']} → {row['end_date']}, "
        f"{row['dates']} dates, {row['rows']} rows"
        for row in report["split_summary"]
    )
    content = f"""# Existing-project mapping for T1

The T1 experiment is independent and writes only to:

`{config.output_directory.resolve()}`

It reads, without modifying:

- Market and HAR features: `{config.market_path.resolve()}`
- Target-company metadata: `{config.metadata_path.resolve()}`
- Target-company semantic representation: `{config.semantic_path.resolve()}`

## Discovered schema

- Date: `feature_date`
- Ticker: `ticker`
- Current volatility: `log_variance`
- Existing one-day target: `target_log_variance`
- Split: `split`
- Price/HAR features: {len(report['price_columns'])}
- Metadata features: {len(report['metadata_columns'])}
- Target semantic features: {len(report['semantic_columns'])}

## Existing chronological split

{split_lines}

## Isolation guarantee

No file in `fintexts_semiconductor_prototype` or `original_volatility_targets`
is written or modified by this experiment.
"""
    path.write_text(content, encoding="utf-8")
    return path
