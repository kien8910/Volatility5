"""Build canonical, leakage-safe news events from normalized occurrences."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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


def _nested(config: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def _path(
    config: Mapping[str, Any],
    candidates: Iterable[tuple[str, ...]],
    default: str,
) -> Path:
    for keys in candidates:
        value = _nested(config, *keys)
        if value not in (None, ""):
            return project_path(config, str(value))
    return project_path(config, default)


def _json_list(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return sorted({str(item) for item in value if str(item)})
    if isinstance(value, (tuple, set, np.ndarray)):
        return sorted({str(item) for item in value if str(item)})
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = [part.strip() for part in text.split(",") if part.strip()]
    if isinstance(parsed, str):
        parsed = [parsed]
    if not isinstance(parsed, Sequence):
        return []
    return sorted({str(item) for item in parsed if str(item)})


def _date_split_map(config: Mapping[str, Any]) -> pd.DataFrame:
    """Return unique ``date, split`` rows from supervised data or split metadata."""

    market_path = _path(
        config,
        (("paths", "market_supervised"), ("market", "supervised_path")),
        "data/processed/market_supervised.parquet",
    )
    if market_path.exists():
        market = safe_read_table(market_path)
        date_candidates = ("feature_date", "date", "predictor_date")
        split_candidates = ("split", "dataset_split", "set")
        date_col = next((column for column in date_candidates if column in market), None)
        split_col = next((column for column in split_candidates if column in market), None)
        if date_col and split_col:
            result = market[[date_col, split_col]].copy()
            result.columns = ["date", "split"]
            result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
            result["split"] = result["split"].astype(str).str.lower()
            conflicts = result.groupby("date", observed=True)["split"].nunique()
            if int(conflicts.max()) > 1:
                bad = conflicts[conflicts > 1].index.astype(str).tolist()[:5]
                raise ValueError(f"Dates assigned to multiple splits: {bad}")
            return result.drop_duplicates().sort_values("date").reset_index(drop=True)

    metadata_path = _path(
        config,
        (("paths", "split_metadata"), ("splits", "metadata_path")),
        "data/processed/split_metadata.csv",
    )
    if not metadata_path.exists():
        raise FileNotFoundError(
            "Cannot assign events to train/validation/test: neither market_supervised "
            f"with a split column nor {metadata_path} is available."
        )
    metadata = safe_read_table(metadata_path)
    if {"date", "split"}.issubset(metadata.columns):
        result = metadata[["date", "split"]].copy()
        result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
        result["split"] = result["split"].astype(str).str.lower()
        return result.drop_duplicates().sort_values("date").reset_index(drop=True)
    validate_required_columns(
        metadata, ["split", "start_date", "end_date"], "split metadata"
    )
    rows: list[dict[str, Any]] = []
    for record in metadata.itertuples(index=False):
        start = pd.Timestamp(getattr(record, "start_date")).normalize()
        end = pd.Timestamp(getattr(record, "end_date")).normalize()
        for date in pd.date_range(start, end, freq="D"):
            rows.append({"date": date, "split": str(getattr(record, "split")).lower()})
    return pd.DataFrame(rows).drop_duplicates("date", keep="last")


def _combine_lists(series: pd.Series) -> list[str]:
    values: set[str] = set()
    for item in series:
        values.update(_json_list(item))
    return sorted(values)


def _exact_canonicalize(
    occurrences: pd.DataFrame, all_tickers: Sequence[str]
) -> pd.DataFrame:
    group_columns = ["date", "news_level", "text_hash"]
    target = occurrences["target_ticker"].fillna("").astype(str)
    occurrences = occurrences.assign(__target_scope=target)
    # Target-company news is intentionally never merged across different target
    # tickers. Related news may be shared by several affected row tickers.
    occurrences.loc[
        occurrences["news_level"] != "target", "__target_scope"
    ] = "__shared__"
    rows: list[dict[str, Any]] = []
    for keys, group in occurrences.groupby(
        [*group_columns, "__target_scope"], sort=True, dropna=False, observed=True
    ):
        date, level, text_hash, target_scope = keys
        group = group.sort_values("source_occurrence_id", kind="mergesort")
        source_tickers = sorted(
            {
                str(value)
                for value in group["source_ticker"].dropna()
                if str(value).strip()
            }
        )
        related = _combine_lists(group["related_tickers"])
        available = _combine_lists(group["available_to_tickers"])
        targets = sorted(
            {
                str(value)
                for value in group["target_ticker"].dropna()
                if str(value).strip()
            }
        )
        if level in {"macro", "sector"}:
            available = sorted(set(all_tickers))
            target_ticker: str | None = None
        elif level == "target":
            target_ticker = str(target_scope)
            available = [target_ticker]
        else:
            target_ticker = targets[0] if len(targets) == 1 else None
            available = sorted(set(available) | set(targets))
        representative = group.iloc[0]
        rows.append(
            {
                "date": pd.Timestamp(date).normalize(),
                "split": str(representative["split"]),
                "news_level": str(level),
                "text": str(representative["text_normalized"]),
                "text_hash": str(text_hash),
                "target_ticker": target_ticker,
                "target_tickers": json.dumps(targets, separators=(",", ":")),
                "source_tickers": json.dumps(source_tickers, separators=(",", ":")),
                "related_tickers": json.dumps(related, separators=(",", ":")),
                "available_to_tickers": json.dumps(available, separators=(",", ":")),
                "source_occurrence_count": int(len(group)),
                "source_occurrence_ids": json.dumps(
                    group["source_occurrence_id"].astype(str).tolist(),
                    separators=(",", ":"),
                ),
                "exact_member_hashes": json.dumps([str(text_hash)], separators=(",", ":")),
                "near_duplicate_merged": False,
            }
        )
    return pd.DataFrame(rows)


def _raw_events(occurrences: pd.DataFrame) -> pd.DataFrame:
    """Represent every source occurrence as an event for no-dedup robustness."""

    rows: list[dict[str, Any]] = []
    for occurrence in occurrences.sort_values(
        ["date", "news_level", "source_occurrence_id"], kind="mergesort"
    ).itertuples(index=False):
        target = (
            None
            if pd.isna(occurrence.target_ticker)
            or not str(occurrence.target_ticker).strip()
            else str(occurrence.target_ticker)
        )
        targets = [] if target is None else [target]
        rows.append(
            {
                "date": pd.Timestamp(occurrence.date).normalize(),
                "split": str(occurrence.split),
                "news_level": str(occurrence.news_level),
                "text": str(occurrence.text_normalized),
                "text_hash": str(occurrence.text_hash),
                "target_ticker": target,
                "target_tickers": json.dumps(targets, separators=(",", ":")),
                "source_tickers": json.dumps(
                    [str(occurrence.source_ticker)], separators=(",", ":")
                ),
                "related_tickers": str(occurrence.related_tickers),
                "available_to_tickers": str(occurrence.available_to_tickers),
                "source_occurrence_count": 1,
                "source_occurrence_ids": json.dumps(
                    [str(occurrence.source_occurrence_id)],
                    separators=(",", ":"),
                ),
                "exact_member_hashes": json.dumps(
                    [str(occurrence.text_hash)], separators=(",", ":")
                ),
                "near_duplicate_merged": False,
            }
        )
    result = pd.DataFrame(rows)
    result.insert(
        0,
        "event_id",
        [
            hashlib.sha256(
                f"raw|{value}".encode("utf-8")
            ).hexdigest()[:24]
            for value in result["source_occurrence_ids"]
        ],
    )
    if result["event_id"].duplicated().any():
        raise AssertionError("Raw occurrence event IDs are not unique.")
    return result.sort_values(
        ["date", "news_level", "event_id"], kind="mergesort"
    ).reset_index(drop=True)


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def _near_scope(row: pd.Series) -> str:
    if row["news_level"] == "target":
        return str(row["target_ticker"])
    return "__shared__"


def _merge_canonical_group(group: pd.DataFrame) -> dict[str, Any]:
    group = group.sort_values(["text_hash", "text"], kind="mergesort")
    representative = group.iloc[0].to_dict()
    for column in (
        "target_tickers",
        "source_tickers",
        "related_tickers",
        "available_to_tickers",
        "source_occurrence_ids",
        "exact_member_hashes",
    ):
        representative[column] = json.dumps(
            _combine_lists(group[column]), separators=(",", ":")
        )
    targets = _json_list(representative["target_tickers"])
    if representative["news_level"] == "related":
        representative["target_ticker"] = targets[0] if len(targets) == 1 else None
    representative["source_occurrence_count"] = int(
        group["source_occurrence_count"].sum()
    )
    representative["near_duplicate_merged"] = bool(len(group) > 1)
    return representative


def _near_deduplicate(
    exact_events: pd.DataFrame, threshold: float, n_features: int
) -> pd.DataFrame:
    """Merge near duplicates only within date, level, split and target scope.

    ``HashingVectorizer`` is stateless, hence it cannot learn vocabulary from
    validation/test.  Groups never cross a chronological split, which keeps
    near-deduplication independent across train, validation and test.
    """

    vectorizer = HashingVectorizer(
        n_features=n_features,
        alternate_sign=False,
        analyzer="char_wb",
        ngram_range=(3, 5),
        norm="l2",
        lowercase=True,
    )
    scoped = exact_events.copy()
    scoped["__near_scope"] = scoped.apply(_near_scope, axis=1)
    merged_rows: list[dict[str, Any]] = []
    group_keys = ["split", "date", "news_level", "__near_scope"]
    for _, group in scoped.groupby(group_keys, sort=True, observed=True):
        group = group.reset_index(drop=True)
        if len(group) == 1:
            merged_rows.append(group.iloc[0].drop(labels="__near_scope").to_dict())
            continue
        matrix: sparse.csr_matrix = vectorizer.transform(group["text"].astype(str))
        similarities = cosine_similarity(matrix, dense_output=True)
        union_find = _UnionFind(len(group))
        left_indices, right_indices = np.where(
            np.triu(similarities >= threshold, k=1)
        )
        for left, right in zip(left_indices.tolist(), right_indices.tolist()):
            union_find.union(left, right)
        components: dict[int, list[int]] = {}
        for index in range(len(group)):
            components.setdefault(union_find.find(index), []).append(index)
        for indices in components.values():
            component = group.iloc[indices].drop(columns="__near_scope")
            merged_rows.append(_merge_canonical_group(component))
    return pd.DataFrame(merged_rows)


def _assign_event_ids(events: pd.DataFrame) -> pd.DataFrame:
    result = events.copy()
    event_ids: list[str] = []
    for row in result.itertuples(index=False):
        scope = (
            getattr(row, "target_ticker")
            if getattr(row, "news_level") == "target"
            else getattr(row, "available_to_tickers")
        )
        signature = (
            f"{pd.Timestamp(getattr(row, 'date')).date()}|"
            f"{getattr(row, 'news_level')}|{scope}|"
            f"{getattr(row, 'exact_member_hashes')}"
        )
        event_ids.append(hashlib.sha256(signature.encode("utf-8")).hexdigest()[:24])
    result.insert(0, "event_id", event_ids)
    if result["event_id"].duplicated().any():
        raise AssertionError("Canonical event IDs are not unique.")
    return result.sort_values(
        ["date", "news_level", "event_id"], kind="mergesort"
    ).reset_index(drop=True)


def _dedup_summary(
    occurrences: pd.DataFrame,
    exact_events: pd.DataFrame,
    final_events: pd.DataFrame,
    near_enabled: bool,
    threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for level in (*NEWS_LEVELS, "all"):
        source = occurrences if level == "all" else occurrences.query("news_level == @level")
        exact = exact_events if level == "all" else exact_events.query("news_level == @level")
        final = final_events if level == "all" else final_events.query("news_level == @level")
        duplicated_across_tickers = (
            source.groupby(["date", "text_hash"], observed=True)["source_ticker"]
            .nunique()
            .gt(1)
            .sum()
        )
        rows.append(
            {
                "news_level": level,
                "source_occurrences": int(len(source)),
                "unique_date_text_hash": int(
                    source[["date", "text_hash"]].drop_duplicates().shape[0]
                ),
                "cross_ticker_duplicate_groups": int(duplicated_across_tickers),
                "events_after_exact_dedup": int(len(exact)),
                "exact_duplicates_removed": int(len(source) - len(exact)),
                "events_without_near_dedup": int(len(exact)),
                "events_with_near_dedup": int(len(final)),
                "near_duplicates_removed": int(len(exact) - len(final)),
                "near_dedup_enabled": bool(near_enabled),
                "near_dedup_cosine_threshold": float(threshold),
                "final_canonical_events": int(len(final)),
                "train_events": int(final["split"].eq("train").sum()),
                "validation_events": int(
                    final["split"].eq("validation").sum()
                ),
                "test_events": int(final["split"].eq("test").sum()),
                "outside_supervised_events": int(
                    final["split"].eq("outside_supervised").sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def run(config: dict) -> dict[str, Path]:
    """Create exact- and optionally near-deduplicated canonical events."""

    seed = int(_nested(config, "project", "seed", default=config.get("seed", 42)))
    set_global_seed(seed)
    ensure_directories(config)
    input_path = _path(
        config,
        (("paths", "news_records"), ("news", "records_path")),
        "data/processed/news_records.parquet",
    )
    occurrences = safe_read_table(input_path)
    required = [
        "source_occurrence_id",
        "date",
        "news_level",
        "text_normalized",
        "text_hash",
        "source_ticker",
        "target_ticker",
        "related_tickers",
        "available_to_tickers",
    ]
    validate_required_columns(occurrences, required, "normalized news occurrences")
    occurrences = occurrences.copy()
    occurrences["date"] = pd.to_datetime(
        occurrences["date"], errors="raise"
    ).dt.normalize()
    unknown_levels = sorted(set(occurrences["news_level"]) - set(NEWS_LEVELS))
    if unknown_levels:
        raise ValueError(f"Unknown news levels: {unknown_levels}")

    split_map = _date_split_map(config)
    occurrences = occurrences.merge(
        split_map, on="date", how="left", validate="many_to_one"
    )
    missing_split = occurrences["split"].isna()
    if missing_split.any():
        # Pre-history used to construct the first 22 lags and the final raw date
        # have no supervised sample.  They remain in the canonical audit/cache,
        # but are never treated as train, validation or test observations.
        occurrences.loc[missing_split, "split"] = "outside_supervised"
        LOGGER.info(
            "Retaining %d news occurrences outside the supervised date span.",
            int(missing_split.sum()),
        )
    all_tickers = tuple(
        str(value).upper()
        for value in _nested(
            config,
            "data",
            "tickers",
            default=("ADI", "AMAT", "AMD", "AVGO", "INTC", "KLAC", "LRCX", "MU", "NVDA", "QCOM", "TXN"),
        )
    )
    exact_events = _exact_canonicalize(occurrences, all_tickers)
    raw_events = _raw_events(occurrences)
    near_enabled = bool(
        _nested(
            config,
            "events",
            "near_duplicate",
            "enabled",
            default=_nested(
                config, "news", "near_duplicate", "enabled", default=False
            ),
        )
    )
    threshold = float(
        _nested(
            config,
            "events",
            "near_duplicate",
            "cosine_threshold",
            default=_nested(
                config,
                "news",
                "near_duplicate",
                "cosine_threshold",
                default=0.94,
            ),
        )
    )
    if not 0.0 < threshold <= 1.0:
        raise ValueError("news.near_duplicate.cosine_threshold must be in (0, 1].")
    n_features = int(
        _nested(
            config,
            "events",
            "near_duplicate",
            "hashing_features",
            default=_nested(
                config,
                "news",
                "near_duplicate",
                "hashing_features",
                default=2**18,
            ),
        )
    )
    final_events = (
        _near_deduplicate(exact_events, threshold, n_features)
        if near_enabled
        else exact_events.copy()
    )
    final_events = _assign_event_ids(final_events)

    canonical_csv = project_path(config, "outputs/tables/canonical_events.csv")
    canonical_parquet = _path(
        config,
        (("paths", "canonical_events"), ("news", "canonical_events_path")),
        "data/processed/canonical_events.parquet",
    )
    exact_path = project_path(config, "data/processed/canonical_events_exact.parquet")
    raw_path = project_path(config, "data/processed/canonical_events_raw.parquet")
    summary_path = project_path(
        config, "outputs/tables/event_deduplication_summary.csv"
    )
    variant_manifest_path = project_path(
        config, "data/processed/event_variant_manifest.csv"
    )
    write_table(_assign_event_ids(exact_events), exact_path)
    write_table(raw_events, raw_path)
    write_table(final_events, canonical_parquet)
    csv_events = final_events.copy()
    csv_events["date"] = csv_events["date"].dt.strftime("%Y-%m-%d")
    atomic_write_csv(csv_events, canonical_csv, index=False)
    summary = _dedup_summary(
        occurrences, exact_events, final_events, near_enabled, threshold
    )
    atomic_write_csv(summary, summary_path, index=False)
    atomic_write_csv(
        pd.DataFrame(
            [
                {
                    "events_variant": "raw",
                    "path": str(raw_path),
                    "event_count": len(raw_events),
                    "near_duplicate_merging": False,
                    "exact_deduplication": False,
                    "selected_default": False,
                },
                {
                    "events_variant": "exact",
                    "path": str(exact_path),
                    "event_count": len(exact_events),
                    "near_duplicate_merging": False,
                    "exact_deduplication": True,
                    "selected_default": False,
                },
                {
                    "events_variant": "canonical",
                    "path": str(canonical_parquet),
                    "event_count": len(final_events),
                    "near_duplicate_merging": near_enabled,
                    "exact_deduplication": True,
                    "selected_default": True,
                },
            ]
        ),
        variant_manifest_path,
        index=False,
    )
    LOGGER.info(
        "Canonicalized %d occurrences into %d events (near dedup=%s).",
        len(occurrences),
        len(final_events),
        near_enabled,
    )
    return {
        "canonical_events": canonical_csv,
        "canonical_events_parquet": canonical_parquet,
        "canonical_events_exact": exact_path,
        "canonical_events_raw": raw_path,
        "event_deduplication_summary": summary_path,
        "event_variant_manifest": variant_manifest_path,
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
