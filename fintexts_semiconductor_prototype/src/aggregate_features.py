"""Aggregate event embeddings and prototypes to leakage-safe ticker-day features."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.utils import (
    atomic_joblib_dump,
    atomic_write_csv,
    ensure_directories,
    get_logger,
    l2_normalize,
    load_config,
    project_path,
    safe_read_table,
    set_global_seed,
    validate_required_columns,
    write_table,
)

LOGGER = get_logger(__name__)
NEWS_LEVELS = ("macro", "sector", "related", "target")
REPRESENTATIONS = tuple(f"R{index}" for index in range(12))


def _events_variant(config: Mapping[str, Any]) -> str:
    variant = str(
        _nested(
            config,
            "events",
            "variant",
            default=_nested(config, "events_variant", default="canonical"),
        )
    ).lower()
    if variant not in {"canonical", "near", "exact", "raw"}:
        raise ValueError("events.variant must be canonical, near, exact, or raw.")
    return variant


def _variant_suffix(variant: str) -> str:
    return "" if variant == "canonical" else f"_{variant}"


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
        return [str(item) for item in value]
    if isinstance(value, (tuple, set, np.ndarray)):
        return [str(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [part.strip() for part in text.split(",") if part.strip()]
    if isinstance(parsed, str):
        return [parsed]
    if isinstance(parsed, Sequence):
        return [str(item) for item in parsed]
    return []


def _load_data(
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, pd.DataFrame]:
    events_variant = _events_variant(config)
    suffix = _variant_suffix(events_variant)
    events_default = (
        f"data/processed/canonical_events_{events_variant}.parquet"
        if events_variant in {"exact", "raw"}
        else "data/processed/canonical_events.parquet"
    )
    market_path = _path(
        config,
        (("paths", "market_supervised"),),
        "data/processed/market_supervised.parquet",
    )
    events_path_candidates = (
        (("paths", f"canonical_events_{events_variant}"),)
        if events_variant in {"exact", "raw"}
        else (
            (("paths", f"canonical_events_{events_variant}"), ("paths", "canonical_events"))
        )
    )
    events_path = _path(
        config,
        events_path_candidates,
        events_default,
    )
    if not events_path.exists() and events_variant not in {"exact", "raw"}:
        events_path = project_path(config, "outputs/tables/canonical_events.csv")
    embedding_path = _path(
        config,
        (("paths", f"event_embeddings_{events_variant}"),),
        f"data/embeddings/event_embeddings{suffix}.npy",
    )
    embedding_metadata_path = _path(
        config,
        (("paths", f"event_embedding_metadata_{events_variant}"),),
        f"data/embeddings/event_embedding_metadata{suffix}.csv",
    )
    assignment_path = _path(
        config,
        (("paths", f"prototype_assignments_{events_variant}"),),
        f"data/processed/prototype_assignments{suffix}.parquet",
    )
    market = safe_read_table(market_path)
    events = safe_read_table(events_path)
    metadata = safe_read_table(embedding_metadata_path)
    assignments = safe_read_table(assignment_path)
    validate_required_columns(
        market, ["ticker", "feature_date", "split"], "market supervised data"
    )
    validate_required_columns(
        events,
        [
            "event_id",
            "date",
            "split",
            "news_level",
            "available_to_tickers",
        ],
        "canonical events",
    )
    validate_required_columns(
        metadata, ["event_id", "embedding_index"], "embedding metadata"
    )
    validate_required_columns(
        assignments,
        [
            "event_id",
            "news_level",
            "k",
            "hard_cluster_id",
            "soft_assignment",
            "assignment_entropy",
            "novelty",
            "nearest_distance",
        ],
        "selected prototype assignments",
    )
    if market.duplicated(["ticker", "feature_date"]).any():
        raise ValueError("market_supervised must be unique by ticker and feature_date.")
    if events["event_id"].duplicated().any():
        raise ValueError("canonical events must have unique event_id.")
    if assignments["event_id"].duplicated().any():
        raise ValueError(
            "Selected prototype assignments must have one row per event_id."
        )
    embeddings_file = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
    if embeddings_file.ndim != 2:
        raise ValueError(f"Expected 2-D event embeddings, got {embeddings_file.shape}.")
    events = events.merge(
        metadata[["event_id", "embedding_index"]],
        on="event_id",
        how="left",
        validate="one_to_one",
    ).merge(
        assignments[
            [
                "event_id",
                "candidate_id",
                "k",
                "hard_cluster_id",
                "soft_assignment",
                "assignment_entropy",
                "novelty",
                "nearest_distance",
                "nearest_similarity",
                "effective_soft_prototypes",
            ]
        ],
        on="event_id",
        how="left",
        validate="one_to_one",
    )
    if events["embedding_index"].isna().any():
        raise ValueError("Some canonical events have no embedding metadata.")
    indices = events["embedding_index"].to_numpy(dtype=int)
    if indices.min(initial=0) < 0 or indices.max(initial=-1) >= len(embeddings_file):
        raise ValueError("Embedding index out of bounds.")
    embeddings = np.asarray(embeddings_file[indices], dtype=np.float64)
    embeddings = l2_normalize(embeddings)
    market = market.copy()
    market["feature_date"] = pd.to_datetime(
        market["feature_date"], errors="raise"
    ).dt.normalize()
    market["ticker"] = market["ticker"].astype(str).str.upper()
    market["split"] = market["split"].astype(str).str.lower()
    market = market.sort_values(
        ["feature_date", "ticker"], kind="mergesort"
    ).reset_index(drop=True)
    market.insert(0, "__row_id", np.arange(len(market), dtype=np.int64))
    events["date"] = pd.to_datetime(events["date"], errors="raise").dt.normalize()
    events["news_level"] = events["news_level"].astype(str).str.lower()
    events.insert(0, "event_pos", np.arange(len(events), dtype=np.int64))
    return market, events, embeddings, assignments


def _soft_matrices(
    events: pd.DataFrame,
    train_mask: np.ndarray,
    seed: int,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, int],
]:
    soft: dict[str, np.ndarray] = {}
    hard: dict[str, np.ndarray] = {}
    random_placebo: dict[str, np.ndarray] = {}
    dimensions: dict[str, int] = {}
    rng = np.random.default_rng(seed + 991)
    for level in NEWS_LEVELS:
        level_mask = events["news_level"].to_numpy() == level
        assigned = level_mask & events["k"].notna().to_numpy()
        if not assigned.any():
            LOGGER.warning("No selected prototype assignments for level %s.", level)
            dimensions[level] = 0
            soft[level] = np.zeros((len(events), 0), dtype=np.float32)
            hard[level] = np.zeros((len(events), 0), dtype=np.float32)
            random_placebo[level] = np.zeros((len(events), 0), dtype=np.float32)
            continue
        k_values = events.loc[assigned, "k"].astype(int).unique()
        if len(k_values) != 1:
            raise ValueError(
                f"Expected one selected k for level {level}, found {k_values.tolist()}."
            )
        k = int(k_values[0])
        dimensions[level] = k
        soft_matrix = np.zeros((len(events), k), dtype=np.float32)
        hard_matrix = np.zeros((len(events), k), dtype=np.float32)
        for index, row in events.loc[assigned].iterrows():
            vector = np.asarray(_json_list(row["soft_assignment"]), dtype=float)
            if vector.shape != (k,):
                raise ValueError(
                    f"Event {row['event_id']} has soft assignment shape {vector.shape}, "
                    f"expected {(k,)}."
                )
            soft_matrix[index] = vector
            cluster_id = int(row["hard_cluster_id"])
            if not 0 <= cluster_id < k:
                raise ValueError(
                    f"Event {row['event_id']} has invalid hard cluster {cluster_id}."
                )
            hard_matrix[index, cluster_id] = 1.0
        training_labels = events.loc[
            assigned & train_mask, "hard_cluster_id"
        ].astype(int)
        if training_labels.empty:
            raise ValueError(f"No train assignments available for level {level}.")
        probabilities = (
            np.bincount(training_labels, minlength=k).astype(float)
            / len(training_labels)
        )
        placebo_matrix = np.zeros((len(events), k), dtype=np.float32)
        placebo_labels = rng.choice(
            np.arange(k), size=int(level_mask.sum()), p=probabilities
        )
        placebo_matrix[
            np.flatnonzero(level_mask), placebo_labels
        ] = 1.0
        soft[level] = soft_matrix
        hard[level] = hard_matrix
        random_placebo[level] = placebo_matrix
    return soft, hard, random_placebo, dimensions


def _eligible_grid_variants(
    config: Mapping[str, Any],
    events: pd.DataFrame,
    primary_seed: int,
) -> list[
    tuple[
        str,
        dict[str, str],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        pd.DataFrame,
        dict[str, Any],
    ]
]:
    """Load aligned all-level candidates for validation-time configuration choice."""

    enabled = bool(
        _nested(
            config,
            "prototype",
            "aggregate_all_eligible_candidates",
            default=True,
        )
    )
    if not enabled:
        return []
    manifest_path = project_path(
        config,
        f"data/processed/prototype_manifest{_variant_suffix(_events_variant(config))}.csv",
    )
    manifest = safe_read_table(manifest_path)
    validate_required_columns(
        manifest,
        [
            "candidate_id",
            "news_level",
            "seed",
            "pca_dim",
            "k",
            "temperature",
            "eligible",
            "assignment_path",
        ],
        "prototype candidate manifest",
    )
    aggregate_all_seeds = bool(
        _nested(
            config,
            "prototype",
            "aggregate_all_prototype_seeds",
            default=True,
        )
    )
    eligible_mask = manifest["eligible"].astype(bool)
    if not aggregate_all_seeds:
        eligible_mask &= manifest["seed"].astype(int).eq(primary_seed)
    eligible = manifest.loc[eligible_mask].copy()
    if eligible.empty:
        return []
    maximum = _nested(
        config, "prototype", "maximum_aggregate_candidates", default=None
    )
    variants: list[
        tuple[
            str,
            dict[str, str],
            dict[str, np.ndarray],
            dict[str, np.ndarray],
            pd.DataFrame,
            dict[str, Any],
        ]
    ] = []
    grouped = eligible.groupby(
        ["k", "pca_dim", "temperature", "seed"],
        dropna=False,
        sort=True,
        observed=True,
    )
    for (k_value, pca_value, temperature_value, seed_value), group in grouped:
        by_level = {
            str(row.news_level): row for row in group.itertuples(index=False)
        }
        if not set(NEWS_LEVELS).issubset(by_level):
            continue
        k = int(k_value)
        pca_token = "none" if pd.isna(pca_value) else str(int(pca_value))
        temperature = float(temperature_value)
        variant_id = (
            f"grid_k{k}_pca{pca_token}_tau"
            f"{str(f'{temperature:.6g}').replace('.', 'p')}_seed{int(seed_value)}"
        )
        variant_family = (
            f"grid_k{k}_pca{pca_token}_tau"
            f"{str(f'{temperature:.6g}').replace('.', 'p')}"
        )
        soft: dict[str, np.ndarray] = {}
        hard: dict[str, np.ndarray] = {}
        variant_events = events.copy()
        for diagnostic in (
            "assignment_entropy",
            "novelty",
            "nearest_distance",
            "nearest_similarity",
            "effective_soft_prototypes",
        ):
            variant_events[diagnostic] = np.nan
        candidate_ids: dict[str, str] = {}
        for level in NEWS_LEVELS:
            row = by_level[level]
            candidate_ids[level] = str(row.candidate_id)
            arrays = np.load(Path(str(row.assignment_path)), allow_pickle=False)
            event_indices = arrays["event_indices"].astype(int)
            if (
                event_indices.min(initial=0) < 0
                or event_indices.max(initial=-1) >= len(events)
            ):
                raise ValueError(
                    f"Candidate {row.candidate_id} contains invalid event indices."
                )
            if not events.iloc[event_indices]["news_level"].eq(level).all():
                raise ValueError(
                    f"Candidate {row.candidate_id} event order no longer matches "
                    "canonical_events; rebuild prototypes."
                )
            if "event_ids" not in arrays.files or not np.array_equal(
                arrays["event_ids"].astype(str),
                events.iloc[event_indices]["event_id"].astype(str).to_numpy(),
            ):
                raise ValueError(
                    f"Candidate {row.candidate_id} event IDs do not match "
                    "canonical_events; rebuild prototypes."
                )
            soft_matrix = np.zeros((len(events), k), dtype=np.float32)
            hard_matrix = np.zeros((len(events), k), dtype=np.float32)
            candidate_soft = arrays["soft_assignment"].astype(np.float32)
            if candidate_soft.shape != (len(event_indices), k):
                raise ValueError(
                    f"Candidate {row.candidate_id} assignment shape "
                    f"{candidate_soft.shape}, expected {(len(event_indices), k)}."
                )
            soft_matrix[event_indices] = candidate_soft
            labels = arrays["hard_cluster_id"].astype(int)
            hard_matrix[event_indices, labels] = 1.0
            soft[level] = soft_matrix
            hard[level] = hard_matrix
            for diagnostic in (
                "assignment_entropy",
                "novelty",
                "nearest_distance",
                "nearest_similarity",
                "effective_soft_prototypes",
            ):
                variant_events.loc[event_indices, diagnostic] = arrays[
                    diagnostic
                ].astype(float)
        details = {
            "k": k,
            "pca_dim": None if pd.isna(pca_value) else int(pca_value),
            "temperature": temperature,
            "seed": int(seed_value),
            "prototype_seed": int(seed_value),
            "representation_variant_family": variant_family,
        }
        variants.append(
            (
                variant_id,
                candidate_ids,
                soft,
                hard,
                variant_events,
                details,
            )
        )
        if maximum is not None and len(variants) >= int(maximum):
            break
    return variants


def _fit_embedding_comparators(
    config: Mapping[str, Any],
    events: pd.DataFrame,
    embeddings: np.ndarray,
    prototype_dims: Mapping[str, int],
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    pca_values: dict[str, np.ndarray] = {}
    random_values: dict[str, np.ndarray] = {}
    bundles: dict[str, Any] = {}
    for level in NEWS_LEVELS:
        level_mask = events["news_level"].to_numpy() == level
        train_mask = level_mask & (events["split"].to_numpy() == "train")
        requested = int(
            prototype_dims.get(level, 0)
            or _nested(config, "prototype", "random_projection_dim", default=64)
        )
        if not level_mask.any() or not train_mask.any():
            pca_values[level] = np.zeros((len(events), 0), dtype=np.float32)
            random_values[level] = np.zeros((len(events), 0), dtype=np.float32)
            bundles[level] = {
                "pca": None,
                "random_projection": None,
                "output_dimension": 0,
                "fit_split": "train",
            }
            continue
        output_dim = min(
            requested,
            int(train_mask.sum()),
            int(embeddings.shape[1]),
        )
        if output_dim < 1:
            raise ValueError(f"No valid comparator dimension for level {level}.")
        solver = (
            "randomized"
            if output_dim < min(int(train_mask.sum()), embeddings.shape[1])
            else "full"
        )
        pca = PCA(n_components=output_dim, svd_solver=solver, random_state=seed)
        pca.fit(embeddings[train_mask])
        pca_level = l2_normalize(pca.transform(embeddings[level_mask]))
        pca_matrix = np.zeros((len(events), output_dim), dtype=np.float32)
        pca_matrix[level_mask] = pca_level.astype(np.float32)

        level_seed = seed + sum(ord(character) for character in level) * 101
        rng = np.random.default_rng(level_seed)
        projection = rng.normal(
            loc=0.0,
            scale=1.0 / math_sqrt(output_dim),
            size=(embeddings.shape[1], output_dim),
        )
        projected = l2_normalize(embeddings[level_mask] @ projection)
        random_matrix = np.zeros((len(events), output_dim), dtype=np.float32)
        random_matrix[level_mask] = projected.astype(np.float32)
        pca_values[level] = pca_matrix
        random_values[level] = random_matrix
        bundles[level] = {
            "pca": pca,
            "random_projection": projection.astype(np.float32),
            "output_dimension": output_dim,
            "requested_dimension": requested,
            "fit_split": "train",
        }
    return pca_values, random_values, bundles


def _random_prototype_placebo(
    events: pd.DataFrame,
    hard: Mapping[str, np.ndarray],
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed + 991)
    result: dict[str, np.ndarray] = {}
    train_mask = events["split"].to_numpy() == "train"
    for level in NEWS_LEVELS:
        matrix = hard[level]
        if matrix.shape[1] == 0:
            result[level] = matrix.copy()
            continue
        level_mask = events["news_level"].to_numpy() == level
        train_level = level_mask & train_mask
        labels = np.argmax(matrix[train_level], axis=1)
        probabilities = (
            np.bincount(labels, minlength=matrix.shape[1]).astype(float)
            / max(len(labels), 1)
        )
        placebo = np.zeros_like(matrix, dtype=np.float32)
        sampled = rng.choice(
            np.arange(matrix.shape[1]),
            size=int(level_mask.sum()),
            p=probabilities,
        )
        placebo[np.flatnonzero(level_mask), sampled] = 1.0
        result[level] = placebo
    return result


def math_sqrt(value: int | float) -> float:
    return float(np.sqrt(float(value)))


def _build_edges(
    events: pd.DataFrame,
    tickers: Sequence[str],
    mode: str,
    market: pd.DataFrame,
    seed: int,
    shuffle_holdout_dates: bool = False,
) -> pd.DataFrame:
    ticker_set = set(tickers)
    ticker_mapping: dict[str, str] | None = None
    date_mapping: dict[tuple[str, pd.Timestamp], pd.Timestamp] | None = None
    if mode == "shuffled_ticker":
        rng = np.random.default_rng(seed + 1709)
        if len(tickers) < 2:
            raise ValueError("Shuffled-ticker placebo requires at least two tickers.")
        offset = int(rng.integers(1, len(tickers)))
        # A non-zero cyclic rotation is a guaranteed derangement.
        permuted = list(tickers[offset:]) + list(tickers[:offset])
        ticker_mapping = dict(zip(tickers, permuted))
    elif mode == "shuffled_date":
        date_mapping = {}
        train_dates = (
            market.loc[market["split"].eq("train"), "feature_date"]
            .drop_duplicates()
            .sort_values()
            .tolist()
        )
        if len(train_dates) < 2:
            raise ValueError("Shuffled-date placebo requires at least two train dates.")
        rng = np.random.default_rng(seed + 1213)
        base_offset = int(rng.integers(1, len(train_dates)))
        for split, split_market in market.groupby("split", observed=True):
            if str(split) != "train" and not shuffle_holdout_dates:
                continue
            dates = (
                split_market["feature_date"].drop_duplicates().sort_values().tolist()
            )
            if len(dates) < 2:
                for date in dates:
                    date_mapping[(str(split), pd.Timestamp(date))] = pd.Timestamp(date)
                continue
            offset = base_offset % len(dates)
            if offset == 0:
                offset = 1
            for index, date in enumerate(dates):
                date_mapping[(str(split), pd.Timestamp(date))] = pd.Timestamp(
                    dates[(index + offset) % len(dates)]
                )
    rows: list[dict[str, Any]] = []
    for event in events.itertuples(index=False):
        available = [
            ticker.upper()
            for ticker in _json_list(event.available_to_tickers)
            if ticker.upper() in ticker_set
        ]
        if mode == "shuffled_ticker" and event.news_level in {"target", "related"}:
            assert ticker_mapping is not None
            available = [ticker_mapping[ticker] for ticker in available]
        event_date = pd.Timestamp(event.date)
        if mode == "shuffled_date":
            assert date_mapping is not None
            event_date = date_mapping.get((str(event.split), event_date), event_date)
        for ticker in sorted(set(available)):
            rows.append(
                {
                    "event_pos": int(event.event_pos),
                    "event_id": event.event_id,
                    "news_level": event.news_level,
                    "ticker": ticker,
                    "event_date": event_date,
                    "event_split": event.split,
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "event_pos",
            "event_id",
            "news_level",
            "ticker",
            "event_date",
            "event_split",
        ],
    )


def _same_day_instances(
    market: pd.DataFrame, edges: pd.DataFrame
) -> pd.DataFrame:
    keys = market[["__row_id", "ticker", "feature_date"]]
    result = edges.merge(
        keys,
        left_on=["ticker", "event_date"],
        right_on=["ticker", "feature_date"],
        how="inner",
        validate="many_to_one",
    )
    result["lag_days"] = 0.0
    result["weight"] = 1.0
    return result


def _decayed_instances(
    market: pd.DataFrame,
    edges: pd.DataFrame,
    half_life_days: float,
    max_lag_days: int,
) -> pd.DataFrame:
    if half_life_days <= 0 or max_lag_days < 0:
        raise ValueError("Exponential half-life must be positive and max lag non-negative.")
    calendars = {
        ticker: group[["__row_id", "feature_date"]]
        .sort_values("feature_date")
        .reset_index(drop=True)
        for ticker, group in market.groupby("ticker", observed=True)
    }
    rows: list[dict[str, Any]] = []
    for edge in edges.itertuples(index=False):
        calendar = calendars.get(edge.ticker)
        if calendar is None:
            continue
        dates = calendar["feature_date"].to_numpy(dtype="datetime64[ns]")
        event_date = np.datetime64(pd.Timestamp(edge.event_date), "ns")
        start = int(np.searchsorted(dates, event_date, side="left"))
        for position in range(start, len(calendar)):
            feature_date = pd.Timestamp(calendar.iloc[position]["feature_date"])
            lag = int((feature_date - pd.Timestamp(edge.event_date)).days)
            if lag > max_lag_days:
                break
            if lag < 0:
                raise AssertionError("A future event entered exponential-decay pooling.")
            rows.append(
                {
                    "event_pos": int(edge.event_pos),
                    "event_id": edge.event_id,
                    "news_level": edge.news_level,
                    "ticker": edge.ticker,
                    "event_date": edge.event_date,
                    "__row_id": int(calendar.iloc[position]["__row_id"]),
                    "feature_date": feature_date,
                    "lag_days": float(lag),
                    "weight": float(np.exp(-np.log(2.0) * lag / half_life_days)),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "event_pos",
            "event_id",
            "news_level",
            "ticker",
            "event_date",
            "__row_id",
            "feature_date",
            "lag_days",
            "weight",
        ],
    )


def _instances(
    market: pd.DataFrame,
    edges: pd.DataFrame,
    pooling: str,
    half_life_days: float,
    max_lag_days: int,
) -> pd.DataFrame:
    if pooling == "exponential_decay":
        return _decayed_instances(
            market, edges, half_life_days, max_lag_days
        )
    if pooling not in {"mean", "normalized_sum", "max"}:
        raise ValueError(f"Unsupported pooling method: {pooling}")
    return _same_day_instances(market, edges)


def _aggregate_matrix(
    instances: pd.DataFrame,
    matrix: np.ndarray,
    level: str,
    n_rows: int,
    pooling: str,
    prefix: str,
) -> pd.DataFrame:
    dimension = matrix.shape[1]
    if dimension == 0:
        return pd.DataFrame(index=np.arange(n_rows))
    selected = instances.loc[instances["news_level"].eq(level)]
    result = np.zeros((n_rows, dimension), dtype=np.float32)
    if not selected.empty:
        row_ids = selected["__row_id"].to_numpy(dtype=int)
        event_ids = selected["event_pos"].to_numpy(dtype=int)
        weights = selected["weight"].to_numpy(dtype=float)
        values = matrix[event_ids]
        if pooling == "max":
            result.fill(-np.inf)
            np.maximum.at(result, row_ids, values * weights[:, None])
            result[~np.isfinite(result)] = 0.0
        else:
            np.add.at(result, row_ids, values * weights[:, None])
            denominator = np.zeros(n_rows, dtype=float)
            if pooling == "normalized_sum":
                np.add.at(denominator, row_ids, 1.0)
                denominator = np.sqrt(denominator)
            else:
                np.add.at(denominator, row_ids, weights)
            nonzero = denominator > 0
            result[nonzero] /= denominator[nonzero, None]
    columns = [f"{prefix}__{level}__{index:04d}" for index in range(dimension)]
    return pd.DataFrame(result, columns=columns)


def _days_since(
    market: pd.DataFrame,
    edges: pd.DataFrame,
    level: str,
    sentinel: float,
) -> tuple[np.ndarray, np.ndarray]:
    result = np.full(len(market), sentinel, dtype=float)
    has_prior = np.zeros(len(market), dtype=float)
    level_edges = edges.loc[edges["news_level"].eq(level)]
    by_ticker = {
        ticker: np.sort(group["event_date"].drop_duplicates().to_numpy(dtype="datetime64[ns]"))
        for ticker, group in level_edges.groupby("ticker", observed=True)
    }
    for ticker, group in market.groupby("ticker", observed=True):
        news_dates = by_ticker.get(ticker)
        if news_dates is None or not len(news_dates):
            continue
        market_dates = group["feature_date"].to_numpy(dtype="datetime64[ns]")
        positions = np.searchsorted(news_dates, market_dates, side="right") - 1
        valid = positions >= 0
        row_ids = group["__row_id"].to_numpy(dtype=int)
        if valid.any():
            deltas = (
                market_dates[valid] - news_dates[positions[valid]]
            ).astype("timedelta64[D]").astype(float)
            if (deltas < 0).any():
                raise AssertionError("days_since used future news.")
            result[row_ids[valid]] = deltas
            has_prior[row_ids[valid]] = 1.0
    return result, has_prior


def _aggregate_metadata(
    market: pd.DataFrame,
    events: pd.DataFrame,
    edges: pd.DataFrame,
    instances: pd.DataFrame,
    level: str,
    sentinel: float,
) -> pd.DataFrame:
    n_rows = len(market)
    selected = instances.loc[instances["news_level"].eq(level)]
    count = np.zeros(n_rows, dtype=float)
    canonical_count = np.zeros(n_rows, dtype=float)
    entropy_sum = np.zeros(n_rows, dtype=float)
    novelty_sum = np.zeros(n_rows, dtype=float)
    distance_sum = np.zeros(n_rows, dtype=float)
    novelty_max = np.zeros(n_rows, dtype=float)
    distance_min = np.full(n_rows, np.inf, dtype=float)
    lag_sum = np.zeros(n_rows, dtype=float)
    lag_max = np.zeros(n_rows, dtype=float)
    diagnostic_weight = np.zeros(n_rows, dtype=float)
    if not selected.empty:
        rows = selected["__row_id"].to_numpy(dtype=int)
        event_positions = selected["event_pos"].to_numpy(dtype=int)
        weights = selected["weight"].to_numpy(dtype=float)
        lags = selected["lag_days"].to_numpy(dtype=float)
        occurrence_counts = pd.to_numeric(
            events.iloc[event_positions].get(
                "source_occurrence_count",
                pd.Series(1.0, index=np.arange(len(event_positions))),
            ),
            errors="coerce",
        ).fillna(1.0).to_numpy(dtype=float)
        np.add.at(count, rows, occurrence_counts)
        np.add.at(lag_sum, rows, lags * weights)
        np.maximum.at(lag_max, rows, lags)
        unique = selected.drop_duplicates(["__row_id", "event_id"])
        np.add.at(
            canonical_count,
            unique["__row_id"].to_numpy(dtype=int),
            1.0,
        )
        entropy = pd.to_numeric(
            events.iloc[event_positions]["assignment_entropy"], errors="coerce"
        ).to_numpy(dtype=float)
        novelty = pd.to_numeric(
            events.iloc[event_positions]["novelty"], errors="coerce"
        ).to_numpy(dtype=float)
        distance = pd.to_numeric(
            events.iloc[event_positions]["nearest_distance"], errors="coerce"
        ).to_numpy(dtype=float)
        valid = np.isfinite(entropy) & np.isfinite(novelty) & np.isfinite(distance)
        if valid.any():
            valid_rows = rows[valid]
            valid_weights = weights[valid]
            np.add.at(entropy_sum, valid_rows, entropy[valid] * valid_weights)
            np.add.at(novelty_sum, valid_rows, novelty[valid] * valid_weights)
            np.add.at(distance_sum, valid_rows, distance[valid] * valid_weights)
            np.add.at(diagnostic_weight, valid_rows, valid_weights)
            np.maximum.at(novelty_max, valid_rows, novelty[valid])
            np.minimum.at(distance_min, valid_rows, distance[valid])
    divisor = np.clip(diagnostic_weight, 1.0e-12, None)
    distance_min[~np.isfinite(distance_min)] = 0.0
    days_since, has_prior = _days_since(market, edges, level, sentinel)
    prefix = f"meta__{level}"
    return pd.DataFrame(
        {
            f"{prefix}__news_count": count,
            f"{prefix}__canonical_event_count": canonical_count,
            f"{prefix}__mean_entropy": entropy_sum / divisor,
            f"{prefix}__mean_novelty": novelty_sum / divisor,
            f"{prefix}__max_novelty": novelty_max,
            f"{prefix}__mean_nearest_distance": distance_sum / divisor,
            f"{prefix}__min_nearest_distance": distance_min,
            f"{prefix}__mean_event_lag_days": lag_sum
            / np.clip(
                selected.groupby("__row_id", observed=True)["weight"]
                .sum()
                .reindex(np.arange(n_rows), fill_value=0.0)
                .to_numpy(dtype=float),
                1.0e-12,
                None,
            )
            if not selected.empty
            else lag_sum,
            f"{prefix}__max_event_lag_days": lag_max,
            f"{prefix}__no_news_mask": (count == 0).astype(float),
            f"{prefix}__days_since_last_news": days_since,
            f"{prefix}__has_prior_news": has_prior,
        }
    )


def _join_feature_blocks(
    market: pd.DataFrame, blocks: Sequence[pd.DataFrame]
) -> pd.DataFrame:
    keys = market[["ticker", "feature_date", "split"]].reset_index(drop=True)
    nonempty = [block.reset_index(drop=True) for block in blocks if block.shape[1]]
    result = pd.concat([keys, *nonempty], axis=1)
    feature_columns = [
        column for column in result if column not in {"ticker", "feature_date", "split"}
    ]
    if feature_columns:
        result[feature_columns] = (
            result[feature_columns]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(np.float32)
        )
    if result.duplicated(["ticker", "feature_date"]).any():
        raise AssertionError("Aggregated features are not unique by ticker-date.")
    return result


def _features_for_pool(
    market: pd.DataFrame,
    events: pd.DataFrame,
    embeddings: np.ndarray,
    edges: pd.DataFrame,
    pooling: str,
    matrices: Mapping[str, Mapping[str, np.ndarray]],
    half_life_days: float,
    max_lag_days: int,
    days_since_sentinel: float,
) -> dict[str, pd.DataFrame]:
    instances = _instances(
        market, edges, pooling, half_life_days, max_lag_days
    )
    metadata_blocks = [
        _aggregate_metadata(
            market, events, edges, instances, level, days_since_sentinel
        )
        for level in NEWS_LEVELS
    ]

    def vector_blocks(kind: str, prefix: str) -> list[pd.DataFrame]:
        return [
            _aggregate_matrix(
                instances,
                matrices[kind][level],
                level,
                len(market),
                pooling,
                prefix,
            )
            for level in NEWS_LEVELS
        ]

    raw_matrices = {
        level: embeddings
        for level in NEWS_LEVELS
    }
    combined_matrices: dict[str, Mapping[str, np.ndarray]] = dict(matrices)
    combined_matrices["raw"] = raw_matrices
    basic_metadata_blocks = [
        block[
            [
                column
                for column in block
                if not any(
                    token in column
                    for token in (
                        "mean_entropy",
                        "mean_nearest_distance",
                        "min_nearest_distance",
                    )
                )
            ]
        ]
        for block in metadata_blocks
    ]
    outputs: dict[str, pd.DataFrame] = {}
    outputs["R1"] = _join_feature_blocks(market, basic_metadata_blocks)
    outputs["R2"] = _join_feature_blocks(
        market, vector_blocks_from(combined_matrices, "raw", "raw", instances, pooling, len(market))
    )
    outputs["R3"] = _join_feature_blocks(
        market, vector_blocks("pca", "pca")
    )
    outputs["R4"] = _join_feature_blocks(
        market, vector_blocks("random_projection", "randproj")
    )
    outputs["R5"] = _join_feature_blocks(
        market, vector_blocks("hard", "hardproto")
    )
    outputs["R6"] = _join_feature_blocks(
        market, vector_blocks("soft", "softproto")
    )
    outputs["R7"] = _join_feature_blocks(
        market, [*vector_blocks("soft", "softproto"), *metadata_blocks]
    )
    outputs["R8"] = _join_feature_blocks(
        market,
        [
            *vector_blocks("pca", "pca"),
            *vector_blocks("soft", "softproto"),
            *metadata_blocks,
        ],
    )
    outputs["R9"] = _join_feature_blocks(
        market,
        vector_blocks("random_prototype", "randomproto"),
    )
    return outputs


def _prototype_features_for_pool(
    market: pd.DataFrame,
    events: pd.DataFrame,
    edges: pd.DataFrame,
    pooling: str,
    soft: Mapping[str, np.ndarray],
    hard: Mapping[str, np.ndarray],
    half_life_days: float,
    max_lag_days: int,
    days_since_sentinel: float,
) -> dict[str, pd.DataFrame]:
    instances = _instances(
        market, edges, pooling, half_life_days, max_lag_days
    )
    metadata = [
        _aggregate_metadata(
            market,
            events,
            edges,
            instances,
            level,
            days_since_sentinel,
        )
        for level in NEWS_LEVELS
    ]
    hard_blocks = [
        _aggregate_matrix(
            instances,
            hard[level],
            level,
            len(market),
            pooling,
            "hardproto",
        )
        for level in NEWS_LEVELS
    ]
    soft_blocks = [
        _aggregate_matrix(
            instances,
            soft[level],
            level,
            len(market),
            pooling,
            "softproto",
        )
        for level in NEWS_LEVELS
    ]
    return {
        "R5": _join_feature_blocks(market, hard_blocks),
        "R6": _join_feature_blocks(market, soft_blocks),
        "R7": _join_feature_blocks(market, [*soft_blocks, *metadata]),
    }


def vector_blocks_from(
    matrices: Mapping[str, Mapping[str, np.ndarray]],
    kind: str,
    prefix: str,
    instances: pd.DataFrame,
    pooling: str,
    n_rows: int,
) -> list[pd.DataFrame]:
    return [
        _aggregate_matrix(
            instances,
            matrices[kind][level],
            level,
            n_rows,
            pooling,
            prefix,
        )
        for level in NEWS_LEVELS
    ]


def _write_representation(
    config: Mapping[str, Any],
    representation: str,
    pooling: str,
    frame: pd.DataFrame,
    variant: str | None = None,
) -> Path:
    events_variant = _events_variant(config)
    events_token = (
        "" if events_variant == "canonical" else f"__events{events_variant}"
    )
    variant_token = "" if variant is None else f"__{variant}"
    path = project_path(
        config,
        "data",
        "processed",
        (
            f"features_{representation}_{pooling}"
            f"{events_token}{variant_token}.parquet"
        ),
    )
    write_table(frame, path)
    return path


def aggregate_fold_features(
    config: dict,
    fold_id: int,
    representation_variant_family: str,
    prototype_seed: int | None = None,
    pooling: str | None = None,
    representations: Sequence[str] | None = None,
) -> dict[str, Path]:
    """Materialize fold-safe representations from one fold-train codebook.

    The caller supplies a hyperparameter *family* chosen without looking at
    test. Prototype seed remains a replication dimension, not a selectable
    hyperparameter. PCA, random projection, random-prototype and shuffled-news
    comparators are rebuilt inside the same fold, so no main-train transform is
    reused for an earlier validation window.
    """

    seed = int(
        prototype_seed
        if prototype_seed is not None
        else _nested(config, "project", "seed", default=42)
    )
    set_global_seed(seed)
    events_variant = _events_variant(config)
    suffix = _variant_suffix(events_variant)
    fold_manifest_path = project_path(
        config, f"data/processed/prototype_fold_manifest{suffix}.csv"
    )
    fold_manifest = safe_read_table(fold_manifest_path)
    validate_required_columns(
        fold_manifest,
        [
            "fold_id",
            "representation_variant_family",
            "news_level",
            "prototype_seed",
            "eligible",
            "assignment_path",
            "train_start",
            "train_end",
            "validation_start",
            "validation_end",
        ],
        "prototype fold manifest",
    )
    selected = fold_manifest.loc[
        fold_manifest["fold_id"].astype(int).eq(int(fold_id))
        & fold_manifest["representation_variant_family"].astype(str).eq(
            representation_variant_family
        )
        & fold_manifest["prototype_seed"].astype(int).eq(seed)
        & fold_manifest["eligible"].astype(bool)
    ].copy()
    by_level = {
        str(row.news_level): row for row in selected.itertuples(index=False)
    }
    missing_levels = sorted(set(NEWS_LEVELS).difference(by_level))
    if missing_levels:
        raise ValueError(
            f"Fold {fold_id} family {representation_variant_family!r} seed {seed} "
            f"is not eligible for levels: {missing_levels}"
        )
    market, events, embeddings, _ = _load_data(config)
    market["target_date"] = pd.to_datetime(
        market["target_date"], errors="raise"
    ).dt.normalize()
    reference = next(iter(by_level.values()))
    train_start = pd.Timestamp(reference.train_start).normalize()
    train_end = pd.Timestamp(reference.train_end).normalize()
    validation_start = pd.Timestamp(reference.validation_start).normalize()
    validation_end = pd.Timestamp(reference.validation_end).normalize()
    if not train_end < validation_start:
        raise AssertionError("Fold feature aggregation received overlapping ranges.")
    fold_train_mask = market["target_date"].between(
        train_start, train_end, inclusive="both"
    )
    fold_validation_mask = market["target_date"].between(
        validation_start, validation_end, inclusive="both"
    )
    fold_market = market.loc[fold_train_mask | fold_validation_mask].copy()
    fold_market["split"] = np.where(
        fold_market["target_date"].le(train_end),
        "train",
        "validation",
    )
    fold_market = fold_market.drop(columns="__row_id").sort_values(
        ["feature_date", "ticker"], kind="mergesort"
    ).reset_index(drop=True)
    fold_market.insert(0, "__row_id", np.arange(len(fold_market), dtype=np.int64))
    if fold_market.empty:
        raise ValueError(f"Fold {fold_id} has no market rows.")

    soft: dict[str, np.ndarray] = {}
    hard: dict[str, np.ndarray] = {}
    candidate_events = events.copy()
    # Erase the main holdout split before assigning fold roles. Retaining it
    # here would let fold-PCA see later main-train events.
    candidate_events["split"] = "outside_fold"
    for diagnostic in (
        "assignment_entropy",
        "novelty",
        "nearest_distance",
        "nearest_similarity",
        "effective_soft_prototypes",
    ):
        candidate_events[diagnostic] = np.nan
    assigned_event_mask = np.zeros(len(events), dtype=bool)
    candidate_ids: dict[str, str] = {}
    prototype_dims: dict[str, int] = {}
    for level in NEWS_LEVELS:
        row = by_level[level]
        candidate_ids[level] = str(row.candidate_id)
        arrays = np.load(Path(str(row.assignment_path)), allow_pickle=False)
        event_indices = arrays["event_indices"].astype(int)
        if "event_ids" not in arrays.files or not np.array_equal(
            arrays["event_ids"].astype(str),
            events.iloc[event_indices]["event_id"].astype(str).to_numpy(),
        ):
            raise ValueError(
                f"Fold candidate {row.candidate_id} has stale event ordering."
            )
        if "fold_role" not in arrays.files:
            raise ValueError(
                f"Fold candidate {row.candidate_id} lacks fold_role; rebuild "
                "fold codebooks with the current pipeline."
            )
        fold_role = arrays["fold_role"].astype(int)
        if fold_role.shape != (len(event_indices),):
            raise ValueError(
                f"Fold candidate {row.candidate_id} has invalid fold_role shape."
            )
        if not np.isin(fold_role, [0, 1]).all():
            raise ValueError(
                f"Fold candidate {row.candidate_id} contains unknown fold roles."
            )
        candidate_events.loc[event_indices, "split"] = np.where(
            fold_role == 0, "train", "validation"
        )
        assigned_event_mask[event_indices] = True
        k = int(row.k)
        prototype_dims[level] = k
        soft_matrix = np.zeros((len(events), k), dtype=np.float32)
        hard_matrix = np.zeros((len(events), k), dtype=np.float32)
        soft_matrix[event_indices] = arrays["soft_assignment"].astype(np.float32)
        labels = arrays["hard_cluster_id"].astype(int)
        hard_matrix[event_indices, labels] = 1.0
        soft[level] = soft_matrix
        hard[level] = hard_matrix
        for diagnostic in (
            "assignment_entropy",
            "novelty",
            "nearest_distance",
            "nearest_similarity",
            "effective_soft_prototypes",
        ):
            candidate_events.loc[event_indices, diagnostic] = arrays[
                diagnostic
            ].astype(float)
    tickers = tuple(
        str(value).upper()
        for value in _nested(
            config,
            "data",
            "tickers",
            default=sorted(fold_market["ticker"].unique()),
        )
    )
    edges = _build_edges(
        candidate_events.loc[assigned_event_mask],
        tickers,
        "true",
        fold_market,
        seed,
    )
    shuffled_date_edges = _build_edges(
        candidate_events.loc[assigned_event_mask],
        tickers,
        "shuffled_date",
        fold_market,
        seed,
        shuffle_holdout_dates=True,
    )
    shuffled_ticker_edges = _build_edges(
        candidate_events.loc[assigned_event_mask],
        tickers,
        "shuffled_ticker",
        fold_market,
        seed,
    )
    selected_pooling = str(
        pooling
        or _nested(config, "prototype", "pooling", default="mean")
    )
    half_life = float(
        _nested(
            config,
            "prototype",
            "exponential_half_life_days",
            default=3.0,
        )
    )
    max_lag = int(
        _nested(config, "prototype", "max_lag_days", default=5)
    )
    sentinel = float(
        _nested(
            config,
            "prototype",
            "days_since_sentinel",
            default=3650.0,
        )
    )
    pca, random_projection, _ = _fit_embedding_comparators(
        config,
        candidate_events,
        embeddings,
        prototype_dims,
        seed,
    )
    random_prototype = _random_prototype_placebo(
        candidate_events,
        hard,
        seed,
    )
    matrices = {
        "soft": soft,
        "hard": hard,
        "random_prototype": random_prototype,
        "pca": pca,
        "random_projection": random_projection,
    }
    true_outputs = _features_for_pool(
        fold_market,
        candidate_events,
        embeddings,
        edges,
        selected_pooling,
        matrices,
        half_life,
        max_lag,
        sentinel,
    )
    date_outputs = _prototype_features_for_pool(
        fold_market,
        candidate_events,
        shuffled_date_edges,
        selected_pooling,
        soft,
        hard,
        half_life,
        max_lag,
        sentinel,
    )
    ticker_outputs = _prototype_features_for_pool(
        fold_market,
        candidate_events,
        shuffled_ticker_edges,
        selected_pooling,
        soft,
        hard,
        half_life,
        max_lag,
        sentinel,
    )
    available_outputs = {
        **true_outputs,
        # Dimension-matched placebos for R6: the semantic codebook is fixed,
        # while only dates/tickers are shuffled inside the current fold.
        "R10": date_outputs["R6"],
        "R11": ticker_outputs["R6"],
    }
    requested = tuple(
        representations
        or (
            "R5",
            "R6",
            "R7",
        )
    )
    supported = {
        "R3",
        "R4",
        "R5",
        "R6",
        "R7",
        "R9",
        "R10",
        "R11",
        "P_LAGGED",
        "P_PERMUTED",
    }
    unknown = sorted(set(requested).difference(supported))
    if unknown:
        raise ValueError(
            f"Unsupported fold representations requested: {unknown}"
        )
    outputs = {
        representation: (
            available_outputs["R6"]
            if representation in {"P_LAGGED", "P_PERMUTED"}
            else available_outputs[representation]
        )
        for representation in requested
    }
    output_dir = project_path(
        config,
        (
            f"data/processed/fold_representations{suffix}/"
            f"fold_{int(fold_id)}"
        ),
    )
    ensure_directories([output_dir])
    paths: dict[str, Path] = {}
    manifest_rows: list[dict[str, Any]] = []
    for representation, frame in outputs.items():
        destination = (
            output_dir
            / (
                f"features_{representation}_{selected_pooling}__"
                f"{representation_variant_family}_seed{seed}.parquet"
            )
        )
        write_table(frame, destination)
        paths[representation] = destination
        manifest_rows.append(
            {
                "fold_id": int(fold_id),
                "representation": representation,
                "representation_variant": (
                    f"{representation_variant_family}_seed{seed}"
                ),
                "representation_variant_family": representation_variant_family,
                "prototype_seed": seed,
                "model_seed": "",
                "pooling": selected_pooling,
                "path": str(destination),
                "candidate_ids": json.dumps(candidate_ids, sort_keys=True),
                "train_start": train_start,
                "train_end": train_end,
                "validation_start": validation_start,
                "validation_end": validation_end,
                "fit_scope": "fold_train_only",
                "events_variant": events_variant,
                "selected": False,
                "placebo": representation
                in {"R9", "R10", "R11", "P_LAGGED", "P_PERMUTED"},
                "placebo_kind": {
                    "R9": "random_prototype_fold_train_distribution",
                    "R10": "within_fold_split_shuffled_date",
                    "R11": "fold_seed_deranged_ticker",
                    "P_LAGGED": "lagged_fold_r6",
                    "P_PERMUTED": "within_fold_split_permuted_r6",
                }.get(representation, ""),
            }
        )
    manifest_path = project_path(
        config, f"data/processed/fold_representation_manifest{suffix}.csv"
    )
    new_manifest = pd.DataFrame(manifest_rows)
    if manifest_path.exists():
        old_manifest = safe_read_table(manifest_path)
        new_manifest = pd.concat([old_manifest, new_manifest], ignore_index=True)
        new_manifest = new_manifest.drop_duplicates(
            [
                "fold_id",
                "representation",
                "representation_variant",
                "pooling",
            ],
            keep="last",
        )
    atomic_write_csv(new_manifest, manifest_path, index=False)
    return {"fold_representation_manifest": manifest_path, **paths}


def _aggregate_response_aware_variants(
    config: Mapping[str, Any],
    market: pd.DataFrame,
    events: pd.DataFrame,
    edges: pd.DataFrame,
    pooling: str,
    half_life: float,
    max_lag: int,
) -> list[dict[str, Any]]:
    events_variant = _events_variant(config)
    suffix = _variant_suffix(events_variant)
    assignment_path = project_path(
        config, f"data/processed/response_aware_assignments{suffix}.parquet"
    )
    if not assignment_path.exists():
        return []
    assignments = safe_read_table(assignment_path)
    required = {
        "event_id",
        "news_level",
        "response_candidate_id",
        "response_aware_group_text_only",
        "shuffled_response_group_text_only",
    }
    if assignments.empty:
        return []
    validate_required_columns(
        assignments, sorted(required), "response-aware assignments"
    )
    event_positions = events[["event_id", "event_pos"]]
    assignments = assignments.merge(
        event_positions,
        on="event_id",
        how="left",
        validate="many_to_one",
    )
    if assignments["event_pos"].isna().any():
        raise ValueError("Response-aware assignments contain unknown event IDs.")
    instances = _instances(
        market, edges, pooling, half_life, max_lag
    )
    rows: list[dict[str, Any]] = []
    seed = int(_nested(config, "project", "seed", default=42))
    for candidate_id, candidate in assignments.groupby(
        "response_candidate_id", sort=True, observed=True
    ):
        true_matrices: dict[str, np.ndarray] = {}
        shuffled_matrices: dict[str, np.ndarray] = {}
        for level in NEWS_LEVELS:
            level_assignments = candidate.loc[
                candidate["news_level"].eq(level)
            ]
            if level_assignments.empty:
                true_matrices[level] = np.zeros((len(events), 0), dtype=np.float32)
                shuffled_matrices[level] = np.zeros(
                    (len(events), 0), dtype=np.float32
                )
                continue
            true_labels = level_assignments[
                "response_aware_group_text_only"
            ].to_numpy(dtype=int)
            shuffled_labels = level_assignments[
                "shuffled_response_group_text_only"
            ].to_numpy(dtype=int)
            k = int(max(true_labels.max(), shuffled_labels.max()) + 1)
            true_matrix = np.zeros((len(events), k), dtype=np.float32)
            shuffled_matrix = np.zeros((len(events), k), dtype=np.float32)
            positions = level_assignments["event_pos"].to_numpy(dtype=int)
            true_matrix[positions, true_labels] = 1.0
            shuffled_matrix[positions, shuffled_labels] = 1.0
            true_matrices[level] = true_matrix
            shuffled_matrices[level] = shuffled_matrix
        true_blocks = [
            _aggregate_matrix(
                instances,
                true_matrices[level],
                level,
                len(market),
                pooling,
                "responseproto",
            )
            for level in NEWS_LEVELS
        ]
        shuffled_blocks = [
            _aggregate_matrix(
                instances,
                shuffled_matrices[level],
                level,
                len(market),
                pooling,
                "shuffledresponse",
            )
            for level in NEWS_LEVELS
        ]
        variants = {
            "R6": (
                f"response_aware__{candidate_id}",
                _join_feature_blocks(market, true_blocks),
                "response_aware",
            ),
            "R9": (
                f"shuffled_response__{candidate_id}",
                _join_feature_blocks(market, shuffled_blocks),
                "shuffled_response",
            ),
        }
        for representation, (variant, frame, kind) in variants.items():
            path = _write_representation(
                config,
                representation,
                pooling,
                frame,
                variant=variant,
            )
            feature_columns = [
                column
                for column in frame
                if column not in {"ticker", "feature_date", "split"}
            ]
            rows.append(
                {
                    "representation": representation,
                    "representation_variant": variant,
                    "representation_variant_family": variant,
                    "pooling": pooling,
                    "path": str(path),
                    "selected": False,
                    "feature_count": len(feature_columns),
                    "feature_columns": json.dumps(feature_columns),
                    "fit_split": "train_response_and_text_centroid_only",
                    "placebo": kind == "shuffled_response",
                    "placebo_kind": (
                        "shuffled_train_response"
                        if kind == "shuffled_response"
                        else ""
                    ),
                    "prototype_candidate_id": str(candidate_id),
                    "events_variant": events_variant,
                    "prototype_seed": seed,
                    "response_aware": kind == "response_aware",
                    "response_validation_or_test_used": False,
                }
            )
    return rows


def run(config: dict) -> dict[str, Path]:
    """Build flat, numeric R0--R11 ticker-date feature artifacts."""

    seed = int(_nested(config, "project", "seed", default=config.get("seed", 42)))
    deterministic = bool(
        _nested(config, "project", "deterministic", default=True)
    )
    set_global_seed(seed, deterministic=deterministic)
    ensure_directories(config)
    events_variant = _events_variant(config)
    suffix = _variant_suffix(events_variant)
    market, events, embeddings, _ = _load_data(config)
    tickers = tuple(
        str(value).upper()
        for value in _nested(
            config, "data", "tickers", default=sorted(market["ticker"].unique())
        )
    )
    unknown_tickers = sorted(set(market["ticker"]).difference(tickers))
    if unknown_tickers:
        raise ValueError(f"Market data contains unconfigured tickers: {unknown_tickers}")
    train_event_mask = events["split"].to_numpy() == "train"
    soft, hard, random_prototype, prototype_dims = _soft_matrices(
        events, train_event_mask, seed
    )
    pca, random_projection, comparator_models = _fit_embedding_comparators(
        config, events, embeddings, prototype_dims, seed
    )
    comparator_path = project_path(
        config,
        f"outputs/models/aggregate_embedding_transforms{suffix}.joblib",
    )
    atomic_joblib_dump(
        {
            "pca_and_random_projection_by_level": comparator_models,
            "fit_split": "train",
            "seed": seed,
            "events_variant": events_variant,
        },
        comparator_path,
    )
    matrices = {
        "soft": soft,
        "hard": hard,
        "random_prototype": random_prototype,
        "pca": pca,
        "random_projection": random_projection,
    }
    true_edges = _build_edges(events, tickers, "true", market, seed)
    shuffle_holdout_dates = bool(
        _nested(
            config,
            "placebo",
            "shuffle_holdout_dates",
            default=False,
        )
    )
    shuffled_date_edges = _build_edges(
        events,
        tickers,
        "shuffled_date",
        market,
        seed,
        shuffle_holdout_dates=shuffle_holdout_dates,
    )
    shuffled_ticker_edges = _build_edges(
        events, tickers, "shuffled_ticker", market, seed
    )
    if true_edges.empty:
        raise ValueError("No event-to-ticker edges could be built.")

    configured_pooling = str(
        _nested(config, "prototype", "pooling", default="mean")
    )
    available_poolings = [
        str(value)
        for value in _nested(
            config,
            "prototype",
            "pooling_options",
            default=["mean", "normalized_sum", "max", "exponential_decay"],
        )
    ]
    if configured_pooling not in available_poolings:
        available_poolings.insert(0, configured_pooling)
    generate_all = bool(
        _nested(
            config,
            "prototype",
            "generate_all_pooling_representations",
            default=False,
        )
    )
    poolings = available_poolings if generate_all else [configured_pooling]
    half_life = float(
        _nested(
            config, "prototype", "exponential_half_life_days", default=3.0
        )
    )
    max_lag = int(
        _nested(config, "prototype", "max_lag_days", default=5)
    )
    sentinel = float(
        _nested(config, "prototype", "days_since_sentinel", default=3650.0)
    )

    manifest_rows: list[dict[str, Any]] = []
    result_paths: dict[str, Path] = {}
    r0 = market[["ticker", "feature_date", "split"]].copy()
    r0["r0_no_news"] = np.float32(0.0)
    r0_path = _write_representation(config, "R0", configured_pooling, r0)
    result_paths["R0"] = r0_path
    manifest_rows.append(
        {
            "representation": "R0",
            "representation_variant": "selected_default",
            "representation_variant_family": "selected_default",
            "pooling": configured_pooling,
            "path": str(r0_path),
            "selected": True,
            "feature_count": 1,
            "feature_columns": json.dumps(["r0_no_news"]),
            "fit_split": "train",
            "placebo": False,
            "prototype_candidate_id": "",
            "events_variant": events_variant,
            "prototype_seed": seed,
        }
    )

    for pooling in poolings:
        true_outputs = _features_for_pool(
            market,
            events,
            embeddings,
            true_edges,
            pooling,
            matrices,
            half_life,
            max_lag,
            sentinel,
        )
        # R10/R11 deliberately reuse the locked semantic codebook; only the
        # availability date/ticker is permuted, independently of any response.
        date_outputs = _prototype_features_for_pool(
            market,
            events,
            shuffled_date_edges,
            pooling,
            soft,
            hard,
            half_life,
            max_lag,
            sentinel,
        )
        ticker_outputs = _prototype_features_for_pool(
            market,
            events,
            shuffled_ticker_edges,
            pooling,
            soft,
            hard,
            half_life,
            max_lag,
            sentinel,
        )
        outputs = {
            **true_outputs,
            "R10": date_outputs["R7"],
            "R11": ticker_outputs["R7"],
        }
        for representation in REPRESENTATIONS[1:]:
            frame = outputs[representation]
            path = _write_representation(
                config, representation, pooling, frame
            )
            selected = pooling == configured_pooling
            if selected:
                result_paths[representation] = path
            feature_columns = [
                column
                for column in frame
                if column not in {"ticker", "feature_date", "split"}
            ]
            manifest_rows.append(
                {
                    "representation": representation,
                    "representation_variant": "selected_default",
                    "representation_variant_family": "selected_default",
                    "pooling": pooling,
                    "path": str(path),
                    "selected": selected,
                    "feature_count": len(feature_columns),
                    "feature_columns": json.dumps(feature_columns),
                    "fit_split": "train",
                    "placebo": representation in {"R9", "R10", "R11"},
                    "placebo_kind": {
                        "R9": "random_prototype_train_distribution",
                        "R10": (
                            "split_isolated_shuffled_date"
                            if shuffle_holdout_dates
                            else "train_only_shuffled_date"
                        ),
                        "R11": "fixed_shuffled_ticker",
                    }.get(representation, ""),
                    "prototype_candidate_id": (
                        json.dumps(
                            {
                                level: str(
                                    events.loc[
                                        events["news_level"].eq(level)
                                        & events["candidate_id"].notna(),
                                        "candidate_id",
                                    ].iloc[0]
                                )
                                for level in NEWS_LEVELS
                                if (
                                    events["news_level"].eq(level)
                                    & events["candidate_id"].notna()
                                ).any()
                            },
                            sort_keys=True,
                        )
                        if representation in {"R5", "R6", "R7", "R8", "R9", "R10", "R11"}
                        else ""
                    ),
                    "events_variant": events_variant,
                    "prototype_seed": seed,
                    "shuffled_date_splits": (
                        "train_validation_test"
                        if representation == "R10" and shuffle_holdout_dates
                        else ("train_only" if representation == "R10" else "")
                    ),
                }
            )

    grid_variants = _eligible_grid_variants(config, events, seed)
    for (
        variant_id,
        candidate_ids,
        candidate_soft,
        candidate_hard,
        candidate_events,
        details,
    ) in grid_variants:
        candidate_pca, candidate_random_projection, _ = (
            _fit_embedding_comparators(
                config,
                candidate_events,
                embeddings,
                {level: int(details["k"]) for level in NEWS_LEVELS},
                int(details["prototype_seed"]),
            )
        )
        candidate_random_prototype = _random_prototype_placebo(
            candidate_events,
            candidate_hard,
            int(details["prototype_seed"]),
        )
        candidate_matrices = {
            "soft": candidate_soft,
            "hard": candidate_hard,
            "random_prototype": candidate_random_prototype,
            "pca": candidate_pca,
            "random_projection": candidate_random_projection,
        }
        candidate_outputs = _features_for_pool(
            market,
            candidate_events,
            embeddings,
            true_edges,
            configured_pooling,
            candidate_matrices,
            half_life,
            max_lag,
            sentinel,
        )
        candidate_date_outputs = _prototype_features_for_pool(
            market,
            candidate_events,
            shuffled_date_edges,
            configured_pooling,
            candidate_soft,
            candidate_hard,
            half_life,
            max_lag,
            sentinel,
        )
        candidate_ticker_outputs = _prototype_features_for_pool(
            market,
            candidate_events,
            shuffled_ticker_edges,
            configured_pooling,
            candidate_soft,
            candidate_hard,
            half_life,
            max_lag,
            sentinel,
        )
        matched_outputs = {
            representation: candidate_outputs[representation]
            for representation in ("R3", "R4", "R5", "R6", "R7", "R8", "R9")
        }
        matched_outputs["R10"] = candidate_date_outputs["R7"]
        matched_outputs["R11"] = candidate_ticker_outputs["R7"]
        for representation, frame in matched_outputs.items():
            path = _write_representation(
                config,
                representation,
                configured_pooling,
                frame,
                variant=variant_id,
            )
            feature_columns = [
                column
                for column in frame
                if column not in {"ticker", "feature_date", "split"}
            ]
            manifest_rows.append(
                {
                    "representation": representation,
                    "representation_variant": variant_id,
                    "pooling": configured_pooling,
                    "path": str(path),
                    "selected": False,
                    "feature_count": len(feature_columns),
                    "feature_columns": json.dumps(feature_columns),
                    "fit_split": "train",
                    "placebo": representation in {"R9", "R10", "R11"},
                    "placebo_kind": {
                        "R9": "random_prototype_train_distribution",
                        "R10": (
                            "split_isolated_shuffled_date"
                            if shuffle_holdout_dates
                            else "train_only_shuffled_date"
                        ),
                        "R11": "fixed_deranged_ticker",
                    }.get(representation, ""),
                    "prototype_candidate_id": json.dumps(
                        candidate_ids, sort_keys=True
                    ),
                    **details,
                    "events_variant": events_variant,
                }
            )

    manifest_rows.extend(
        _aggregate_response_aware_variants(
            config=config,
            market=market,
            events=events,
            edges=true_edges,
            pooling=configured_pooling,
            half_life=half_life,
            max_lag=max_lag,
        )
    )

    if set(result_paths) != set(REPRESENTATIONS):
        missing = sorted(set(REPRESENTATIONS).difference(result_paths))
        raise AssertionError(f"Missing selected representation artifacts: {missing}")
    default_news_path = project_path(
        config, f"data/processed/news_features{suffix}.parquet"
    )
    write_table(safe_read_table(result_paths["R7"]), default_news_path)
    manifest = pd.DataFrame(manifest_rows).sort_values(
        ["representation", "selected", "pooling"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    manifest_path = project_path(
        config, f"data/processed/representation_manifest{suffix}.csv"
    )
    table_manifest_path = project_path(
        config, f"outputs/tables/representation_manifest{suffix}.csv"
    )
    atomic_write_csv(manifest, manifest_path, index=False)
    atomic_write_csv(manifest, table_manifest_path, index=False)
    # The unsuffixed manifest is the active configuration pointer consumed by
    # train_targets. Variant-specific manifests remain available for robustness.
    active_manifest_path = project_path(
        config, "data/processed/representation_manifest.csv"
    )
    if active_manifest_path != manifest_path:
        atomic_write_csv(manifest, active_manifest_path, index=False)
    LOGGER.info(
        "Wrote %d ticker-day representation artifacts (%s selected pooling).",
        len(manifest),
        configured_pooling,
    )
    return {
        "representation_manifest": manifest_path,
        "active_representation_manifest": active_manifest_path,
        "representation_manifest_table": table_manifest_path,
        "news_features": default_news_path,
        "aggregate_embedding_transforms": comparator_path,
        **{f"features_{key}": value for key, value in result_paths.items()},
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
