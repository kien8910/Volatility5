"""Fit leakage-safe, level-specific semantic prototype codebooks.

PCA and spherical K-means are fitted on train events only.  Every accepted
candidate is persisted so downstream target selection can use validation
scores without ever consulting test performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    silhouette_score,
)

from src.utils import (
    atomic_joblib_dump,
    atomic_write_csv,
    chronological_assertions,
    ensure_directories,
    get_logger,
    l2_normalize,
    load_config,
    project_path,
    safe_read_table,
    set_global_seed,
    stable_softmax,
    validate_required_columns,
    write_table,
)

LOGGER = get_logger(__name__)
NEWS_LEVELS = ("macro", "sector", "related", "target")
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "test": "test",
    "testing": "test",
    "outside_supervised": "outside_supervised",
}


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


def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".npz", prefix=f".{path.stem}.", dir=path.parent, delete=False
        ) as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_name = handle.name
        os.replace(temporary_name, path)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _normalize_pca_dims(values: Sequence[Any]) -> list[int | None]:
    result: list[int | None] = []
    for value in values:
        if value is None or str(value).strip().lower() in {"none", "null", "raw"}:
            parsed = None
        else:
            parsed = int(value)
            if parsed < 1:
                raise ValueError("prototype.pca_dims values must be positive or null.")
        if parsed not in result:
            result.append(parsed)
    return result


def _float_token(value: float) -> str:
    return f"{value:.6g}".replace("-", "m").replace(".", "p")


def _candidate_id(
    level: str, k: int, pca_dim: int | None, temperature: float, seed: int
) -> str:
    pca_token = "none" if pca_dim is None else str(pca_dim)
    return (
        f"{level}_k{k}_pca{pca_token}_tau{_float_token(temperature)}_seed{seed}"
    )


def _base_id(level: str, k: int, pca_dim: int | None, seed: int) -> str:
    return f"{level}_k{k}_pca{'none' if pca_dim is None else pca_dim}_seed{seed}"


def _load_inputs(config: Mapping[str, Any]) -> tuple[pd.DataFrame, np.ndarray]:
    events_variant = _events_variant(config)
    events_default = (
        f"data/processed/canonical_events_{events_variant}.parquet"
        if events_variant in {"exact", "raw"}
        else "data/processed/canonical_events.parquet"
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
    metadata_path = _path(
        config,
        (("paths", f"event_embedding_metadata_{events_variant}"),),
        (
            "data/embeddings/"
            f"event_embedding_metadata{_variant_suffix(events_variant)}.csv"
        ),
    )
    embedding_path = _path(
        config,
        (("paths", f"event_embeddings_{events_variant}"),),
        f"data/embeddings/event_embeddings{_variant_suffix(events_variant)}.npy",
    )
    events = safe_read_table(events_path)
    metadata = safe_read_table(metadata_path)
    validate_required_columns(
        events,
        ["event_id", "date", "split", "news_level", "text"],
        "canonical events",
    )
    validate_required_columns(
        metadata,
        ["event_id", "embedding_index", "text_hash"],
        "event embedding metadata",
    )
    if events["event_id"].duplicated().any() or metadata["event_id"].duplicated().any():
        raise ValueError("event_id must be unique in events and embedding metadata.")
    merged = events.merge(
        metadata[["event_id", "embedding_index", "text_hash"]],
        on="event_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_embedding"),
    )
    if merged["embedding_index"].isna().any():
        missing = merged.loc[merged["embedding_index"].isna(), "event_id"].tolist()[:5]
        raise ValueError(f"Canonical events missing embeddings: {missing}")
    embeddings = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2-D embeddings, found {embeddings.shape}.")
    indices = merged["embedding_index"].to_numpy(dtype=int)
    if indices.min(initial=0) < 0 or indices.max(initial=-1) >= len(embeddings):
        raise ValueError("Embedding metadata contains out-of-bounds indices.")
    matrix = np.asarray(embeddings[indices], dtype=np.float64)
    matrix = l2_normalize(matrix)
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite event embeddings cannot be clustered.")
    merged["date"] = pd.to_datetime(merged["date"], errors="raise").dt.normalize()
    normalized_splits = merged["split"].astype(str).str.lower().map(SPLIT_ALIASES)
    if normalized_splits.isna().any():
        unknown = sorted(merged.loc[normalized_splits.isna(), "split"].astype(str).unique())
        raise ValueError(f"Unknown event split labels: {unknown}")
    merged["split"] = normalized_splits
    if set(NEWS_LEVELS).difference(set(merged["news_level"])):
        LOGGER.warning(
            "Some configured news levels have no canonical events: %s",
            sorted(set(NEWS_LEVELS).difference(set(merged["news_level"]))),
        )
    return merged.reset_index(drop=True), matrix


def _assert_chronology(events: pd.DataFrame) -> None:
    available = set(events["split"])
    if {"train", "validation", "test"}.issubset(available):
        chronological_assertions(
            events.loc[events["split"] == "train", "date"],
            events.loc[events["split"] == "validation", "date"],
            events.loc[events["split"] == "test", "date"],
        )


def _fit_projection(
    train: np.ndarray,
    all_values: np.ndarray,
    pca_dim: int | None,
    seed: int,
) -> tuple[PCA | None, np.ndarray, float]:
    if pca_dim is None:
        return None, l2_normalize(all_values), 1.0
    upper = min(train.shape[0], train.shape[1])
    if pca_dim > upper:
        raise ValueError(
            f"PCA dimension {pca_dim} exceeds min(train rows, embedding dim)={upper}."
        )
    solver = "randomized" if pca_dim < upper else "full"
    pca = PCA(n_components=pca_dim, svd_solver=solver, random_state=seed)
    pca.fit(train)
    transformed = pca.transform(all_values)
    return (
        pca,
        l2_normalize(transformed),
        float(np.sum(pca.explained_variance_ratio_)),
    )


def _fit_spherical_kmeans(
    train: np.ndarray,
    k: int,
    seed: int,
    n_init: int,
    max_iter: int,
) -> np.ndarray:
    estimator = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=n_init,
        max_iter=max_iter,
        random_state=seed,
        algorithm="lloyd",
    )
    estimator.fit(train)
    centers = l2_normalize(estimator.cluster_centers_)
    if not np.isfinite(centers).all():
        raise ValueError("K-means produced non-finite centroids.")
    return centers


def _assign(
    values: np.ndarray, centers: np.ndarray, temperature: float
) -> dict[str, np.ndarray]:
    similarities = np.clip(values @ centers.T, -1.0, 1.0)
    hard = np.argmax(similarities, axis=1).astype(np.int32)
    nearest_similarity = similarities[np.arange(len(values)), hard]
    soft = stable_softmax(similarities / temperature, axis=1).astype(np.float32)
    entropy = -np.sum(soft * np.log(np.clip(soft, 1.0e-12, None)), axis=1)
    return {
        "hard_cluster_id": hard,
        "soft_assignment": soft,
        "nearest_similarity": nearest_similarity.astype(np.float32),
        "nearest_distance": (1.0 - nearest_similarity).astype(np.float32),
        "novelty": (1.0 - nearest_similarity).astype(np.float32),
        "assignment_entropy": entropy.astype(np.float32),
        "effective_soft_prototypes": np.exp(entropy).astype(np.float32),
    }


def _usage_diagnostics(
    train_values: np.ndarray,
    centers: np.ndarray,
    min_events_per_cluster: int,
    max_small_fraction: float,
    min_effective_fraction: float,
    silhouette_sample_size: int,
    seed: int,
) -> dict[str, Any]:
    similarities = np.clip(train_values @ centers.T, -1.0, 1.0)
    hard = np.argmax(similarities, axis=1)
    k = centers.shape[0]
    counts = np.bincount(hard, minlength=k)
    proportions = counts / max(counts.sum(), 1)
    nonzero = proportions[proportions > 0]
    effective = float(np.exp(-np.sum(nonzero * np.log(nonzero))))
    dead_fraction = float(np.mean(counts == 0))
    small_fraction = float(np.mean(counts < min_events_per_cluster))
    reasons: list[str] = []
    if dead_fraction > 0:
        reasons.append(f"dead_prototype_fraction={dead_fraction:.3f}")
    if small_fraction > max_small_fraction:
        reasons.append(
            f"small_prototype_fraction={small_fraction:.3f}>{max_small_fraction:.3f}"
        )
    if effective / k < min_effective_fraction:
        reasons.append(
            f"effective_fraction={effective / k:.3f}<{min_effective_fraction:.3f}"
        )

    silhouette = np.nan
    davies_bouldin = np.nan
    unique_clusters = np.unique(hard)
    if 1 < len(unique_clusters) < len(train_values):
        if len(train_values) > silhouette_sample_size:
            rng = np.random.default_rng(seed)
            sample_indices = np.sort(
                rng.choice(
                    len(train_values), size=silhouette_sample_size, replace=False
                )
            )
        else:
            sample_indices = np.arange(len(train_values))
        sampled_labels = hard[sample_indices]
        if 1 < np.unique(sampled_labels).size < len(sample_indices):
            silhouette = float(
                silhouette_score(
                    train_values[sample_indices],
                    sampled_labels,
                    metric="cosine",
                )
            )
            davies_bouldin = float(
                davies_bouldin_score(
                    train_values[sample_indices], sampled_labels
                )
            )
    return {
        "hard": hard.astype(np.int32),
        "counts": counts.astype(int),
        "effective_number": effective,
        "effective_fraction": effective / k,
        "dead_fraction": dead_fraction,
        "small_fraction": small_fraction,
        "silhouette": silhouette,
        "davies_bouldin": davies_bouldin,
        "rejection_reasons": reasons,
    }


def _example_rows(
    level_events: pd.DataFrame,
    transformed: np.ndarray,
    train_mask: np.ndarray,
    centers: np.ndarray,
    base_id: str,
    top_n: int,
) -> list[dict[str, Any]]:
    train_indices = np.flatnonzero(train_mask)
    train_values = transformed[train_mask]
    similarities = np.clip(train_values @ centers.T, -1.0, 1.0)
    rows: list[dict[str, Any]] = []
    for cluster_id in range(centers.shape[0]):
        ranking = np.argsort(-similarities[:, cluster_id], kind="mergesort")[:top_n]
        for rank, local_index in enumerate(ranking, start=1):
            event = level_events.iloc[int(train_indices[local_index])]
            rows.append(
                {
                    "base_candidate_id": base_id,
                    "news_level": event["news_level"],
                    "prototype_id": cluster_id,
                    "rank": rank,
                    "event_id": event["event_id"],
                    "date": event["date"],
                    "text": event["text"],
                    "cosine_similarity": float(
                        similarities[local_index, cluster_id]
                    ),
                    "split": "train",
                }
            )
    return rows


def _selection_preferences(
    config: Mapping[str, Any], level: str
) -> tuple[int, int | None, float, int | None]:
    selected = _nested(config, "prototype", "selected", level, default={})
    if not isinstance(selected, Mapping):
        selected = {}
    pca_value = selected.get(
        "pca_dim", _nested(config, "prototype", "preferred_pca_dim", default=64)
    )
    pca_dim = (
        None
        if pca_value is None or str(pca_value).lower() in {"none", "null"}
        else int(pca_value)
    )
    return (
        int(selected.get("k", _nested(config, "prototype", "preferred_k", default=8))),
        pca_dim,
        float(
            selected.get(
                "temperature",
                _nested(config, "prototype", "preferred_temperature", default=0.1),
            )
        ),
        int(selected["seed"]) if "seed" in selected else None,
    )


def _choose_candidate(
    manifest: pd.DataFrame,
    config: Mapping[str, Any],
    level: str,
    primary_seed: int,
) -> pd.Series | None:
    eligible = manifest.loc[
        (manifest["news_level"] == level) & manifest["eligible"].astype(bool)
    ].copy()
    if eligible.empty:
        return None
    preferred_k, preferred_pca, preferred_temperature, preferred_seed = (
        _selection_preferences(config, level)
    )
    desired_seed = primary_seed if preferred_seed is None else preferred_seed
    eligible["seed_penalty"] = (eligible["seed"].astype(int) != desired_seed).astype(int)
    eligible["k_penalty"] = (eligible["k"].astype(int) - preferred_k).abs()
    eligible["pca_penalty"] = eligible["pca_dim"].apply(
        lambda value: (
            0
            if (pd.isna(value) and preferred_pca is None)
            else (
                abs(int(value) - int(preferred_pca))
                if not pd.isna(value) and preferred_pca is not None
                else 10_000
            )
        )
    )
    eligible["temperature_penalty"] = (
        eligible["temperature"].astype(float) - preferred_temperature
    ).abs()
    # Diagnostic quality only breaks ties after explicit/default preferences.
    eligible = eligible.sort_values(
        [
            "seed_penalty",
            "k_penalty",
            "pca_penalty",
            "temperature_penalty",
            "effective_fraction",
            "candidate_id",
        ],
        ascending=[True, True, True, True, False, True],
        kind="mergesort",
    )
    return eligible.iloc[0]


def _centroid_stability(left: np.ndarray, right: np.ndarray) -> float:
    similarities = np.clip(left @ right.T, -1.0, 1.0)
    rows, columns = linear_sum_assignment(-similarities)
    return float(similarities[rows, columns].mean())


def _build_stability(
    cache: Mapping[tuple[str, int | None, int, int], dict[str, np.ndarray]],
    primary_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_keys = sorted({key[:3] for key in cache}, key=str)
    for level, pca_dim, k in group_keys:
        seeds = sorted(
            key[3] for key in cache if key[:3] == (level, pca_dim, k)
        )
        reference_seed = primary_seed if primary_seed in seeds else seeds[0]
        reference = cache[(level, pca_dim, k, reference_seed)]
        for seed in seeds:
            candidate = cache[(level, pca_dim, k, seed)]
            rows.append(
                {
                    "news_level": level,
                    "pca_dim": pca_dim,
                    "k": k,
                    "reference_seed": reference_seed,
                    "comparison_seed": seed,
                    "adjusted_rand_index": float(
                        adjusted_rand_score(reference["hard"], candidate["hard"])
                    ),
                    "matched_centroid_cosine": _centroid_stability(
                        reference["centers"], candidate["centers"]
                    ),
                    "n_train_events": len(reference["hard"]),
                }
            )
    return pd.DataFrame(rows)


def _response_matrix_from_train(
    config: Mapping[str, Any],
    response_kind: str,
) -> tuple[pd.DataFrame, list[str], str]:
    residual_path = _path(
        config,
        (("paths", "residual_targets"),),
        "data/processed/residual_targets.parquet",
    )
    if not residual_path.exists():
        raise FileNotFoundError(residual_path)
    residuals = safe_read_table(residual_path)
    validate_required_columns(
        residuals,
        ["ticker", "feature_date", "split", "signed_residual"],
        "response-aware residual targets",
    )
    # This filter is the central leakage guard for response-aware prototypes.
    train = residuals.loc[
        residuals["split"].astype(str).str.lower().eq("train")
    ].copy()
    if train.empty:
        raise ValueError("No train residuals are available for response-aware grouping.")
    train["feature_date"] = pd.to_datetime(
        train["feature_date"], errors="raise"
    ).dt.normalize()
    tickers = [
        str(value).upper()
        for value in _nested(
            config,
            "data",
            "tickers",
            default=sorted(train["ticker"].astype(str).str.upper().unique()),
        )
    ]
    train["ticker"] = train["ticker"].astype(str).str.upper()
    response_kind = str(response_kind).lower()
    if response_kind == "signed":
        values = pd.to_numeric(train["signed_residual"], errors="coerce")
    elif response_kind in {"absolute", "magnitude"}:
        if "residual_magnitude" in train:
            values = pd.to_numeric(train["residual_magnitude"], errors="coerce")
        else:
            values = pd.to_numeric(
                train["signed_residual"], errors="coerce"
            ).abs()
        response_kind = "absolute"
    elif response_kind in {"spike", "spike_q90"}:
        if "spike_q90" not in train:
            raise ValueError(
                "response-aware spike signatures require residual_targets.spike_q90."
            )
        values = pd.to_numeric(train["spike_q90"], errors="coerce")
        response_kind = "spike_q90"
    else:
        raise ValueError(
            "prototype.response_aware.response must be signed, absolute, or spike_q90."
        )
    if values.isna().any():
        raise ValueError("Non-numeric train residuals in response-aware signature.")
    train["__response"] = values.to_numpy(dtype=float)
    pivot = train.pivot_table(
        index="feature_date",
        columns="ticker",
        values="__response",
        aggfunc="mean",
    ).reindex(columns=tickers)
    coverage = pivot.notna().sum(axis=1)
    minimum_coverage = int(
        _nested(
            config,
            "prototype",
            "response_aware",
            "minimum_ticker_coverage",
            default=max(3, len(tickers) // 2),
        )
    )
    pivot = pivot.loc[coverage >= minimum_coverage]
    if pivot.empty:
        raise ValueError(
            "No train dates meet response-aware ticker coverage threshold "
            f"{minimum_coverage}."
        )
    pivot = pivot.fillna(pivot.mean(axis=0)).fillna(0.0)
    response_values = pivot.to_numpy(dtype=float)
    response = pd.DataFrame(
        response_values, index=pivot.index, columns=tickers
    ).reset_index()
    return response, tickers, response_kind


def _text_centroids(
    text_values: np.ndarray, labels: np.ndarray, k: int
) -> np.ndarray:
    centroids = np.zeros((k, text_values.shape[1]), dtype=float)
    for cluster_id in range(k):
        selected = text_values[labels == cluster_id]
        if not len(selected):
            raise ValueError(
                f"Response-aware cluster {cluster_id} has no text members."
            )
        centroids[cluster_id] = selected.mean(axis=0)
    return l2_normalize(centroids)


def _chronological_text_predictability(
    text_values: np.ndarray,
    responses: np.ndarray,
    dates: np.ndarray,
    k: int,
    lambda_z: float,
    lambda_q: float,
    seed: int,
    n_init: int,
    max_iter: int,
    tail_fraction: float,
) -> tuple[float, float, int]:
    unique_dates = np.sort(np.unique(dates))
    if len(unique_dates) < 5:
        return np.nan, np.nan, 0
    cutoff_index = int(np.floor(len(unique_dates) * (1.0 - tail_fraction)))
    cutoff_index = min(max(cutoff_index, 1), len(unique_dates) - 1)
    cutoff = unique_dates[cutoff_index]
    early = dates < cutoff
    tail = ~early
    if int(early.sum()) < max(k, 2 * k) or int(tail.sum()) < 1:
        return np.nan, np.nan, int(tail.sum())
    response_center = responses[early].mean(axis=0, keepdims=True)
    response_scale = responses[early].std(
        axis=0, ddof=1, keepdims=True
    )
    response_scale = np.where(
        np.isfinite(response_scale) & (response_scale > 1.0e-8),
        response_scale,
        1.0,
    )
    normalized_responses = l2_normalize(
        (responses - response_center) / response_scale
    )
    combined = l2_normalize(
        np.concatenate(
            [
                np.sqrt(lambda_z) * text_values,
                np.sqrt(lambda_q) * normalized_responses,
            ],
            axis=1,
        )
    )
    centers = _fit_spherical_kmeans(
        combined[early], k, seed, n_init, max_iter
    )
    early_labels = np.argmax(combined[early] @ centers.T, axis=1)
    if np.unique(early_labels).size < k:
        return np.nan, np.nan, int(tail.sum())
    text_centers = _text_centroids(text_values[early], early_labels, k)
    impact_tail = np.argmax(combined[tail] @ centers.T, axis=1)
    text_tail = np.argmax(text_values[tail] @ text_centers.T, axis=1)
    return (
        float(np.mean(impact_tail == text_tail)),
        float(adjusted_rand_score(impact_tail, text_tail)),
        int(tail.sum()),
    )


def _fit_response_candidate(
    events: pd.DataFrame,
    embeddings: np.ndarray,
    response_frame: pd.DataFrame,
    tickers: Sequence[str],
    response_kind: str,
    requested_k: int,
    lambda_z: float,
    lambda_q: float,
    tail_fraction: float,
    predictability_threshold: float,
    seed: int,
    n_init: int,
    max_iter: int,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame], dict[str, Any]]:
    response_dates = pd.to_datetime(response_frame["feature_date"]).dt.normalize()
    response_lookup = {
        pd.Timestamp(date): vector
        for date, vector in zip(
            response_dates,
            response_frame[list(tickers)].to_numpy(dtype=float),
        )
    }
    candidate_id = (
        f"response_{response_kind}_lz{_float_token(lambda_z)}"
        f"_lq{_float_token(lambda_q)}_k{requested_k}"
    )
    rng = np.random.default_rng(
        seed
        + int(round(lambda_z * 1000))
        + int(round(lambda_q * 10000))
        + sum(ord(character) for character in response_kind)
    )
    summary_rows: list[dict[str, Any]] = []
    assignment_frames: list[pd.DataFrame] = []
    level_models: dict[str, Any] = {}
    for level in NEWS_LEVELS:
        level_mask = events["news_level"].to_numpy() == level
        matched_mask = (
            level_mask
            & (events["split"].to_numpy() == "train")
            & events["date"].isin(response_lookup).to_numpy()
        )
        matched_indices = np.flatnonzero(matched_mask)
        if len(matched_indices) < max(2, requested_k):
            summary_rows.append(
                {
                    "response_candidate_id": candidate_id,
                    "status": "skipped_too_few_train_response_events",
                    "news_level": level,
                    "n_train_response_events": len(matched_indices),
                    "requested_k": requested_k,
                    "response_kind": response_kind,
                    "lambda_z": lambda_z,
                    "lambda_q": lambda_q,
                    "response_validation_or_test_used": False,
                }
            )
            continue
        k = min(requested_k, len(matched_indices))
        text_train = embeddings[matched_indices]
        raw_response_train = np.vstack(
            [
                response_lookup[pd.Timestamp(events.iloc[index]["date"])]
                for index in matched_indices
            ]
        )
        response_center = raw_response_train.mean(axis=0, keepdims=True)
        response_scale = raw_response_train.std(
            axis=0, ddof=1, keepdims=True
        )
        response_scale = np.where(
            np.isfinite(response_scale) & (response_scale > 1.0e-8),
            response_scale,
            1.0,
        )
        response_train = l2_normalize(
            (raw_response_train - response_center) / response_scale
        )
        combined = l2_normalize(
            np.concatenate(
                [
                    np.sqrt(lambda_z) * text_train,
                    np.sqrt(lambda_q) * response_train,
                ],
                axis=1,
            )
        )
        combined_centers = _fit_spherical_kmeans(
            combined, k, seed, n_init, max_iter
        )
        impact_labels = np.argmax(combined @ combined_centers.T, axis=1)
        if np.unique(impact_labels).size < k:
            summary_rows.append(
                {
                    "response_candidate_id": candidate_id,
                    "status": "rejected_dead_impact_group",
                    "news_level": level,
                    "response_kind": response_kind,
                    "k": k,
                    "lambda_z": lambda_z,
                    "lambda_q": lambda_q,
                    "n_train_response_events": len(matched_indices),
                    "response_validation_or_test_used": False,
                }
            )
            continue
        text_centers = _text_centroids(text_train, impact_labels, k)
        level_indices = np.flatnonzero(level_mask)
        text_all = embeddings[level_indices]
        text_similarities = np.clip(text_all @ text_centers.T, -1.0, 1.0)
        text_groups = np.argmax(text_similarities, axis=1)
        nearest_text_similarity = text_similarities[
            np.arange(len(text_all)), text_groups
        ]
        text_train_groups = np.argmax(text_train @ text_centers.T, axis=1)
        train_accuracy = float(np.mean(text_train_groups == impact_labels))
        train_ari = float(
            adjusted_rand_score(impact_labels, text_train_groups)
        )
        tail_accuracy, tail_ari, tail_count = _chronological_text_predictability(
            text_train,
            raw_response_train,
            events.iloc[matched_indices]["date"].to_numpy(),
            k,
            lambda_z,
            lambda_q,
            seed,
            n_init,
            max_iter,
            tail_fraction,
        )

        shuffled_raw_response = raw_response_train[
            rng.permutation(len(raw_response_train))
        ]
        shuffled_response = l2_normalize(
            (shuffled_raw_response - response_center) / response_scale
        )
        shuffled_combined = l2_normalize(
            np.concatenate(
                [
                    np.sqrt(lambda_z) * text_train,
                    np.sqrt(lambda_q) * shuffled_response,
                ],
                axis=1,
            )
        )
        shuffled_centers = _fit_spherical_kmeans(
            shuffled_combined, k, seed + 1, n_init, max_iter
        )
        shuffled_impact = np.argmax(
            shuffled_combined @ shuffled_centers.T, axis=1
        )
        if np.unique(shuffled_impact).size < k:
            summary_rows.append(
                {
                    "response_candidate_id": candidate_id,
                    "status": "rejected_dead_shuffled_impact_group",
                    "news_level": level,
                    "response_kind": response_kind,
                    "k": k,
                    "lambda_z": lambda_z,
                    "lambda_q": lambda_q,
                    "n_train_response_events": len(matched_indices),
                    "response_validation_or_test_used": False,
                }
            )
            continue
        shuffled_text_centers = _text_centroids(
            text_train, shuffled_impact, k
        )
        shuffled_train_groups = np.argmax(
            text_train @ shuffled_text_centers.T, axis=1
        )
        shuffled_accuracy = float(
            np.mean(shuffled_train_groups == shuffled_impact)
        )
        (
            shuffled_tail_accuracy,
            shuffled_tail_ari,
            shuffled_tail_count,
        ) = _chronological_text_predictability(
            text_train,
            shuffled_raw_response,
            events.iloc[matched_indices]["date"].to_numpy(),
            k,
            lambda_z,
            lambda_q,
            seed + 1,
            n_init,
            max_iter,
            tail_fraction,
        )
        shuffled_all_groups = np.argmax(
            text_all @ shuffled_text_centers.T, axis=1
        )

        impact_by_event = {
            int(index): int(label)
            for index, label in zip(matched_indices, impact_labels)
        }
        frame = events.iloc[level_indices][
            ["event_id", "date", "split", "news_level"]
        ].copy()
        frame["response_candidate_id"] = candidate_id
        frame["response_kind"] = response_kind
        frame["lambda_z"] = lambda_z
        frame["lambda_q"] = lambda_q
        frame["response_aware_group_text_only"] = text_groups.astype(int)
        frame["response_aware_text_novelty"] = (
            1.0 - nearest_text_similarity
        )
        frame["impact_group_true_train_only"] = [
            impact_by_event.get(int(index), -1) for index in level_indices
        ]
        frame["shuffled_response_group_text_only"] = shuffled_all_groups.astype(
            int
        )
        assignment_frames.append(frame)
        forecasting_feasible = bool(
            np.isfinite(tail_accuracy)
            and tail_accuracy >= predictability_threshold
            and (
                not np.isfinite(shuffled_tail_accuracy)
                or tail_accuracy > shuffled_tail_accuracy
            )
        )
        conclusion = (
            "Response-aware text assignment shows train-tail forecasting potential."
            if forecasting_feasible
            else (
                "Response-aware grouping chỉ có giá trị hậu nghiệm, "
                "không khả thi cho forecasting."
            )
        )
        summary_rows.append(
            {
                "response_candidate_id": candidate_id,
                "status": "completed",
                "news_level": level,
                "response_kind": response_kind,
                "k": k,
                "lambda_z": lambda_z,
                "lambda_q": lambda_q,
                "n_train_response_events": len(matched_indices),
                "n_train_tail_events": tail_count,
                "train_text_assignment_accuracy": train_accuracy,
                "train_text_assignment_ari": train_ari,
                "chronological_train_tail_accuracy": tail_accuracy,
                "chronological_train_tail_ari": tail_ari,
                "shuffled_response_train_accuracy": shuffled_accuracy,
                "shuffled_response_train_tail_accuracy": (
                    shuffled_tail_accuracy
                ),
                "shuffled_response_train_tail_ari": shuffled_tail_ari,
                "shuffled_response_train_tail_events": shuffled_tail_count,
                "minimum_text_predictability": predictability_threshold,
                "forecasting_feasible": forecasting_feasible,
                "response_validation_or_test_used": False,
                "validation_test_assignment_method": "text_centroid_only",
                "conclusion": conclusion,
            }
        )
        level_models[level] = {
            "k": k,
            "lambda_z": lambda_z,
            "lambda_q": lambda_q,
            "combined_centers_train_only": combined_centers.astype(np.float32),
            "text_centers_for_all_holdout_assignment": text_centers.astype(
                np.float32
            ),
            "shuffled_response_text_centers": shuffled_text_centers.astype(
                np.float32
            ),
        }
    return summary_rows, assignment_frames, level_models


def _response_aware_experiment_legacy(
    config: Mapping[str, Any],
    events: pd.DataFrame,
    embeddings: np.ndarray,
    seed: int,
    n_init: int,
    max_iter: int,
) -> tuple[Path, Path, Path]:
    events_variant = _events_variant(config)
    suffix = _variant_suffix(events_variant)
    summary_path = project_path(
        config,
        f"outputs/tables/response_aware_prototype_summary{suffix}.csv",
    )
    assignment_path = project_path(
        config, f"data/processed/response_aware_assignments{suffix}.parquet"
    )
    model_path = project_path(
        config, f"outputs/models/prototypes/response_aware{suffix}.joblib"
    )
    enabled = bool(
        _nested(
            config,
            "prototype",
            "response_aware",
            "enabled",
            default=True,
        )
    )
    if not enabled:
        summary = pd.DataFrame(
            [
                {
                    "status": "disabled",
                    "response_validation_or_test_used": False,
                }
            ]
        )
        atomic_write_csv(summary, summary_path, index=False)
        write_table(
            pd.DataFrame(
                columns=[
                    "event_id",
                    "date",
                    "split",
                    "news_level",
                    "response_aware_group_text_only",
                ]
            ),
            assignment_path,
        )
        atomic_joblib_dump({"enabled": False}, model_path)
        return summary_path, assignment_path, model_path
    try:
        response_frame, tickers, response_kind = _response_matrix_from_train(
            config, "signed"
        )
    except FileNotFoundError as exc:
        summary = pd.DataFrame(
            [
                {
                    "status": "skipped_missing_residual_targets",
                    "missing_path": str(exc),
                    "response_validation_or_test_used": False,
                }
            ]
        )
        atomic_write_csv(summary, summary_path, index=False)
        write_table(
            pd.DataFrame(
                columns=[
                    "event_id",
                    "date",
                    "split",
                    "news_level",
                    "response_aware_group_text_only",
                ]
            ),
            assignment_path,
        )
        atomic_joblib_dump({"enabled": True, "status": "missing_residuals"}, model_path)
        LOGGER.warning(
            "Response-aware prototype experiment skipped because %s is absent.",
            exc,
        )
        return summary_path, assignment_path, model_path

    response_dates = pd.to_datetime(response_frame["feature_date"]).dt.normalize()
    response_lookup = {
        pd.Timestamp(date): vector
        for date, vector in zip(
            response_dates,
            response_frame[tickers].to_numpy(dtype=float),
        )
    }
    response_config = _nested(
        config, "prototype", "response_aware", default={}
    )
    if not isinstance(response_config, Mapping):
        response_config = {}
    requested_k = int(
        response_config.get(
            "k", _nested(config, "prototype", "preferred_k", default=8)
        )
    )
    lambda_z = float(response_config.get("lambda_z", 1.0))
    lambda_q = float(response_config.get("lambda_q", 1.0))
    if lambda_z <= 0 or lambda_q <= 0:
        raise ValueError("Response-aware lambda_z and lambda_q must be positive.")
    tail_fraction = float(response_config.get("train_tail_fraction", 0.2))
    if not 0.0 < tail_fraction < 1.0:
        raise ValueError("response-aware train_tail_fraction must be in (0, 1).")
    predictability_threshold = float(
        response_config.get("minimum_text_predictability", 0.35)
    )
    rng = np.random.default_rng(seed + 4049)
    summary_rows: list[dict[str, Any]] = []
    assignment_frames: list[pd.DataFrame] = []
    model_bundles: dict[str, Any] = {
        "kind": "response_aware",
        "fit_split": "train_oof_residuals_only",
        "response_kind": response_kind,
        "tickers": tickers,
        "levels": {},
    }
    for level in NEWS_LEVELS:
        level_mask = events["news_level"].to_numpy() == level
        matched_mask = (
            level_mask
            & (events["split"].to_numpy() == "train")
            & events["date"].isin(response_lookup).to_numpy()
        )
        matched_indices = np.flatnonzero(matched_mask)
        if len(matched_indices) < max(2, requested_k):
            summary_rows.append(
                {
                    "status": "skipped_too_few_train_response_events",
                    "news_level": level,
                    "n_train_response_events": len(matched_indices),
                    "requested_k": requested_k,
                    "response_kind": response_kind,
                    "response_validation_or_test_used": False,
                }
            )
            continue
        k = min(requested_k, len(matched_indices))
        text_train = embeddings[matched_indices]
        response_train = np.vstack(
            [response_lookup[pd.Timestamp(events.iloc[index]["date"])] for index in matched_indices]
        )
        combined = l2_normalize(
            np.concatenate(
                [
                    np.sqrt(lambda_z) * text_train,
                    np.sqrt(lambda_q) * response_train,
                ],
                axis=1,
            )
        )
        combined_centers = _fit_spherical_kmeans(
            combined, k, seed, n_init, max_iter
        )
        impact_labels = np.argmax(combined @ combined_centers.T, axis=1)
        text_centers = _text_centroids(text_train, impact_labels, k)
        level_indices = np.flatnonzero(level_mask)
        text_all = embeddings[level_indices]
        text_similarities = np.clip(text_all @ text_centers.T, -1.0, 1.0)
        text_groups = np.argmax(text_similarities, axis=1)
        nearest_text_similarity = text_similarities[
            np.arange(len(text_all)), text_groups
        ]
        text_train_groups = np.argmax(text_train @ text_centers.T, axis=1)
        train_accuracy = float(np.mean(text_train_groups == impact_labels))
        train_ari = float(
            adjusted_rand_score(impact_labels, text_train_groups)
        )
        tail_accuracy, tail_ari, tail_count = _chronological_text_predictability(
            text_train,
            response_train,
            events.iloc[matched_indices]["date"].to_numpy(),
            k,
            lambda_z,
            lambda_q,
            seed,
            n_init,
            max_iter,
            tail_fraction,
        )

        shuffled_response = response_train[rng.permutation(len(response_train))]
        shuffled_combined = l2_normalize(
            np.concatenate(
                [
                    np.sqrt(lambda_z) * text_train,
                    np.sqrt(lambda_q) * shuffled_response,
                ],
                axis=1,
            )
        )
        shuffled_centers = _fit_spherical_kmeans(
            shuffled_combined, k, seed + 1, n_init, max_iter
        )
        shuffled_impact = np.argmax(
            shuffled_combined @ shuffled_centers.T, axis=1
        )
        shuffled_text_centers = _text_centroids(
            text_train, shuffled_impact, k
        )
        shuffled_train_groups = np.argmax(
            text_train @ shuffled_text_centers.T, axis=1
        )
        shuffled_accuracy = float(
            np.mean(shuffled_train_groups == shuffled_impact)
        )
        (
            shuffled_tail_accuracy,
            shuffled_tail_ari,
            shuffled_tail_count,
        ) = _chronological_text_predictability(
            text_train,
            shuffled_response,
            events.iloc[matched_indices]["date"].to_numpy(),
            k,
            lambda_z,
            lambda_q,
            seed + 1,
            n_init,
            max_iter,
            tail_fraction,
        )
        shuffled_all_groups = np.argmax(
            text_all @ shuffled_text_centers.T, axis=1
        )

        impact_by_event = {
            int(index): int(label)
            for index, label in zip(matched_indices, impact_labels)
        }
        frame = events.iloc[level_indices][
            ["event_id", "date", "split", "news_level"]
        ].copy()
        frame["response_aware_group_text_only"] = text_groups.astype(int)
        frame["response_aware_text_novelty"] = (
            1.0 - nearest_text_similarity
        )
        frame["impact_group_true_train_only"] = [
            impact_by_event.get(int(index), -1) for index in level_indices
        ]
        frame["shuffled_response_group_text_only"] = shuffled_all_groups.astype(
            int
        )
        assignment_frames.append(frame)
        forecasting_feasible = bool(
            np.isfinite(tail_accuracy)
            and tail_accuracy >= predictability_threshold
            and (
                not np.isfinite(shuffled_tail_accuracy)
                or tail_accuracy > shuffled_tail_accuracy
            )
        )
        conclusion = (
            "Response-aware text assignment shows train-tail forecasting potential."
            if forecasting_feasible
            else (
                "Response-aware grouping chỉ có giá trị hậu nghiệm, "
                "không khả thi cho forecasting."
            )
        )
        summary_rows.append(
            {
                "status": "completed",
                "news_level": level,
                "response_kind": response_kind,
                "k": k,
                "lambda_z": lambda_z,
                "lambda_q": lambda_q,
                "n_train_response_events": len(matched_indices),
                "n_train_tail_events": tail_count,
                "train_text_assignment_accuracy": train_accuracy,
                "train_text_assignment_ari": train_ari,
                "chronological_train_tail_accuracy": tail_accuracy,
                "chronological_train_tail_ari": tail_ari,
                "shuffled_response_train_accuracy": shuffled_accuracy,
                "shuffled_response_train_tail_accuracy": (
                    shuffled_tail_accuracy
                ),
                "shuffled_response_train_tail_ari": shuffled_tail_ari,
                "shuffled_response_train_tail_events": shuffled_tail_count,
                "minimum_text_predictability": predictability_threshold,
                "forecasting_feasible": forecasting_feasible,
                "response_validation_or_test_used": False,
                "validation_test_assignment_method": "text_centroid_only",
                "conclusion": conclusion,
            }
        )
        model_bundles["levels"][level] = {
            "k": k,
            "lambda_z": lambda_z,
            "lambda_q": lambda_q,
            "combined_centers_train_only": combined_centers.astype(np.float32),
            "text_centers_for_all_holdout_assignment": text_centers.astype(
                np.float32
            ),
            "shuffled_response_text_centers": shuffled_text_centers.astype(
                np.float32
            ),
        }
    response_summary = pd.DataFrame(summary_rows)
    response_assignments = (
        pd.concat(assignment_frames, ignore_index=True)
        if assignment_frames
        else pd.DataFrame(
            columns=[
                "event_id",
                "date",
                "split",
                "news_level",
                "response_aware_group_text_only",
                "response_aware_text_novelty",
                "impact_group_true_train_only",
                "shuffled_response_group_text_only",
            ]
        )
    )
    atomic_write_csv(response_summary, summary_path, index=False)
    write_table(response_assignments, assignment_path)
    atomic_joblib_dump(model_bundles, model_path)
    return summary_path, assignment_path, model_path


def _response_aware_experiment(
    config: Mapping[str, Any],
    events: pd.DataFrame,
    embeddings: np.ndarray,
    seed: int,
    n_init: int,
    max_iter: int,
) -> tuple[Path, Path, Path]:
    """Run the configured train-only response-signature/lambda grid."""

    events_variant = _events_variant(config)
    suffix = _variant_suffix(events_variant)
    summary_path = project_path(
        config,
        f"outputs/tables/response_aware_prototype_summary{suffix}.csv",
    )
    assignment_path = project_path(
        config, f"data/processed/response_aware_assignments{suffix}.parquet"
    )
    model_path = project_path(
        config, f"outputs/models/prototypes/response_aware{suffix}.joblib"
    )
    response_config = _nested(
        config, "prototype", "response_aware", default={}
    )
    if not isinstance(response_config, Mapping):
        response_config = {}
    if not bool(response_config.get("enabled", True)):
        atomic_write_csv(
            pd.DataFrame(
                [
                    {
                        "status": "disabled",
                        "response_validation_or_test_used": False,
                    }
                ]
            ),
            summary_path,
            index=False,
        )
        write_table(
            pd.DataFrame(
                columns=[
                    "event_id",
                    "date",
                    "split",
                    "news_level",
                    "response_candidate_id",
                    "response_aware_group_text_only",
                ]
            ),
            assignment_path,
        )
        atomic_joblib_dump({"enabled": False}, model_path)
        return summary_path, assignment_path, model_path

    signatures = [
        str(value)
        for value in response_config.get(
            "signature_types",
            [response_config.get("response", "signed")],
        )
    ]
    lambda_embedding = [
        float(value)
        for value in response_config.get(
            "lambda_embedding",
            [response_config.get("lambda_z", 1.0)],
        )
    ]
    lambda_response = [
        float(value)
        for value in response_config.get(
            "lambda_response",
            [response_config.get("lambda_q", 1.0)],
        )
    ]
    if not signatures:
        raise ValueError("response_aware.signature_types cannot be empty.")
    if (
        not lambda_embedding
        or not lambda_response
        or min(lambda_embedding) <= 0
        or min(lambda_response) <= 0
    ):
        raise ValueError(
            "response_aware lambda_embedding/lambda_response must be positive."
        )
    requested_k = int(
        response_config.get(
            "k", _nested(config, "prototype", "preferred_k", default=8)
        )
    )
    tail_fraction = float(response_config.get("train_tail_fraction", 0.2))
    if not 0.0 < tail_fraction < 1.0:
        raise ValueError("response-aware train_tail_fraction must be in (0, 1).")
    predictability_threshold = float(
        response_config.get(
            "min_validation_assignment_accuracy",
            response_config.get("minimum_text_predictability", 0.35),
        )
    )
    summary_rows: list[dict[str, Any]] = []
    assignment_frames: list[pd.DataFrame] = []
    model_bundle: dict[str, Any] = {
        "kind": "response_aware_grid",
        "fit_split": "train_oof_residuals_only",
        "response_validation_or_test_used": False,
        "events_variant": events_variant,
        "candidates": {},
    }
    for signature in signatures:
        try:
            response_frame, tickers, normalized_signature = (
                _response_matrix_from_train(config, signature)
            )
        except FileNotFoundError as exc:
            summary = pd.DataFrame(
                [
                    {
                        "status": "skipped_missing_residual_targets",
                        "missing_path": str(exc),
                        "response_validation_or_test_used": False,
                    }
                ]
            )
            atomic_write_csv(summary, summary_path, index=False)
            write_table(
                pd.DataFrame(
                    columns=[
                        "event_id",
                        "date",
                        "split",
                        "news_level",
                        "response_candidate_id",
                        "response_aware_group_text_only",
                    ]
                ),
                assignment_path,
            )
            atomic_joblib_dump(
                {"enabled": True, "status": "missing_residuals"}, model_path
            )
            LOGGER.warning(
                "Response-aware prototype experiment skipped because %s is absent.",
                exc,
            )
            return summary_path, assignment_path, model_path
        for lambda_z in lambda_embedding:
            for lambda_q in lambda_response:
                (
                    candidate_summary,
                    candidate_assignments,
                    candidate_models,
                ) = _fit_response_candidate(
                    events=events,
                    embeddings=embeddings,
                    response_frame=response_frame,
                    tickers=tickers,
                    response_kind=normalized_signature,
                    requested_k=requested_k,
                    lambda_z=lambda_z,
                    lambda_q=lambda_q,
                    tail_fraction=tail_fraction,
                    predictability_threshold=predictability_threshold,
                    seed=seed,
                    n_init=n_init,
                    max_iter=max_iter,
                )
                summary_rows.extend(candidate_summary)
                assignment_frames.extend(candidate_assignments)
                candidate_id = (
                    f"response_{normalized_signature}"
                    f"_lz{_float_token(lambda_z)}"
                    f"_lq{_float_token(lambda_q)}_k{requested_k}"
                )
                model_bundle["candidates"][candidate_id] = {
                    "signature_requested": signature,
                    "response_kind": normalized_signature,
                    "lambda_z": lambda_z,
                    "lambda_q": lambda_q,
                    "tickers": tickers,
                    "levels": candidate_models,
                }
    response_summary = pd.DataFrame(summary_rows)
    response_assignments = (
        pd.concat(assignment_frames, ignore_index=True)
        if assignment_frames
        else pd.DataFrame(
            columns=[
                "event_id",
                "date",
                "split",
                "news_level",
                "response_candidate_id",
                "response_aware_group_text_only",
                "response_aware_text_novelty",
                "impact_group_true_train_only",
                "shuffled_response_group_text_only",
            ]
        )
    )
    if not response_summary.empty:
        response_summary["events_variant"] = events_variant
    atomic_write_csv(response_summary, summary_path, index=False)
    write_table(response_assignments, assignment_path)
    atomic_joblib_dump(model_bundle, model_path)
    return summary_path, assignment_path, model_path


def _selected_assignments(
    events: pd.DataFrame,
    selected_rows: list[pd.Series],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for selected in selected_rows:
        assignment_path = Path(str(selected["assignment_path"]))
        arrays = np.load(assignment_path, allow_pickle=False)
        event_indices = arrays["event_indices"].astype(int)
        level_events = events.iloc[event_indices].reset_index(drop=True)
        if "event_ids" not in arrays.files or not np.array_equal(
            arrays["event_ids"].astype(str),
            level_events["event_id"].astype(str).to_numpy(),
        ):
            raise ValueError(
                f"Stale event ordering in prototype assignment {assignment_path}."
            )
        soft = arrays["soft_assignment"]
        if len(level_events) != len(soft):
            raise ValueError(f"Assignment length mismatch in {assignment_path}.")
        frame = level_events[["event_id", "date", "split", "news_level"]].copy()
        frame["candidate_id"] = selected["candidate_id"]
        frame["k"] = int(selected["k"])
        frame["pca_dim"] = selected["pca_dim"]
        frame["temperature"] = float(selected["temperature"])
        frame["seed"] = int(selected["seed"])
        frame["hard_cluster_id"] = arrays["hard_cluster_id"].astype(int)
        frame["soft_assignment"] = [
            json.dumps(vector.tolist(), separators=(",", ":")) for vector in soft
        ]
        for column in (
            "nearest_similarity",
            "nearest_distance",
            "novelty",
            "assignment_entropy",
            "effective_soft_prototypes",
        ):
            frame[column] = arrays[column].astype(float)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(
            columns=[
                "event_id",
                "date",
                "split",
                "news_level",
                "candidate_id",
                "k",
                "pca_dim",
                "temperature",
                "seed",
                "hard_cluster_id",
                "soft_assignment",
                "nearest_similarity",
                "nearest_distance",
                "novelty",
                "assignment_entropy",
                "effective_soft_prototypes",
            ]
        )
    return pd.concat(frames, ignore_index=True).sort_values(
        ["date", "news_level", "event_id"], kind="mergesort"
    )


def _build_fold_codebooks(
    config: Mapping[str, Any],
    events: pd.DataFrame,
    embeddings: np.ndarray,
    seed: int,
    levels: Sequence[str],
    k_values: Sequence[int],
    pca_dims: Sequence[int | None],
    temperatures: Sequence[float],
    n_init: int,
    max_iter: int,
    min_cluster_events: int,
    max_small_fraction: float,
    min_effective_fraction: float,
    silhouette_sample_size: int,
    append: bool = False,
) -> Path:
    """Fit an independent semantic grid inside every expanding fold."""

    events_variant = _events_variant(config)
    suffix = _variant_suffix(events_variant)
    output_path = project_path(
        config, f"data/processed/prototype_fold_manifest{suffix}.csv"
    )
    enabled = bool(
        _nested(
            config,
            "prototype",
            "build_fold_codebooks",
            default=True,
        )
    )
    if not enabled:
        atomic_write_csv(
            pd.DataFrame(
                [
                    {
                        "status": "disabled",
                        "events_variant": events_variant,
                    }
                ]
            ),
            output_path,
            index=False,
        )
        return output_path
    folds_path = _path(
        config,
        (("paths", "chronological_folds"),),
        "outputs/tables/chronological_folds.csv",
    )
    market_path = _path(
        config,
        (("paths", "market_supervised"),),
        "data/processed/market_supervised.parquet",
    )
    folds = safe_read_table(folds_path)
    market = safe_read_table(market_path)
    validate_required_columns(
        folds,
        [
            "fold",
            "train_start",
            "train_end",
            "validation_start",
            "validation_end",
        ],
        "chronological folds",
    )
    validate_required_columns(
        market,
        ["feature_date", "target_date"],
        "market supervised fold mapping",
    )
    date_map = market[["feature_date", "target_date"]].copy()
    date_map["feature_date"] = pd.to_datetime(
        date_map["feature_date"], errors="raise"
    ).dt.normalize()
    date_map["target_date"] = pd.to_datetime(
        date_map["target_date"], errors="raise"
    ).dt.normalize()
    conflicts = date_map.groupby("feature_date", observed=True)[
        "target_date"
    ].nunique()
    if int(conflicts.max()) > 1:
        raise ValueError("A feature_date maps to multiple target dates.")
    target_lookup = (
        date_map.drop_duplicates("feature_date")
        .set_index("feature_date")["target_date"]
        .to_dict()
    )
    event_target_dates = events["date"].map(target_lookup)
    grid_mode = str(
        _nested(
            config,
            "prototype",
            "fold_grid_mode",
            default="full",
        )
    ).lower()
    if grid_mode not in {"full", "preferred"}:
        raise ValueError("prototype.fold_grid_mode must be full or preferred.")
    fold_assignment_root = project_path(
        config, f"data/processed/prototype_assignments{suffix}/folds"
    )
    fold_model_root = project_path(
        config, f"outputs/models/prototypes{suffix}/folds"
    )
    ensure_directories([fold_assignment_root, fold_model_root])
    rows: list[dict[str, Any]] = []
    for fold in folds.itertuples(index=False):
        fold_id = int(fold.fold)
        train_start = pd.Timestamp(fold.train_start).normalize()
        train_end = pd.Timestamp(fold.train_end).normalize()
        validation_start = pd.Timestamp(fold.validation_start).normalize()
        validation_end = pd.Timestamp(fold.validation_end).normalize()
        if not train_end < validation_start:
            raise AssertionError(
                f"Fold {fold_id} leaks: train_end={train_end}, "
                f"validation_start={validation_start}."
            )
        fold_train = event_target_dates.between(
            train_start, train_end, inclusive="both"
        ).to_numpy()
        fold_validation = event_target_dates.between(
            validation_start, validation_end, inclusive="both"
        ).to_numpy()
        fold_scope = fold_train | fold_validation
        fold_assignment_dir = fold_assignment_root / f"fold_{fold_id}"
        fold_model_dir = fold_model_root / f"fold_{fold_id}"
        ensure_directories([fold_assignment_dir, fold_model_dir])
        for level in levels:
            level_mask = events["news_level"].to_numpy() == level
            train_mask_global = level_mask & fold_train
            scope_indices = np.flatnonzero(level_mask & fold_scope)
            train_values = embeddings[train_mask_global]
            scope_values = embeddings[scope_indices]
            if not len(train_values) or not len(scope_values):
                rows.append(
                    {
                        "fold_id": fold_id,
                        "news_level": level,
                        "status": "no_fold_train_events",
                        "eligible": False,
                        "n_train_events": len(train_values),
                        "n_validation_events": int(
                            (level_mask & fold_validation).sum()
                        ),
                        "train_start": train_start,
                        "train_end": train_end,
                        "validation_start": validation_start,
                        "validation_end": validation_end,
                        "events_variant": events_variant,
                        "prototype_seed": seed,
                    }
                )
                continue
            if grid_mode == "preferred":
                preferred_k, preferred_pca, preferred_tau, _ = (
                    _selection_preferences(config, level)
                )
                fold_k_values = [preferred_k]
                fold_pca_dims = [preferred_pca]
                fold_temperatures = [preferred_tau]
            else:
                fold_k_values = list(k_values)
                fold_pca_dims = list(pca_dims)
                fold_temperatures = list(temperatures)
            scope_train_mask = fold_train[scope_indices]
            for pca_dim in fold_pca_dims:
                projection_error: str | None = None
                try:
                    pca, transformed, explained_variance = _fit_projection(
                        train_values,
                        scope_values,
                        pca_dim,
                        seed + fold_id,
                    )
                except ValueError as exc:
                    projection_error = str(exc)
                    pca, transformed, explained_variance = (
                        None,
                        np.empty((0, 0)),
                        np.nan,
                    )
                for k in fold_k_values:
                    base_id = _base_id(level, k, pca_dim, seed)
                    model_path = fold_model_dir / f"{base_id}.joblib"
                    if projection_error is not None or len(train_values) < k:
                        reason = (
                            projection_error
                            if projection_error is not None
                            else f"train_events={len(train_values)}<k={k}"
                        )
                        for temperature in fold_temperatures:
                            rows.append(
                                {
                                    "fold_id": fold_id,
                                    "candidate_id": _candidate_id(
                                        level,
                                        k,
                                        pca_dim,
                                        temperature,
                                        seed,
                                    ),
                                    "news_level": level,
                                    "k": k,
                                    "pca_dim": pca_dim,
                                    "temperature": temperature,
                                    "prototype_seed": seed,
                                    "status": "rejected",
                                    "eligible": False,
                                    "rejection_reason": reason,
                                    "n_train_events": len(train_values),
                                    "n_validation_events": int(
                                        (level_mask & fold_validation).sum()
                                    ),
                                    "train_start": train_start,
                                    "train_end": train_end,
                                    "validation_start": validation_start,
                                    "validation_end": validation_end,
                                    "events_variant": events_variant,
                                    "fit_scope": "fold_train_only",
                                }
                            )
                        continue
                    centers = _fit_spherical_kmeans(
                        transformed[scope_train_mask],
                        k,
                        seed + fold_id,
                        n_init,
                        max_iter,
                    )
                    diagnostics = _usage_diagnostics(
                        transformed[scope_train_mask],
                        centers,
                        min_cluster_events,
                        max_small_fraction,
                        min_effective_fraction,
                        silhouette_sample_size,
                        seed + fold_id,
                    )
                    eligible = not diagnostics["rejection_reasons"]
                    if eligible:
                        atomic_joblib_dump(
                            {
                                "kind": "semantic_prototype_fold",
                                "fold_id": fold_id,
                                "news_level": level,
                                "seed": seed,
                                "k": k,
                                "pca_dim": pca_dim,
                                "pca": pca,
                                "centroids": centers.astype(np.float32),
                                "fit_scope": "fold_train_only",
                                "train_start": train_start,
                                "train_end": train_end,
                                "validation_start": validation_start,
                                "validation_end": validation_end,
                                "events_variant": events_variant,
                            },
                            model_path,
                        )
                    for temperature in fold_temperatures:
                        candidate_id = _candidate_id(
                            level, k, pca_dim, temperature, seed
                        )
                        assignment_path = (
                            fold_assignment_dir / f"{candidate_id}.npz"
                        )
                        if eligible:
                            assignment = _assign(
                                transformed, centers, temperature
                            )
                            _atomic_save_npz(
                                assignment_path,
                                event_indices=scope_indices.astype(np.int64),
                                event_ids=events.iloc[scope_indices][
                                    "event_id"
                                ].astype(str).to_numpy(dtype="U"),
                                fold_role=np.where(
                                    scope_train_mask, 0, 1
                                ).astype(np.int8),
                                **assignment,
                            )
                        pca_token = (
                            "none" if pca_dim is None else str(pca_dim)
                        )
                        family = (
                            f"grid_k{k}_pca{pca_token}"
                            f"_tau{_float_token(temperature)}"
                        )
                        rows.append(
                            {
                                "fold_id": fold_id,
                                "candidate_id": candidate_id,
                                "representation_variant_family": family,
                                "news_level": level,
                                "k": k,
                                "pca_dim": pca_dim,
                                "temperature": temperature,
                                "prototype_seed": seed,
                                "status": (
                                    "eligible" if eligible else "rejected"
                                ),
                                "eligible": eligible,
                                "rejection_reason": ";".join(
                                    diagnostics["rejection_reasons"]
                                ),
                                "n_train_events": len(train_values),
                                "n_validation_events": int(
                                    (level_mask & fold_validation).sum()
                                ),
                                "effective_fraction": diagnostics[
                                    "effective_fraction"
                                ],
                                "small_prototype_fraction": diagnostics[
                                    "small_fraction"
                                ],
                                "silhouette_score": diagnostics["silhouette"],
                                "pca_explained_variance": explained_variance,
                                "model_path": (
                                    str(model_path) if eligible else ""
                                ),
                                "assignment_path": (
                                    str(assignment_path) if eligible else ""
                                ),
                                "train_start": train_start,
                                "train_end": train_end,
                                "validation_start": validation_start,
                                "validation_end": validation_end,
                                "events_variant": events_variant,
                                "fit_scope": "fold_train_only",
                            }
                        )
    manifest = pd.DataFrame(rows)
    if append and output_path.exists():
        previous = safe_read_table(output_path)
        manifest = pd.concat([previous, manifest], ignore_index=True)
        identity_columns = [
            column
            for column in (
                "fold_id",
                "candidate_id",
                "news_level",
                "prototype_seed",
            )
            if column in manifest
        ]
        if identity_columns:
            manifest = manifest.drop_duplicates(
                identity_columns, keep="last"
            )
    atomic_write_csv(manifest, output_path, index=False)
    return output_path


def run(config: dict) -> dict[str, Path]:
    """Fit the full semantic prototype grid and transform all event splits."""

    primary_seed = int(
        _nested(config, "project", "seed", default=config.get("seed", 42))
    )
    deterministic = bool(
        _nested(config, "project", "deterministic", default=True)
    )
    set_global_seed(primary_seed, deterministic=deterministic)
    ensure_directories(config)
    events_variant = _events_variant(config)
    suffix = _variant_suffix(events_variant)
    events, embeddings = _load_inputs(config)
    _assert_chronology(events)

    prototype_config = _nested(config, "prototype", default={})
    if not isinstance(prototype_config, Mapping):
        raise TypeError("prototype configuration must be a mapping.")
    levels = tuple(prototype_config.get("levels", NEWS_LEVELS))
    unknown_levels = sorted(set(levels).difference(NEWS_LEVELS))
    if unknown_levels:
        raise ValueError(f"Unknown prototype levels: {unknown_levels}")
    k_values = sorted({int(value) for value in prototype_config.get("k_values", [4, 8, 12, 16, 24, 32])})
    if not k_values or min(k_values) < 2:
        raise ValueError("prototype.k_values must contain integers >= 2.")
    pca_dims = _normalize_pca_dims(
        prototype_config.get("pca_dims", [None, 32, 64, 128])
    )
    temperatures = sorted(
        {float(value) for value in prototype_config.get("temperatures", [0.05, 0.1, 0.2, 0.5])}
    )
    if not temperatures or min(temperatures) <= 0:
        raise ValueError("prototype.temperatures must be positive.")
    configured_seeds = _nested(config, "robustness", "seeds", default=[primary_seed])
    seeds = list(dict.fromkeys([primary_seed, *(int(value) for value in configured_seeds)]))
    if not bool(prototype_config.get("fit_stability_seeds", True)):
        seeds = [primary_seed]

    n_init = int(prototype_config.get("n_init", 20))
    max_iter = int(prototype_config.get("max_iter", 500))
    min_cluster_events = int(prototype_config.get("min_events_per_cluster", 5))
    max_small_fraction = float(
        prototype_config.get("max_small_cluster_fraction", 0.5)
    )
    min_effective_fraction = float(
        prototype_config.get("min_effective_fraction", 0.35)
    )
    silhouette_sample_size = int(
        prototype_config.get("silhouette_sample_size", 5000)
    )
    top_examples = int(prototype_config.get("examples_per_prototype", 10))

    assignment_dir = project_path(
        config, f"data/processed/prototype_assignments{suffix}"
    )
    model_dir = project_path(
        config, f"outputs/models/prototypes{suffix}"
    )
    ensure_directories([assignment_dir, model_dir])
    summary_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    example_rows: list[dict[str, Any]] = []
    stability_cache: dict[
        tuple[str, int | None, int, int], dict[str, np.ndarray]
    ] = {}

    for level in levels:
        global_indices = np.flatnonzero(events["news_level"].to_numpy() == level)
        level_events = events.iloc[global_indices].reset_index(drop=True)
        level_values = embeddings[global_indices]
        train_mask = level_events["split"].to_numpy() == "train"
        train_values = level_values[train_mask]
        if not len(train_values):
            for seed in seeds:
                for pca_dim in pca_dims:
                    for k in k_values:
                        for temperature in temperatures:
                            summary_rows.append(
                                {
                                    "candidate_id": _candidate_id(
                                        level, k, pca_dim, temperature, seed
                                    ),
                                    "news_level": level,
                                    "seed": seed,
                                    "pca_dim": pca_dim,
                                    "k": k,
                                    "temperature": temperature,
                                    "n_train_events": 0,
                                    "n_all_events": len(level_events),
                                    "eligible": False,
                                    "rejection_reason": "no_train_events",
                                }
                            )
            continue
        for seed in seeds:
            for pca_dim in pca_dims:
                projection_error: str | None = None
                try:
                    pca, transformed, explained_variance = _fit_projection(
                        train_values, level_values, pca_dim, seed
                    )
                except ValueError as exc:
                    projection_error = str(exc)
                    pca, transformed, explained_variance = None, np.empty((0, 0)), np.nan
                for k in k_values:
                    if projection_error is not None:
                        for temperature in temperatures:
                            summary_rows.append(
                                {
                                    "candidate_id": _candidate_id(
                                        level, k, pca_dim, temperature, seed
                                    ),
                                    "news_level": level,
                                    "seed": seed,
                                    "pca_dim": pca_dim,
                                    "k": k,
                                    "temperature": temperature,
                                    "n_train_events": len(train_values),
                                    "n_all_events": len(level_events),
                                    "eligible": False,
                                    "rejection_reason": projection_error,
                                }
                            )
                        continue
                    if len(train_values) < k:
                        for temperature in temperatures:
                            summary_rows.append(
                                {
                                    "candidate_id": _candidate_id(
                                        level, k, pca_dim, temperature, seed
                                    ),
                                    "news_level": level,
                                    "seed": seed,
                                    "pca_dim": pca_dim,
                                    "k": k,
                                    "temperature": temperature,
                                    "n_train_events": len(train_values),
                                    "n_all_events": len(level_events),
                                    "eligible": False,
                                    "rejection_reason": (
                                        f"train_events={len(train_values)}<k={k}"
                                    ),
                                }
                            )
                        continue
                    centers = _fit_spherical_kmeans(
                        transformed[train_mask], k, seed, n_init, max_iter
                    )
                    diagnostics = _usage_diagnostics(
                        transformed[train_mask],
                        centers,
                        min_cluster_events,
                        max_small_fraction,
                        min_effective_fraction,
                        silhouette_sample_size,
                        seed,
                    )
                    base_candidate = _base_id(level, k, pca_dim, seed)
                    if np.all(diagnostics["counts"] > 0):
                        original_space_centers = np.vstack(
                            [
                                level_values[train_mask][
                                    diagnostics["hard"] == cluster_id
                                ].mean(axis=0)
                                for cluster_id in range(k)
                            ]
                        )
                        original_space_centers = l2_normalize(
                            original_space_centers
                        )
                        stability_cache[(level, pca_dim, k, seed)] = {
                            "hard": diagnostics["hard"],
                            # PCA bases differ by seed; stability cosine is only
                            # meaningful after mapping memberships back to the
                            # common original embedding space.
                            "centers": original_space_centers,
                        }
                    eligible = not diagnostics["rejection_reasons"]
                    model_path = model_dir / f"{base_candidate}.joblib"
                    if eligible:
                        atomic_joblib_dump(
                            {
                                "kind": "semantic_prototype",
                                "news_level": level,
                                "seed": seed,
                                "k": k,
                                "pca_dim": pca_dim,
                                "pca": pca,
                                "centroids": centers.astype(np.float32),
                                "embedding_dimension": embeddings.shape[1],
                                "fit_split": "train",
                                "events_variant": events_variant,
                                "train_event_ids_sha256": hashlib.sha256(
                                    "\n".join(
                                        level_events.loc[
                                            train_mask, "event_id"
                                        ].astype(str)
                                    ).encode("utf-8")
                                ).hexdigest(),
                            },
                            model_path,
                        )
                        if seed == primary_seed:
                            example_rows.extend(
                                _example_rows(
                                    level_events,
                                    transformed,
                                    train_mask,
                                    centers,
                                    base_candidate,
                                    top_examples,
                                )
                            )
                    for temperature in temperatures:
                        candidate = _candidate_id(
                            level, k, pca_dim, temperature, seed
                        )
                        assignment_path = assignment_dir / f"{candidate}.npz"
                        mean_entropy = np.nan
                        mean_novelty = np.nan
                        if eligible:
                            assignment = _assign(
                                transformed, centers, temperature
                            )
                            _atomic_save_npz(
                                assignment_path,
                                event_indices=global_indices.astype(np.int64),
                                event_ids=level_events["event_id"]
                                .astype(str)
                                .to_numpy(dtype="U"),
                                **assignment,
                            )
                            mean_entropy = float(
                                assignment["assignment_entropy"][train_mask].mean()
                            )
                            mean_novelty = float(
                                assignment["novelty"][train_mask].mean()
                            )
                            manifest_rows.append(
                                {
                                    "candidate_id": candidate,
                                    "base_candidate_id": base_candidate,
                                    "news_level": level,
                                    "seed": seed,
                                    "pca_dim": pca_dim,
                                    "k": k,
                                    "temperature": temperature,
                                    "eligible": True,
                                    "model_path": str(model_path),
                                    "assignment_path": str(assignment_path),
                                    "effective_fraction": diagnostics[
                                        "effective_fraction"
                                    ],
                                    "silhouette": diagnostics["silhouette"],
                                }
                            )
                        summary_rows.append(
                            {
                                "candidate_id": candidate,
                                "base_candidate_id": base_candidate,
                                "news_level": level,
                                "seed": seed,
                                "pca_dim": pca_dim,
                                "k": k,
                                "temperature": temperature,
                                "n_train_events": len(train_values),
                                "n_all_events": len(level_events),
                                "embedding_dimension": embeddings.shape[1],
                                "projected_dimension": transformed.shape[1],
                                "pca_explained_variance": explained_variance,
                                "min_cluster_size": int(
                                    diagnostics["counts"].min()
                                ),
                                "median_cluster_size": float(
                                    np.median(diagnostics["counts"])
                                ),
                                "max_cluster_size": int(
                                    diagnostics["counts"].max()
                                ),
                                "dead_prototype_fraction": diagnostics[
                                    "dead_fraction"
                                ],
                                "small_prototype_fraction": diagnostics[
                                    "small_fraction"
                                ],
                                "effective_number_of_prototypes": diagnostics[
                                    "effective_number"
                                ],
                                "effective_fraction": diagnostics[
                                    "effective_fraction"
                                ],
                                "silhouette_score": diagnostics["silhouette"],
                                "davies_bouldin_index": diagnostics[
                                    "davies_bouldin"
                                ],
                                "mean_train_assignment_entropy": mean_entropy,
                                "mean_train_novelty": mean_novelty,
                                "eligible": eligible,
                                "rejection_reason": ";".join(
                                    diagnostics["rejection_reasons"]
                                ),
                                "fit_split": "train",
                            }
                        )

    summary = pd.DataFrame(summary_rows)
    manifest = pd.DataFrame(manifest_rows)
    if manifest.empty:
        manifest = pd.DataFrame(
            columns=[
                "candidate_id",
                "base_candidate_id",
                "news_level",
                "seed",
                "pca_dim",
                "k",
                "temperature",
                "eligible",
                "model_path",
                "assignment_path",
                "effective_fraction",
                "silhouette",
                "selected",
            ]
        )
    selected_rows: list[pd.Series] = []
    if not manifest.empty:
        for level in levels:
            selected = _choose_candidate(
                manifest, config, level, primary_seed
            )
            if selected is not None:
                selected_rows.append(selected)
                manifest.loc[
                    manifest["candidate_id"] == selected["candidate_id"], "selected"
                ] = True
        manifest["selected"] = manifest.get("selected", False)
        manifest["selected"] = manifest["selected"].fillna(False).astype(bool)

    selected_assignments = _selected_assignments(events, selected_rows)
    selected_path = project_path(
        config, f"data/processed/prototype_assignments{suffix}.parquet"
    )
    summary_path = project_path(
        config, f"outputs/tables/prototype_summary{suffix}.csv"
    )
    examples_path = project_path(
        config, f"outputs/tables/prototype_examples{suffix}.csv"
    )
    stability_path = project_path(
        config, f"outputs/tables/prototype_stability{suffix}.csv"
    )
    manifest_path = project_path(
        config, f"data/processed/prototype_manifest{suffix}.csv"
    )
    if not summary.empty:
        summary["events_variant"] = events_variant
    if not manifest.empty:
        manifest["events_variant"] = events_variant
    write_table(selected_assignments, selected_path)
    atomic_write_csv(summary, summary_path, index=False)
    examples = pd.DataFrame(
        example_rows,
        columns=[
            "base_candidate_id",
            "news_level",
            "prototype_id",
            "rank",
            "event_id",
            "date",
            "text",
            "cosine_similarity",
            "split",
        ],
    )
    atomic_write_csv(examples, examples_path, index=False)
    stability = _build_stability(stability_cache, primary_seed)
    if stability.empty:
        stability = pd.DataFrame(
            columns=[
                "news_level",
                "pca_dim",
                "k",
                "reference_seed",
                "comparison_seed",
                "adjusted_rand_index",
                "matched_centroid_cosine",
                "n_train_events",
            ]
        )
    atomic_write_csv(stability, stability_path, index=False)
    atomic_write_csv(manifest, manifest_path, index=False)
    build_fold_codebooks = bool(
        prototype_config.get("build_fold_codebooks", True)
    )
    fold_seeds = (
        seeds
        if build_fold_codebooks
        and bool(
            prototype_config.get(
                "build_fold_codebooks_all_seeds", True
            )
        )
        else [primary_seed]
    )
    fold_manifest_path: Path | None = None
    for fold_seed_index, fold_seed in enumerate(fold_seeds):
        fold_manifest_path = _build_fold_codebooks(
            config=config,
            events=events,
            embeddings=embeddings,
            seed=fold_seed,
            levels=levels,
            k_values=k_values,
            pca_dims=pca_dims,
            temperatures=temperatures,
            n_init=n_init,
            max_iter=max_iter,
            min_cluster_events=min_cluster_events,
            max_small_fraction=max_small_fraction,
            min_effective_fraction=min_effective_fraction,
            silhouette_sample_size=silhouette_sample_size,
            append=fold_seed_index > 0,
        )
    if fold_manifest_path is None:
        raise AssertionError("No fold prototype seed was configured.")
    (
        response_summary_path,
        response_assignment_path,
        response_model_path,
    ) = _response_aware_experiment(
        config,
        events,
        embeddings,
        primary_seed,
        n_init,
        max_iter,
    )
    LOGGER.info(
        "Fitted %d eligible semantic prototype candidates; selected %d level defaults.",
        len(manifest),
        len(selected_rows),
    )
    return {
        "prototype_assignments": selected_path,
        "prototype_manifest": manifest_path,
        "prototype_summary": summary_path,
        "prototype_examples": examples_path,
        "prototype_stability": stability_path,
        "prototype_fold_manifest": fold_manifest_path,
        "response_aware_summary": response_summary_path,
        "response_aware_assignments": response_assignment_path,
        "response_aware_model": response_model_path,
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
