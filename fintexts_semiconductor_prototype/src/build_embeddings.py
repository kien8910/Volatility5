"""Encode canonical news events with a frozen, cached sentence encoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.utils import (
    atomic_write_csv,
    ensure_directories,
    get_logger,
    l2_normalize,
    load_config,
    project_path,
    safe_read_table,
    set_global_seed,
    validate_required_columns,
)

LOGGER = get_logger(__name__)


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


def _atomic_save_npy(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".npy", prefix=f".{path.stem}.", dir=path.parent, delete=False
        ) as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_name = handle.name
        os.replace(temporary_name, path)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _event_fingerprint(events: pd.DataFrame, model_name: str) -> str:
    digest = hashlib.sha256(model_name.encode("utf-8"))
    for row in events[["event_id", "text_hash"]].itertuples(index=False):
        digest.update(str(row.event_id).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row.text_hash).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _load_reusable_cache(
    embedding_path: Path,
    metadata_path: Path,
    model_name: str,
    encoder_config_fingerprint: str,
) -> tuple[dict[tuple[str, str], np.ndarray], int | None]:
    if not embedding_path.exists() or not metadata_path.exists():
        return {}, None
    metadata = pd.read_csv(metadata_path)
    required = {
        "event_id",
        "text_hash",
        "embedding_index",
        "model_name",
        "encoder_config_fingerprint",
    }
    if not required.issubset(metadata.columns):
        LOGGER.warning("Ignoring legacy embedding cache with incomplete metadata.")
        return {}, None
    if not metadata["model_name"].astype(str).eq(model_name).all():
        LOGGER.info("Embedding model changed; existing cache will not be reused.")
        return {}, None
    if not metadata["encoder_config_fingerprint"].astype(str).eq(
        encoder_config_fingerprint
    ).all():
        LOGGER.info("Encoder configuration changed; cache will not be reused.")
        return {}, None
    cached = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
    if cached.ndim != 2 or len(cached) != len(metadata):
        LOGGER.warning("Ignoring inconsistent embedding cache at %s.", embedding_path)
        return {}, None
    indices = metadata["embedding_index"].to_numpy(dtype=int)
    if indices.size and (
        indices.min() < 0
        or indices.max() >= len(cached)
        or len(np.unique(indices)) != len(indices)
    ):
        LOGGER.warning("Ignoring embedding cache with invalid indices.")
        return {}, None
    result = {
        (str(row.event_id), str(row.text_hash)): np.asarray(
            cached[int(row.embedding_index)], dtype=np.float32
        ).copy()
        for row in metadata.itertuples(index=False)
    }
    return result, int(cached.shape[1])


def _build_encoder(model_name: str, config: Mapping[str, Any], device: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for the embedding stage. "
            "Install requirements.txt before running the pipeline."
        ) from exc
    trust_remote_code = bool(
        _nested(
            config,
            "embedding",
            "trust_remote_code",
            default=_nested(
                config, "embeddings", "trust_remote_code", default=False
            ),
        )
    )
    cache_folder = _nested(config, "embedding", "cache_dir")
    if cache_folder not in (None, ""):
        cache_folder = str(project_path(config, str(cache_folder)))
    revision = _nested(config, "embedding", "revision")
    encoder = SentenceTransformer(
        model_name,
        device=device,
        trust_remote_code=trust_remote_code,
        cache_folder=cache_folder,
        revision=None if revision in (None, "") else str(revision),
    )
    maximum_length = _nested(config, "embeddings", "max_sequence_length")
    if maximum_length is None:
        maximum_length = _nested(config, "embedding", "max_length")
    if maximum_length is not None:
        encoder.max_seq_length = int(maximum_length)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder


def run(config: dict) -> dict[str, Path]:
    """Create L2-normalized event embeddings, reusing unchanged cache rows."""

    seed = int(_nested(config, "project", "seed", default=config.get("seed", 42)))
    deterministic = bool(
        _nested(config, "project", "deterministic", default=True)
    )
    set_global_seed(seed, deterministic=deterministic)
    ensure_directories(config)

    events_variant = _events_variant(config)
    default_events_path = (
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
        default_events_path,
    )
    if not events_path.exists() and events_variant not in {"exact", "raw"}:
        events_path = project_path(config, "outputs/tables/canonical_events.csv")
    events = safe_read_table(events_path)
    validate_required_columns(
        events,
        ["event_id", "date", "news_level", "text", "text_hash"],
        "canonical events",
    )
    if events["event_id"].duplicated().any():
        raise ValueError("Canonical event_id must be unique before embedding.")
    events = events.sort_values(
        ["date", "news_level", "event_id"], kind="mergesort"
    ).reset_index(drop=True)
    if events.empty:
        raise ValueError("Cannot create embeddings for an empty event table.")

    use_official = bool(
        _nested(
            config,
            "embedding",
            "use_official_fintexts_model",
            default=False,
        )
    )
    if use_official:
        model_name = str(
            _nested(
                config,
                "embedding",
                "official_model_name",
                default=_nested(
                    config,
                    "embedding",
                    "official_fintexts_model_name",
                    default="EXAONE-BI/FinTexTS-Embedding",
                ),
            )
        )
    else:
        configured_model = _nested(
            config,
            "embedding",
            "model_name",
            default=_nested(config, "embeddings", "model_name"),
        )
        if configured_model in (None, ""):
            raise KeyError(
                "embedding.model_name is required unless "
                "embedding.use_official_fintexts_model=true."
            )
        model_name = str(configured_model)
    encoder_configuration = {
        "model_name": model_name,
        "revision": _nested(config, "embedding", "revision"),
        "max_length": _nested(
            config,
            "embedding",
            "max_length",
            default=_nested(config, "embeddings", "max_sequence_length"),
        ),
        "trust_remote_code": bool(
            _nested(config, "embedding", "trust_remote_code", default=False)
        ),
        "backend": str(
            _nested(
                config,
                "embedding",
                "backend",
                default="sentence_transformers",
            )
        ),
        "normalize": True,
        "events_variant": events_variant,
    }
    encoder_config_fingerprint = hashlib.sha256(
        json.dumps(
            encoder_configuration, sort_keys=True, default=str
        ).encode("utf-8")
    ).hexdigest()
    batch_size = int(
        _nested(
            config,
            "embedding",
            "batch_size",
            default=_nested(config, "embeddings", "batch_size", default=64),
        )
    )
    if batch_size < 1:
        raise ValueError("embeddings.batch_size must be positive.")
    configured_device = str(
        _nested(config, "embedding", "device", default="auto")
    ).lower()
    gpu_allowed = bool(_nested(config, "embeddings", "use_gpu", default=True))
    if configured_device not in {"auto", "cpu", "cuda"}:
        raise ValueError("embedding.device must be one of: auto, cpu, cuda.")
    if configured_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("embedding.device=cuda but CUDA is not available.")
    device = (
        configured_device
        if configured_device != "auto"
        else ("cuda" if gpu_allowed and torch.cuda.is_available() else "cpu")
    )
    embedding_path = _path(
        config,
        (
            ("paths", f"event_embeddings_{events_variant}"),
            ("embeddings", "output_path"),
        ),
        f"data/embeddings/event_embeddings{_variant_suffix(events_variant)}.npy",
    )
    metadata_path = _path(
        config,
        (("paths", f"event_embedding_metadata_{events_variant}"),),
        (
            "data/embeddings/"
            f"event_embedding_metadata{_variant_suffix(events_variant)}.csv"
        ),
    )
    force = bool(
        _nested(
            config,
            "embedding",
            "force_rebuild",
            default=_nested(
                config, "embeddings", "force_rebuild", default=False
            ),
        )
    )
    reusable, cached_dimension = (
        ({}, None)
        if force
        else _load_reusable_cache(
            embedding_path,
            metadata_path,
            model_name,
            encoder_config_fingerprint,
        )
    )

    keys = list(
        zip(events["event_id"].astype(str), events["text_hash"].astype(str))
    )
    missing_indices = [index for index, key in enumerate(keys) if key not in reusable]
    encoded_missing: np.ndarray | None = None
    encoder: Any | None = None
    if missing_indices:
        LOGGER.info(
            "Encoding %d/%d events with %s on %s.",
            len(missing_indices),
            len(events),
            model_name,
            device,
        )
        encoder = _build_encoder(model_name, config, device)
        texts = events.loc[missing_indices, "text"].astype(str).tolist()
        encoded_missing = encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=bool(
                _nested(
                    config,
                    "embedding",
                    "show_progress_bar",
                    default=_nested(
                        config,
                        "embeddings",
                        "show_progress_bar",
                        default=True,
                    ),
                )
            ),
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        encoded_missing = np.asarray(encoded_missing, dtype=np.float32)
        if encoded_missing.ndim != 2 or len(encoded_missing) != len(missing_indices):
            raise RuntimeError(
                "Sentence encoder returned an unexpected embedding array shape "
                f"{encoded_missing.shape}."
            )
        encoded_missing = np.asarray(l2_normalize(encoded_missing), dtype=np.float32)
        dimension = int(encoded_missing.shape[1])
        if cached_dimension is not None and cached_dimension != dimension:
            LOGGER.warning(
                "Cached embedding dimension %d differs from encoder dimension %d; "
                "the cache will be rebuilt.",
                cached_dimension,
                dimension,
            )
            reusable = {}
            missing_indices = list(range(len(events)))
            encoded_missing = encoder.encode(
                events["text"].astype(str).tolist(),
                batch_size=batch_size,
                show_progress_bar=bool(
                    _nested(
                        config,
                        "embedding",
                        "show_progress_bar",
                        default=True,
                    )
                ),
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
            encoded_missing = np.asarray(
                l2_normalize(np.asarray(encoded_missing, dtype=np.float32)),
                dtype=np.float32,
            )
    elif cached_dimension is None:
        raise RuntimeError("Embedding cache was reported reusable without a dimension.")

    dimension = (
        int(encoded_missing.shape[1])
        if encoded_missing is not None
        else int(cached_dimension)
    )
    matrix = np.empty((len(events), dimension), dtype=np.float32)
    missing_lookup = {
        event_index: encoded_index
        for encoded_index, event_index in enumerate(missing_indices)
    }
    for event_index, key in enumerate(keys):
        vector = reusable.get(key)
        if vector is None:
            if encoded_missing is None:
                raise RuntimeError(f"Missing embedding for event {key[0]}.")
            vector = encoded_missing[missing_lookup[event_index]]
        if vector.shape != (dimension,):
            raise ValueError(
                f"Embedding dimension mismatch for event {key[0]}: {vector.shape}."
            )
        matrix[event_index] = vector
    matrix = np.asarray(l2_normalize(matrix), dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite values found in event embeddings.")

    fingerprint = _event_fingerprint(events, encoder_config_fingerprint)
    metadata = events[
        ["event_id", "date", "news_level", "text_hash"]
    ].copy()
    metadata.insert(0, "embedding_index", np.arange(len(metadata), dtype=int))
    metadata["model_name"] = model_name
    metadata["embedding_dimension"] = dimension
    metadata["event_set_fingerprint"] = fingerprint
    metadata["l2_norm"] = np.linalg.norm(matrix, axis=1)
    metadata["device_used"] = device
    metadata["encoder_frozen"] = True
    metadata["events_variant"] = events_variant
    metadata["official_fintexts_model_requested"] = use_official
    metadata["encoder_config_fingerprint"] = encoder_config_fingerprint
    metadata["date"] = pd.to_datetime(metadata["date"]).dt.strftime("%Y-%m-%d")

    _atomic_save_npy(matrix, embedding_path)
    atomic_write_csv(metadata, metadata_path, index=False)
    LOGGER.info(
        "Saved %d normalized %d-dimensional event embeddings (%d cache hits).",
        len(matrix),
        dimension,
        len(events) - len(missing_indices),
    )
    return {
        "event_embeddings": embedding_path,
        "event_embedding_metadata": metadata_path,
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
