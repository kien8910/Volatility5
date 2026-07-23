"""Download and persist the raw FinTexTS dataset.

This module deliberately performs no schema assumptions.  Schema discovery is
handled by :mod:`src.inspect_schema`; here we only materialise every requested
Hugging Face split into one reproducible Parquet file and record enough
metadata to audit the download.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

from src.utils import get_logger, load_config, project_path, set_global_seed


DEFAULT_DATASET_ID = "EXAONE-BI/FinTexTS"
DEFAULT_RAW_FILENAME = "fintexts_raw.parquet"
DEFAULT_INFO_FILENAME = "dataset_info.json"
SOURCE_SPLIT_COLUMN = "__hf_split__"


def _first_config_value(
    config: Mapping[str, Any],
    paths: Sequence[Sequence[str]],
    default: Any = None,
) -> Any:
    """Return the first non-``None`` value found at one of ``paths``."""

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


def _as_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _ordered_split_names(
    available: Sequence[str],
    requested: Sequence[str] | None,
) -> list[str]:
    available_set = set(available)
    if requested:
        missing = [name for name in requested if name not in available_set]
        if missing:
            raise ValueError(
                "Requested Hugging Face splits are unavailable: "
                f"{missing}. Available splits: {list(available)}"
            )
        return list(dict.fromkeys(requested))

    conventional = ["train", "validation", "valid", "dev", "test"]
    ordered = [name for name in conventional if name in available_set]
    ordered.extend(sorted(available_set.difference(ordered)))
    return ordered


def _attach_split_column(dataset: Dataset, split_name: str) -> Dataset:
    if SOURCE_SPLIT_COLUMN in dataset.column_names:
        observed = set(dataset.unique(SOURCE_SPLIT_COLUMN))
        if observed != {split_name}:
            raise ValueError(
                f"Dataset already contains reserved column {SOURCE_SPLIT_COLUMN!r} "
                f"with values {sorted(map(str, observed))}; expected only "
                f"{split_name!r}."
            )
        return dataset
    return dataset.add_column(SOURCE_SPLIT_COLUMN, [split_name] * len(dataset))


def _materialise_splits(
    loaded: Dataset | DatasetDict,
    requested_splits: Sequence[str] | None,
) -> tuple[Dataset, dict[str, int]]:
    """Create one Dataset and retain each row's original HF split."""

    if isinstance(loaded, Dataset):
        if requested_splits and len(requested_splits) > 1:
            raise ValueError(
                "A single Hugging Face Dataset was returned, but multiple splits "
                f"were requested: {list(requested_splits)}"
            )
        split_name = requested_splits[0] if requested_splits else "all"
        result = _attach_split_column(loaded, split_name)
        return result, {split_name: len(result)}

    if not isinstance(loaded, DatasetDict):
        raise TypeError(
            "Expected datasets.Dataset or datasets.DatasetDict from load_dataset, "
            f"received {type(loaded).__name__}."
        )

    names = _ordered_split_names(list(loaded.keys()), requested_splits)
    if not names:
        raise ValueError("FinTexTS returned an empty DatasetDict.")

    split_sizes = {name: len(loaded[name]) for name in names}
    datasets = [_attach_split_column(loaded[name], name) for name in names]
    if len(datasets) == 1:
        return datasets[0], split_sizes
    return concatenate_datasets(datasets), split_sizes


def _feature_metadata(dataset: Dataset) -> dict[str, Any]:
    features = dataset.features
    if hasattr(features, "to_dict"):
        feature_dict = features.to_dict()
    else:
        feature_dict = {name: str(feature) for name, feature in features.items()}
    return {
        "column_names": list(dataset.column_names),
        "features": feature_dict,
        "arrow_schema": str(dataset.data.schema),
    }


def _atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_parquet(dataset: Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.parquet")
    if temporary.exists():
        temporary.unlink()
    try:
        dataset.to_parquet(str(temporary))
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise IOError(
                f"Writing the raw dataset produced an empty file: {temporary}"
            )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run(config: dict) -> dict[str, Path]:
    """Download FinTexTS and cache a raw, unmodified Parquet snapshot.

    Parameters
    ----------
    config:
        Project configuration loaded from ``config/config.yaml``.

    Returns
    -------
    dict[str, pathlib.Path]
        Paths keyed by ``raw_data`` and ``dataset_info``.
    """

    logger = get_logger(__name__, config)
    seed = int(
        _first_config_value(
            config,
            (("seed",), ("project", "seed"), ("runtime", "seed")),
            42,
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

    raw_filename = str(
        _first_config_value(
            config,
            (
                ("paths", "raw_filename"),
                ("dataset", "raw_filename"),
                ("data", "raw_filename"),
            ),
            DEFAULT_RAW_FILENAME,
        )
    )
    info_filename = str(
        _first_config_value(
            config,
            (("paths", "dataset_info_filename"), ("dataset", "info_filename")),
            DEFAULT_INFO_FILENAME,
        )
    )
    raw_path = project_path(config, "data", "raw", raw_filename)
    info_path = project_path(config, "data", "raw", info_filename)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    force = bool(
        _first_config_value(
            config,
            (
                ("download", "force"),
                ("dataset", "force_download"),
                ("data", "force_download"),
            ),
            False,
        )
    )
    if raw_path.is_file() and info_path.is_file() and not force:
        logger.info("Using cached FinTexTS snapshot at %s", raw_path)
        return {"raw_data": raw_path, "dataset_info": info_path}

    dataset_id = str(
        _first_config_value(
            config,
            (("dataset", "id"), ("dataset", "name"), ("data", "dataset_id")),
            DEFAULT_DATASET_ID,
        )
    )
    dataset_config_name = _as_optional_string(
        _first_config_value(
            config,
            (
                ("dataset", "config_name"),
                ("dataset", "subset"),
                ("data", "dataset_config"),
            ),
        )
    )
    revision = _as_optional_string(
        _first_config_value(config, (("dataset", "revision"),))
    )
    cache_dir_value = _first_config_value(
        config,
        (("dataset", "hf_cache_dir"), ("paths", "hf_cache_dir")),
    )
    cache_dir = (
        str(project_path(config, str(cache_dir_value)))
        if cache_dir_value
        else str(project_path(config, "data", "raw", "hf_cache"))
    )

    configured_splits = _first_config_value(
        config,
        (
            ("dataset", "splits"),
            ("download", "splits"),
            ("data", "dataset_split"),
        ),
    )
    if isinstance(configured_splits, str):
        requested_splits = [configured_splits]
    elif configured_splits is None:
        requested_splits = None
    elif isinstance(configured_splits, (list, tuple)):
        requested_splits = [str(value) for value in configured_splits]
    else:
        raise TypeError(
            "dataset.splits/data.dataset_split must be a split name, a list "
            "of split names, or null."
        )

    token_env = _as_optional_string(
        _first_config_value(
            config,
            (("dataset", "token_env"),),
            "HF_TOKEN",
        )
    )
    token = os.environ.get(token_env) if token_env else None
    load_kwargs: dict[str, Any] = {
        "path": dataset_id,
        "cache_dir": cache_dir,
        "streaming": False,
    }
    if dataset_config_name is not None:
        load_kwargs["name"] = dataset_config_name
    if revision is not None:
        load_kwargs["revision"] = revision
    if token:
        load_kwargs["token"] = token
    trust_remote_code = bool(
        _first_config_value(
            config,
            (("dataset", "trust_remote_code"), ("data", "trust_remote_code")),
            False,
        )
    )
    if trust_remote_code:
        load_kwargs["trust_remote_code"] = True
    if force:
        load_kwargs["download_mode"] = "force_redownload"

    logger.info(
        "Downloading Hugging Face dataset %s%s",
        dataset_id,
        f" (config={dataset_config_name})" if dataset_config_name else "",
    )
    loaded = load_dataset(**load_kwargs)
    materialised, split_sizes = _materialise_splits(loaded, requested_splits)
    if len(materialised) == 0:
        raise ValueError(f"Hugging Face dataset {dataset_id!r} contains no rows.")
    if len(set(materialised.column_names)) != len(materialised.column_names):
        raise ValueError("FinTexTS contains duplicate column names.")

    _atomic_write_parquet(materialised, raw_path)
    metadata: dict[str, Any] = {
        "dataset_id": dataset_id,
        "dataset_config_name": dataset_config_name,
        "revision": revision,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "num_rows": len(materialised),
        "num_columns": len(materialised.column_names),
        "source_split_column": SOURCE_SPLIT_COLUMN,
        "split_sizes": split_sizes,
        "dataset_fingerprint": getattr(materialised, "_fingerprint", None),
        "raw_parquet": str(raw_path),
        "seed": seed,
        **_feature_metadata(materialised),
    }
    _atomic_write_json(metadata, info_path)
    logger.info(
        "Saved %,d FinTexTS rows across %d columns to %s",
        len(materialised),
        len(materialised.column_names),
        raw_path,
    )
    return {"raw_data": raw_path, "dataset_info": info_path}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download EXAONE-BI/FinTexTS from Hugging Face."
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
