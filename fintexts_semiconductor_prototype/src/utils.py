"""Shared utilities for deterministic, leakage-safe experiments."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _deep_merge_config(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> dict[str, Any]:
    """Recursively merge a small experiment profile over the base config."""

    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if (
            isinstance(value, Mapping)
            and isinstance(merged.get(key), Mapping)
        ):
            merged[key] = _deep_merge_config(
                merged[key],  # type: ignore[arg-type]
                value,
            )
        else:
            merged[key] = value
    return merged


def _load_yaml_config(
    config_path: Path,
    active_paths: frozenset[Path] = frozenset(),
) -> dict[str, Any]:
    """Load a YAML config, resolving an optional relative ``extends`` chain."""

    config_path = config_path.expanduser().resolve()
    if config_path in active_paths:
        chain = " -> ".join(str(path) for path in (*active_paths, config_path))
        raise ValueError(f"Circular configuration inheritance: {chain}")
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    parent_value = loaded.pop("extends", None)
    if parent_value is None:
        return loaded
    if not isinstance(parent_value, str) or not parent_value.strip():
        raise ValueError(
            f"Configuration 'extends' must be a non-empty path: {config_path}"
        )
    parent_path = Path(parent_value)
    if not parent_path.is_absolute():
        parent_path = config_path.parent / parent_path
    parent = _load_yaml_config(
        parent_path,
        active_paths | frozenset({config_path}),
    )
    return _deep_merge_config(parent, loaded)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML configuration and attach absolute project/config paths."""
    config_path = Path(path) if path is not None else PROJECT_ROOT / "config" / "config.yaml"
    config_path = config_path.expanduser().resolve()
    config = _load_yaml_config(config_path)
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(config_path.parent.parent)
    return config


def project_path(config: Mapping[str, Any], *parts: str | os.PathLike[str]) -> Path:
    """Resolve a path relative to the configured project root."""
    root = Path(str(config.get("_project_root", PROJECT_ROOT))).resolve()
    relative = Path(*map(Path, parts))
    project_config = config.get("project", {})
    artifact_profile = (
        str(project_config.get("artifact_profile", "")).strip()
        if isinstance(project_config, Mapping)
        else ""
    )
    if artifact_profile:
        profile_path = Path(artifact_profile)
        if (
            artifact_profile in {".", ".."}
            or profile_path.is_absolute()
            or len(profile_path.parts) != 1
            or profile_path.name != artifact_profile
        ):
            raise ValueError(
                "project.artifact_profile must be one safe directory name"
            )
        relative_parts = relative.parts
        is_output = bool(relative_parts and relative_parts[0] == "outputs")
        is_processed = bool(
            len(relative_parts) >= 2
            and relative_parts[0] == "data"
            and relative_parts[1] == "processed"
        )
        if is_output or is_processed:
            relative = Path("runs") / artifact_profile / relative
    return root / relative


def ensure_directories(paths_or_config: Iterable[str | Path] | Mapping[str, Any]) -> None:
    """Create standard project directories or an explicit iterable of paths."""
    if isinstance(paths_or_config, Mapping):
        config = paths_or_config
        paths: Iterable[Path] = (
            project_path(config, "data", "raw"),
            project_path(config, "data", "processed"),
            project_path(config, "data", "embeddings"),
            project_path(config, "outputs", "figures"),
            project_path(config, "outputs", "tables"),
            project_path(config, "outputs", "models"),
            project_path(config, "outputs", "logs"),
        )
    else:
        paths = (Path(path) for path in paths_or_config)
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def get_logger(
    name: str,
    config: Mapping[str, Any] | None = None,
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Return a non-duplicating stdout/file logger."""
    logger = logging.getLogger(name)
    level_name = str((config or {}).get("project", {}).get("log_level", "INFO"))
    logger.setLevel(getattr(logging, level_name.upper(), logging.INFO))
    logger.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if not any(getattr(handler, "_fintexts_console", False) for handler in logger.handlers):
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        console._fintexts_console = True  # type: ignore[attr-defined]
        logger.addHandler(console)
    if log_file is not None:
        destination = Path(log_file)
        destination.parent.mkdir(parents=True, exist_ok=True)
        resolved = destination.resolve()
        if not any(
            getattr(handler, "baseFilename", None) == str(resolved)
            for handler in logger.handlers
        ):
            file_handler = logging.FileHandler(resolved, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    return logger


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch when available."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def validate_required_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    context: str,
) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}")


def safe_read_table(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Required table does not exist: {source}")
    suffix = source.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(source, **kwargs)
    if suffix == ".csv":
        return pd.read_csv(source, **kwargs)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(source, lines=True, **kwargs)
    if suffix == ".json":
        return pd.read_json(source, **kwargs)
    raise ValueError(f"Unsupported table format: {source.suffix}")


def _temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(
        prefix=f".{destination.stem}.",
        suffix=destination.suffix,
        dir=destination.parent,
    )
    os.close(handle)
    return Path(name)


def atomic_write_csv(
    frame: pd.DataFrame,
    path: str | Path,
    index: bool = False,
    **kwargs: Any,
) -> Path:
    destination = Path(path)
    temporary = _temporary_path(destination)
    try:
        frame.to_csv(temporary, index=index, **kwargs)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def write_table(
    frame: pd.DataFrame,
    path: str | Path,
    index: bool = False,
    **kwargs: Any,
) -> Path:
    destination = Path(path)
    temporary = _temporary_path(destination)
    try:
        if destination.suffix.lower() in {".parquet", ".pq"}:
            frame.to_parquet(temporary, index=index, **kwargs)
        elif destination.suffix.lower() == ".csv":
            frame.to_csv(temporary, index=index, **kwargs)
        else:
            raise ValueError(f"Unsupported output table format: {destination.suffix}")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def write_json(payload: Any, path: str | Path, indent: int = 2) -> Path:
    destination = Path(path)
    temporary = _temporary_path(destination)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=indent, default=str)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def read_json(path: str | Path) -> Any:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    with source.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_yaml(payload: Any, path: str | Path) -> Path:
    destination = Path(path)
    temporary = _temporary_path(destination)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def atomic_joblib_dump(payload: Any, path: str | Path) -> Path:
    destination = Path(path)
    temporary = _temporary_path(destination)
    try:
        joblib.dump(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def l2_normalize(values: np.ndarray, epsilon: float = 1.0e-12) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.clip(norms, epsilon, None)


def stable_softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    shifted = array - np.max(array, axis=axis, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / np.clip(
        exponentiated.sum(axis=axis, keepdims=True), 1.0e-12, None
    )


def chronological_assertions(
    train_dates: Sequence[Any],
    validation_dates: Sequence[Any],
    test_dates: Sequence[Any],
) -> None:
    """Enforce the two strict temporal boundaries required by the experiment."""
    train = pd.to_datetime(pd.Series(train_dates).dropna().unique())
    validation = pd.to_datetime(pd.Series(validation_dates).dropna().unique())
    test = pd.to_datetime(pd.Series(test_dates).dropna().unique())
    if not len(train) or not len(validation) or not len(test):
        raise ValueError("Train, validation, and test must each contain at least one date")
    train_end, validation_start = train.max(), validation.min()
    validation_end, test_start = validation.max(), test.min()
    if not train_end < validation_start:
        raise AssertionError(
            f"Leakage boundary failed: train_end={train_end} >= validation_start={validation_start}"
        )
    if not validation_end < test_start:
        raise AssertionError(
            f"Leakage boundary failed: validation_end={validation_end} >= test_start={test_start}"
        )


def qlike(y_true_log_variance: Sequence[float], y_pred_log_variance: Sequence[float]) -> float:
    """QLIKE for log-variance forecasts, evaluated in variance space stably."""
    truth = np.asarray(y_true_log_variance, dtype=float)
    prediction = np.asarray(y_pred_log_variance, dtype=float)
    valid = np.isfinite(truth) & np.isfinite(prediction)
    if not valid.any():
        return float("nan")
    log_ratio = np.clip(truth[valid] - prediction[valid], -50.0, 50.0)
    return float(np.mean(np.exp(log_ratio) - log_ratio - 1.0))


def expected_calibration_error(
    y_true: Sequence[int],
    probabilities: Sequence[float],
    n_bins: int = 10,
) -> float:
    labels = np.asarray(y_true, dtype=int)
    probs = np.asarray(probabilities, dtype=float)
    valid = np.isfinite(probs)
    labels, probs = labels[valid], np.clip(probs[valid], 0.0, 1.0)
    if not len(labels):
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(probs, edges[1:-1]), 0, n_bins - 1)
    error = 0.0
    for bin_id in range(n_bins):
        selected = bin_ids == bin_id
        if selected.any():
            error += selected.mean() * abs(
                labels[selected].mean() - probs[selected].mean()
            )
    return float(error)


def regression_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(truth) & np.isfinite(prediction)
    truth, prediction = truth[valid], prediction[valid]
    if not len(truth):
        return {key: float("nan") for key in ("mae", "mse", "rmse", "r2", "pearson", "spearman")}
    mse = mean_squared_error(truth, prediction)
    denominator = np.sum((truth - truth.mean()) ** 2)
    r2 = 1.0 - np.sum((truth - prediction) ** 2) / denominator if denominator > 0 else np.nan
    pearson = stats.pearsonr(truth, prediction).statistic if len(truth) > 1 else np.nan
    spearman = stats.spearmanr(truth, prediction).statistic if len(truth) > 1 else np.nan
    return {
        "mae": float(mean_absolute_error(truth, prediction)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2),
        "pearson": float(pearson),
        "spearman": float(spearman),
    }


def classification_metrics(
    y_true: Sequence[int],
    probabilities: Sequence[float],
    threshold: float = 0.5,
) -> dict[str, float]:
    labels = np.asarray(y_true, dtype=int)
    probs = np.asarray(probabilities, dtype=float)
    valid = np.isfinite(probs)
    labels, probs = labels[valid], np.clip(probs[valid], 0.0, 1.0)
    predictions = (probs >= threshold).astype(int)
    has_two_classes = np.unique(labels).size == 2
    return {
        "pr_auc": float(average_precision_score(labels, probs)) if has_two_classes else np.nan,
        "roc_auc": float(roc_auc_score(labels, probs)) if has_two_classes else np.nan,
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "brier": float(brier_score_loss(labels, probs)),
        "ece": expected_calibration_error(labels, probs),
    }


def confidence_interval(
    values: Sequence[float],
    confidence: float = 0.95,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return float("nan"), float("nan")
    if len(array) == 1:
        return float(array[0]), float(array[0])
    sem = stats.sem(array)
    width = stats.t.ppf((1.0 + confidence) / 2.0, len(array) - 1) * sem
    return float(array.mean() - width), float(array.mean() + width)


def finite_or_raise(frame: pd.DataFrame, columns: Sequence[str], context: str) -> None:
    validate_required_columns(frame, columns, context)
    non_finite: dict[str, int] = {}
    for column in columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        count = int((~np.isfinite(numeric)).sum())
        if count:
            non_finite[column] = count
    if non_finite:
        raise ValueError(f"{context} contains non-finite/missing values: {non_finite}")
