"""Shared, side-effect-free utilities for the original-volatility experiment."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import tempfile
import warnings
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
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IDENTIFIER_COLUMNS = {
    "ticker",
    "date",
    "feature_date",
    "target_date",
    "input_date",
    "expected_next_trading_date",
    "split",
    "industry",
    "industry_source",
    "representation",
    "representation_variant",
}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = (
        Path(path)
        if path is not None
        else PROJECT_ROOT / "config" / "config_original_volatility.yaml"
    )
    config_path = config_path.expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"Configuration must be a mapping: {config_path}")
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(config_path.parent.parent)
    return config


def project_path(config: Mapping[str, Any], *parts: str | os.PathLike[str]) -> Path:
    return Path(str(config.get("_project_root", PROJECT_ROOT))).joinpath(*parts).resolve()


def shared_root(config: Mapping[str, Any]) -> Path:
    value = Path(str(config["shared"]["residual_project_root"]))
    if not value.is_absolute():
        value = project_path(config, value)
    return value.resolve()


def ensure_directories(config: Mapping[str, Any]) -> None:
    for relative in (
        "data/processed",
        "outputs/tables",
        "outputs/figures",
        "outputs/models",
        "outputs/checkpoints",
        "outputs/checkpoints/tasks",
        "outputs/logs",
    ):
        project_path(config, relative).mkdir(parents=True, exist_ok=True)


def get_logger(
    name: str,
    config: Mapping[str, Any],
    log_file: str | Path | None = None,
) -> logging.Logger:
    logger = logging.getLogger(f"original_volatility.{name}")
    level = getattr(
        logging,
        str(config.get("project", {}).get("log_level", "INFO")).upper(),
        logging.INFO,
    )
    logger.setLevel(level)
    logger.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if not any(getattr(handler, "_original_console", False) for handler in logger.handlers):
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        console._original_console = True  # type: ignore[attr-defined]
        logger.addHandler(console)
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        resolved = str(path.resolve())
        if not any(
            isinstance(handler, logging.FileHandler)
            and getattr(handler, "baseFilename", None) == resolved
            for handler in logger.handlers
        ):
            file_handler = logging.FileHandler(path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    return logger


def set_global_seed(seed: int, deterministic: bool = True) -> None:
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
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def _temporary_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    return Path(name)


def atomic_write_json(payload: Any, path: str | Path) -> Path:
    destination = Path(path)
    temporary = _temporary_path(destination)
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def atomic_write_csv(
    frame: pd.DataFrame, path: str | Path, *, index: bool = False
) -> Path:
    destination = Path(path)
    temporary = _temporary_path(destination)
    try:
        frame.to_csv(temporary, index=index)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def write_table(
    frame: pd.DataFrame, path: str | Path, *, index: bool = False
) -> Path:
    destination = Path(path)
    temporary = _temporary_path(destination)
    try:
        if destination.suffix.lower() == ".parquet":
            frame.to_parquet(temporary, index=index)
        elif destination.suffix.lower() == ".csv":
            frame.to_csv(temporary, index=index)
        else:
            raise ValueError(f"Unsupported table extension: {destination}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def atomic_joblib_dump(payload: Any, path: str | Path) -> Path:
    destination = Path(path)
    temporary = _temporary_path(destination)
    try:
        joblib.dump(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def read_table(path: str | Path, columns: Sequence[str] | None = None) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    if source.suffix.lower() == ".parquet":
        return pd.read_parquet(source, columns=columns)
    if source.suffix.lower() == ".csv":
        frame = pd.read_csv(source)
        return frame if columns is None else frame[list(columns)]
    raise ValueError(f"Unsupported table extension: {source}")


def validate_columns(
    frame: pd.DataFrame, required: Iterable[str], label: str
) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise KeyError(f"{label} is missing columns: {missing}")


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(payload: Mapping[str, Any], prefix: str = "task") -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(encoded).hexdigest()[:16]}"


def _shared_candidates(config: Mapping[str, Any], kind: str) -> list[Path]:
    root = shared_root(config)
    key = {
        "processed": "processed_candidates",
        "tables": "table_candidates",
        "embeddings": "embedding_candidates",
    }[kind]
    return [root / str(value) for value in config["shared"].get(key, [])]


def resolve_shared_file(
    config: Mapping[str, Any],
    filename: str | Path,
    *,
    kinds: Sequence[str] = ("processed", "tables"),
    required: bool = True,
) -> Path | None:
    value = Path(filename)
    candidates: list[Path] = []
    if value.is_absolute():
        candidates.append(value)
    else:
        candidates.append(shared_root(config) / value)
    for kind in kinds:
        candidates.extend(
            directory / value.name for directory in _shared_candidates(config, kind)
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    if required:
        raise FileNotFoundError(
            f"Shared artifact {value.name!r} was not found. Checked: "
            + "; ".join(str(candidate) for candidate in candidates)
        )
    return None


def load_shared_market(config: Mapping[str, Any]) -> tuple[pd.DataFrame, Path]:
    path = resolve_shared_file(
        config,
        str(config["shared"]["market_file"]),
        kinds=("processed",),
    )
    assert path is not None
    market = read_table(path)
    validate_columns(
        market,
        (
            "ticker",
            "feature_date",
            "target_date",
            "split",
            "target_log_variance",
            "target_gk_variance",
        ),
        "shared market panel",
    )
    market["ticker"] = market["ticker"].astype(str)
    for column in ("feature_date", "target_date"):
        market[column] = pd.to_datetime(market[column], errors="raise").dt.normalize()
    market = market.sort_values(["feature_date", "ticker"], kind="mergesort")
    return market.reset_index(drop=True), path


def _resolve_manifest_artifact(
    config: Mapping[str, Any], raw_path: Any
) -> Path:
    value = Path(str(raw_path))
    if value.is_file():
        return value.resolve()
    resolved = resolve_shared_file(
        config, value.name, kinds=("processed",), required=False
    )
    if resolved is None:
        raise FileNotFoundError(
            f"Representation artifact cannot be resolved from manifest path {raw_path!r}"
        )
    return resolved


def load_representation_catalog(config: Mapping[str, Any]) -> pd.DataFrame:
    manifest_path = resolve_shared_file(
        config,
        str(config["shared"]["representation_manifest"]),
        kinds=("processed", "tables"),
    )
    assert manifest_path is not None
    manifest = read_table(manifest_path)
    validate_columns(
        manifest,
        ("representation", "representation_variant", "path"),
        "representation manifest",
    )
    manifest = manifest.copy()
    manifest["resolved_path"] = [
        str(_resolve_manifest_artifact(config, value)) for value in manifest["path"]
    ]
    if "selected" not in manifest.columns:
        manifest["selected"] = (
            manifest["representation_variant"].astype(str) == "selected_default"
        )
    manifest["selected"] = manifest["selected"].fillna(False).astype(bool)
    virtual_rows = []
    base_r7 = manifest.loc[
        (manifest["representation"].astype(str) == "R7") & manifest["selected"]
    ]
    if not base_r7.empty:
        template = base_r7.iloc[0].to_dict()
        for representation, variant in (
            ("P_LAGGED", "lagged_news_placebo"),
            ("P_PERMUTED", "within_split_permuted_assignment"),
        ):
            row = dict(template)
            row.update(
                {
                    "representation": representation,
                    "representation_variant": variant,
                    "representation_variant_family": variant,
                    "placebo": True,
                    "placebo_kind": variant,
                    "selected": True,
                }
            )
            virtual_rows.append(row)
    if virtual_rows:
        manifest = pd.concat(
            [manifest, pd.DataFrame(virtual_rows)],
            ignore_index=True,
            sort=False,
        )
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def selected_representation_rows(
    catalog: pd.DataFrame,
    representations: Sequence[str],
    *,
    selected_only: bool,
) -> pd.DataFrame:
    output = catalog.loc[
        catalog["representation"].astype(str).isin(set(map(str, representations)))
    ].copy()
    if selected_only:
        output = output.loc[output["selected"].fillna(False).astype(bool)]
    order = {value: index for index, value in enumerate(representations)}
    output["__order"] = output["representation"].map(order)
    return output.sort_values(
        ["__order", "representation_variant"], kind="mergesort"
    ).drop(columns="__order")


def load_representation_frame(
    config: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    seed: int,
) -> pd.DataFrame:
    frame = read_table(str(row["resolved_path"]))
    validate_columns(
        frame,
        ("ticker", "feature_date", "split"),
        f"representation {row['representation']}",
    )
    frame = frame.copy()
    frame["ticker"] = frame["ticker"].astype(str)
    frame["feature_date"] = pd.to_datetime(
        frame["feature_date"], errors="raise"
    ).dt.normalize()
    representation = str(row["representation"])
    feature_columns = representation_feature_columns(frame)
    if representation == "P_LAGGED":
        lag = int(config["features"]["lagged_placebo_days"])
        frame = frame.sort_values(["ticker", "feature_date"], kind="mergesort")
        frame[feature_columns] = (
            frame.groupby("ticker", sort=False)[feature_columns].shift(lag).fillna(0.0)
        )
    elif representation == "P_PERMUTED":
        rng = np.random.default_rng(seed)
        pieces = []
        for _, group in frame.groupby("split", sort=False, dropna=False):
            shuffled = group.copy()
            permutation = rng.permutation(len(group))
            shuffled.loc[:, feature_columns] = (
                group.iloc[permutation][feature_columns].to_numpy()
            )
            pieces.append(shuffled)
        frame = pd.concat(pieces, ignore_index=True)
    if frame.duplicated(["ticker", "feature_date"]).any():
        raise ValueError(
            f"Representation {representation} has duplicate ticker/feature_date keys"
        )
    return frame


def representation_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column not in IDENTIFIER_COLUMNS
        and pd.api.types.is_numeric_dtype(frame[column])
    ]


def price_feature_columns(
    frame: pd.DataFrame, config: Mapping[str, Any]
) -> list[str]:
    prefixes = tuple(map(str, config["features"]["price_prefixes"]))
    excludes = tuple(map(str, config["features"]["exclude_tokens"]))
    columns = []
    for column in frame.columns:
        lower = column.lower()
        if not pd.api.types.is_numeric_dtype(frame[column]):
            continue
        if any(token in lower for token in excludes):
            continue
        if column == "log_variance" or lower.startswith(prefixes):
            columns.append(column)
    if not columns:
        raise ValueError("No causal price feature columns were detected")
    return list(dict.fromkeys(columns))


def assert_split_contract(frame: pd.DataFrame, tickers: Sequence[str]) -> None:
    if frame.duplicated(["ticker", "feature_date"]).any():
        raise AssertionError("Market panel contains duplicate ticker/feature_date")
    if (frame["feature_date"] >= frame["target_date"]).any():
        examples = frame.loc[
            frame["feature_date"] >= frame["target_date"],
            ["ticker", "feature_date", "target_date"],
        ].head(5)
        raise AssertionError(
            "feature_date must precede target_date by one trading step: "
            f"{examples.to_dict(orient='records')}"
        )
    missing_tickers = sorted(set(tickers).difference(frame["ticker"].unique()))
    if missing_tickers:
        raise AssertionError(f"Missing configured tickers: {missing_tickers}")
    bounds = (
        frame.groupby("split", observed=True)["feature_date"]
        .agg(["min", "max"])
        .to_dict(orient="index")
    )
    if not {"train", "validation", "test"}.issubset(bounds):
        raise AssertionError(f"Incomplete chronological splits: {sorted(bounds)}")
    if not (
        bounds["train"]["max"] < bounds["validation"]["min"]
        and bounds["validation"]["max"] < bounds["test"]["min"]
    ):
        raise AssertionError(f"Chronological split overlap: {bounds}")


def trim_split_dates(
    frame: pd.DataFrame,
    split: str,
    limit: int | None,
    *,
    keep_latest: bool,
) -> pd.DataFrame:
    subset = frame.loc[frame["split"].astype(str) == split].copy()
    if limit is None:
        return subset
    dates = np.sort(subset["feature_date"].dropna().unique())
    selected = dates[-limit:] if keep_latest else dates[:limit]
    return subset.loc[subset["feature_date"].isin(selected)].copy()


def task_split_frames(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    profile: Mapping[str, Any],
    fold: str | int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if str(fold) == "holdout":
        train = trim_split_dates(
            frame,
            "train",
            profile.get("max_train_dates"),
            keep_latest=True,
        )
        validation = trim_split_dates(
            frame,
            "validation",
            profile.get("max_validation_dates"),
            keep_latest=False,
        )
        test = trim_split_dates(
            frame,
            "test",
            profile.get("max_test_dates"),
            keep_latest=False,
        )
    else:
        fold_path = resolve_shared_file(
            config,
            str(config["shared"]["chronological_folds"]),
            kinds=("tables",),
        )
        assert fold_path is not None
        folds = read_table(fold_path)
        selected = folds.loc[
            pd.to_numeric(folds["fold"], errors="coerce") == int(fold)
        ]
        if len(selected) != 1:
            raise ValueError(f"Chronological fold {fold} is unavailable")
        row = selected.iloc[0]
        train_start = pd.Timestamp(row["train_start"])
        train_end = pd.Timestamp(row["train_end"])
        validation_start = pd.Timestamp(row["validation_start"])
        validation_end = pd.Timestamp(row["validation_end"])
        if not train_end < validation_start:
            raise AssertionError(f"Fold {fold} overlaps train and validation")
        # chronological_folds.csv is defined from target_date in the residual
        # project. Filtering feature_date here shifts each boundary by one
        # trading day and makes it disagree with fold-fitted text artifacts.
        date_column = "target_date"
        train = frame.loc[
            frame[date_column].between(train_start, train_end)
        ].copy()
        validation = frame.loc[
            frame[date_column].between(validation_start, validation_end)
        ].copy()
        test = frame.iloc[0:0].copy()
    if train.empty or validation.empty:
        raise ValueError(f"Task split {fold} has empty train/validation data")
    chronology_column = (
        "target_date" if str(fold) != "holdout" else "feature_date"
    )
    if not train[chronology_column].max() < validation[chronology_column].min():
        raise AssertionError(f"Task split {fold} violates chronology")
    if not test.empty and not validation["feature_date"].max() < test["feature_date"].min():
        raise AssertionError(f"Task split {fold} validation/test overlap")
    return train, validation, test


def qlike_from_log(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_variance = np.exp(np.clip(np.asarray(y_true, dtype=float), -40, 40))
    predicted_variance = np.exp(np.clip(np.asarray(y_pred, dtype=float), -40, 40))
    ratio = true_variance / np.clip(predicted_variance, 1.0e-12, None)
    return float(np.mean(ratio - np.log(np.clip(ratio, 1.0e-12, None)) - 1.0))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[valid], y_pred[valid]
    if not len(y_true):
        return {}
    mse = mean_squared_error(y_true, y_pred)
    denominator = np.sum(np.square(y_true - y_true.mean()))
    r2 = (
        1.0 - np.sum(np.square(y_true - y_pred)) / denominator
        if denominator > 0
        else np.nan
    )
    return {
        "n": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2),
        "pearson": _safe_correlation(y_true, y_pred, "pearson"),
        "spearman": _safe_correlation(y_true, y_pred, "spearman"),
        "qlike": qlike_from_log(y_true, y_pred),
    }


def _safe_correlation(
    left: np.ndarray, right: np.ndarray, method: str
) -> float:
    if len(left) < 3 or np.std(left) == 0 or np.std(right) == 0:
        return np.nan
    result = (
        stats.pearsonr(left, right)
        if method == "pearson"
        else stats.spearmanr(left, right)
    )
    return float(result.statistic)


def expected_calibration_error(
    y_true: np.ndarray, probability: np.ndarray, bins: int = 10
) -> float:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = max(len(y_true), 1)
    error = 0.0
    for index in range(bins):
        mask = (
            (probability >= edges[index])
            & (
                probability <= edges[index + 1]
                if index == bins - 1
                else probability < edges[index + 1]
            )
        )
        if mask.any():
            error += (
                np.sum(mask)
                / total
                * abs(float(y_true[mask].mean() - probability[mask].mean()))
            )
    return float(error)


def binary_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    valid = np.isfinite(probability)
    y_true, probability = y_true[valid], probability[valid]
    predicted = (probability >= threshold).astype(int)
    output: dict[str, Any] = {
        "n": int(len(y_true)),
        "positive_rate": float(y_true.mean()) if len(y_true) else np.nan,
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(y_true, predicted))
            if len(np.unique(y_true)) > 1
            else np.nan
        ),
        "brier": float(brier_score_loss(y_true, probability)),
        "ece": expected_calibration_error(y_true, probability),
        "confusion_matrix": json.dumps(confusion_matrix(y_true, predicted).tolist()),
    }
    if len(np.unique(y_true)) > 1:
        output["pr_auc"] = float(average_precision_score(y_true, probability))
        output["roc_auc"] = float(roc_auc_score(y_true, probability))
    else:
        output["pr_auc"] = np.nan
        output["roc_auc"] = np.nan
    for precision_level in (0.25, 0.50):
        order = np.argsort(-probability)
        sorted_y = y_true[order]
        cumulative_positive = np.cumsum(sorted_y)
        ranks = np.arange(1, len(sorted_y) + 1)
        precision_curve = cumulative_positive / ranks
        eligible = np.flatnonzero(precision_curve >= precision_level)
        recall_value = (
            cumulative_positive[eligible[-1]] / max(np.sum(y_true), 1)
            if len(eligible)
            else 0.0
        )
        output[f"recall_at_precision_{int(precision_level * 100)}"] = float(
            recall_value
        )
    return output


def multiclass_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    predicted = np.argmax(probability, axis=1)
    classes = np.arange(probability.shape[1])
    one_hot = np.eye(probability.shape[1])[y_true]
    precision = precision_score(
        y_true, predicted, labels=classes, average=None, zero_division=0
    )
    recall = recall_score(
        y_true, predicted, labels=classes, average=None, zero_division=0
    )
    class_calibration = [
        expected_calibration_error(one_hot[:, index], probability[:, index])
        for index in classes
    ]
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="y_pred contains classes not in y_true",
            category=UserWarning,
        )
        balanced = float(balanced_accuracy_score(y_true, predicted))
    return {
        "n": int(len(y_true)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, predicted, average="weighted", zero_division=0)
        ),
        "balanced_accuracy": balanced,
        "multiclass_brier": float(np.mean(np.sum(np.square(probability - one_hot), axis=1))),
        "multiclass_ece": float(np.mean(class_calibration)),
        "per_class_ece": json.dumps(class_calibration),
        "per_class_precision": json.dumps(precision.tolist()),
        "per_class_recall": json.dumps(recall.tolist()),
        "high_volatility_recall": float(recall[-1]),
        "confusion_matrix": json.dumps(
            confusion_matrix(y_true, predicted, labels=classes).tolist()
        ),
    }


def gaussian_crps(
    y: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    scale = np.clip(np.asarray(scale, dtype=float), 1.0e-8, None)
    z = (np.asarray(y, dtype=float) - np.asarray(mean, dtype=float)) / scale
    return scale * (
        z * (2.0 * stats.norm.cdf(z) - 1.0)
        + 2.0 * stats.norm.pdf(z)
        - 1.0 / np.sqrt(np.pi)
    )


def uncertainty_metrics(
    y: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    distribution: str,
    df: np.ndarray | float = 5.0,
) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    mean = np.asarray(mean, dtype=float)
    scale = np.clip(np.asarray(scale, dtype=float), 1.0e-8, None)
    if distribution == "gaussian":
        standardized = (y - mean) / scale
        nll = -stats.norm.logpdf(y, loc=mean, scale=scale)
        pit = stats.norm.cdf(standardized)
        crps = gaussian_crps(y, mean, scale)
        def quantile(probability: float) -> np.ndarray:
            return mean + scale * stats.norm.ppf(probability)
    elif distribution == "student_t":
        df_array = np.broadcast_to(np.asarray(df, dtype=float), y.shape)
        standardized = (y - mean) / scale
        nll = -stats.t.logpdf(standardized, df=df_array) + np.log(scale)
        pit = stats.t.cdf(standardized, df=df_array)
        # Deterministic quantile approximation of CRPS.
        probabilities = np.linspace(0.01, 0.99, 99)
        samples = mean[:, None] + scale[:, None] * stats.t.ppf(
            probabilities[None, :], df=df_array[:, None]
        )
        first = np.mean(np.abs(samples - y[:, None]), axis=1)
        sorted_samples = np.sort(samples, axis=1)
        weights = 2 * np.arange(1, samples.shape[1] + 1) - samples.shape[1] - 1
        pairwise = (
            2.0
            * np.sum(sorted_samples * weights[None, :], axis=1)
            / samples.shape[1] ** 2
        )
        crps = first - 0.5 * pairwise
        def quantile(probability: float) -> np.ndarray:
            return mean + scale * stats.t.ppf(probability, df=df_array)
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")
    output: dict[str, float] = {
        "n": int(len(y)),
        "nll": float(np.mean(nll)),
        "crps": float(np.mean(crps)),
        "pit_ks": float(stats.kstest(pit, "uniform").statistic),
    }
    coverage_errors = []
    for level in (0.80, 0.90, 0.95, 0.99):
        alpha = (1.0 - level) / 2.0
        lower, upper = quantile(alpha), quantile(1.0 - alpha)
        coverage = float(np.mean((y >= lower) & (y <= upper)))
        output[f"coverage_{int(level * 100)}"] = coverage
        output[f"width_{int(level * 100)}"] = float(np.mean(upper - lower))
        coverage_errors.append(abs(coverage - level))
    output["coverage_calibration_error"] = float(np.mean(coverage_errors))
    for probability in (0.95, 0.99):
        quantile_value = quantile(probability)
        exceedance = y > quantile_value
        output[f"var_{int(probability * 100)}_exceedance"] = float(exceedance.mean())
        output[f"quantile_loss_{int(probability * 100)}"] = float(
            np.mean(
                np.maximum(
                    probability * (y - quantile_value),
                    (probability - 1.0) * (y - quantile_value),
                )
            )
        )
        kupiec_statistic, kupiec_pvalue = kupiec_test(
            exceedance, 1.0 - probability
        )
        christoffersen_statistic, christoffersen_pvalue = (
            christoffersen_independence_test(exceedance)
        )
        output[f"kupiec_{int(probability * 100)}_statistic"] = kupiec_statistic
        output[f"kupiec_{int(probability * 100)}_pvalue"] = kupiec_pvalue
        output[
            f"christoffersen_{int(probability * 100)}_statistic"
        ] = christoffersen_statistic
        output[
            f"christoffersen_{int(probability * 100)}_pvalue"
        ] = christoffersen_pvalue
    return output


def _xlog(value: float, probability: float) -> float:
    if value <= 0:
        return 0.0
    return value * np.log(np.clip(probability, 1.0e-12, 1.0))


def kupiec_test(
    exceedance: np.ndarray, expected_rate: float
) -> tuple[float, float]:
    values = np.asarray(exceedance, dtype=int)
    observations = len(values)
    failures = int(values.sum())
    if observations == 0:
        return np.nan, np.nan
    empirical = failures / observations
    null_log_likelihood = _xlog(
        observations - failures, 1.0 - expected_rate
    ) + _xlog(failures, expected_rate)
    fitted_log_likelihood = _xlog(
        observations - failures, 1.0 - empirical
    ) + _xlog(failures, empirical)
    statistic = max(-2.0 * (null_log_likelihood - fitted_log_likelihood), 0.0)
    return float(statistic), float(stats.chi2.sf(statistic, df=1))


def christoffersen_independence_test(
    exceedance: np.ndarray,
) -> tuple[float, float]:
    values = np.asarray(exceedance, dtype=int)
    if len(values) < 3:
        return np.nan, np.nan
    previous, current = values[:-1], values[1:]
    n00 = int(np.sum((previous == 0) & (current == 0)))
    n01 = int(np.sum((previous == 0) & (current == 1)))
    n10 = int(np.sum((previous == 1) & (current == 0)))
    n11 = int(np.sum((previous == 1) & (current == 1)))
    total = n00 + n01 + n10 + n11
    pooled = (n01 + n11) / max(total, 1)
    probability_01 = n01 / max(n00 + n01, 1)
    probability_11 = n11 / max(n10 + n11, 1)
    independent_log_likelihood = _xlog(n00 + n10, 1.0 - pooled) + _xlog(
        n01 + n11, pooled
    )
    markov_log_likelihood = (
        _xlog(n00, 1.0 - probability_01)
        + _xlog(n01, probability_01)
        + _xlog(n10, 1.0 - probability_11)
        + _xlog(n11, probability_11)
    )
    statistic = max(
        -2.0 * (independent_log_likelihood - markov_log_likelihood),
        0.0,
    )
    return float(statistic), float(stats.chi2.sf(statistic, df=1))


def confidence_interval(values: Sequence[float], level: float = 0.95) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return np.nan, np.nan
    if len(array) == 1:
        return float(array[0]), float(array[0])
    mean = float(array.mean())
    half = float(
        stats.t.ppf((1 + level) / 2, len(array) - 1)
        * stats.sem(array)
    )
    return mean - half, mean + half
