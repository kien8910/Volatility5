"""Train simple leakage-safe models for every candidate forecasting target."""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import (
    atomic_joblib_dump,
    atomic_write_csv,
    ensure_directories,
    get_logger,
    load_config,
    project_path,
    qlike,
    regression_metrics,
    safe_read_table,
    set_global_seed,
    sha256_text,
    validate_required_columns,
    write_json,
    write_table,
)

IDENTIFIER_COLUMNS = {
    "ticker",
    "feature_date",
    "target_date",
    "date",
    "split",
    "fold",
    "event_id",
    "news_level",
    "representation",
    "target_gk_variance",
    "target_log_parkinson_variance",
    "y_t_plus_1",
    "next_trading_day_aligned",
    "model_ready",
}
LEAKAGE_TOKENS = (
    "target_log_variance",
    "baseline_prediction",
    "residual_origin",
    "signed_residual",
    "residual_magnitude",
    "squared_residual",
    "log_squared_residual",
    "standardized_residual",
    "spike_q",
    "regime",
    "future",
    "target_y",
)


def _experiment_group(representation_variant_family: str) -> str:
    family = str(representation_variant_family).lower()
    if family.startswith("response_aware"):
        return "response_aware"
    if family.startswith("shuffled_response"):
        return "shuffled_response"
    return "semantic"


def _numpy_softplus(values: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, np.asarray(values, dtype=float))


@dataclass(frozen=True)
class Candidate:
    model: str
    parameters: dict[str, Any]

    @property
    def identifier(self) -> str:
        payload = json.dumps(
            {"model": self.model, **self.parameters}, sort_keys=True
        )
        return f"{self.model}-{sha256_text(payload)[:10]}"


def _format_duration(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(seconds) or seconds < 0:
        return "unknown"
    rounded = int(round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{seconds_part:02d}s"
    if minutes:
        return f"{minutes:d}m{seconds_part:02d}s"
    return f"{seconds_part:d}s"


class _ProgressReporter:
    """Thread-safe phase progress with periodic heartbeat and ETA logging."""

    def __init__(
        self,
        logger: Any,
        phase: str,
        total: int,
        *,
        enabled: bool,
        log_every_jobs: int,
        heartbeat_seconds: float,
    ) -> None:
        self.logger = logger
        self.phase = phase
        self.total = max(0, int(total))
        self.enabled = bool(enabled)
        self.log_every_jobs = max(1, int(log_every_jobs))
        self.heartbeat_seconds = max(5.0, float(heartbeat_seconds))
        self.completed = 0
        self.current_label = "preparing"
        self.started = time.monotonic()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "_ProgressReporter":
        if not self.enabled:
            return self
        self.logger.info(
            "PROGRESS %s started | total_jobs=%d | heartbeat=%.0fs",
            self.phase,
            self.total,
            self.heartbeat_seconds,
        )
        self._thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"progress-{self.phase}",
            daemon=True,
        )
        self._thread.start()
        return self

    def start(self) -> "_ProgressReporter":
        return self.__enter__()

    def finish(self) -> None:
        self.__exit__(None, None, None)

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=min(5.0, self.heartbeat_seconds))
        if self.enabled:
            self._emit("aborted" if exc_type is not None else "finished")

    def start_job(self, label: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            self.current_label = str(label)

    def advance(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self.completed += 1
            completed = self.completed
            total = self.total
        if (
            completed == 1
            or completed == total
            or completed % self.log_every_jobs == 0
        ):
            self._emit("update")

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_seconds):
            self._emit("heartbeat")

    def _emit(self, event: str) -> None:
        with self._lock:
            completed = self.completed
            total = self.total
            label = self.current_label
        elapsed = max(time.monotonic() - self.started, 1.0e-9)
        rate = completed / elapsed
        remaining = (
            max(total - completed, 0) / rate
            if completed > 0 and total >= completed and rate > 0
            else None
        )
        percentage = 100.0 * completed / total if total else 100.0
        self.logger.info(
            "PROGRESS %s %s | %d/%d (%.1f%%) | elapsed=%s | ETA=%s | current=%s",
            self.phase,
            event,
            completed,
            total,
            percentage,
            _format_duration(elapsed),
            _format_duration(remaining),
            label,
        )


def _progress_options(config: Mapping[str, Any]) -> dict[str, Any]:
    models = config.get("models", {})
    if not isinstance(models, Mapping):
        models = {}
    progress = models.get("progress", {})
    if not isinstance(progress, Mapping):
        progress = {}
    return {
        "enabled": bool(progress.get("enabled", True)),
        "log_every_jobs": int(progress.get("log_every_jobs", 10)),
        "heartbeat_seconds": float(progress.get("heartbeat_seconds", 30.0)),
    }


def _is_usable_numeric(column: str, series: pd.Series) -> bool:
    lower = column.lower()
    if column in IDENTIFIER_COLUMNS or any(token in lower for token in LEAKAGE_TOKENS):
        return False
    return pd.api.types.is_numeric_dtype(series)


def _representation_artifacts(
    config: Mapping[str, Any],
) -> list[tuple[str, str, str, int | None, Path]]:
    processed = project_path(config, "data", "processed")
    manifest_candidates = [
        processed / "representation_manifest.csv",
        project_path(config, "outputs", "tables", "representation_manifest.csv"),
    ]
    discovered: list[tuple[str, str, str, int | None, Path]] = []
    seen: set[tuple[str, str, str]] = set()
    for manifest_path in manifest_candidates:
        if not manifest_path.exists():
            continue
        manifest = safe_read_table(manifest_path)
        if not {"representation", "path"}.issubset(manifest.columns):
            continue
        for row in manifest.itertuples(index=False):
            path = Path(str(row.path))
            if not path.is_absolute():
                path = project_path(config, str(path))
            representation = str(row.representation)
            raw_variant = str(
                getattr(row, "representation_variant", "selected_default")
            )
            raw_family = str(
                getattr(row, "representation_variant_family", raw_variant)
            )
            raw_prototype_seed = getattr(row, "prototype_seed", None)
            prototype_seed = (
                None
                if raw_prototype_seed is None or pd.isna(raw_prototype_seed)
                else int(raw_prototype_seed)
            )
            pooling = str(getattr(row, "pooling", "mean"))
            selected = bool(getattr(row, "selected", False))
            variant = (
                "selected_default"
                if selected
                else f"{raw_variant}__pool_{pooling}"
            )
            variant_family = (
                "selected_default"
                if selected
                else f"{raw_family}__pool_{pooling}"
            )
            key = (representation, variant, str(path))
            if key not in seen:
                discovered.append(
                    (
                        representation,
                        variant,
                        variant_family,
                        prototype_seed,
                        path,
                    )
                )
                seen.add(key)
        if discovered:
            break
    if not discovered:
        search_roots = [processed / "representations", processed]
        for root in search_roots:
            if not root.exists():
                continue
            for pattern in ("features_R*.parquet", "representation_R*.parquet", "R*.parquet"):
                for path in sorted(root.glob(pattern)):
                    token = next(
                        (
                            part
                            for part in path.stem.replace("-", "_").split("_")
                            if len(part) >= 2 and part[0] == "R" and part[1:].isdigit()
                        ),
                        None,
                    )
                    if token:
                        key = (token, "selected_default", str(path))
                        if key not in seen:
                            discovered.append(
                                (
                                    token,
                                    "selected_default",
                                    "selected_default",
                                    None,
                                    path,
                                )
                            )
                            seen.add(key)
    return discovered


def _load_joined_representation(
    config: Mapping[str, Any],
    residuals: pd.DataFrame,
    representation: str,
    path: Path | None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    base = residuals.copy()
    price_columns = [
        column
        for column in base.columns
        if _is_usable_numeric(column, base[column])
    ]
    if representation == "R0" or path is None:
        return base, price_columns, []
    features = safe_read_table(path)
    validate_required_columns(features, ["ticker", "feature_date"], str(path))
    features = features.copy()
    features["ticker"] = features["ticker"].astype(str).str.upper()
    features["feature_date"] = pd.to_datetime(features["feature_date"])
    if features.duplicated(["ticker", "feature_date"]).any():
        duplicates = int(features.duplicated(["ticker", "feature_date"]).sum())
        raise ValueError(
            f"{path} has {duplicates} duplicate ticker/feature_date rows; "
            "choose one pooling/prototype configuration in the manifest"
        )
    new_columns = [
        column
        for column in features.columns
        if column not in base.columns
        and column not in IDENTIFIER_COLUMNS
        and _is_usable_numeric(column, features[column])
    ]
    joined = base.merge(
        features[["ticker", "feature_date", *new_columns]],
        on=["ticker", "feature_date"],
        how="left",
        validate="one_to_one",
        indicator="__feature_merge",
    )
    missing_rows = joined["__feature_merge"].ne("both")
    if missing_rows.any():
        examples = joined.loc[
            missing_rows, ["ticker", "feature_date"]
        ].head(10)
        raise ValueError(
            f"{path} is missing {int(missing_rows.sum())} residual ticker-date "
            f"rows: {examples.to_dict(orient='records')}"
        )
    joined = joined.drop(columns="__feature_merge")
    return joined, price_columns, new_columns


def _feature_sets(
    representation: str,
    price_columns: Sequence[str],
    text_columns: Sequence[str],
) -> dict[str, list[str]]:
    if representation == "R0" or not text_columns:
        return {"price_only": list(price_columns)}
    return {
        "price_plus_text": list(dict.fromkeys([*price_columns, *text_columns])),
        "text_only": list(text_columns),
    }


def _make_preprocessor(feature_columns: Sequence[str]) -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        dtype=np.float32,
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, list(feature_columns)),
            ("ticker", categorical, ["ticker"]),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def _candidate_grid(target: str, config: Mapping[str, Any]) -> list[Candidate]:
    model_config = config["models"]
    target_key = (
        target
        if target in {"signed", "magnitude", "squared", "uncertainty"}
        else "spike"
        if target.startswith("spike")
        else "regime"
        if target.startswith("regime")
        else target
    )
    configured_models = {
        str(value).lower()
        for value in model_config.get(target_key, [])
    }
    known_models = {
        "signed": {"ridge", "elastic_net", "mlp"},
        "magnitude": {"ridge", "elastic_net", "mlp"},
        "squared": {"ridge", "elastic_net", "mlp"},
        "spike": {"logistic", "weighted_logistic", "mlp"},
        "regime": {"multinomial_logistic", "mlp"},
        "uncertainty": {"gaussian", "student_t"},
    }
    unknown = configured_models.difference(known_models.get(target_key, set()))
    if unknown:
        raise ValueError(
            f"Unsupported models for {target_key}: {sorted(unknown)}"
        )
    candidates: list[Candidate] = []
    if target in {"signed", "magnitude", "squared"}:
        if "ridge" in configured_models:
            candidates.extend(
                Candidate("ridge", {"alpha": float(alpha)})
                for alpha in model_config["ridge_alphas"]
            )
        if "elastic_net" in configured_models:
            candidates.extend(
                Candidate(
                    "elastic_net",
                    {
                        "alpha": float(alpha),
                        "l1_ratio": float(l1_ratio),
                    },
                )
                for alpha in model_config["elastic_net_alphas"]
                for l1_ratio in model_config["elastic_net_l1_ratios"]
            )
        if "mlp" in configured_models:
            candidates.extend(
                Candidate("mlp_regression", {"hidden_sizes": list(hidden)})
                for hidden in model_config["mlp_hidden_sizes"]
            )
    if target.startswith("spike"):
        if "logistic" in configured_models:
            candidates.extend(
                Candidate(
                    "logistic",
                    {"C": float(value), "class_weight": None},
                )
                for value in model_config["logistic_cs"]
            )
        if "weighted_logistic" in configured_models:
            candidates.extend(
                Candidate(
                    "weighted_logistic",
                    {"C": float(value), "class_weight": "balanced"},
                )
                for value in model_config["logistic_cs"]
            )
        if "mlp" in configured_models:
            candidates.extend(
                Candidate(
                    "mlp_binary",
                    {"hidden_sizes": list(hidden), "weighted": True},
                )
                for hidden in model_config["mlp_hidden_sizes"]
            )
    if target.startswith("regime"):
        if "multinomial_logistic" in configured_models:
            candidates.extend(
                Candidate("multinomial_logistic", {"C": float(value)})
                for value in model_config["logistic_cs"]
            )
        if "mlp" in configured_models:
            candidates.extend(
                Candidate("mlp_multiclass", {"hidden_sizes": list(hidden)})
                for hidden in model_config["mlp_hidden_sizes"]
            )
    if target == "uncertainty":
        candidates.extend(
            Candidate(
                distribution,
                {
                    "hidden_sizes": list(hidden),
                    "distribution": distribution,
                },
            )
            for distribution in config["uncertainty"]["distributions"]
            if str(distribution).lower() in configured_models
            for hidden in model_config["mlp_hidden_sizes"]
        )
    if not candidates:
        raise ValueError(
            f"No model candidates enabled for target {target!r}; "
            f"check models.{target_key}."
        )
    return candidates


def _target_column(target: str) -> str:
    return {
        "signed": "signed_residual",
        "magnitude": "residual_magnitude",
        "squared": "log_squared_residual",
        "spike_q90": "spike_q90",
        "spike_q95": "spike_q95",
        "spike_q90_pooled_standardized": "spike_q90_pooled_standardized",
        "spike_q95_pooled_standardized": "spike_q95_pooled_standardized",
        "regime": "regime",
        "regime_pooled_standardized": "regime_pooled_standardized",
        "uncertainty": "target_log_variance",
    }[target]


def _fit_sklearn(candidate: Candidate, x: np.ndarray, y: np.ndarray, seed: int) -> Any:
    if candidate.model == "ridge":
        model = Ridge(alpha=candidate.parameters["alpha"])
    elif candidate.model == "elastic_net":
        model = ElasticNet(
            alpha=candidate.parameters["alpha"],
            l1_ratio=candidate.parameters["l1_ratio"],
            max_iter=20_000,
            random_state=seed,
        )
    elif candidate.model in {"logistic", "weighted_logistic"}:
        model = LogisticRegression(
            C=candidate.parameters["C"],
            class_weight=candidate.parameters["class_weight"],
            max_iter=5_000,
            solver="lbfgs",
            random_state=seed,
        )
    elif candidate.model == "multinomial_logistic":
        model = LogisticRegression(
            C=candidate.parameters["C"],
            class_weight="balanced",
            max_iter=5_000,
            solver="lbfgs",
            random_state=seed,
        )
    else:
        raise ValueError(f"{candidate.model} is not a scikit-learn candidate")
    model.fit(x, y)
    return model


def _torch_modules() -> tuple[Any, Any]:
    import torch
    from torch import nn

    return torch, nn


class _TorchModelHolder:
    """Serializable description/state for a small PyTorch network."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Sequence[int],
        output_dim: int,
        state_dict: Mapping[str, Any] | None = None,
    ) -> None:
        self.input_dim = int(input_dim)
        self.hidden_sizes = tuple(int(value) for value in hidden_sizes)
        self.output_dim = int(output_dim)
        self.state_dict = state_dict

    def build(self) -> Any:
        torch, nn = _torch_modules()
        layers: list[Any] = []
        previous = self.input_dim
        for width in self.hidden_sizes:
            layers.extend([nn.Linear(previous, width), nn.ReLU()])
            previous = width
        layers.append(nn.Linear(previous, self.output_dim))
        model = nn.Sequential(*layers)
        if self.state_dict is not None:
            model.load_state_dict(self.state_dict)
        return model


def _chronological_inner_masks(dates: pd.Series, fraction: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    unique_dates = np.sort(pd.to_datetime(dates).unique())
    cut = max(1, int(math.floor(len(unique_dates) * (1.0 - fraction))))
    cut = min(cut, len(unique_dates) - 1)
    boundary = unique_dates[cut]
    validation = pd.to_datetime(dates).to_numpy() >= boundary
    training = ~validation
    if not training.any() or not validation.any():
        raise ValueError("Not enough chronological dates for MLP early stopping")
    return training, validation


def _fit_torch(
    candidate: Candidate,
    x: np.ndarray,
    y: np.ndarray,
    dates: pd.Series,
    config: Mapping[str, Any],
    seed: int,
    baseline_prediction: np.ndarray | None = None,
) -> _TorchModelHolder:
    torch, nn = _torch_modules()
    set_global_seed(seed, bool(config["project"].get("deterministic", True)))
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if candidate.model in {"mlp_regression", "mlp_binary"}:
        output_dim = 1
    elif candidate.model == "mlp_multiclass":
        output_dim = 3
    elif candidate.model == "gaussian":
        output_dim = 1
    elif candidate.model == "student_t":
        output_dim = 2
    else:
        raise ValueError(candidate.model)
    holder = _TorchModelHolder(
        x.shape[1], candidate.parameters["hidden_sizes"], output_dim
    )
    model = holder.build().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config["models"]["learning_rate"])
    )
    train_mask, early_mask = _chronological_inner_masks(dates)
    x_tensor = torch.as_tensor(x, dtype=torch.float32)
    if candidate.model == "mlp_multiclass":
        y_tensor = torch.as_tensor(y, dtype=torch.long)
    else:
        y_tensor = torch.as_tensor(y, dtype=torch.float32)
    baseline_tensor = (
        torch.as_tensor(baseline_prediction, dtype=torch.float32)
        if baseline_prediction is not None
        else None
    )
    train_indices = np.flatnonzero(train_mask)
    early_indices = np.flatnonzero(early_mask)
    generator = torch.Generator().manual_seed(seed)
    dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(train_indices, dtype=torch.long)
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(config["models"]["batch_size"]),
        shuffle=True,
        generator=generator,
    )
    if candidate.model == "mlp_binary":
        positives = max(float(y[train_mask].sum()), 1.0)
        negatives = max(float(train_mask.sum() - positives), 1.0)
        binary_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(negatives / positives, device=device)
        )
    else:
        binary_loss = None

    min_scale = float(config["uncertainty"]["min_scale"])
    min_df = float(config["uncertainty"]["student_t_min_df"])

    def loss_for(indices: np.ndarray | Any) -> Any:
        index = torch.as_tensor(indices, dtype=torch.long)
        features = x_tensor[index].to(device)
        target = y_tensor[index].to(device)
        raw = model(features)
        if candidate.model == "mlp_regression":
            return nn.functional.mse_loss(raw.squeeze(-1), target)
        if candidate.model == "mlp_binary":
            assert binary_loss is not None
            return binary_loss(raw.squeeze(-1), target)
        if candidate.model == "mlp_multiclass":
            return nn.functional.cross_entropy(raw, target)
        if baseline_tensor is None:
            raise ValueError("Uncertainty model requires fixed baseline predictions")
        mean = baseline_tensor[index].to(device)
        scale = nn.functional.softplus(raw[:, 0]) + min_scale
        if candidate.model == "gaussian":
            distribution = torch.distributions.Normal(mean, scale)
        else:
            degrees = nn.functional.softplus(raw[:, 1]) + min_df
            distribution = torch.distributions.StudentT(degrees, mean, scale)
        return -distribution.log_prob(target).mean()

    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    remaining_patience = int(config["models"]["patience"])
    max_epochs = int(config["models"]["max_epochs"])
    min_delta = float(config["models"]["min_delta"])
    for _ in range(max_epochs):
        model.train()
        for (batch_indices,) in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for(batch_indices)
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), float(config["models"]["gradient_clip"])
            )
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_for(early_indices).detach().cpu())
        if validation_loss < best_loss - min_delta:
            best_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            remaining_patience = int(config["models"]["patience"])
        else:
            remaining_patience -= 1
        if remaining_patience <= 0:
            break
    if best_state is None:
        raise RuntimeError("MLP early stopping did not produce a valid state")
    holder.state_dict = best_state
    return holder


def _predict_candidate(
    candidate: Candidate,
    model: Any,
    x: np.ndarray,
    config: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    if candidate.model in {"ridge", "elastic_net"}:
        return {"prediction": np.asarray(model.predict(x), dtype=float)}
    if candidate.model in {"logistic", "weighted_logistic"}:
        return {"probability": np.asarray(model.predict_proba(x)[:, 1], dtype=float)}
    if candidate.model == "multinomial_logistic":
        probabilities = np.asarray(model.predict_proba(x), dtype=float)
        output = {"prediction": np.argmax(probabilities, axis=1)}
        for class_id in range(3):
            if class_id in model.classes_:
                position = int(np.flatnonzero(model.classes_ == class_id)[0])
                output[f"prob_{class_id}"] = probabilities[:, position]
            else:
                output[f"prob_{class_id}"] = np.zeros(len(x))
        return output

    torch, nn = _torch_modules()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = model.build().to(device)
    network.eval()
    outputs: list[np.ndarray] = []
    batch_size = int(config["models"]["batch_size"])
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.as_tensor(
                x[start : start + batch_size], dtype=torch.float32, device=device
            )
            outputs.append(network(batch).cpu().numpy())
    raw = np.concatenate(outputs, axis=0)
    if candidate.model == "mlp_regression":
        return {"prediction": raw[:, 0]}
    if candidate.model == "mlp_binary":
        return {"probability": 1.0 / (1.0 + np.exp(-raw[:, 0]))}
    if candidate.model == "mlp_multiclass":
        shifted = raw - raw.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return {
            "prediction": np.argmax(probabilities, axis=1),
            **{f"prob_{class_id}": probabilities[:, class_id] for class_id in range(3)},
        }
    scale = _numpy_softplus(raw[:, 0]) + float(config["uncertainty"]["min_scale"])
    result = {"scale": np.asarray(scale, dtype=float)}
    if candidate.model == "student_t":
        result["degrees_of_freedom"] = (
            _numpy_softplus(raw[:, 1])
            + float(config["uncertainty"]["student_t_min_df"])
        )
    return result


def _primary_score(
    target: str,
    frame: pd.DataFrame,
    predictions: Mapping[str, np.ndarray],
) -> tuple[str, float, bool]:
    """Return metric, value, and whether larger values are better."""
    if target == "signed":
        final = frame["baseline_prediction"].to_numpy() + predictions["prediction"]
        return "qlike", qlike(frame["target_log_variance"], final), False
    if target in {"magnitude", "squared"}:
        column = _target_column(target)
        score = regression_metrics(frame[column], predictions["prediction"])["spearman"]
        return "spearman", score, True
    if target.startswith("spike"):
        score = average_precision_score(
            frame[_target_column(target)], predictions["probability"]
        )
        return "pr_auc", float(score), True
    if target.startswith("regime"):
        score = f1_score(
            frame[_target_column(target)],
            predictions["prediction"],
            average="macro",
            zero_division=0,
        )
        return "macro_f1", float(score), True
    if target == "uncertainty":
        residual = (
            frame["target_log_variance"].to_numpy()
            - frame["baseline_prediction"].to_numpy()
        )
        scale = predictions["scale"]
        if "degrees_of_freedom" in predictions:
            from scipy import stats

            nll = -stats.t.logpdf(
                residual / scale,
                df=predictions["degrees_of_freedom"],
            ) + np.log(scale)
        else:
            nll = (
                0.5 * np.log(2.0 * np.pi)
                + np.log(scale)
                + 0.5 * np.square(residual / scale)
            )
        return "nll", float(np.mean(nll)), False
    raise ValueError(target)


def _prediction_frame(
    source: pd.DataFrame,
    target: str,
    representation: str,
    representation_variant: str,
    representation_variant_family: str,
    prototype_seed: int | None,
    input_variant: str,
    candidate: Candidate,
    seed: int,
    split: str,
    predictions: Mapping[str, np.ndarray],
    selected: bool,
) -> pd.DataFrame:
    columns = [
        "ticker",
        "feature_date",
        "target_date",
        "split",
        "target_log_variance",
        "baseline_prediction",
        _target_column(target),
    ]
    columns = list(dict.fromkeys(columns))
    output = source[columns].copy()
    output = output.rename(columns={_target_column(target): "y_true"})
    output["target"] = target
    output["representation"] = representation
    output["representation_variant"] = representation_variant
    output["representation_variant_family"] = representation_variant_family
    output["prototype_seed"] = prototype_seed
    output["experiment_group"] = _experiment_group(
        representation_variant_family
    )
    output["input_variant"] = input_variant
    output["model"] = candidate.model
    output["config_id"] = candidate.identifier
    output["seed"] = seed
    output["evaluation_split"] = split
    output["selected_on_validation"] = selected
    for name, values in predictions.items():
        output[name] = values
    if target == "signed" and "prediction" in predictions:
        output["final_prediction"] = (
            output["baseline_prediction"] + output["prediction"]
        )
    return output


def _fit_candidate(
    candidate: Candidate,
    x: np.ndarray,
    frame: pd.DataFrame,
    target: str,
    config: Mapping[str, Any],
    seed: int,
) -> Any:
    y = frame[_target_column(target)].to_numpy()
    if candidate.model.startswith("mlp_") or candidate.model in {"gaussian", "student_t"}:
        return _fit_torch(
            candidate,
            x,
            y,
            frame["target_date"],
            config,
            seed,
            baseline_prediction=(
                frame["baseline_prediction"].to_numpy()
                if target == "uncertainty"
                else None
            ),
        )
    return _fit_sklearn(candidate, x, y, seed)


def _is_better(value: float, incumbent: float | None, larger_is_better: bool) -> bool:
    if not np.isfinite(value):
        return False
    if incumbent is None or not np.isfinite(incumbent):
        return True
    return value > incumbent if larger_is_better else value < incumbent


def _chronological_screening_frames(
    train: pd.DataFrame,
    fraction: float,
    minimum_prefix_dates: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split TRAIN only into a prefix fit window and a later screening window."""

    if not 0.0 < fraction < 0.5:
        raise ValueError("models.validation_screening.inner_fraction must be in (0, 0.5)")
    dates = np.sort(pd.to_datetime(train["target_date"]).unique())
    if len(dates) <= minimum_prefix_dates:
        raise ValueError(
            "Not enough train dates for validation screening: "
            f"{len(dates)} <= minimum_prefix_dates={minimum_prefix_dates}"
        )
    cut = int(math.floor(len(dates) * (1.0 - fraction)))
    cut = max(cut, int(minimum_prefix_dates))
    cut = min(cut, len(dates) - 1)
    boundary = pd.Timestamp(dates[cut])
    prefix = train.loc[pd.to_datetime(train["target_date"]) < boundary].copy()
    tail = train.loc[pd.to_datetime(train["target_date"]) >= boundary].copy()
    if prefix.empty or tail.empty:
        raise ValueError("Chronological validation screening produced an empty window")
    if not (
        pd.to_datetime(prefix["target_date"]).max()
        < pd.to_datetime(tail["target_date"]).min()
    ):
        raise AssertionError("Screening prefix and tail are not strictly chronological")
    return prefix, tail


def _cheap_screening_score(
    target: str,
    x_prefix: np.ndarray,
    prefix: pd.DataFrame,
    x_tail: np.ndarray,
    tail: pd.DataFrame,
    config: Mapping[str, Any],
    seed: int,
) -> tuple[str, float, bool, str]:
    """Score one representation with a cheap model fitted on TRAIN-prefix only."""

    screening = config["models"].get("validation_screening", {})
    ridge_alpha = float(screening.get("ridge_alpha", 1.0))
    logistic_c = float(screening.get("logistic_c", 1.0))
    if target in {"signed", "magnitude", "squared"}:
        model = Ridge(alpha=ridge_alpha)
        model.fit(x_prefix, prefix[_target_column(target)].to_numpy())
        predictions = {"prediction": np.asarray(model.predict(x_tail), dtype=float)}
        metric, value, larger = _primary_score(target, tail, predictions)
        return metric, value, larger, "ridge"
    if target.startswith("spike"):
        y_prefix = prefix[_target_column(target)].to_numpy()
        y_tail = tail[_target_column(target)].to_numpy()
        if len(np.unique(y_prefix)) < 2 or len(np.unique(y_tail)) < 2:
            return "pr_auc", float("nan"), True, "class_support_unavailable"
        model = LogisticRegression(
            C=logistic_c,
            class_weight="balanced",
            max_iter=5_000,
            solver="lbfgs",
            random_state=seed,
        )
        model.fit(x_prefix, y_prefix)
        predictions = {
            "probability": np.asarray(model.predict_proba(x_tail)[:, 1], dtype=float)
        }
        metric, value, larger = _primary_score(target, tail, predictions)
        return metric, value, larger, "weighted_logistic"
    if target.startswith("regime"):
        y_prefix = prefix[_target_column(target)].to_numpy()
        y_tail = tail[_target_column(target)].to_numpy()
        if len(np.unique(y_prefix)) < 2 or len(np.unique(y_tail)) < 2:
            return "macro_f1", float("nan"), True, "class_support_unavailable"
        model = LogisticRegression(
            C=logistic_c,
            class_weight="balanced",
            max_iter=5_000,
            solver="lbfgs",
            random_state=seed,
        )
        model.fit(x_prefix, y_prefix)
        predictions = {"prediction": np.asarray(model.predict(x_tail), dtype=int)}
        metric, value, larger = _primary_score(target, tail, predictions)
        return metric, value, larger, "multinomial_logistic"
    if target == "uncertainty":
        epsilon = float(config["targets"].get("residual_epsilon", 1.0e-8))
        log_scale_target = np.log(
            np.abs(prefix["signed_residual"].to_numpy(dtype=float)) + epsilon
        )
        model = Ridge(alpha=ridge_alpha)
        model.fit(x_prefix, log_scale_target)
        log_scale = np.clip(
            np.asarray(model.predict(x_tail), dtype=float), -20.0, 20.0
        )
        predictions = {
            "scale": np.exp(log_scale)
            + float(config["uncertainty"]["min_scale"])
        }
        metric, value, larger = _primary_score(target, tail, predictions)
        return metric, value, larger, "ridge_log_absolute_residual"
    raise ValueError(f"Unsupported screening target: {target}")


def _rank_screening_families(
    rows: list[dict[str, Any]],
    top_n: int,
    logger: Any,
) -> tuple[pd.DataFrame, set[tuple[str, str, str, str, str]]]:
    """Rank seed-free families by mean TRAIN-tail score and retain all repeats."""

    audit = pd.DataFrame(rows)
    if audit.empty:
        raise RuntimeError("Validation screening did not produce any audit rows")
    family_keys = [
        "target",
        "representation",
        "input_variant",
        "experiment_group",
        "representation_variant_family",
    ]
    grouped = (
        audit.groupby(family_keys, dropna=False, sort=True, observed=True)
        .agg(
            family_mean_primary_value=("primary_value", "mean"),
            prototype_seed_repeats=("prototype_seed", "size"),
            finite_seed_repeats=(
                "primary_value",
                lambda values: int(np.isfinite(np.asarray(values, dtype=float)).sum()),
            ),
            larger_is_better=("larger_is_better", "first"),
        )
        .reset_index()
    )
    selected: set[tuple[str, str, str, str, str]] = set()
    grouping_keys = family_keys[:4]
    rank_pieces: list[pd.DataFrame] = []
    for group_key, frame in grouped.groupby(
        grouping_keys, dropna=False, sort=True, observed=True
    ):
        directions = set(frame["larger_is_better"].astype(bool))
        if len(directions) != 1:
            raise AssertionError(
                f"Screening metric direction differs within group {group_key}"
            )
        complete = frame.loc[
            frame["finite_seed_repeats"].eq(frame["prototype_seed_repeats"])
            & np.isfinite(frame["family_mean_primary_value"])
        ].copy()
        if complete.empty:
            logger.warning(
                "No fully scored screening family for %s; retaining all families "
                "for full validation instead of making an unsupported choice.",
                group_key,
            )
            ranked = frame.sort_values(
                "representation_variant_family", kind="mergesort"
            ).copy()
            ranked["screening_rank"] = np.nan
            ranked["selected_for_full_validation"] = True
        else:
            larger = directions.pop()
            ranked_complete = complete.sort_values(
                [
                    "family_mean_primary_value",
                    "representation_variant_family",
                ],
                ascending=[not larger, True],
                kind="mergesort",
            ).copy()
            ranked_complete["screening_rank"] = np.arange(
                1, len(ranked_complete) + 1
            )
            ranked_complete["selected_for_full_validation"] = (
                ranked_complete["screening_rank"] <= top_n
            )
            incomplete = frame.loc[~frame.index.isin(complete.index)].copy()
            incomplete["screening_rank"] = np.nan
            incomplete["selected_for_full_validation"] = False
            ranked = pd.concat([ranked_complete, incomplete], ignore_index=True)
        for row in ranked.loc[
            ranked["selected_for_full_validation"]
        ].itertuples(index=False):
            selected.add(
                (
                    str(row.target),
                    str(row.representation),
                    str(row.input_variant),
                    str(row.experiment_group),
                    str(row.representation_variant_family),
                )
            )
        rank_pieces.append(ranked)
    family_audit = pd.concat(rank_pieces, ignore_index=True)
    audit = audit.merge(
        family_audit,
        on=family_keys,
        how="left",
        validate="many_to_one",
        suffixes=("", "_family"),
    )
    return audit, selected


def run(config: dict[str, Any]) -> dict[str, Path]:
    ensure_directories(config)
    logger = get_logger(
        __name__, config, project_path(config, "outputs", "logs", "targets.log")
    )
    base_seed = int(config["project"]["seed"])
    set_global_seed(base_seed, bool(config["project"].get("deterministic", True)))
    residual_path = project_path(
        config, "data", "processed", "residual_targets.parquet"
    )
    residuals = safe_read_table(residual_path)
    validate_required_columns(
        residuals,
        [
            "ticker",
            "feature_date",
            "target_date",
            "split",
            "target_log_variance",
            "baseline_prediction",
            "signed_residual",
            "residual_magnitude",
            "log_squared_residual",
            "spike_q90",
            "spike_q95",
            "regime",
            "spike_q90_pooled_standardized",
            "spike_q95_pooled_standardized",
            "regime_pooled_standardized",
        ],
        "residual targets",
    )
    residuals["feature_date"] = pd.to_datetime(residuals["feature_date"])
    residuals["target_date"] = pd.to_datetime(residuals["target_date"])
    residuals = residuals.sort_values(["target_date", "ticker"]).reset_index(drop=True)

    artifacts = _representation_artifacts(config)
    requested = [str(value) for value in config["models"]["representations"]]
    targets = [
        "signed",
        "magnitude",
        "squared",
        "spike_q90",
        "spike_q95",
        "regime",
        "spike_q90_pooled_standardized",
        "spike_q95_pooled_standardized",
        "regime_pooled_standardized",
        "uncertainty",
    ]
    seeds = [base_seed]
    if bool(config["models"].get("run_all_robustness_seeds", True)):
        seeds = list(
            dict.fromkeys(
                [
                    base_seed,
                    *(
                        int(value)
                        for value in config.get("robustness", {}).get("seeds", [])
                    ),
                ]
            )
        )
    prediction_pieces: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    selected_models: dict[str, Any] = {}
    pending_test_jobs: list[dict[str, Any]] = []

    execution_artifacts = [
        artifact for artifact in artifacts if artifact[0] in set(requested)
    ]
    available_representations = {artifact[0] for artifact in execution_artifacts}
    for missing in sorted(set(requested).difference(available_representations)):
        if missing == "R0":
            execution_artifacts.append(
                (
                    "R0",
                    "generated_price_only",
                    "generated_price_only",
                    None,
                    None,
                )
            )
        else:
            logger.warning("Skipping %s because no feature artifact was found", missing)

    progress_options = _progress_options(config)
    planned_input_sets = sum(
        1 if artifact[0] == "R0" else 2
        for artifact in execution_artifacts
    )
    planned_screening_jobs = planned_input_sets * len(targets)
    logger.info(
        "Target training started | residual_rows=%d | artifacts=%d | "
        "targets=%d | seeds=%d | planned_screening_jobs=%d",
        len(residuals),
        len(execution_artifacts),
        len(targets),
        len(seeds),
        planned_screening_jobs,
    )

    screening_config = config["models"].get("validation_screening", {})
    screening_enabled = bool(screening_config.get("enabled", True))
    screening_selected: set[tuple[str, str, str, str, str]] | None = None
    tables = project_path(config, "outputs", "tables")
    screening_audit_path = tables / "target_screening_manifest.csv"
    if screening_enabled:
        top_n = max(1, int(screening_config.get("top_n_families", 2)))
        inner_fraction = float(screening_config.get("inner_fraction", 0.15))
        minimum_prefix_dates = int(
            screening_config.get("minimum_prefix_dates", 63)
        )
        screening_rows: list[dict[str, Any]] = []
        screening_progress = _ProgressReporter(
            logger,
            "screening",
            planned_screening_jobs,
            **progress_options,
        ).start()
        for (
            representation,
            representation_variant,
            representation_variant_family,
            prototype_seed,
            path,
        ) in execution_artifacts:
            screening_progress.start_job(
                f"load {representation}/{representation_variant}"
            )
            if path is not None and not path.exists():
                raise FileNotFoundError(
                    f"Representation artifact in manifest does not exist: {path}"
                )
            joined, price_columns, text_columns = _load_joined_representation(
                config, residuals, representation, path
            )
            train = joined.loc[joined["split"] == "train"].copy()
            screen_prefix, screen_tail = _chronological_screening_frames(
                train, inner_fraction, minimum_prefix_dates
            )
            for input_variant, feature_columns in _feature_sets(
                representation, price_columns, text_columns
            ).items():
                if not feature_columns:
                    continue
                processor = _make_preprocessor(feature_columns)
                x_prefix = np.asarray(
                    processor.fit_transform(
                        screen_prefix[[*feature_columns, "ticker"]]
                    ),
                    dtype=np.float32,
                )
                x_tail = np.asarray(
                    processor.transform(screen_tail[[*feature_columns, "ticker"]]),
                    dtype=np.float32,
                )
                experiment_group = _experiment_group(
                    representation_variant_family
                )
                for target in targets:
                    screening_progress.start_job(
                        f"{target} | {representation}/"
                        f"{representation_variant}/{input_variant}"
                    )
                    metric, score, larger, screening_model = (
                        _cheap_screening_score(
                            target,
                            x_prefix,
                            screen_prefix,
                            x_tail,
                            screen_tail,
                            config,
                            base_seed,
                        )
                    )
                    screening_rows.append(
                        {
                            "target": target,
                            "representation": representation,
                            "representation_variant": representation_variant,
                            "representation_variant_family": (
                                representation_variant_family
                            ),
                            "prototype_seed": prototype_seed,
                            "prototype_seed_role": "replicate_not_hyperparameter",
                            "experiment_group": experiment_group,
                            "input_variant": input_variant,
                            "artifact_path": "" if path is None else str(path),
                            "screening_model": screening_model,
                            "primary_metric": metric,
                            "primary_value": score,
                            "larger_is_better": larger,
                            "fit_scope": (
                                "representation_main_train__downstream_train_prefix"
                            ),
                            "representation_fit_scope": "main_train_only",
                            "downstream_fit_scope": "train_prefix_only",
                            "evaluation_scope": "train_tail_only",
                            "screening_seed": base_seed,
                            "n_features": x_prefix.shape[1],
                            "n_fit": len(screen_prefix),
                            "n_evaluation": len(screen_tail),
                            "fit_start": pd.to_datetime(
                                screen_prefix["target_date"]
                            ).min(),
                            "fit_end": pd.to_datetime(
                                screen_prefix["target_date"]
                            ).max(),
                            "evaluation_start": pd.to_datetime(
                                screen_tail["target_date"]
                            ).min(),
                            "evaluation_end": pd.to_datetime(
                                screen_tail["target_date"]
                            ).max(),
                            "uses_validation": False,
                            "uses_test": False,
                        }
                    )
                    screening_progress.advance()
        screening_progress.finish()
        screening_audit, screening_selected = _rank_screening_families(
            screening_rows, top_n, logger
        )
        screening_audit = screening_audit.sort_values(
            [
                "target",
                "representation",
                "input_variant",
                "experiment_group",
                "screening_rank",
                "representation_variant_family",
                "prototype_seed",
            ],
            kind="mergesort",
            na_position="last",
        )
        screening_audit_path = atomic_write_csv(
            screening_audit, screening_audit_path, index=False
        )
        logger.info(
            "TRAIN-prefix screening retained %d seed-free target/representation "
            "families (top_n=%d); full model search starts only after this lock.",
            len(screening_selected),
            top_n,
        )
    else:
        screening_audit_path = atomic_write_csv(
            pd.DataFrame(
                [
                    {
                        "screening_enabled": False,
                        "reason": "models.validation_screening.enabled=false",
                        "uses_validation": False,
                        "uses_test": False,
                    }
                ]
            ),
            screening_audit_path,
            index=False,
        )

    planned_full_fit_jobs = 0
    for (
        representation,
        _representation_variant,
        representation_variant_family,
        _prototype_seed,
        _path,
    ) in execution_artifacts:
        input_variants = (
            ("price_only",)
            if representation == "R0"
            else ("price_plus_text", "text_only")
        )
        experiment_group = _experiment_group(
            representation_variant_family
        )
        for input_variant in input_variants:
            for target in targets:
                if (
                    screening_selected is not None
                    and (
                        target,
                        representation,
                        input_variant,
                        experiment_group,
                        representation_variant_family,
                    )
                    not in screening_selected
                ):
                    continue
                planned_full_fit_jobs += len(_candidate_grid(target, config))
                planned_full_fit_jobs += max(len(seeds) - 1, 0)
    logger.info(
        "Full validation search plan | candidate_fits=%d",
        planned_full_fit_jobs,
    )
    validation_progress = _ProgressReporter(
        logger,
        "validation-search",
        planned_full_fit_jobs,
        **progress_options,
    ).start()
    for (
        representation,
        representation_variant,
        representation_variant_family,
        prototype_seed,
        path,
    ) in execution_artifacts:
        validation_progress.start_job(
            f"load {representation}/{representation_variant}"
        )
        if path is not None and not path.exists():
            raise FileNotFoundError(
                f"Representation artifact in manifest does not exist: {path}"
            )
        joined, price_columns, text_columns = _load_joined_representation(
            config, residuals, representation, path
        )
        for input_variant, feature_columns in _feature_sets(
            representation, price_columns, text_columns
        ).items():
            if not feature_columns:
                logger.warning("Skipping empty feature set %s/%s", representation, input_variant)
                continue
            experiment_group = _experiment_group(
                representation_variant_family
            )
            eligible_targets = [
                target
                for target in targets
                if screening_selected is None
                or (
                    target,
                    representation,
                    input_variant,
                    experiment_group,
                    representation_variant_family,
                )
                in screening_selected
            ]
            if not eligible_targets:
                continue
            train = joined.loc[joined["split"] == "train"].copy()
            validation = joined.loc[joined["split"] == "validation"].copy()
            test = joined.loc[joined["split"] == "test"].copy()
            primary_choices: dict[str, Candidate] = {}
            processor = _make_preprocessor(feature_columns)
            x_train = np.asarray(
                processor.fit_transform(train[[*feature_columns, "ticker"]]),
                dtype=np.float32,
            )
            x_validation = np.asarray(
                processor.transform(validation[[*feature_columns, "ticker"]]),
                dtype=np.float32,
            )
            for seed in seeds:
                for target in eligible_targets:
                    if seed != base_seed and target not in primary_choices:
                        continue
                    best: tuple[Candidate, Any, float, str, bool] | None = None
                    candidate_grid = (
                        _candidate_grid(target, config)
                        if seed == base_seed
                        else [primary_choices[target]]
                    )
                    for candidate in candidate_grid:
                        validation_progress.start_job(
                            f"{target} | {representation}/"
                            f"{representation_variant}/{input_variant} | "
                            f"seed={seed} | {candidate.identifier}"
                        )
                        model = _fit_candidate(
                            candidate, x_train, train, target, config, seed
                        )
                        prediction = _predict_candidate(
                            candidate, model, x_validation, config
                        )
                        metric_name, score, larger = _primary_score(
                            target, validation, prediction
                        )
                        manifest_rows.append(
                            {
                                "target": target,
                                "representation": representation,
                                "representation_variant": representation_variant,
                                "representation_variant_family": (
                                    representation_variant_family
                                ),
                                "prototype_seed": prototype_seed,
                                "experiment_group": _experiment_group(
                                    representation_variant_family
                                ),
                                "input_variant": input_variant,
                                "seed": seed,
                                "model": candidate.model,
                                "config_id": candidate.identifier,
                                "parameters": json.dumps(candidate.parameters, sort_keys=True),
                                "split": "validation",
                                "primary_metric": metric_name,
                                "primary_value": score,
                                "larger_is_better": larger,
                                "selected": False,
                                "selected_using_test": False,
                                "fit_scope": "train_only",
                                "validation_used_for_fit": False,
                                "n_features": x_train.shape[1],
                                "n_train": len(train),
                                "n_evaluation": len(validation),
                            }
                        )
                        if best is None or _is_better(score, best[2], larger):
                            best = (candidate, model, score, metric_name, larger)
                        validation_progress.advance()
                    if best is None or not np.isfinite(best[2]):
                        logger.warning(
                            "No finite validation candidate for %s/%s/%s/%s",
                            target,
                            representation,
                            representation_variant,
                            input_variant,
                        )
                        continue
                    chosen, validation_model, best_score, metric_name, larger = best
                    if seed == base_seed:
                        primary_choices[target] = chosen
                    for row in reversed(manifest_rows):
                        if (
                            row["target"] == target
                            and row["representation"] == representation
                            and row["representation_variant"] == representation_variant
                            and row["input_variant"] == input_variant
                            and row["seed"] == seed
                            and row["config_id"] == chosen.identifier
                        ):
                            row["selected"] = True
                            break

                    pending_test_jobs.append(
                        {
                            "target": target,
                            "representation": representation,
                            "representation_variant": representation_variant,
                            "representation_variant_family": (
                                representation_variant_family
                            ),
                            "prototype_seed": prototype_seed,
                            "experiment_group": _experiment_group(
                                representation_variant_family
                            ),
                            "input_variant": input_variant,
                            "seed": seed,
                            "candidate": chosen,
                            "processor": processor,
                            "validation_model": validation_model,
                            "validation_prediction": _predict_candidate(
                                chosen, validation_model, x_validation, config
                            ),
                            "validation_primary_value": best_score,
                            "validation_primary_metric": metric_name,
                            "larger_is_better": larger,
                            "feature_columns": feature_columns,
                            "train": train,
                            "validation": validation,
                            "test": test,
                        }
                    )
    validation_progress.finish()

    # K/PCA/temperature/pooling is selected on validation, but a stochastic
    # prototype seed is a replicate rather than a tunable configuration.  Mean
    # the primary-model-seed scores within each seed-free family and lock the
    # family before any test transform or prediction is produced.
    jobs_by_family: dict[
        tuple[str, str, str, str], list[dict[str, Any]]
    ] = {}
    for job in pending_test_jobs:
        if int(job["seed"]) != base_seed:
            continue
        family_key = (
            str(job["target"]),
            str(job["representation"]),
            str(job["input_variant"]),
            str(job["representation_variant_family"]),
        )
        jobs_by_family.setdefault(family_key, []).append(job)

    best_family_jobs: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for family_key, family_jobs in jobs_by_family.items():
        scores = np.asarray(
            [float(job["validation_primary_value"]) for job in family_jobs],
            dtype=float,
        )
        finite_scores = scores[np.isfinite(scores)]
        if not len(finite_scores):
            continue
        larger_values = {bool(job["larger_is_better"]) for job in family_jobs}
        if len(larger_values) != 1:
            raise AssertionError(
                f"Primary metric direction differs within family {family_key}"
            )
        summary = {
            "representation_variant_family": family_key[3],
            "validation_primary_value": float(np.mean(finite_scores)),
            "larger_is_better": larger_values.pop(),
            "prototype_seed_count": len(
                {
                    job["prototype_seed"]
                    for job in family_jobs
                    if job["prototype_seed"] is not None
                }
            ),
        }
        selection_key = (
            *family_key[:3],
            _experiment_group(family_key[3]),
        )
        incumbent = best_family_jobs.get(selection_key)
        if incumbent is None or _is_better(
            float(summary["validation_primary_value"]),
            float(incumbent["validation_primary_value"]),
            bool(summary["larger_is_better"]),
        ):
            best_family_jobs[selection_key] = summary
    if not best_family_jobs:
        raise RuntimeError("Validation did not lock any representation configuration")

    locked_pending_jobs: list[dict[str, Any]] = []
    for job in pending_test_jobs:
        selection_key = (
            str(job["target"]),
            str(job["representation"]),
            str(job["input_variant"]),
            _experiment_group(str(job["representation_variant_family"])),
        )
        locked = best_family_jobs.get(selection_key)
        if locked is None:
            logger.warning(
                "Skipping unlocked validation group with no finite family score: %s",
                selection_key,
            )
            continue
        if (
            str(job["representation_variant_family"])
            == str(locked["representation_variant_family"])
        ):
            locked_pending_jobs.append(job)
    logger.info(
        "Locked test prediction plan | jobs=%d",
        len(locked_pending_jobs),
    )
    test_progress = _ProgressReporter(
        logger,
        "locked-test",
        len(locked_pending_jobs),
        **progress_options,
    ).start()
    locked_config_by_repeat: dict[tuple[str, str, str, str, int], str] = {}
    for job in locked_pending_jobs:
        target = str(job["target"])
        representation = str(job["representation"])
        representation_variant = str(job["representation_variant"])
        representation_variant_family = str(
            job["representation_variant_family"]
        )
        prototype_seed = job["prototype_seed"]
        input_variant = str(job["input_variant"])
        seed = int(job["seed"])
        chosen = job["candidate"]
        test_progress.start_job(
            f"{target} | {representation}/{representation_variant}/"
            f"{input_variant} | seed={seed}"
        )
        processor = job["processor"]
        validation_model = job["validation_model"]
        validation_prediction = job["validation_prediction"]
        feature_columns = list(job["feature_columns"])
        train = job["train"]
        validation = job["validation"]
        test = job["test"]
        prediction_pieces.append(
            _prediction_frame(
                validation,
                target,
                representation,
                representation_variant,
                representation_variant_family,
                prototype_seed,
                input_variant,
                chosen,
                seed,
                "validation",
                validation_prediction,
                selected=True,
            )
        )
        x_test = np.asarray(
            processor.transform(test[[*feature_columns, "ticker"]]),
            dtype=np.float32,
        )
        test_prediction = _predict_candidate(
            chosen, validation_model, x_test, config
        )
        test_metric_name, test_score, _ = _primary_score(
            target, test, test_prediction
        )
        prediction_pieces.append(
            _prediction_frame(
                test,
                target,
                representation,
                representation_variant,
                representation_variant_family,
                prototype_seed,
                input_variant,
                chosen,
                seed,
                "test",
                test_prediction,
                selected=True,
            )
        )
        manifest_rows.append(
            {
                "target": target,
                "representation": representation,
                "representation_variant": representation_variant,
                "representation_variant_family": (
                    representation_variant_family
                ),
                "prototype_seed": prototype_seed,
                "experiment_group": _experiment_group(
                    representation_variant_family
                ),
                "input_variant": input_variant,
                "seed": seed,
                "model": chosen.model,
                "config_id": chosen.identifier,
                "parameters": json.dumps(chosen.parameters, sort_keys=True),
                "split": "test",
                "primary_metric": test_metric_name,
                "primary_value": test_score,
                "larger_is_better": bool(job["larger_is_better"]),
                "selected": True,
                "selected_using_test": False,
                "fit_scope": "train_only",
                "validation_used_for_fit": False,
                "n_features": x_test.shape[1],
                "n_train": len(train),
                "n_evaluation": len(test),
            }
        )
        artifact_key = (
            f"{target}__{representation}__{representation_variant}"
            f"__{input_variant}__seed{seed}"
        )
        selected_models[artifact_key] = {
            "candidate": chosen,
            "processor": processor,
            "model": validation_model,
            "features": feature_columns,
            "validation_primary_value": job["validation_primary_value"],
            "validation_primary_metric": job["validation_primary_metric"],
        }
        locked_config_by_repeat[
            (
                target,
                representation,
                input_variant,
                representation_variant,
                seed,
            )
        ] = chosen.identifier
        logger.info(
            "Locked %s | %s/%s/%s | %s | validation %s=%.6f | test=%.6f",
            target,
            representation,
            representation_variant,
            input_variant,
            chosen.identifier,
            job["validation_primary_metric"],
            job["validation_primary_value"],
            test_score,
        )
        test_progress.advance()
    test_progress.finish()

    if not prediction_pieces:
        raise RuntimeError("No target models were trained; check representation artifacts")
    logger.info(
        "Finalizing target artifacts | prediction_blocks=%d | manifest_rows=%d "
        "| selected_models=%d",
        len(prediction_pieces),
        len(manifest_rows),
        len(selected_models),
    )
    predictions = pd.concat(prediction_pieces, ignore_index=True)
    manifest = pd.DataFrame(manifest_rows)
    validation_mask = predictions["evaluation_split"].eq("validation")

    def is_locked_prediction(row: Any) -> bool:
        selection_key = (
            str(row.target),
            str(row.representation),
            str(row.input_variant),
            _experiment_group(str(row.representation_variant_family)),
        )
        locked = best_family_jobs.get(selection_key)
        if locked is None:
            return False
        return bool(
            str(row.representation_variant_family)
            == str(locked["representation_variant_family"])
            and str(row.config_id)
            == locked_config_by_repeat.get(
                (
                    str(row.target),
                    str(row.representation),
                    str(row.input_variant),
                    str(row.representation_variant),
                    int(row.seed),
                ),
                "",
            )
        )

    predictions.loc[validation_mask, "selected_on_validation"] = [
        is_locked_prediction(row)
        for row in predictions.loc[validation_mask].itertuples(index=False)
    ]
    manifest.loc[manifest["split"].eq("validation"), "selected"] = False
    for index, row in manifest.loc[manifest["split"].eq("validation")].iterrows():
        key = (
            str(row["target"]),
            str(row["representation"]),
            str(row["input_variant"]),
            str(row["representation_variant"]),
            int(row["seed"]),
        )
        family_selection_key = (
            *key[:3],
            _experiment_group(str(row["representation_variant_family"])),
        )
        locked_summary = best_family_jobs.get(family_selection_key)
        if locked_summary is None:
            manifest.loc[index, "selected"] = False
            continue
        locked_family = locked_summary["representation_variant_family"]
        manifest.loc[index, "selected"] = bool(
            str(row["representation_variant_family"]) == str(locked_family)
            and str(row["config_id"]) == locked_config_by_repeat.get(key, "")
        )
    processed = project_path(config, "data", "processed")
    tables = project_path(config, "outputs", "tables")
    models = project_path(config, "outputs", "models")
    prediction_path = write_table(
        predictions, processed / "target_predictions.parquet", index=False
    )
    manifest_path = atomic_write_csv(
        manifest, tables / "target_training_manifest.csv", index=False
    )
    model_path = atomic_joblib_dump(
        selected_models, models / "selected_target_models.joblib"
    )
    feature_audit = {
        key: {
            "features": value["features"],
            "validation_primary_metric": value["validation_primary_metric"],
            "validation_primary_value": value["validation_primary_value"],
        }
        for key, value in selected_models.items()
    }
    audit_path = write_json(feature_audit, models / "target_feature_audit.json")
    logger.info(
        "Target training artifacts completed | predictions=%s | manifest=%s "
        "| models=%s",
        prediction_path,
        manifest_path,
        model_path,
    )
    return {
        "target_predictions": prediction_path,
        "target_screening_manifest": screening_audit_path,
        "training_manifest": manifest_path,
        "selected_models": model_path,
        "feature_audit": audit_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to config YAML")
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
