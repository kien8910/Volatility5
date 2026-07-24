"""Leakage-safe feature assembly and small-model training primitives."""

from __future__ import annotations

import copy
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.progress_tracker import ProgressTracker, TaskSpec
from src.utils import (
    atomic_joblib_dump,
    load_representation_catalog,
    load_representation_frame,
    price_feature_columns,
    project_path,
    read_table,
    representation_feature_columns,
    resolve_shared_file,
    selected_representation_rows,
    shared_root,
    task_split_frames,
    validate_columns,
    write_table,
)


@dataclass
class PreparedData:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    feature_columns: list[str]
    x_train: np.ndarray
    x_validation: np.ndarray
    x_test: np.ndarray
    processor: ColumnTransformer
    representation_fit_scope: str
    qualifies_for_robustness: bool


FOLD_MANIFEST_REPRESENTATIONS = {
    "R3",
    "R4",
    "R5",
    "R6",
    "R7",
    "R8",
    "R9",
    "R10",
    "R11",
    "P_LAGGED",
    "P_PERMUTED",
}

NEWS_LEVELS = ("macro", "sector", "related", "target")
_NEWS_GATE_CACHE: dict[
    tuple[str, str, float],
    tuple[pd.DataFrame, str],
] = {}


def experiment_task(
    stage: str,
    action: str,
    task_config: Mapping[str, Any],
    *,
    required: bool = False,
) -> TaskSpec:
    provisional = TaskSpec(
        stage=stage,
        action=action,
        config=dict(task_config),
        required=required,
    ).with_id()
    return TaskSpec(
        stage=provisional.stage,
        action=provisional.action,
        config=provisional.config,
        outputs=(
            f"outputs/checkpoints/tasks/{provisional.task_id}.parquet",
            f"outputs/models/{provisional.task_id}.joblib",
        ),
        required=required,
        task_id=provisional.task_id,
    )


def task_checkpoint_paths(
    config: Mapping[str, Any], task: TaskSpec
) -> tuple[Path, Path]:
    directory = project_path(config, "outputs", "checkpoints", "tasks")
    return (
        directory / f"{task.task_id}.parquet",
        project_path(config, "outputs", "models", f"{task.task_id}.joblib"),
    )


def representation_row(
    config: Mapping[str, Any],
    representation: str,
    representation_variant: str,
    *,
    fold: str | int = "holdout",
    seed: int | None = None,
    representation_variant_family: str | None = None,
) -> pd.Series:
    if (
        str(fold) != "holdout"
        and representation in FOLD_MANIFEST_REPRESENTATIONS
    ):
        path = resolve_shared_file(
            config,
            str(config["shared"]["fold_representation_manifest"]),
            kinds=("processed", "tables"),
            required=False,
        )
        if path is None:
            raise FileNotFoundError(
                "Fold-train prototype representations are unavailable; "
                "refusing to reuse main-train prototypes in an earlier fold"
            )
        manifest = read_table(path)
        validate_columns(
            manifest,
            (
                "fold_id",
                "representation",
                "representation_variant_family",
                "prototype_seed",
                "path",
                "fit_scope",
            ),
            "fold representation manifest",
        )
        family = str(
            representation_variant_family or representation_variant
        )
        selected = manifest.loc[
            (pd.to_numeric(manifest["fold_id"], errors="coerce") == int(fold))
            & (manifest["representation"].astype(str) == representation)
            & (
                manifest["representation_variant_family"].astype(str)
                == family
            )
            & (
                pd.to_numeric(manifest["prototype_seed"], errors="coerce")
                == int(seed if seed is not None else config["project"]["seed"])
            )
        ].copy()
        if len(selected) != 1:
            raise ValueError(
                f"Expected one fold artifact for fold={fold}, "
                f"{representation}/{family}, seed={seed}; found {len(selected)}"
            )
        row = selected.iloc[0].copy()
        source = Path(str(row["path"]))
        if not source.is_file():
            matches = list(
                shared_root(config).rglob(source.name)
            )
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"Cannot resolve fold representation artifact {source}"
                )
            source = matches[0]
        row["resolved_path"] = str(source.resolve())
        row["fit_split"] = str(row["fit_scope"])
        return row
    catalog = load_representation_catalog(config)
    selected = catalog.loc[
        (catalog["representation"].astype(str) == representation)
        & (
            catalog["representation_variant"].astype(str)
            == representation_variant
        )
    ]
    if selected.empty and representation_variant == "selected_default":
        selected = catalog.loc[
            (catalog["representation"].astype(str) == representation)
            & catalog["selected"].fillna(False).astype(bool)
        ]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one artifact for {representation}/{representation_variant}; "
            f"found {len(selected)}"
        )
    return selected.iloc[0]


def fold_task_supported(
    config: Mapping[str, Any],
    representation: str,
    representation_variant_family: str,
    fold: str | int,
    seed: int,
) -> bool:
    """Return False instead of ever evaluating a fold with future-fitted text."""

    if str(fold) == "holdout" or representation in {"R0", "R1", "R2"}:
        return True
    if representation not in FOLD_MANIFEST_REPRESENTATIONS:
        return False
    try:
        representation_row(
            config,
            representation,
            representation_variant_family,
            fold=fold,
            seed=seed,
            representation_variant_family=representation_variant_family,
        )
    except (FileNotFoundError, ValueError, KeyError):
        return False
    return True


def plan_representation_variants(
    config: Mapping[str, Any],
    representations: Sequence[str],
    *,
    quick: bool,
    fixed_family: str | None = None,
) -> list[dict[str, str]]:
    catalog = load_representation_catalog(config)
    if fixed_family is not None:
        rows: list[dict[str, str]] = []
        for representation in representations:
            name = str(representation)
            if name == "R0":
                selected = selected_representation_rows(
                    catalog, [name], selected_only=True
                )
                if len(selected) != 1:
                    raise ValueError(
                        "The confirmatory grid requires exactly one selected R0."
                    )
                row = selected.iloc[0]
                rows.append(
                    {
                        "representation": name,
                        "representation_variant": str(
                            row["representation_variant"]
                        ),
                        "representation_variant_family": str(
                            row.get(
                                "representation_variant_family",
                                row["representation_variant"],
                            )
                        ),
                    }
                )
            else:
                rows.append(
                    {
                        "representation": name,
                        "representation_variant": str(fixed_family),
                        "representation_variant_family": str(fixed_family),
                    }
                )
        return rows
    rows = selected_representation_rows(
        catalog, representations, selected_only=quick
    )
    return [
        {
            "representation": str(row.representation),
            "representation_variant": str(row.representation_variant),
            "representation_variant_family": str(
                getattr(
                    row,
                    "representation_variant_family",
                    row.representation_variant,
                )
            ),
        }
        for row in rows.itertuples(index=False)
    ]


def feature_columns_for_news_levels(
    feature_columns: Sequence[str],
    news_levels: Sequence[str],
) -> list[str]:
    """Keep only feature blocks whose explicit level token is requested."""
    levels = tuple(str(level).strip().lower() for level in news_levels)
    unknown = sorted(set(levels).difference(NEWS_LEVELS))
    if unknown:
        raise ValueError(f"Unknown news levels: {unknown}")
    if not levels:
        raise ValueError("At least one news level must be selected.")
    tokens = tuple(f"__{level}__" for level in levels)
    return [
        column
        for column in feature_columns
        if any(token in str(column).lower() for token in tokens)
    ]


def _news_presence_frame(
    config: Mapping[str, Any],
    task: TaskSpec,
    profile: Mapping[str, Any],
) -> tuple[pd.DataFrame, str]:
    level = str(profile["evaluation_news_level"]).strip().lower()
    if level not in NEWS_LEVELS:
        raise ValueError(f"Unknown evaluation news level: {level!r}")
    representation = str(
        profile.get("evaluation_gate_representation", "R6")
    )
    family = str(profile["representation_variant_family"])
    row = representation_row(
        config,
        representation,
        family,
        fold=task.config["fold"],
        seed=int(task.config["seed"]),
        representation_variant_family=family,
    )
    required_pooling = profile.get("required_pooling")
    artifact_pooling = row.get("pooling")
    if (
        required_pooling is not None
        and pd.notna(artifact_pooling)
        and str(artifact_pooling) != str(required_pooling)
    ):
        raise ValueError(
            f"Target-news gate requires pooling={required_pooling!r}, "
            f"but the fold artifact uses {artifact_pooling!r}."
        )
    epsilon = float(profile.get("news_presence_epsilon", 1.0e-10))
    cache_key = (str(row["resolved_path"]), level, epsilon)
    cached = _NEWS_GATE_CACHE.get(cache_key)
    if cached is not None:
        cached_frame, cached_column = cached
        return cached_frame.copy(), cached_column
    frame = load_representation_frame(
        config,
        row,
        seed=int(task.config["seed"]),
    )
    numeric_columns = representation_feature_columns(frame)
    level_columns = feature_columns_for_news_levels(
        numeric_columns,
        [level],
    )
    if not level_columns:
        raise ValueError(
            f"Gate representation {representation} has no {level!r} features."
        )
    values = (
        frame[level_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    gate_column = f"has_{level}_news"
    gate = frame[["ticker", "feature_date"]].copy()
    gate[gate_column] = np.abs(values).sum(axis=1) > epsilon
    gate["__news_gate_observed"] = True
    if gate.duplicated(["ticker", "feature_date"]).any():
        raise AssertionError("News-presence gate contains duplicate row keys.")
    _NEWS_GATE_CACHE[cache_key] = (gate.copy(), gate_column)
    return gate, gate_column


def _make_processor(feature_columns: Sequence[str]) -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    ticker = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        dtype=np.float32,
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, list(feature_columns)),
            ("ticker", ticker, ["ticker"]),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def prepare_stock_data(
    config: Mapping[str, Any],
    task: TaskSpec,
    profile: Mapping[str, Any],
) -> PreparedData:
    market = read_table(
        project_path(config, "data", "processed", "original_market_targets.parquet")
    )
    validate_columns(
        market,
        ("ticker", "feature_date", "target_date", "split", "volatility_level"),
        "original market targets",
    )
    market["feature_date"] = pd.to_datetime(market["feature_date"]).dt.normalize()
    representation = str(task.config["representation"])
    variant = str(task.config.get("representation_variant", "selected_default"))
    row = representation_row(
        config,
        representation,
        variant,
        fold=task.config["fold"],
        seed=int(task.config["seed"]),
        representation_variant_family=str(
            task.config.get("representation_variant_family", variant)
        ),
    )
    representation_frame = load_representation_frame(
        config, row, seed=int(task.config["seed"])
    )
    text_columns = representation_feature_columns(representation_frame)
    selected_news_levels = task.config.get("text_news_levels")
    if selected_news_levels is not None and representation != "R0":
        text_columns = feature_columns_for_news_levels(
            text_columns,
            selected_news_levels,
        )
        if not text_columns:
            raise ValueError(
                f"No {list(selected_news_levels)!r} text features found for "
                f"{representation}/{variant}."
            )
    joined = market.merge(
        representation_frame.drop(columns=["split"], errors="ignore"),
        on=["ticker", "feature_date"],
        how="left",
        validate="one_to_one",
    )
    gate_column: str | None = None
    if profile.get("evaluation_news_level") is not None:
        gate, gate_column = _news_presence_frame(
            config,
            task,
            profile,
        )
        joined = joined.merge(
            gate,
            on=["ticker", "feature_date"],
            how="left",
            validate="one_to_one",
        )
    if text_columns:
        joined[text_columns] = joined[text_columns].replace(
            [np.inf, -np.inf], np.nan
        )
    price_columns = price_feature_columns(joined, config)
    input_variant = str(task.config.get("input_variant", "price_plus_text"))
    if representation == "R0" or input_variant == "price_only":
        feature_columns = price_columns
    elif input_variant == "text_only":
        feature_columns = text_columns
    elif input_variant == "price_plus_text":
        feature_columns = list(dict.fromkeys([*price_columns, *text_columns]))
    else:
        raise ValueError(f"Unsupported input variant: {input_variant}")
    if not feature_columns:
        raise ValueError(
            f"No features for {representation}/{variant}/{input_variant}"
        )
    train, validation, test = task_split_frames(
        joined, config, profile, task.config["fold"]
    )
    if gate_column is not None:
        for split_name, split_frame in (
            ("train", train),
            ("validation", validation),
            ("test", test),
        ):
            if split_frame.empty:
                continue
            if split_frame["__news_gate_observed"].isna().any():
                raise AssertionError(
                    f"{split_name} rows are missing the true-news gate for "
                    f"fold={task.config['fold']}, seed={task.config['seed']}."
                )
            split_frame[gate_column] = (
                split_frame[gate_column].fillna(False).astype(bool)
            )
    processor = _make_processor(feature_columns)
    x_train = np.asarray(
        processor.fit_transform(train[[*feature_columns, "ticker"]]),
        dtype=np.float32,
    )
    x_validation = np.asarray(
        processor.transform(validation[[*feature_columns, "ticker"]]),
        dtype=np.float32,
    )
    x_test = (
        np.asarray(
            processor.transform(test[[*feature_columns, "ticker"]]),
            dtype=np.float32,
        )
        if not test.empty
        else np.empty((0, x_train.shape[1]), dtype=np.float32)
    )
    fold = str(task.config["fold"])
    fit_scope = str(row.get("fit_split", "train"))
    train_dependent = representation in FOLD_MANIFEST_REPRESENTATIONS
    fold_safe = (
        fold == "holdout"
        or not train_dependent
        or fit_scope == "fold_train_only"
    )
    representation_fit_scope = (
        "shared_main_train"
        if fold == "holdout"
        else "target_fold_train_independent_representation"
        if fold_safe
        else "shared_main_train_not_refit_in_target_fold"
    )
    if fold == "holdout" and fit_scope not in {"train", "train_only", "main_train_only"}:
        representation_fit_scope = fit_scope
    return PreparedData(
        train=train,
        validation=validation,
        test=test,
        feature_columns=feature_columns,
        x_train=x_train,
        x_validation=x_validation,
        x_test=x_test,
        processor=processor,
        representation_fit_scope=representation_fit_scope,
        qualifies_for_robustness=fold_safe,
    )


def prepare_sector_data(
    config: Mapping[str, Any],
    task: TaskSpec,
    profile: Mapping[str, Any],
) -> PreparedData:
    sector = read_table(
        project_path(config, "data", "processed", "sector_targets.parquet")
    )
    sector["ticker"] = "SEMICONDUCTOR_SECTOR"
    sector["target_date"] = pd.to_datetime(sector["target_date"]).dt.normalize()
    sector["feature_date"] = pd.to_datetime(sector["feature_date"]).dt.normalize()
    representation = str(task.config["representation"])
    variant = str(task.config.get("representation_variant", "selected_default"))
    row = representation_row(
        config,
        representation,
        variant,
        fold=task.config["fold"],
        seed=int(task.config["seed"]),
        representation_variant_family=str(
            task.config.get("representation_variant_family", variant)
        ),
    )
    ticker_features = load_representation_frame(
        config, row, seed=int(task.config["seed"])
    )
    text_columns = representation_feature_columns(ticker_features)
    news_scope = str(task.config.get("news_scope", "all"))
    if representation != "R0" and news_scope != "all":
        allowed_levels = {
            "macro": ("macro",),
            "sector": ("sector",),
            "macro_sector": ("macro", "sector"),
        }.get(news_scope)
        if allowed_levels is None:
            raise ValueError(f"Unsupported sector news scope: {news_scope}")
        text_columns = [
            column
            for column in text_columns
            if any(
                f"__{level}__" in column
                or column.startswith(f"meta__{level}__")
                for level in allowed_levels
            )
        ]
    aggregation = {column: "mean" for column in text_columns}
    daily_text = (
        ticker_features.groupby("feature_date", sort=True, observed=True)
        .agg(aggregation)
        .reset_index()
        if text_columns
        else ticker_features[["feature_date"]].drop_duplicates()
    )
    joined = sector.merge(
        daily_text, on="feature_date", how="left", validate="one_to_one"
    )
    sector_price_columns = [
        column
        for column in joined.columns
        if column.startswith(("sector_price_mean__", "sector_price_std__"))
    ]
    input_variant = str(task.config.get("input_variant", "price_plus_text"))
    if representation == "R0" or input_variant == "price_only":
        feature_columns = sector_price_columns
    elif input_variant == "text_only":
        feature_columns = text_columns
    else:
        feature_columns = list(dict.fromkeys([*sector_price_columns, *text_columns]))
    train, validation, test = task_split_frames(
        joined, config, profile, task.config["fold"]
    )
    processor = _make_processor(feature_columns)
    x_train = np.asarray(
        processor.fit_transform(train[[*feature_columns, "ticker"]]),
        dtype=np.float32,
    )
    x_validation = np.asarray(
        processor.transform(validation[[*feature_columns, "ticker"]]),
        dtype=np.float32,
    )
    x_test = (
        np.asarray(
            processor.transform(test[[*feature_columns, "ticker"]]),
            dtype=np.float32,
        )
        if not test.empty
        else np.empty((0, x_train.shape[1]), dtype=np.float32)
    )
    fold = str(task.config["fold"])
    train_dependent = representation in {"R3", "R5", "R6", "R7", "R8", "R9", "R10", "R11"}
    fit_scope = str(row.get("fit_split", "train"))
    fold_safe = (
        fold == "holdout"
        or not train_dependent
        or fit_scope == "fold_train_only"
    )
    return PreparedData(
        train=train,
        validation=validation,
        test=test,
        feature_columns=feature_columns,
        x_train=x_train,
        x_validation=x_validation,
        x_test=x_test,
        processor=processor,
        representation_fit_scope=(
            "shared_main_train"
            if fold == "holdout"
            else "fold_train_only"
            if fit_scope == "fold_train_only"
            else "shared_main_train_not_refit_in_target_fold"
            if train_dependent
            else "target_fold_train_independent_representation"
        ),
        qualifies_for_robustness=fold_safe,
    )


def stock_spike_labels(
    data: PreparedData,
    quantile: float,
    mode: str,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    train = data.train
    frames = [data.train, data.validation, data.test]
    if mode == "ticker":
        thresholds = (
            train.groupby("ticker", observed=True)["volatility_level"]
            .quantile(quantile)
            .to_dict()
        )
        labels = [
            (
                frame["volatility_level"]
                > frame["ticker"].map(thresholds).astype(float)
            )
            .astype(np.int8)
            .to_numpy()
            for frame in frames
        ]
        metadata = {"ticker_thresholds": thresholds}
    elif mode == "pooled_standardized":
        statistics = train.groupby("ticker", observed=True)["volatility_level"].agg(
            ["mean", "std"]
        )
        standardized = []
        for frame in frames:
            mean = frame["ticker"].map(statistics["mean"])
            std = frame["ticker"].map(statistics["std"])
            standardized.append(
                (frame["volatility_level"] - mean) / (std + epsilon)
            )
        threshold = float(standardized[0].quantile(quantile))
        labels = [
            (values > threshold).astype(np.int8).to_numpy()
            for values in standardized
        ]
        metadata = {"pooled_standardized_threshold": threshold}
    else:
        raise ValueError(f"Unsupported spike threshold mode: {mode}")
    return labels[0], labels[1], labels[2], metadata


def stock_regime_labels(
    data: PreparedData,
    definition: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    quantiles = (0.50, 0.90) if definition == "q50_q90" else (0.33, 0.67)
    thresholds = (
        data.train.groupby("ticker", observed=True)["volatility_level"]
        .quantile(list(quantiles))
        .unstack()
    )
    outputs = []
    for frame in (data.train, data.validation, data.test):
        lower = frame["ticker"].map(thresholds[quantiles[0]])
        upper = frame["ticker"].map(thresholds[quantiles[1]])
        outputs.append(
            np.select(
                [
                    frame["volatility_level"] <= lower,
                    frame["volatility_level"] <= upper,
                ],
                [0, 1],
                default=2,
            ).astype(np.int8)
        )
    metadata = {
        "lower_quantile": quantiles[0],
        "upper_quantile": quantiles[1],
        "ticker_thresholds": thresholds.to_dict(),
    }
    return outputs[0], outputs[1], outputs[2], metadata


def _mlp_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    config: Mapping[str, Any],
    task: TaskSpec,
    tracker: ProgressTracker,
    seed: int,
    *,
    quick: bool,
) -> MLPRegressor:
    model_config = config["models"]
    max_epochs = int(
        model_config["quick_max_epochs"] if quick else model_config["max_epochs"]
    )
    model = MLPRegressor(
        hidden_layer_sizes=tuple(model_config["mlp_hidden_sizes"]),
        learning_rate_init=float(model_config["learning_rate"]),
        batch_size=min(int(model_config["batch_size"]), len(x_train)),
        max_iter=1,
        warm_start=True,
        shuffle=False,
        random_state=seed,
    )
    best: MLPRegressor | None = None
    best_loss = np.inf
    stale = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for epoch in range(1, max_epochs + 1):
            model.fit(x_train, y_train)
            train_loss = mean_squared_error(y_train, model.predict(x_train))
            validation_loss = mean_squared_error(
                y_validation, model.predict(x_validation)
            )
            if validation_loss < best_loss - float(model_config["min_delta"]):
                best_loss = float(validation_loss)
                best = copy.deepcopy(model)
                stale = 0
            else:
                stale += 1
            if (
                epoch == 1
                or epoch % int(config["progress"]["epoch_log_every"]) == 0
                or stale >= int(model_config["patience"])
            ):
                tracker.log_epoch(
                    task,
                    epoch=epoch,
                    max_epochs=max_epochs,
                    train_loss=float(train_loss),
                    validation_loss=float(validation_loss),
                    best_validation_loss=float(best_loss),
                    patience_remaining=max(int(model_config["patience"]) - stale, 0),
                )
            if stale >= int(model_config["patience"]):
                break
    if best is None:
        raise RuntimeError("MLP regression did not produce a finite validation model")
    return best


def _mlp_classification(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    classes: np.ndarray,
    config: Mapping[str, Any],
    task: TaskSpec,
    tracker: ProgressTracker,
    seed: int,
    *,
    quick: bool,
) -> MLPClassifier:
    model_config = config["models"]
    max_epochs = int(
        model_config["quick_max_epochs"] if quick else model_config["max_epochs"]
    )
    model = MLPClassifier(
        hidden_layer_sizes=tuple(model_config["mlp_hidden_sizes"]),
        learning_rate_init=float(model_config["learning_rate"]),
        batch_size=min(int(model_config["batch_size"]), len(x_train)),
        max_iter=1,
        warm_start=True,
        shuffle=False,
        random_state=seed,
    )
    best: MLPClassifier | None = None
    best_loss = np.inf
    stale = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for epoch in range(1, max_epochs + 1):
            model.partial_fit(x_train, y_train, classes=classes)
            train_loss = log_loss(
                y_train, model.predict_proba(x_train), labels=classes
            )
            validation_loss = log_loss(
                y_validation,
                model.predict_proba(x_validation),
                labels=classes,
            )
            if validation_loss < best_loss - float(model_config["min_delta"]):
                best_loss = float(validation_loss)
                best = copy.deepcopy(model)
                stale = 0
            else:
                stale += 1
            if (
                epoch == 1
                or epoch % int(config["progress"]["epoch_log_every"]) == 0
                or stale >= int(model_config["patience"])
            ):
                tracker.log_epoch(
                    task,
                    epoch=epoch,
                    max_epochs=max_epochs,
                    train_loss=float(train_loss),
                    validation_loss=float(validation_loss),
                    best_validation_loss=float(best_loss),
                    patience_remaining=max(int(model_config["patience"]) - stale, 0),
                )
            if stale >= int(model_config["patience"]):
                break
    if best is None:
        raise RuntimeError("MLP classification did not produce a validation model")
    return best


def fit_regressor(
    model_name: str,
    data: PreparedData,
    y_train: np.ndarray,
    y_validation: np.ndarray,
    config: Mapping[str, Any],
    task: TaskSpec,
    tracker: ProgressTracker,
    *,
    quick: bool,
) -> Any:
    seed = int(task.config["seed"])
    if model_name == "historical_mean":
        return {"mean": float(np.mean(y_train))}
    if model_name == "ridge":
        model = Ridge(alpha=float(task.config.get("alpha", 1.0)))
        return model.fit(data.x_train, y_train)
    if model_name == "elastic_net":
        model = ElasticNet(
            alpha=float(config["models"]["elastic_net_alpha"]),
            l1_ratio=float(config["models"]["elastic_net_l1_ratio"]),
            max_iter=10_000,
            random_state=seed,
        )
        return model.fit(data.x_train, y_train)
    if model_name == "mlp":
        return _mlp_regression(
            data.x_train,
            y_train,
            data.x_validation,
            y_validation,
            config,
            task,
            tracker,
            seed,
            quick=quick,
        )
    raise ValueError(f"Unsupported regression model: {model_name}")


def predict_regressor(model_name: str, model: Any, x: np.ndarray) -> np.ndarray:
    if model_name == "historical_mean":
        return np.full(len(x), float(model["mean"]))
    return np.asarray(model.predict(x), dtype=float)


def fit_classifier(
    model_name: str,
    data: PreparedData,
    y_train: np.ndarray,
    y_validation: np.ndarray,
    config: Mapping[str, Any],
    task: TaskSpec,
    tracker: ProgressTracker,
    *,
    multiclass: bool,
    quick: bool,
) -> Any:
    seed = int(task.config["seed"])
    if len(np.unique(y_train)) < (3 if multiclass else 2):
        raise ValueError(
            f"Training labels lack class support for {task.task_id}: "
            f"{np.unique(y_train).tolist()}"
        )
    if model_name in {"logistic", "weighted_logistic", "multinomial_logistic"}:
        class_weight = "balanced" if model_name == "weighted_logistic" else None
        model = LogisticRegression(
            C=float(config["models"]["logistic_c"]),
            class_weight=class_weight,
            max_iter=5_000,
            random_state=seed,
            solver="lbfgs",
        )
        return model.fit(data.x_train, y_train)
    if model_name == "mlp":
        classes = np.arange(3) if multiclass else np.arange(2)
        return _mlp_classification(
            data.x_train,
            y_train,
            data.x_validation,
            y_validation,
            classes,
            config,
            task,
            tracker,
            seed,
            quick=quick,
        )
    raise ValueError(f"Unsupported classification model: {model_name}")


def prediction_frame(
    source: pd.DataFrame,
    task: TaskSpec,
    *,
    evaluation_split: str,
    y_true: np.ndarray,
    prediction: np.ndarray | None = None,
    probability: np.ndarray | None = None,
    class_probability: np.ndarray | None = None,
    mean: np.ndarray | None = None,
    scale: np.ndarray | None = None,
    df: np.ndarray | None = None,
    distribution: str | None = None,
    representation_fit_scope: str,
    qualifies_for_robustness: bool,
) -> pd.DataFrame:
    columns = [
        column
        for column in (
            "ticker",
            "feature_date",
            "target_date",
            "split",
            "has_target_news",
        )
        if column in source.columns
    ]
    frame = source[columns].copy().reset_index(drop=True)
    frame["evaluation_split"] = evaluation_split
    frame["y_true"] = np.asarray(y_true)
    if prediction is not None:
        frame["prediction"] = np.asarray(prediction)
    if probability is not None:
        frame["probability"] = np.asarray(probability)
    if class_probability is not None:
        for index in range(class_probability.shape[1]):
            frame[f"probability_class_{index}"] = class_probability[:, index]
    if mean is not None:
        frame["distribution_mean"] = np.asarray(mean)
    if scale is not None:
        frame["distribution_scale"] = np.asarray(scale)
    if df is not None:
        frame["distribution_df"] = np.asarray(df)
    if distribution is not None:
        frame["distribution"] = distribution
    for key, value in task.config.items():
        frame[key] = (
            json.dumps(value, sort_keys=True)
            if isinstance(value, (dict, list))
            else value
        )
    frame["task_id"] = task.task_id
    frame["stage"] = task.stage
    frame["representation_fit_scope"] = representation_fit_scope
    frame["qualifies_for_robustness"] = bool(qualifies_for_robustness)
    return frame


def save_task_artifacts(
    config: Mapping[str, Any],
    task: TaskSpec,
    predictions: pd.DataFrame,
    model_payload: Mapping[str, Any],
) -> tuple[Path, Path]:
    prediction_path, model_path = task_checkpoint_paths(config, task)
    write_table(predictions, prediction_path, index=False)
    atomic_joblib_dump(dict(model_payload), model_path)
    return prediction_path, model_path
