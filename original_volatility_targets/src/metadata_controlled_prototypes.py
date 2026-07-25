"""Metadata-controlled, regime-conditioned target-news prototype experiment.

This module deliberately keeps the locked holdout closed.  Every nuisance
model, scaler, volatility-regime threshold and classifier is fitted inside one
chronological training fold.  Semantic features used to train the downstream
classifier are expanding-window residuals; validation features are transformed
by a nuisance model fitted on the complete fold-training interval.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.evaluate_original_targets import aggregate_metrics, load_predictions
from src.modeling import (
    PreparedData,
    experiment_task,
    fit_classifier,
    fold_task_supported,
    prediction_frame,
    representation_row,
    save_task_artifacts,
    stock_spike_labels,
)
from src.progress_tracker import ProgressTracker, TaskSpec
from src.utils import (
    atomic_write_csv,
    atomic_write_json,
    load_representation_frame,
    price_feature_columns,
    project_path,
    read_table,
    representation_feature_columns,
    task_split_frames,
    validate_columns,
)


EXPERIMENT_PROFILE = "metadata_controlled_prototypes"
BASELINE_REPRESENTATIONS = {"R0", "META_BASIC"}
OUTPUT_NAMES = (
    "metadata_controlled_fold_results.csv",
    "metadata_controlled_comparisons.csv",
    "metadata_controlled_random_reference.csv",
    "metadata_controlled_stability.csv",
    "metadata_controlled_decision.csv",
    "metadata_controlled_report.json",
)
FIGURE_NAME = "metadata_controlled_calibration.png"


def evaluation_outputs() -> tuple[str, ...]:
    return tuple(
        [
            *(f"outputs/tables/{name}" for name in OUTPUT_NAMES),
            f"outputs/figures/{FIGURE_NAME}",
        ]
    )


def _profile(config: Mapping[str, Any]) -> dict[str, Any]:
    section = config.get(EXPERIMENT_PROFILE)
    if not isinstance(section, Mapping):
        raise KeyError(f"Missing {EXPERIMENT_PROFILE} configuration section.")
    required = {
        "representations",
        "representation_source_map",
        "source_feature_prefixes",
        "representation_variant_family",
        "folds",
        "seeds",
        "random_prototype_seeds",
        "spike_models",
        "spike_quantile",
        "spike_threshold_mode",
        "metadata_feature_names",
        "nuisance_ridge_alpha",
        "nuisance_warmup_fraction",
        "nuisance_expanding_blocks",
        "primary_evaluation_cohort",
    }
    missing = sorted(required.difference(section))
    if missing:
        raise KeyError(
            f"{EXPERIMENT_PROFILE} configuration is missing: {missing}"
        )
    profile = dict(section)
    profile["folds"] = tuple(dict.fromkeys(int(v) for v in section["folds"]))
    profile["seeds"] = tuple(dict.fromkeys(int(v) for v in section["seeds"]))
    profile["random_prototype_seeds"] = tuple(
        dict.fromkeys(int(v) for v in section["random_prototype_seeds"])
    )
    if len(profile["folds"]) < 3:
        raise ValueError("At least three chronological folds are required.")
    if len(profile["seeds"]) < 5:
        raise ValueError("At least five prototype/model seeds are required.")
    if len(profile["random_prototype_seeds"]) < int(
        section.get("minimum_random_prototype_seeds", 30)
    ):
        raise ValueError("The empirical random-prototype null is too small.")
    if not 0.0 < float(profile["nuisance_warmup_fraction"]) < 0.5:
        raise ValueError("nuisance_warmup_fraction must be in (0, 0.5).")
    if int(profile["nuisance_expanding_blocks"]) < 2:
        raise ValueError("At least two expanding nuisance blocks are required.")
    if str(profile["primary_evaluation_cohort"]) != "target_news_days":
        raise ValueError("Primary evaluation must use true target-news days.")
    if float(profile["spike_quantile"]) != 0.90:
        raise ValueError("This locked exploratory design uses q90 spike only.")
    return profile


def _random_representation(random_seed: int) -> str:
    return f"ORTHO_R9_NULL_{int(random_seed)}"


def _source_representation(
    representation: str,
    profile: Mapping[str, Any],
) -> str:
    random_prefix = "ORTHO_R9_NULL_"
    if representation.startswith(random_prefix):
        return "R9_NULL_" + representation.removeprefix(random_prefix)
    source_map = {
        str(key): str(value)
        for key, value in profile["representation_source_map"].items()
    }
    if representation not in source_map:
        raise KeyError(f"No source representation for {representation}.")
    return source_map[representation]


def plan_tasks(
    config: Mapping[str, Any],
    profile: Mapping[str, Any],
    *,
    quick: bool = False,
) -> list[TaskSpec]:
    del quick
    locked = _profile(config)
    family = str(locked["representation_variant_family"])
    representations = list(map(str, locked["representations"]))
    representations.extend(
        _random_representation(seed)
        for seed in locked["random_prototype_seeds"]
    )
    anchor_seed = int(locked["seeds"][0])
    tasks: list[TaskSpec] = []
    for representation in representations:
        source = _source_representation(representation, locked)
        seeds: Sequence[int] = (
            (anchor_seed,)
            if representation in BASELINE_REPRESENTATIONS
            else locked["seeds"]
        )
        for fold in locked["folds"]:
            for seed in seeds:
                if not fold_task_supported(
                    config,
                    source,
                    family,
                    fold,
                    int(seed),
                ):
                    raise FileNotFoundError(
                        "Missing fold-safe source artifact for "
                        f"{representation} (source={source}), fold={fold}, "
                        f"prototype_seed={seed}, family={family}. Rebuild the "
                        "fold representations in the residual project first."
                    )
                for model in locked["spike_models"]:
                    payload = {
                        "experiment_profile": EXPERIMENT_PROFILE,
                        "target_family": "spike",
                        "target": "volatility_spike_q90_ticker",
                        "representation": representation,
                        "representation_source": source,
                        "representation_variant": family,
                        "representation_variant_family": family,
                        "input_variant": "price_plus_text",
                        "model": str(model),
                        "fold": int(fold),
                        "seed": int(seed),
                        "quantile": 0.90,
                        "threshold_mode": str(
                            locked["spike_threshold_mode"]
                        ),
                        "text_news_levels": ["target"],
                        "training_cohort": "all_days_after_nuisance_warmup",
                        "primary_evaluation_cohort": "target_news_days",
                        "quick": False,
                    }
                    if representation.startswith("ORTHO_R9_NULL_"):
                        payload["random_prototype_seed"] = int(
                            representation.rsplit("_", 1)[1]
                        )
                    tasks.append(
                        experiment_task(
                            "spike",
                            "train_metadata_controlled_spike",
                            payload,
                            required=True,
                        )
                    )
    return tasks


def _target_feature_columns(
    frame: pd.DataFrame,
    source: str,
    profile: Mapping[str, Any],
) -> list[str]:
    prefix_map = {
        str(key): tuple(map(str, value))
        for key, value in profile["source_feature_prefixes"].items()
    }
    prefix_key = (
        "R9_NULL"
        if source.startswith("R9_NULL_")
        else source
    )
    prefixes = prefix_map.get(prefix_key)
    if not prefixes:
        raise KeyError(f"No target feature prefixes configured for {source}.")
    columns = [
        column
        for column in representation_feature_columns(frame)
        if str(column).startswith(prefixes)
    ]
    if not columns:
        raise ValueError(
            f"Source {source} has no target features with prefixes={prefixes}."
        )
    return columns


def _load_fold_representation(
    config: Mapping[str, Any],
    source: str,
    family: str,
    fold: int,
    seed: int,
) -> tuple[pd.DataFrame, str]:
    row = representation_row(
        config,
        source,
        family,
        fold=fold,
        seed=seed,
        representation_variant_family=family,
    )
    fit_scope = str(row.get("fit_scope", row.get("fit_split", "")))
    if fit_scope != "fold_train_only":
        raise AssertionError(
            f"{source}, fold={fold}, seed={seed} is not fold-train fitted: "
            f"fit_scope={fit_scope!r}."
        )
    return load_representation_frame(config, row, seed=seed), fit_scope


def _calendar_features(frame: pd.DataFrame) -> list[str]:
    date = pd.to_datetime(frame["feature_date"], errors="raise")
    day = date.dt.dayofweek.to_numpy(dtype=float)
    month = date.dt.month.to_numpy(dtype=float)
    frame["nuisance__dow_sin"] = np.sin(2.0 * np.pi * day / 7.0)
    frame["nuisance__dow_cos"] = np.cos(2.0 * np.pi * day / 7.0)
    frame["nuisance__month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
    frame["nuisance__month_cos"] = np.cos(2.0 * np.pi * month / 12.0)
    # A fixed epoch/scale keeps this feature deterministic.  Using the maximum
    # date in the assembled panel would leak validation chronology into train.
    ordinal = (
        date - pd.Timestamp("2000-01-01")
    ).dt.days.to_numpy(dtype=float)
    frame["nuisance__time_trend"] = ordinal / 365.25
    return [
        "nuisance__dow_sin",
        "nuisance__dow_cos",
        "nuisance__month_sin",
        "nuisance__month_cos",
        "nuisance__time_trend",
    ]


def _numeric_ticker_processor(columns: Sequence[str]) -> ColumnTransformer:
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
            ("numeric", numeric, list(columns)),
            ("ticker", ticker, ["ticker"]),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def _nuisance_model(
    columns: Sequence[str],
    alpha: float,
) -> Pipeline:
    return Pipeline(
        [
            ("processor", _numeric_ticker_processor(columns)),
            ("ridge", Ridge(alpha=float(alpha))),
        ]
    )


def _expanding_orthogonalize(
    train_full: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    semantic_columns: Sequence[str],
    nuisance_columns: Sequence[str],
    profile: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    date_column = "target_date"
    dates = np.sort(train_full[date_column].dropna().unique())
    warmup_count = max(
        int(profile.get("nuisance_minimum_warmup_dates", 40)),
        int(math.ceil(len(dates) * float(profile["nuisance_warmup_fraction"]))),
    )
    if warmup_count >= len(dates) - int(profile["nuisance_expanding_blocks"]):
        raise ValueError(
            f"Only {len(dates)} train dates are available; nuisance warmup "
            f"would consume {warmup_count}."
        )
    warmup_dates = dates[:warmup_count]
    scoring_dates = dates[warmup_count:]
    blocks = [
        values
        for values in np.array_split(
            scoring_dates,
            int(profile["nuisance_expanding_blocks"]),
        )
        if len(values)
    ]
    residual_columns = [
        f"orthogonal__{index:04d}"
        for index in range(len(semantic_columns))
    ]
    train = train_full.loc[
        train_full[date_column].isin(scoring_dates)
    ].copy()
    train.loc[:, residual_columns] = np.nan
    block_rows: list[dict[str, Any]] = []
    alpha = float(profile["nuisance_ridge_alpha"])
    predictors = [*nuisance_columns, "ticker"]
    for block_index, block_dates in enumerate(blocks, start=1):
        block_start = pd.Timestamp(block_dates[0])
        history = train_full.loc[
            train_full[date_column] < block_start
        ]
        scoring = train.loc[train[date_column].isin(block_dates)]
        if history.empty or scoring.empty:
            raise ValueError("An expanding nuisance block is empty.")
        nuisance = _nuisance_model(nuisance_columns, alpha)
        nuisance.fit(
            history[predictors],
            history[list(semantic_columns)].to_numpy(dtype=float),
        )
        predicted = np.asarray(nuisance.predict(scoring[predictors]), dtype=float)
        observed = scoring[list(semantic_columns)].to_numpy(dtype=float)
        train.loc[scoring.index, residual_columns] = observed - predicted
        block_rows.append(
            {
                "block": block_index,
                "history_start": str(history[date_column].min()),
                "history_end": str(history[date_column].max()),
                "scoring_start": str(scoring[date_column].min()),
                "scoring_end": str(scoring[date_column].max()),
                "history_rows": int(len(history)),
                "scoring_rows": int(len(scoring)),
                "semantic_mse": float(np.mean(np.square(observed - predicted))),
            }
        )
    if train[residual_columns].isna().any().any():
        raise AssertionError("Expanding nuisance residuals contain missing values.")
    final_nuisance = _nuisance_model(nuisance_columns, alpha)
    final_nuisance.fit(
        train_full[predictors],
        train_full[list(semantic_columns)].to_numpy(dtype=float),
    )

    def transform(frame: pd.DataFrame) -> pd.DataFrame:
        output = frame.copy()
        if output.empty:
            for column in residual_columns:
                output[column] = pd.Series(dtype=float)
            return output
        predicted = np.asarray(
            final_nuisance.predict(output[predictors]),
            dtype=float,
        )
        observed = output[list(semantic_columns)].to_numpy(dtype=float)
        output.loc[:, residual_columns] = observed - predicted
        return output

    validation = transform(validation)
    test = transform(test)
    diagnostics = {
        "warmup_date_count": int(warmup_count),
        "warmup_start": str(pd.Timestamp(warmup_dates[0])),
        "warmup_end": str(pd.Timestamp(warmup_dates[-1])),
        "scoring_start": str(pd.Timestamp(scoring_dates[0])),
        "expanding_blocks": block_rows,
        "nuisance_predictors": list(nuisance_columns),
        "semantic_columns": list(semantic_columns),
        "residual_columns": residual_columns,
        "final_nuisance_model": final_nuisance,
    }
    return train, validation, test, diagnostics


def _common_warmup(
    train_full: pd.DataFrame,
    profile: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    dates = np.sort(train_full["target_date"].dropna().unique())
    warmup_count = max(
        int(profile.get("nuisance_minimum_warmup_dates", 40)),
        int(math.ceil(len(dates) * float(profile["nuisance_warmup_fraction"]))),
    )
    if warmup_count >= len(dates):
        raise ValueError("Nuisance warmup removes the complete training fold.")
    output = train_full.loc[
        train_full["target_date"].isin(dates[warmup_count:])
    ].copy()
    return output, {
        "warmup_date_count": int(warmup_count),
        "warmup_start": str(pd.Timestamp(dates[0])),
        "warmup_end": str(pd.Timestamp(dates[warmup_count - 1])),
        "scoring_start": str(pd.Timestamp(dates[warmup_count])),
    }


def _add_regime_interactions(
    train_reference: pd.DataFrame,
    frames: Sequence[pd.DataFrame],
    residual_columns: Sequence[str],
) -> tuple[list[pd.DataFrame], dict[str, Any], list[str]]:
    if "log_variance" not in train_reference:
        raise KeyError("Causal current-day log_variance is required for regimes.")
    thresholds = (
        train_reference.groupby("ticker", observed=True)["log_variance"]
        .quantile([0.33, 0.67])
        .unstack()
    )
    if thresholds.isna().any().any():
        raise ValueError("Ticker volatility-regime thresholds contain missing values.")
    interaction_columns: list[str] = []
    outputs: list[pd.DataFrame] = []
    for frame in frames:
        output = frame.copy()
        lower = output["ticker"].map(thresholds[0.33]).astype(float)
        upper = output["ticker"].map(thresholds[0.67]).astype(float)
        middle = (
            (output["log_variance"] > lower)
            & (output["log_variance"] <= upper)
        ).astype(float)
        high = (output["log_variance"] > upper).astype(float)
        output["state__volatility_regime_middle"] = middle
        output["state__volatility_regime_high"] = high
        for residual in residual_columns:
            for suffix, values in (("middle", middle), ("high", high)):
                column = f"interaction__{residual}__{suffix}"
                output[column] = output[residual].to_numpy(dtype=float) * values
                if column not in interaction_columns:
                    interaction_columns.append(column)
        outputs.append(output)
    threshold_payload = {
        ticker: {
            "q33": float(thresholds.loc[ticker, 0.33]),
            "q67": float(thresholds.loc[ticker, 0.67]),
        }
        for ticker in thresholds.index
    }
    state_columns = [
        "state__volatility_regime_middle",
        "state__volatility_regime_high",
    ]
    return outputs, threshold_payload, [*state_columns, *interaction_columns]


def _prepare_data(
    config: Mapping[str, Any],
    task: TaskSpec,
    profile: Mapping[str, Any],
) -> tuple[PreparedData, dict[str, Any]]:
    representation = str(task.config["representation"])
    source = str(task.config["representation_source"])
    family = str(task.config["representation_variant_family"])
    fold = int(task.config["fold"])
    seed = int(task.config["seed"])
    market = read_table(
        project_path(
            config,
            "data",
            "processed",
            "original_market_targets.parquet",
        )
    )
    validate_columns(
        market,
        (
            "ticker",
            "feature_date",
            "target_date",
            "split",
            "volatility_level",
        ),
        "original market targets",
    )
    market = market.copy()
    market["feature_date"] = pd.to_datetime(
        market["feature_date"], format="mixed", errors="raise"
    ).dt.normalize()
    market["target_date"] = pd.to_datetime(
        market["target_date"], format="mixed", errors="raise"
    ).dt.normalize()

    metadata_frame, metadata_scope = _load_fold_representation(
        config,
        "R7",
        family,
        fold,
        seed,
    )
    metadata_names = list(map(str, profile["metadata_feature_names"]))
    missing_metadata = sorted(
        set(metadata_names).difference(metadata_frame.columns)
    )
    if missing_metadata:
        raise ValueError(f"R7 metadata features are missing: {missing_metadata}")
    metadata = metadata_frame[
        ["ticker", "feature_date", *metadata_names]
    ].copy()
    metadata["__metadata_artifact_observed"] = True
    joined = market.merge(
        metadata,
        on=["ticker", "feature_date"],
        how="left",
        validate="one_to_one",
    )

    gate_frame, gate_scope = _load_fold_representation(
        config,
        "R6",
        family,
        fold,
        seed,
    )
    gate_columns = [
        column
        for column in representation_feature_columns(gate_frame)
        if str(column).startswith("softproto__target__")
    ]
    if not gate_columns:
        raise ValueError("R6 has no target prototype columns for the news gate.")
    gate = gate_frame[["ticker", "feature_date"]].copy()
    gate["__gate_artifact_observed"] = True
    gate["has_target_news"] = (
        gate_frame[gate_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .abs()
        .sum(axis=1)
        .gt(float(profile.get("news_presence_epsilon", 1.0e-10)))
        .to_numpy()
    )
    joined = joined.merge(
        gate,
        on=["ticker", "feature_date"],
        how="left",
        validate="one_to_one",
    )

    semantic_columns: list[str] = []
    source_scope = "not_applicable"
    source_required = representation != "R0" and source != "R7"
    if representation != "R0" and source != "R7":
        source_frame, source_scope = _load_fold_representation(
            config,
            source,
            family,
            fold,
            seed,
        )
        source_columns = _target_feature_columns(source_frame, source, profile)
        rename = {
            column: f"semantic_source__{index:04d}"
            for index, column in enumerate(source_columns)
        }
        semantic_columns = list(rename.values())
        semantic = source_frame[
            ["ticker", "feature_date", *source_columns]
        ].rename(columns=rename)
        semantic["__semantic_artifact_observed"] = True
        joined = joined.merge(
            semantic,
            on=["ticker", "feature_date"],
            how="left",
            validate="one_to_one",
        )
    numeric_fill = [*metadata_names, *semantic_columns]
    joined[numeric_fill] = (
        joined[numeric_fill]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    nuisance_calendar = _calendar_features(joined)
    nuisance_columns = [*metadata_names, *nuisance_calendar]
    price_columns = price_feature_columns(joined, config)
    train_full, validation, test = task_split_frames(
        joined,
        config,
        profile,
        fold,
    )
    # Fold-safe representation artifacts intentionally omit market rows outside
    # their train/validation interval.  Coverage must therefore be checked only
    # after task_split_frames has selected the current fold.
    coverage_markers = [
        "__metadata_artifact_observed",
        "__gate_artifact_observed",
    ]
    if source_required:
        coverage_markers.append("__semantic_artifact_observed")
    for split_name, split_frame in (
        ("train", train_full),
        ("validation", validation),
        ("test", test),
    ):
        if split_frame.empty:
            continue
        missing_coverage: dict[str, int] = {}
        for marker in coverage_markers:
            if marker not in split_frame:
                missing_coverage[marker] = int(len(split_frame))
                continue
            missing_count = int(split_frame[marker].isna().sum())
            if missing_count:
                missing_coverage[marker] = missing_count
        if missing_coverage:
            raise AssertionError(
                f"Fold {fold} {split_name} rows lack fold-safe artifact "
                f"coverage: {missing_coverage}"
            )
        if split_frame["has_target_news"].isna().any():
            raise AssertionError(
                f"Fold {fold} {split_name} has an invalid target-news gate."
            )
        split_frame["has_target_news"] = split_frame[
            "has_target_news"
        ].astype(bool)
    preparation: dict[str, Any] = {
        "metadata_fit_scope": metadata_scope,
        "gate_fit_scope": gate_scope,
        "source_fit_scope": source_scope,
        "metadata_columns": metadata_names,
        "price_columns": price_columns,
    }

    if representation.startswith("ORTHO_"):
        train, validation, test, nuisance = _expanding_orthogonalize(
            train_full,
            validation,
            test,
            semantic_columns,
            nuisance_columns,
            profile,
        )
        residual_columns = list(nuisance["residual_columns"])
        preparation["nuisance"] = nuisance
        text_columns = [*metadata_names, *residual_columns]
        if representation == "ORTHO_R6_REGIME":
            transformed, regime_thresholds, regime_columns = (
                _add_regime_interactions(
                    train_full,
                    (train, validation, test),
                    residual_columns,
                )
            )
            train, validation, test = transformed
            text_columns.extend(regime_columns)
            preparation["regime_thresholds"] = regime_thresholds
    else:
        train, warmup = _common_warmup(train_full, profile)
        preparation["nuisance_warmup"] = warmup
        if representation == "R0":
            text_columns = []
        elif representation == "META_BASIC":
            text_columns = metadata_names
        elif representation == "META_R6":
            text_columns = [*metadata_names, *semantic_columns]
        else:
            raise ValueError(f"Unsupported derived representation: {representation}")

    feature_columns = list(dict.fromkeys([*price_columns, *text_columns]))
    if train.empty or validation.empty:
        raise ValueError("Prepared metadata-controlled split is empty.")
    if not train["target_date"].max() < validation["target_date"].min():
        raise AssertionError("Metadata-controlled train/validation overlap.")
    processor = _numeric_ticker_processor(feature_columns)
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
    data = PreparedData(
        train=train,
        validation=validation,
        test=test,
        feature_columns=feature_columns,
        x_train=x_train,
        x_validation=x_validation,
        x_test=x_test,
        processor=processor,
        representation_fit_scope=(
            "fold_train_only_expanding_metadata_orthogonalization"
            if representation.startswith("ORTHO_")
            else "fold_train_only_common_nuisance_warmup"
        ),
        qualifies_for_robustness=True,
    )
    return data, preparation


def run_task(
    task: TaskSpec,
    config: Mapping[str, Any],
    profile: Mapping[str, Any],
    tracker: ProgressTracker,
) -> dict[str, Any]:
    locked = _profile(config)
    data, preparation = _prepare_data(config, task, locked)
    y_train, y_validation, y_test, thresholds = stock_spike_labels(
        data,
        float(locked["spike_quantile"]),
        str(locked["spike_threshold_mode"]),
        float(config["targets"]["epsilon"]),
    )
    model_name = str(task.config["model"])
    model = fit_classifier(
        model_name,
        data,
        y_train,
        y_validation,
        config,
        task,
        tracker,
        multiclass=False,
        quick=False,
    )
    probability = np.asarray(
        model.predict_proba(data.x_validation)[:, 1],
        dtype=float,
    )
    predictions = prediction_frame(
        data.validation,
        task,
        evaluation_split="validation",
        y_true=y_validation,
        probability=probability,
        representation_fit_scope=data.representation_fit_scope,
        qualifies_for_robustness=True,
    )
    if len(data.test) or len(y_test):
        raise AssertionError(
            "The metadata-controlled exploratory experiment must not read test."
        )
    paths = save_task_artifacts(
        config,
        task,
        predictions,
        {
            "model": model,
            "processor": data.processor,
            "features": data.feature_columns,
            "thresholds": thresholds,
            "preparation": preparation,
            "fit_scope": "fold_train_only",
            "locked_test_used": False,
        },
    )
    return {"predictions": paths[0], "model": paths[1]}


def _truthy(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )


def _metric_lookup(metrics: pd.DataFrame) -> dict[tuple[int, int, str], pd.Series]:
    output: dict[tuple[int, int, str], pd.Series] = {}
    for row in metrics.itertuples(index=False):
        key = (int(row.fold), int(row.seed), str(row.representation))
        if key in output:
            raise ValueError(f"Duplicate metric cell: {key}")
        output[key] = pd.Series(row._asdict())
    return output


def _gain(candidate: float, reference: float, *, lower_better: bool) -> float:
    denominator = max(abs(float(reference)), 1.0e-12)
    delta = (
        float(reference) - float(candidate)
        if lower_better
        else float(candidate) - float(reference)
    )
    return 100.0 * delta / denominator


def _comparison_cells(
    metrics: pd.DataFrame,
    profile: Mapping[str, Any],
    pairs: Sequence[tuple[str, str, str]],
) -> pd.DataFrame:
    lookup = _metric_lookup(metrics)
    anchor = int(profile["seeds"][0])
    rows: list[dict[str, Any]] = []
    for candidate, reference, label in pairs:
        for fold in profile["folds"]:
            for seed in profile["seeds"]:
                candidate_seed = (
                    anchor if candidate in BASELINE_REPRESENTATIONS else seed
                )
                reference_seed = (
                    anchor if reference in BASELINE_REPRESENTATIONS else seed
                )
                candidate_row = lookup[(fold, candidate_seed, candidate)]
                reference_row = lookup[(fold, reference_seed, reference)]
                if int(candidate_row["n"]) != int(reference_row["n"]):
                    raise AssertionError(
                        f"Unpaired samples for {label}, fold={fold}, seed={seed}."
                    )
                rows.append(
                    {
                        "comparison": label,
                        "candidate": candidate,
                        "reference": reference,
                        "fold": int(fold),
                        "seed": int(seed),
                        "n": int(candidate_row["n"]),
                        "candidate_pr_auc": float(candidate_row["pr_auc"]),
                        "reference_pr_auc": float(reference_row["pr_auc"]),
                        "pr_auc_gain_percent": _gain(
                            candidate_row["pr_auc"],
                            reference_row["pr_auc"],
                            lower_better=False,
                        ),
                        "candidate_brier": float(candidate_row["brier"]),
                        "reference_brier": float(reference_row["brier"]),
                        "brier_gain_percent": _gain(
                            candidate_row["brier"],
                            reference_row["brier"],
                            lower_better=True,
                        ),
                        "candidate_ece": float(candidate_row["ece"]),
                        "reference_ece": float(reference_row["ece"]),
                        "ece_improvement": float(reference_row["ece"])
                        - float(candidate_row["ece"]),
                    }
                )
    return pd.DataFrame(rows)


def _comparison_summary(
    cells: pd.DataFrame,
    profile: Mapping[str, Any],
) -> pd.DataFrame:
    minimum_gain = float(profile.get("minimum_pr_auc_gain_percent", 1.0))
    brier_tolerance = float(
        profile.get("maximum_brier_degradation_percent", 0.5)
    )
    ece_tolerance = float(
        profile.get("maximum_ece_degradation", 0.01)
    )
    rows: list[dict[str, Any]] = []
    for comparison, group in cells.groupby(
        "comparison",
        sort=True,
        observed=True,
    ):
        fold_values = (
            group.groupby("fold", observed=True)
            .agg(
                pr_auc_gain_percent=("pr_auc_gain_percent", "mean"),
                brier_gain_percent=("brier_gain_percent", "mean"),
                ece_improvement=("ece_improvement", "mean"),
            )
            .reset_index()
        )
        values = fold_values["pr_auc_gain_percent"].to_numpy(dtype=float)
        mean_gain = float(np.mean(values))
        if len(values) > 1:
            half_width = float(
                stats.t.ppf(0.975, len(values) - 1)
                * stats.sem(values)
            )
        else:
            half_width = np.nan
        first = group.iloc[0]
        threshold = (
            float(profile.get("minimum_regime_increment_percent", 0.25))
            if str(comparison).endswith("_vs_ORTHO_R6")
            else minimum_gain
        )
        all_folds_positive = bool((values > 0.0).all())
        mean_brier = float(fold_values["brier_gain_percent"].mean())
        mean_ece = float(fold_values["ece_improvement"].mean())
        passed = bool(
            mean_gain >= threshold
            and all_folds_positive
            and mean_brier >= -brier_tolerance
            and mean_ece >= -ece_tolerance
        )
        rows.append(
            {
                "comparison": comparison,
                "candidate": first["candidate"],
                "reference": first["reference"],
                "fold_count": int(len(fold_values)),
                "cell_count": int(len(group)),
                "mean_pr_auc_gain_percent": mean_gain,
                "std_pr_auc_gain_percent_across_folds": float(
                    np.std(values, ddof=1)
                ),
                "ci95_low_pr_auc_gain_percent": mean_gain - half_width,
                "ci95_high_pr_auc_gain_percent": mean_gain + half_width,
                "mean_brier_gain_percent": mean_brier,
                "mean_ece_improvement": mean_ece,
                "positive_fold_count": int((values > 0.0).sum()),
                "all_folds_positive": all_folds_positive,
                "minimum_required_gain_percent": threshold,
                "comparison_passed": passed,
                "fold_gains_json": json.dumps(
                    {
                        str(int(row.fold)): float(row.pr_auc_gain_percent)
                        for row in fold_values.itertuples(index=False)
                    },
                    sort_keys=True,
                ),
            }
        )
    return pd.DataFrame(rows)


def _random_reference(
    metrics: pd.DataFrame,
    profile: Mapping[str, Any],
    candidate: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    pairs = [
        (
            candidate,
            _random_representation(seed),
            f"{candidate}_vs_RANDOM_{seed}",
        )
        for seed in profile["random_prototype_seeds"]
    ]
    cells = _comparison_cells(metrics, profile, pairs)
    rows: list[dict[str, Any]] = []
    for random_seed in profile["random_prototype_seeds"]:
        label = f"{candidate}_vs_RANDOM_{random_seed}"
        selected = cells.loc[cells["comparison"].eq(label)]
        folds = (
            selected.groupby("fold", observed=True)["pr_auc_gain_percent"]
            .mean()
            .sort_index()
        )
        rows.append(
            {
                "random_prototype_seed": int(random_seed),
                "mean_pr_auc_gain_percent": float(folds.mean()),
                "minimum_fold_gain_percent": float(folds.min()),
                "positive_fold_count": int((folds > 0.0).sum()),
                "all_folds_positive": bool((folds > 0.0).all()),
                "fold_gains_json": json.dumps(
                    {str(int(k)): float(v) for k, v in folds.items()},
                    sort_keys=True,
                ),
            }
        )
    reference = pd.DataFrame(rows)
    mean_gains = reference["mean_pr_auc_gain_percent"].to_numpy(dtype=float)
    percentile = float(np.mean(mean_gains > 0.0))
    empirical_p = float((1 + np.sum(mean_gains <= 0.0)) / (1 + len(mean_gains)))
    fold_win_rate = float(
        reference["positive_fold_count"].sum()
        / (len(reference) * len(profile["folds"]))
    )
    summary = {
        "candidate": candidate,
        "random_seed_count": int(len(reference)),
        "candidate_win_rate_against_random": percentile,
        "empirical_one_sided_p_value": empirical_p,
        "fold_level_win_rate_against_random": fold_win_rate,
        "candidate_beats_random_in_every_fold_rate": float(
            reference["all_folds_positive"].mean()
        ),
        "random_gate_passed": bool(
            percentile
            >= float(profile.get("minimum_semantic_percentile", 0.95))
            and empirical_p
            <= float(profile.get("maximum_empirical_p_value", 0.05))
            and fold_win_rate
            >= float(profile.get("minimum_random_fold_win_rate", 0.80))
        ),
    }
    return reference, summary


def _calibration_figure(
    predictions: pd.DataFrame,
    profile: Mapping[str, Any],
    path: Path,
) -> None:
    anchor = int(profile["seeds"][0])
    representatives = (
        "R0",
        "META_BASIC",
        "ORTHO_R6",
        "ORTHO_R6_REGIME",
    )
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    edges = np.linspace(0.0, 1.0, 11)
    for representation in representatives:
        selected = predictions.loc[
            predictions["representation"].astype(str).eq(representation)
        ].copy()
        if representation in BASELINE_REPRESENTATIONS:
            selected = selected.loc[
                pd.to_numeric(selected["seed"], errors="coerce").eq(anchor)
            ]
        if selected.empty:
            continue
        selected["bin"] = pd.cut(
            selected["probability"],
            bins=edges,
            include_lowest=True,
            duplicates="drop",
        )
        curve = (
            selected.groupby("bin", observed=True)
            .agg(predicted=("probability", "mean"), observed=("y_true", "mean"))
            .dropna()
        )
        ax.plot(
            curve["predicted"],
            curve["observed"],
            marker="o",
            label=representation,
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    ax.set(
        xlabel="Mean predicted q90-spike probability",
        ylabel="Observed q90-spike rate",
        title="Metadata-controlled prototype calibration (validation only)",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_evaluation(config: Mapping[str, Any]) -> dict[str, Path]:
    profile = _profile(config)
    predictions = load_predictions(config, mode=EXPERIMENT_PROFILE)
    if set(predictions["evaluation_split"].astype(str)) != {"validation"}:
        raise AssertionError("Locked test predictions entered the new experiment.")
    validate_columns(
        predictions,
        (
            "fold",
            "seed",
            "representation",
            "has_target_news",
            "probability",
            "y_true",
        ),
        "metadata-controlled predictions",
    )
    predictions = predictions.loc[_truthy(predictions["has_target_news"])].copy()
    minimum = int(profile.get("minimum_primary_samples_per_fold", 20))
    counts = predictions.groupby(
        ["fold", "seed", "representation"],
        observed=True,
    ).size()
    if counts.empty or counts.min() < minimum:
        raise ValueError(
            f"A target-news validation cell has fewer than {minimum} samples."
        )
    metrics = aggregate_metrics(predictions)
    expected = {
        *map(str, profile["representations"]),
        *(
            _random_representation(seed)
            for seed in profile["random_prototype_seeds"]
        ),
    }
    observed = set(metrics["representation"].astype(str))
    missing = sorted(expected.difference(observed))
    if missing:
        raise ValueError(f"Experiment predictions are incomplete: {missing}")

    primary = "ORTHO_R6_REGIME"
    pairs = [
        ("META_BASIC", "R0", "META_BASIC_vs_R0"),
        ("META_R6", "META_BASIC", "META_R6_vs_META_BASIC"),
        ("ORTHO_R6", "META_BASIC", "ORTHO_R6_vs_META_BASIC"),
        ("ORTHO_R6", "ORTHO_R3", "ORTHO_R6_vs_ORTHO_R3"),
        ("ORTHO_R6", "ORTHO_R4", "ORTHO_R6_vs_ORTHO_R4"),
        (primary, "META_BASIC", f"{primary}_vs_META_BASIC"),
        (primary, "ORTHO_R6", f"{primary}_vs_ORTHO_R6"),
        (primary, "ORTHO_R3", f"{primary}_vs_ORTHO_R3"),
        (primary, "ORTHO_R4", f"{primary}_vs_ORTHO_R4"),
        (primary, "ORTHO_R10", f"{primary}_vs_ORTHO_R10"),
        (primary, "ORTHO_R11", f"{primary}_vs_ORTHO_R11"),
        (primary, "ORTHO_P_LAGGED", f"{primary}_vs_ORTHO_P_LAGGED"),
        (primary, "ORTHO_P_PERMUTED", f"{primary}_vs_ORTHO_P_PERMUTED"),
    ]
    comparisons = _comparison_cells(metrics, profile, pairs)
    stability = _comparison_summary(comparisons, profile)
    random_reference, random_summary = _random_reference(
        metrics,
        profile,
        primary,
    )
    passed = {
        str(row.comparison): bool(row.comparison_passed)
        for row in stability.itertuples(index=False)
    }
    metadata_passed = passed.get("META_BASIC_vs_R0", False)
    raw_passed = passed.get("META_R6_vs_META_BASIC", False)
    orthogonal_passed = all(
        passed.get(name, False)
        for name in (
            "ORTHO_R6_vs_META_BASIC",
            "ORTHO_R6_vs_ORTHO_R3",
            "ORTHO_R6_vs_ORTHO_R4",
        )
    )
    regime_passed = all(
        passed.get(f"{primary}_vs_{reference}", False)
        for reference in (
            "META_BASIC",
            "ORTHO_R6",
            "ORTHO_R3",
            "ORTHO_R4",
            "ORTHO_R10",
            "ORTHO_R11",
            "ORTHO_P_LAGGED",
            "ORTHO_P_PERMUTED",
        )
    )
    random_passed = bool(random_summary["random_gate_passed"])
    if regime_passed and random_passed:
        decision = "REGIME-CONDITIONED-PROTOTYPE-SUPPORTED"
        next_step = (
            "Pre-register the locked representation and confirm it on a fresh "
            "future temporal window before reading any holdout."
        )
    elif orthogonal_passed and random_passed:
        decision = "ORTHOGONAL-PROTOTYPE-SUPPORTED"
        next_step = (
            "Keep orthogonal R6 without regime interactions and confirm it on "
            "a fresh future temporal window."
        )
    elif raw_passed:
        decision = "RAW-SEMANTIC-INCREMENT-ONLY"
        next_step = (
            "Treat the semantic increment as exploratory because it did not "
            "survive the orthogonal/placebo gates."
        )
    elif metadata_passed:
        decision = "METADATA-ONLY"
        next_step = (
            "Do not promote prototypes. Report that target-news arrival "
            "metadata explains the validation signal."
        )
    else:
        decision = "NO-STABLE-TARGET-NEWS-SIGNAL"
        next_step = "Stop this branch or collect a genuinely new time period."

    decision_frame = pd.DataFrame(
        [
            {
                "target": "volatility_spike_q90_ticker",
                "primary_metric": "pr_auc",
                "primary_evaluation_cohort": "target_news_days",
                "fold_count": len(profile["folds"]),
                "prototype_model_seed_count": len(profile["seeds"]),
                "random_prototype_seed_count": len(
                    profile["random_prototype_seeds"]
                ),
                "metadata_signal_passed": metadata_passed,
                "raw_semantic_increment_passed": raw_passed,
                "orthogonal_semantic_passed": orthogonal_passed,
                "regime_conditioning_passed": regime_passed,
                "random_prototype_gate_passed": random_passed,
                "random_empirical_p_value": random_summary[
                    "empirical_one_sided_p_value"
                ],
                "random_win_rate": random_summary[
                    "candidate_win_rate_against_random"
                ],
                "decision": decision,
                "locked_test_used": False,
                "next_step": next_step,
            }
        ]
    )
    table_dir = project_path(config, "outputs", "tables")
    figure_path = project_path(config, "outputs", "figures", FIGURE_NAME)
    paths = {
        "fold_results": table_dir / OUTPUT_NAMES[0],
        "comparisons": table_dir / OUTPUT_NAMES[1],
        "random_reference": table_dir / OUTPUT_NAMES[2],
        "stability": table_dir / OUTPUT_NAMES[3],
        "decision": table_dir / OUTPUT_NAMES[4],
        "report": table_dir / OUTPUT_NAMES[5],
        "calibration": figure_path,
    }
    atomic_write_csv(metrics, paths["fold_results"], index=False)
    atomic_write_csv(comparisons, paths["comparisons"], index=False)
    atomic_write_csv(random_reference, paths["random_reference"], index=False)
    atomic_write_csv(stability, paths["stability"], index=False)
    atomic_write_csv(decision_frame, paths["decision"], index=False)
    atomic_write_json(
        {
            "experiment_profile": EXPERIMENT_PROFILE,
            "evidence_scope": "exploratory_chronological_validation_only",
            "target": "volatility_spike_q90_ticker",
            "representations": list(profile["representations"]),
            "folds": list(profile["folds"]),
            "prototype_model_seeds": list(profile["seeds"]),
            "random_prototype_seeds": list(
                profile["random_prototype_seeds"]
            ),
            "nuisance_design": {
                "cross_fit": "expanding_window",
                "warmup_fraction": profile["nuisance_warmup_fraction"],
                "expanding_blocks": profile["nuisance_expanding_blocks"],
                "predictors": [
                    "META_BASIC",
                    "ticker",
                    "calendar_features",
                ],
            },
            "random_null_summary": random_summary,
            "decision": decision,
            "locked_test_used": False,
            "next_step": next_step,
        },
        paths["report"],
    )
    _calibration_figure(predictions, profile, figure_path)
    return paths


def print_report(
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> None:
    path = project_path(
        config,
        "outputs",
        "tables",
        "metadata_controlled_decision.csv",
    )
    if not path.is_file():
        return
    row = pd.read_csv(path).iloc[0]
    print("\nMETADATA-CONTROLLED PROTOTYPE EXPERIMENT")
    print(f"Completed tasks: {summary['completed_tasks']}")
    print(f"Failed tasks: {summary['failed_tasks']}")
    print(f"Skipped tasks: {summary['skipped_tasks']}")
    print(
        f"Grid: {int(row['fold_count'])} folds x "
        f"{int(row['prototype_model_seed_count'])} prototype/model seeds"
    )
    print("Primary target: q90 volatility spike")
    print("Primary cohort: true target-news validation days")
    print(
        "Orthogonal semantic signal passed: "
        f"{bool(row['orthogonal_semantic_passed'])}"
    )
    print(
        "Regime-conditioned prototype passed: "
        f"{bool(row['regime_conditioning_passed'])}"
    )
    print(
        "Random-prototype gate passed: "
        f"{bool(row['random_prototype_gate_passed'])}"
    )
    print(f"Decision: {row['decision']}")
    print("Locked test used: False")
    print(f"Next step: {row['next_step']}")
