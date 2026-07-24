"""Semiconductor-wide volatility level, spike, and regime experiments."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.modeling import (
    experiment_task,
    fit_classifier,
    fit_regressor,
    fold_task_supported,
    plan_representation_variants,
    predict_regressor,
    prediction_frame,
    prepare_sector_data,
    save_task_artifacts,
)
from src.progress_tracker import ProgressTracker, TaskSpec
from src.utils import project_path, read_table


def _scopes(representation: str, quick: bool) -> list[str]:
    if representation == "R0":
        return ["price"]
    if representation in {"R1", "R6", "R7"}:
        return ["macro_sector", "all"] if quick else [
            "macro",
            "sector",
            "macro_sector",
            "all",
        ]
    return ["all"]


def plan_tasks(
    config: Mapping[str, Any],
    profile: Mapping[str, Any],
    *,
    quick: bool,
) -> list[TaskSpec]:
    variants = plan_representation_variants(
        config, profile["sector_representations"], quick=quick
    )
    available_models = set(map(str, profile["sector_models"]))
    tasks: list[TaskSpec] = []
    target_specs: list[tuple[str, str, list[str]]] = [
        (
            "sector_level",
            "sector_mean_volatility",
            [
                model
                for model in ("ridge", "elastic_net", "mlp")
                if model in available_models
            ],
        )
    ]
    for definition in config["targets"]["sector_spike_definitions"]:
        target_specs.append(
            (
                "sector_spike",
                f"sector_spike_{definition}",
                [
                    model
                    for model in ("logistic", "weighted_logistic", "mlp")
                    if model in available_models
                ],
            )
        )
    for definition in profile["regime_definitions"]:
        target_specs.append(
            (
                "sector_regime",
                f"sector_regime_{definition}",
                [
                    model
                    for model in ("multinomial_logistic", "mlp")
                    if model in available_models
                ],
            )
        )
    for variant in variants:
        representation = variant["representation"]
        for scope in _scopes(representation, quick):
            for family, target, models in target_specs:
                for model in models:
                    for fold in profile["folds"]:
                        for seed in profile["seeds"]:
                            if not fold_task_supported(
                                config,
                                representation,
                                variant["representation_variant_family"],
                                fold,
                                int(seed),
                            ):
                                continue
                            payload = {
                                **variant,
                                "target_family": family,
                                "target": target,
                                "model": model,
                                "input_variant": (
                                    "price_only"
                                    if representation == "R0"
                                    else "price_plus_text"
                                ),
                                "news_scope": scope,
                                "fold": fold,
                                "seed": int(seed),
                                "quick": bool(quick),
                            }
                            tasks.append(
                                experiment_task("sector", "train_sector", payload)
                            )
    return tasks


def _sector_spike_labels(
    data: Any,
    config: Mapping[str, Any],
    target: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    minimum = int(
        config["targets"]["sector_spike_definitions"][
            target.removeprefix("sector_spike_")
        ]
    )
    market = read_table(
        project_path(config, "data", "processed", "original_market_targets.parquet")
    )
    market["feature_date"] = pd.to_datetime(market["feature_date"]).dt.normalize()
    train_dates = set(pd.to_datetime(data.train["feature_date"]).dt.normalize())
    threshold_rows = market.loc[market["feature_date"].isin(train_dates)]
    ticker_thresholds = (
        threshold_rows.groupby("ticker", observed=True)["volatility_level"]
        .quantile(0.90)
        .to_dict()
    )
    expected_tickers = set(map(str, config["universe"]["tickers"]))
    if set(ticker_thresholds) != expected_tickers:
        raise ValueError(
            "Sector fold lacks train-only q90 thresholds for all tickers"
        )
    outputs = []
    for frame in (data.train, data.validation, data.test):
        dates = set(pd.to_datetime(frame["feature_date"]).dt.normalize())
        stock_rows = market.loc[market["feature_date"].isin(dates)].copy()
        stock_rows["__spike"] = (
            stock_rows["volatility_level"]
            > stock_rows["ticker"].map(ticker_thresholds).astype(float)
        ).astype(np.int8)
        counts = stock_rows.groupby("feature_date", observed=True)["__spike"].sum()
        aligned_counts = (
            pd.to_datetime(frame["feature_date"]).dt.normalize().map(counts)
        )
        if aligned_counts.isna().any():
            raise ValueError("Sector spike labels are missing ticker-day responses")
        outputs.append(
            (aligned_counts >= minimum)
            .astype(np.int8)
            .to_numpy()
        )
    return outputs[0], outputs[1], outputs[2], {
        "minimum_spike_count": minimum,
        "ticker_q90_thresholds": ticker_thresholds,
        "threshold_fit_scope": "task_train_only",
    }


def _sector_regime_labels(
    data: Any,
    definition: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    quantiles = (0.50, 0.90) if definition == "q50_q90" else (0.33, 0.67)
    lower, upper = data.train["sector_mean_volatility"].quantile(
        list(quantiles)
    ).to_numpy()
    outputs = [
        np.select(
            [
                frame["sector_mean_volatility"] <= lower,
                frame["sector_mean_volatility"] <= upper,
            ],
            [0, 1],
            default=2,
        ).astype(np.int8)
        for frame in (data.train, data.validation, data.test)
    ]
    return outputs[0], outputs[1], outputs[2], {
        "lower_quantile": quantiles[0],
        "upper_quantile": quantiles[1],
        "lower_threshold": float(lower),
        "upper_threshold": float(upper),
    }


def run_task(
    task: TaskSpec,
    config: Mapping[str, Any],
    profile: Mapping[str, Any],
    tracker: ProgressTracker,
) -> dict[str, Any]:
    data = prepare_sector_data(config, task, profile)
    family = str(task.config["target_family"])
    target = str(task.config["target"])
    model_name = str(task.config["model"])
    if family == "sector_level":
        y_train = data.train["sector_mean_volatility"].to_numpy(dtype=float)
        y_validation = data.validation["sector_mean_volatility"].to_numpy(dtype=float)
        y_test = data.test["sector_mean_volatility"].to_numpy(dtype=float)
        model = fit_regressor(
            model_name,
            data,
            y_train,
            y_validation,
            config,
            task,
            tracker,
            quick=bool(task.config["quick"]),
        )
        validation_kwargs = {
            "prediction": predict_regressor(model_name, model, data.x_validation)
        }
        test_kwargs = {
            "prediction": predict_regressor(model_name, model, data.x_test)
        }
        metadata = {}
    elif family == "sector_spike":
        y_train, y_validation, y_test, metadata = _sector_spike_labels(
            data, config, target
        )
        minimum_positive = int(
            config["targets"].get(
                "quick_sector_spike_minimum_positive",
                config["targets"]["sector_spike_minimum_positive"],
            )
            if bool(task.config["quick"])
            else config["targets"]["sector_spike_minimum_positive"]
        )
        if min(int(y_train.sum()), int(y_validation.sum())) < minimum_positive:
            raise ValueError(
                f"{target} has insufficient positive samples: "
                f"train={int(y_train.sum())}, validation={int(y_validation.sum())}"
            )
        model = fit_classifier(
            model_name,
            data,
            y_train,
            y_validation,
            config,
            task,
            tracker,
            multiclass=False,
            quick=bool(task.config["quick"]),
        )
        validation_kwargs = {
            "probability": np.asarray(
                model.predict_proba(data.x_validation)[:, 1], dtype=float
            )
        }
        test_kwargs = {
            "probability": np.asarray(
                model.predict_proba(data.x_test)[:, 1], dtype=float
            )
        }
    elif family == "sector_regime":
        definition = target.removeprefix("sector_regime_")
        y_train, y_validation, y_test, metadata = _sector_regime_labels(
            data, definition
        )
        model = fit_classifier(
            model_name,
            data,
            y_train,
            y_validation,
            config,
            task,
            tracker,
            multiclass=True,
            quick=bool(task.config["quick"]),
        )
        validation_kwargs = {
            "class_probability": np.asarray(
                model.predict_proba(data.x_validation), dtype=float
            )
        }
        test_kwargs = {
            "class_probability": np.asarray(
                model.predict_proba(data.x_test), dtype=float
            )
        }
    else:
        raise ValueError(f"Unsupported sector target family: {family}")
    pieces = [
        prediction_frame(
            data.validation,
            task,
            evaluation_split="validation",
            y_true=y_validation,
            representation_fit_scope=data.representation_fit_scope,
            qualifies_for_robustness=data.qualifies_for_robustness,
            **validation_kwargs,
        )
    ]
    if len(data.test):
        pieces.append(
            prediction_frame(
                data.test,
                task,
                evaluation_split="test",
                y_true=y_test,
                representation_fit_scope=data.representation_fit_scope,
                qualifies_for_robustness=data.qualifies_for_robustness,
                **test_kwargs,
            )
        )
    predictions = pd.concat(pieces, ignore_index=True)
    paths = save_task_artifacts(
        config,
        task,
        predictions,
        {
            "model": model,
            "processor": data.processor,
            "features": data.feature_columns,
            "thresholds": metadata,
            "fit_scope": "task_train_only",
            "representation_fit_scope": data.representation_fit_scope,
        },
    )
    return {"predictions": paths[0], "model": paths[1]}
