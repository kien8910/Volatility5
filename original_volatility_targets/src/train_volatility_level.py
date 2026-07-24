"""Volatility-level regression tasks on the original log-volatility target."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd

from src.modeling import (
    experiment_task,
    fit_regressor,
    fold_task_supported,
    plan_representation_variants,
    predict_regressor,
    prediction_frame,
    prepare_stock_data,
    save_task_artifacts,
)
from src.progress_tracker import ProgressTracker, TaskSpec


def plan_tasks(
    config: Mapping[str, Any],
    profile: Mapping[str, Any],
    *,
    quick: bool,
) -> list[TaskSpec]:
    variants = plan_representation_variants(
        config, profile["representations"], quick=quick
    )
    tasks: list[TaskSpec] = []
    for variant in variants:
        representation = variant["representation"]
        input_variants = (
            ["price_only"]
            if representation == "R0"
            else list(profile["input_variants"])
        )
        for input_variant in input_variants:
            for model in profile["level_models"]:
                if model == "historical_mean" and representation != "R0":
                    continue
                alphas: Sequence[float | None] = (
                    config["models"]["ridge_alphas"]
                    if model == "ridge" and not quick
                    else [1.0]
                    if model == "ridge"
                    else [None]
                )
                for alpha in alphas:
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
                                "target_family": "level",
                                "target": "volatility_level",
                                "model": str(model),
                                "alpha": alpha,
                                "input_variant": input_variant,
                                "fold": fold,
                                "seed": int(seed),
                                "quick": bool(quick),
                            }
                            tasks.append(
                                experiment_task("level", "train_level", payload)
                            )
    return tasks


def run_task(
    task: TaskSpec,
    config: Mapping[str, Any],
    profile: Mapping[str, Any],
    tracker: ProgressTracker,
) -> dict[str, Any]:
    data = prepare_stock_data(config, task, profile)
    y_train = data.train["volatility_level"].to_numpy(dtype=float)
    y_validation = data.validation["volatility_level"].to_numpy(dtype=float)
    y_test = data.test["volatility_level"].to_numpy(dtype=float)
    model_name = str(task.config["model"])
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
    validation_prediction = predict_regressor(
        model_name, model, data.x_validation
    )
    pieces = [
        prediction_frame(
            data.validation,
            task,
            evaluation_split="validation",
            y_true=y_validation,
            prediction=validation_prediction,
            representation_fit_scope=data.representation_fit_scope,
            qualifies_for_robustness=data.qualifies_for_robustness,
        )
    ]
    if len(data.test):
        pieces.append(
            prediction_frame(
                data.test,
                task,
                evaluation_split="test",
                y_true=y_test,
                prediction=predict_regressor(model_name, model, data.x_test),
                representation_fit_scope=data.representation_fit_scope,
                qualifies_for_robustness=data.qualifies_for_robustness,
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
            "fit_scope": "task_train_only",
            "representation_fit_scope": data.representation_fit_scope,
        },
    )
    return {"predictions": paths[0], "model": paths[1]}
