"""Three-class original-volatility regime classification."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.modeling import (
    experiment_task,
    fit_classifier,
    fold_task_supported,
    plan_representation_variants,
    prediction_frame,
    prepare_stock_data,
    save_task_artifacts,
    stock_regime_labels,
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
        for definition in profile["regime_definitions"]:
            for input_variant in input_variants:
                for model in profile["regime_models"]:
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
                                "target_family": "regime",
                                "target": f"volatility_regime_{definition}",
                                "regime_definition": str(definition),
                                "model": str(model),
                                "input_variant": input_variant,
                                "fold": fold,
                                "seed": int(seed),
                                "quick": bool(quick),
                            }
                            tasks.append(
                                experiment_task(
                                    "regime", "train_regime", payload
                                )
                            )
    return tasks


def run_task(
    task: TaskSpec,
    config: Mapping[str, Any],
    profile: Mapping[str, Any],
    tracker: ProgressTracker,
) -> dict[str, Any]:
    data = prepare_stock_data(config, task, profile)
    y_train, y_validation, y_test, thresholds = stock_regime_labels(
        data, str(task.config["regime_definition"])
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
        multiclass=True,
        quick=bool(task.config["quick"]),
    )
    validation_probability = np.asarray(
        model.predict_proba(data.x_validation), dtype=float
    )
    pieces = [
        prediction_frame(
            data.validation,
            task,
            evaluation_split="validation",
            y_true=y_validation,
            class_probability=validation_probability,
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
                class_probability=np.asarray(
                    model.predict_proba(data.x_test), dtype=float
                ),
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
            "thresholds": thresholds,
            "fit_scope": "task_train_only",
            "representation_fit_scope": data.representation_fit_scope,
        },
    )
    return {"predictions": paths[0], "model": paths[1]}
