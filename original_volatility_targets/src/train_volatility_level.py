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
    representations = list(map(str, profile["representations"]))
    random_seeds = tuple(
        dict.fromkeys(
            int(value)
            for value in profile.get("random_prototype_seeds", [])
        )
    )
    if random_seeds:
        random_prefix = str(
            profile.get("random_representation_prefix", "R9_NULL")
        )
        representations.extend(
            f"{random_prefix}_{seed}" for seed in random_seeds
        )
    representations = list(dict.fromkeys(representations))
    variants = plan_representation_variants(
        config,
        representations,
        quick=quick,
        fixed_family=profile.get("representation_variant_family"),
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
                    profile.get(
                        "ridge_alphas",
                        config["models"]["ridge_alphas"],
                    )
                    if model == "ridge" and not quick
                    else [1.0]
                    if model == "ridge"
                    else [None]
                )
                for alpha in alphas:
                    for fold in profile["folds"]:
                        for seed in profile["seeds"]:
                            supported = fold_task_supported(
                                config,
                                representation,
                                variant["representation_variant_family"],
                                fold,
                                int(seed),
                            )
                            if not supported and profile.get(
                                "experiment_profile"
                            ):
                                raise FileNotFoundError(
                                    "Missing fold-safe confirmatory artifact for "
                                    f"representation={representation}, "
                                    f"family={variant['representation_variant_family']}, "
                                    f"fold={fold}, seed={seed}. Run the residual "
                                    "r6-confirmatory artifact stage first."
                                )
                            if not supported:
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
                            if profile.get("experiment_profile"):
                                payload["experiment_profile"] = str(
                                    profile["experiment_profile"]
                                )
                            for key in (
                                "text_news_levels",
                                "training_cohort",
                                "evaluation_news_level",
                                "evaluation_gate_representation",
                                "primary_evaluation_cohort",
                                "required_pooling",
                            ):
                                if key in profile:
                                    payload[key] = profile[key]
                            prefix_map = profile.get(
                                "representation_feature_prefixes",
                                {},
                            )
                            if representation in prefix_map:
                                payload["text_feature_prefixes"] = list(
                                    prefix_map[representation]
                                )
                            random_prefix = str(
                                profile.get(
                                    "random_representation_prefix",
                                    "R9_NULL",
                                )
                            )
                            random_token = f"{random_prefix}_"
                            if representation.startswith(random_token):
                                payload["random_prototype_seed"] = int(
                                    representation.removeprefix(random_token)
                                )
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
