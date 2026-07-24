"""Two-stage Gaussian/Student-t forecasting of original volatility."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.modeling import (
    experiment_task,
    fold_task_supported,
    plan_representation_variants,
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
            for model in profile["uncertainty_models"]:
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
                        distribution = (
                            "student_t" if str(model).startswith("student_t") else "gaussian"
                        )
                        mean_design = (
                            "fixed_price_mean"
                            if str(model).endswith("fixed_mean")
                            else "joint_feature_mean"
                        )
                        payload = {
                            **variant,
                            "target_family": "uncertainty",
                            "target": "volatility_uncertainty",
                            "model": str(model),
                            "distribution": distribution,
                            "mean_design": mean_design,
                            "input_variant": input_variant,
                            "fold": fold,
                            "seed": int(seed),
                            "quick": bool(quick),
                        }
                        tasks.append(
                            experiment_task(
                                "uncertainty", "train_uncertainty", payload
                            )
                        )
    return tasks


def _price_mean_task(task: TaskSpec) -> TaskSpec:
    payload = dict(task.config)
    payload.update(
        {
            "representation": "R0",
            "representation_variant": "selected_default",
            "representation_variant_family": "selected_default",
            "input_variant": "price_only",
        }
    )
    return TaskSpec(
        stage=task.stage,
        action=task.action,
        config=payload,
        task_id=task.task_id,
    )


def _expanding_oof_predictions(
    x: np.ndarray,
    y: np.ndarray,
    dates: pd.Series,
    alpha: float,
) -> np.ndarray:
    unique_dates = np.sort(pd.to_datetime(dates).unique())
    if len(unique_dates) < 25:
        raise ValueError("Uncertainty OOF mean requires at least 25 train dates")
    initial = max(int(np.floor(0.40 * len(unique_dates))), 20)
    remaining = unique_dates[initial:]
    blocks = [block for block in np.array_split(remaining, 4) if len(block)]
    predictions = np.full(len(y), np.nan, dtype=float)
    date_values = pd.to_datetime(dates).to_numpy()
    for block in blocks:
        validation_start = block[0]
        train_mask = date_values < validation_start
        validation_mask = np.isin(date_values, block)
        if not train_mask.any() or not validation_mask.any():
            continue
        model = Ridge(alpha=alpha)
        model.fit(x[train_mask], y[train_mask])
        predictions[validation_mask] = model.predict(x[validation_mask])
    return predictions


def run_task(
    task: TaskSpec,
    config: Mapping[str, Any],
    profile: Mapping[str, Any],
    tracker: ProgressTracker,
) -> dict[str, Any]:
    scale_data = prepare_stock_data(config, task, profile)
    mean_data = (
        prepare_stock_data(config, _price_mean_task(task), profile)
        if task.config["mean_design"] == "fixed_price_mean"
        else scale_data
    )
    for left, right, label in (
        (mean_data.train, scale_data.train, "train"),
        (mean_data.validation, scale_data.validation, "validation"),
        (mean_data.test, scale_data.test, "test"),
    ):
        if not left[["ticker", "feature_date"]].reset_index(drop=True).equals(
            right[["ticker", "feature_date"]].reset_index(drop=True)
        ):
            raise AssertionError(f"Mean/scale feature rows are misaligned in {label}")
    y_train = scale_data.train["volatility_level"].to_numpy(dtype=float)
    y_validation = scale_data.validation["volatility_level"].to_numpy(dtype=float)
    y_test = scale_data.test["volatility_level"].to_numpy(dtype=float)
    alpha = 1.0
    oof_mean = _expanding_oof_predictions(
        mean_data.x_train,
        y_train,
        mean_data.train["target_date"],
        alpha,
    )
    valid_oof = np.isfinite(oof_mean)
    if valid_oof.sum() < 50:
        raise ValueError("Too few OOF mean predictions for scale training")
    scale_target = np.log(
        np.abs(y_train[valid_oof] - oof_mean[valid_oof])
        + float(config["targets"]["epsilon"])
    )
    mean_model = Ridge(alpha=alpha).fit(mean_data.x_train, y_train)
    scale_model = Ridge(alpha=alpha).fit(
        scale_data.x_train[valid_oof], scale_target
    )
    minimum_scale = float(config["models"]["uncertainty_min_scale"])

    def predict(
        x_mean: np.ndarray, x_scale: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean = np.asarray(mean_model.predict(x_mean), dtype=float)
        # E|N(0,sigma)| = sigma*sqrt(2/pi), hence the calibration factor.
        scale = (
            np.exp(np.clip(scale_model.predict(x_scale), -12, 8))
            * np.sqrt(np.pi / 2.0)
        )
        scale = np.clip(scale, minimum_scale, None)
        df = np.full(
            len(mean), float(config["models"]["student_t_df"]), dtype=float
        )
        return mean, scale, df

    distribution = str(task.config["distribution"])
    validation_mean, validation_scale, validation_df = predict(
        mean_data.x_validation, scale_data.x_validation
    )
    qualifies = (
        mean_data.qualifies_for_robustness
        and scale_data.qualifies_for_robustness
    )
    fit_scope = (
        f"mean={mean_data.representation_fit_scope};"
        f"scale={scale_data.representation_fit_scope}"
    )
    pieces = [
        prediction_frame(
            scale_data.validation,
            task,
            evaluation_split="validation",
            y_true=y_validation,
            mean=validation_mean,
            scale=validation_scale,
            df=validation_df,
            distribution=distribution,
            representation_fit_scope=fit_scope,
            qualifies_for_robustness=qualifies,
        )
    ]
    if len(scale_data.test):
        test_mean, test_scale, test_df = predict(
            mean_data.x_test, scale_data.x_test
        )
        pieces.append(
            prediction_frame(
                scale_data.test,
                task,
                evaluation_split="test",
                y_true=y_test,
                mean=test_mean,
                scale=test_scale,
                df=test_df,
                distribution=distribution,
                representation_fit_scope=fit_scope,
                qualifies_for_robustness=qualifies,
            )
        )
    predictions = pd.concat(pieces, ignore_index=True)
    paths = save_task_artifacts(
        config,
        task,
        predictions,
        {
            "mean_model": mean_model,
            "scale_model": scale_model,
            "mean_processor": mean_data.processor,
            "scale_processor": scale_data.processor,
            "mean_features": mean_data.feature_columns,
            "scale_features": scale_data.feature_columns,
            "oof_scale_training_rows": int(valid_oof.sum()),
            "distribution": distribution,
            "fit_scope": "task_train_only_with_expanding_oof_scale_target",
            "representation_fit_scope": fit_scope,
        },
    )
    return {"predictions": paths[0], "model": paths[1]}
