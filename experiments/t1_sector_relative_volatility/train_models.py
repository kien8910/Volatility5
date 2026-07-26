from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import ExperimentConfig
from .evaluation_common import regression_metrics
from .prepare_dataset import FeatureGroups
from .utils import progress, set_seed


@dataclass
class TrainingResult:
    predictions: pd.DataFrame
    selection: pd.DataFrame


def _pipeline(feature_columns: list[str], alpha: float) -> Pipeline:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    processor = ColumnTransformer(
        [
            ("numeric", numeric, feature_columns),
            (
                "ticker",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                ["ticker"],
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("processor", processor),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def _prediction_frame(
    source: pd.DataFrame,
    prediction: np.ndarray,
    *,
    model_name: str,
    seed: int,
) -> pd.DataFrame:
    keep = [
        "date",
        "ticker",
        "split",
        "offset",
        "t1_target",
        "forward_mean_5d",
        "peer_mean_leave_one_out",
    ]
    result = source[keep].copy()
    result = result.rename(
        columns={
            "t1_target": "actual_t1",
            "forward_mean_5d": "base_forward_vol_5d",
            "peer_mean_leave_one_out": "peer_forward_vol_5d",
        }
    )
    result["prediction"] = prediction
    result["model_name"] = model_name
    result["seed"] = int(seed)
    return result[
        [
            "date",
            "ticker",
            "split",
            "offset",
            "actual_t1",
            "prediction",
            "model_name",
            "seed",
            "base_forward_vol_5d",
            "peer_forward_vol_5d",
        ]
    ]


def _mean_offset_mae(predictions: pd.DataFrame, stride: int) -> tuple[float, list[float]]:
    values = []
    for offset in range(stride):
        subset = predictions[predictions["offset"].eq(offset)]
        values.append(regression_metrics(subset)["mae"])
    return float(np.mean(values)), values


def train_models(
    data: pd.DataFrame,
    groups: FeatureGroups,
    config: ExperimentConfig,
) -> TrainingResult:
    train = data[data["split"].eq("train")].copy()
    validation = data[data["split"].eq("validation")].copy()
    test = data[data["split"].eq("test")].copy()
    model_features = groups.model_features()
    seeds = config.seeds[:1] if config.debug else config.seeds
    model_names = (
        ["M0_PRICE", "M2_PRICE_SEMANTIC"]
        if config.debug
        else list(model_features)
    )
    tasks = [(seed, model) for seed in seeds for model in model_names]
    prediction_parts: list[pd.DataFrame] = []
    selection_rows: list[dict[str, Any]] = []
    models_directory = config.output_directory / "models"
    models_directory.mkdir(parents=True, exist_ok=True)

    for seed, model_name in progress(
        tasks,
        total=len(tasks),
        description="[Training] model/seed",
    ):
        set_seed(seed)
        feature_columns = model_features[model_name]
        candidates: list[tuple[float, float, list[float], Pipeline]] = []
        for alpha in config.ridge_alphas:
            estimator = _pipeline(feature_columns, alpha)
            estimator.fit(train[[*feature_columns, "ticker"]], train["t1_target"])
            validation_prediction = estimator.predict(
                validation[[*feature_columns, "ticker"]]
            )
            validation_frame = _prediction_frame(
                validation,
                validation_prediction,
                model_name=model_name,
                seed=seed,
            )
            score, offset_mae = _mean_offset_mae(
                validation_frame, config.eval_stride
            )
            candidates.append((score, alpha, offset_mae, estimator))
            selection_rows.append(
                {
                    "model_name": model_name,
                    "seed": seed,
                    "alpha": alpha,
                    "validation_mean_non_overlapping_mae": score,
                    "validation_offset_mae": str(offset_mae),
                    "selected": False,
                }
            )
        candidates.sort(key=lambda item: (item[0], item[1]))
        best_score, best_alpha, best_offsets, best_model = candidates[0]
        for row in reversed(selection_rows):
            if (
                row["model_name"] == model_name
                and row["seed"] == seed
                and row["alpha"] == best_alpha
            ):
                row["selected"] = True
                break

        validation_prediction = best_model.predict(
            validation[[*feature_columns, "ticker"]]
        )
        test_prediction = best_model.predict(test[[*feature_columns, "ticker"]])
        prediction_parts.extend(
            [
                _prediction_frame(
                    validation,
                    validation_prediction,
                    model_name=model_name,
                    seed=seed,
                ),
                _prediction_frame(
                    test,
                    test_prediction,
                    model_name=model_name,
                    seed=seed,
                ),
            ]
        )
        joblib.dump(
            {
                "pipeline": best_model,
                "features": feature_columns,
                "alpha": best_alpha,
                "seed": seed,
                "validation_score": best_score,
                "validation_offset_mae": best_offsets,
            },
            models_directory / f"{model_name}_seed{seed}.joblib",
        )
    predictions = pd.concat(prediction_parts, ignore_index=True)
    return TrainingResult(
        predictions=predictions,
        selection=pd.DataFrame(selection_rows),
    )
