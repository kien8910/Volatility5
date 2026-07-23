"""Price-only historical mean and HAR baselines.

Model selection is performed exclusively on the chronological validation set.
Only after the winning baseline and hyperparameter are locked is the test set
evaluated.  The helpers in this module are reused by ``generate_residuals`` to
produce genuinely forward, expanding-window training residuals.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import (
    atomic_joblib_dump,
    atomic_write_csv,
    chronological_assertions,
    ensure_directories,
    get_logger,
    load_config,
    project_path,
    qlike,
    regression_metrics,
    safe_read_table,
    set_global_seed,
    validate_required_columns,
    write_json,
    write_table,
)

LOGGER = get_logger(__name__)
HAR_COLUMNS = ["har_daily", "har_weekly", "har_monthly"]
KEY_COLUMNS = ["ticker", "feature_date", "target_date", "split"]
TARGET_COLUMN = "target_log_variance"


def _market_path(config: Mapping[str, Any]) -> Path:
    candidates = [
        project_path(config, "data", "processed", "market_supervised.parquet"),
        project_path(config, "data", "processed", "market_features.parquet"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_market_supervised(config: Mapping[str, Any]) -> pd.DataFrame:
    frame = safe_read_table(_market_path(config))
    validate_required_columns(
        frame, [*KEY_COLUMNS, TARGET_COLUMN, *HAR_COLUMNS], "market supervised table"
    )
    frame = frame.copy()
    frame["feature_date"] = pd.to_datetime(frame["feature_date"])
    frame["target_date"] = pd.to_datetime(frame["target_date"])
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame = frame.sort_values(["target_date", "ticker"]).reset_index(drop=True)
    feature_and_target = [*HAR_COLUMNS, TARGET_COLUMN]
    invalid = ~np.isfinite(frame[feature_and_target].to_numpy(dtype=float)).all(axis=1)
    if invalid.any():
        LOGGER.warning("Dropping %d rows with non-finite HAR/target values", invalid.sum())
        frame = frame.loc[~invalid].reset_index(drop=True)
    chronological_assertions(
        frame.loc[frame["split"] == "train", "target_date"],
        frame.loc[frame["split"] == "validation", "target_date"],
        frame.loc[frame["split"] == "test", "target_date"],
    )
    return frame


def fit_baseline(
    frame: pd.DataFrame,
    method: str,
    alpha: float = 1.0,
) -> dict[str, Any]:
    """Fit one baseline without inspecting any future frame."""
    validate_required_columns(frame, ["ticker", TARGET_COLUMN, *HAR_COLUMNS], "baseline fit")
    training = frame.dropna(subset=[TARGET_COLUMN, *HAR_COLUMNS]).copy()
    if training.empty:
        raise ValueError("Cannot fit a baseline on an empty frame")

    if method == "B0":
        return {
            "method": method,
            "global_mean": float(training[TARGET_COLUMN].mean()),
            "ticker_means": training.groupby("ticker")[TARGET_COLUMN].mean().to_dict(),
        }

    if method == "B1":
        models: dict[str, Pipeline] = {}
        for ticker, ticker_frame in training.groupby("ticker", sort=True):
            pipeline = Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("ridge", Ridge(alpha=float(alpha))),
                ]
            )
            pipeline.fit(ticker_frame[HAR_COLUMNS], ticker_frame[TARGET_COLUMN])
            models[str(ticker)] = pipeline
        return {
            "method": method,
            "alpha": float(alpha),
            "models": models,
            "fallback": fit_baseline(training, "B0"),
        }

    if method == "B2":
        transformer = ColumnTransformer(
            [
                ("har", StandardScaler(), HAR_COLUMNS),
                (
                    "ticker",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                    ["ticker"],
                ),
            ],
            remainder="drop",
        )
        pipeline = Pipeline(
            [
                ("features", transformer),
                ("ridge", Ridge(alpha=float(alpha))),
            ]
        )
        pipeline.fit(training[[*HAR_COLUMNS, "ticker"]], training[TARGET_COLUMN])
        return {"method": method, "alpha": float(alpha), "model": pipeline}

    raise ValueError(f"Unknown baseline method: {method}")


def predict_baseline(bundle: Mapping[str, Any], frame: pd.DataFrame) -> np.ndarray:
    method = str(bundle["method"])
    if method == "B0":
        global_mean = float(bundle["global_mean"])
        ticker_means = bundle["ticker_means"]
        return frame["ticker"].map(ticker_means).fillna(global_mean).to_numpy(dtype=float)

    if method == "B1":
        fallback = predict_baseline(bundle["fallback"], frame)
        predictions = fallback.copy()
        for ticker, positions in frame.groupby("ticker", sort=False).indices.items():
            model = bundle["models"].get(str(ticker))
            if model is not None:
                position_array = np.asarray(positions, dtype=int)
                predictions[position_array] = model.predict(
                    frame.iloc[position_array][HAR_COLUMNS]
                )
        return predictions

    if method == "B2":
        return np.asarray(
            bundle["model"].predict(frame[[*HAR_COLUMNS, "ticker"]]), dtype=float
        )

    raise ValueError(f"Unknown baseline method in artifact: {method}")


def _result_row(
    method: str,
    alpha: float | None,
    split: str,
    frame: pd.DataFrame,
    predictions: Sequence[float],
    selected: bool = False,
    scope: str = "overall",
    ticker: str = "ALL",
) -> dict[str, Any]:
    metrics = regression_metrics(frame[TARGET_COLUMN], predictions)
    return {
        "baseline": method,
        "alpha": alpha,
        "split": split,
        "scope": scope,
        "ticker": ticker,
        "n_samples": len(frame),
        "start_date": frame["target_date"].min(),
        "end_date": frame["target_date"].max(),
        "qlike": qlike(frame[TARGET_COLUMN], predictions),
        **metrics,
        "selected": selected,
    }


def _result_rows(
    method: str,
    alpha: float | None,
    split: str,
    frame: pd.DataFrame,
    predictions: Sequence[float],
    selected: bool = False,
) -> list[dict[str, Any]]:
    prediction_array = np.asarray(predictions, dtype=float)
    rows = [
        _result_row(
            method,
            alpha,
            split,
            frame,
            prediction_array,
            selected=selected,
        )
    ]
    for ticker, positions in frame.groupby("ticker", sort=True).indices.items():
        position_array = np.asarray(positions, dtype=int)
        rows.append(
            _result_row(
                method,
                alpha,
                split,
                frame.iloc[position_array],
                prediction_array[position_array],
                selected=selected,
                scope="ticker",
                ticker=str(ticker),
            )
        )
    return rows


def _fit_selected_for_test(
    selection: Mapping[str, Any],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    refit: bool,
) -> dict[str, Any]:
    if refit:
        raise ValueError(
            "baseline.refit_test_with_train_validation=true violates this "
            "experiment's strict train-fit/holdout-transform contract"
        )
    fit_frame = train
    return fit_baseline(
        fit_frame,
        method=str(selection["baseline"]),
        alpha=float(selection.get("alpha") or 0.0),
    )


def run(config: dict[str, Any]) -> dict[str, Path]:
    ensure_directories(config)
    seed = int(config["project"]["seed"])
    set_global_seed(seed, bool(config["project"].get("deterministic", True)))
    logger = get_logger(
        __name__,
        config,
        project_path(config, "outputs", "logs", "baseline.log"),
    )

    frame = load_market_supervised(config)
    train = frame.loc[frame["split"] == "train"].reset_index(drop=True)
    validation = frame.loc[frame["split"] == "validation"].reset_index(drop=True)
    test = frame.loc[frame["split"] == "test"].reset_index(drop=True)
    logger.info(
        "Chronological samples | train=%d [%s,%s] | validation=%d [%s,%s] | test=%d [%s,%s]",
        len(train),
        train["target_date"].min().date(),
        train["target_date"].max().date(),
        len(validation),
        validation["target_date"].min().date(),
        validation["target_date"].max().date(),
        len(test),
        test["target_date"].min().date(),
        test["target_date"].max().date(),
    )

    candidates: list[tuple[str, float | None]] = [("B0", None)]
    for method in ("B1", "B2"):
        candidates.extend(
            (method, float(alpha)) for alpha in config["baseline"]["ridge_alphas"]
        )

    validation_rows: list[dict[str, Any]] = []
    validation_predictions: dict[tuple[str, float | None], np.ndarray] = {}
    train_models: dict[tuple[str, float | None], dict[str, Any]] = {}
    for method, alpha in candidates:
        model = fit_baseline(train, method, float(alpha or 0.0))
        prediction = predict_baseline(model, validation)
        key = (method, alpha)
        train_models[key] = model
        validation_predictions[key] = prediction
        validation_rows.extend(
            _result_rows(method, alpha, "validation", validation, prediction)
        )

    validation_table = pd.DataFrame(validation_rows)
    overall_validation = validation_table.loc[
        validation_table["scope"] == "overall"
    ]
    chosen_index = overall_validation["qlike"].astype(float).idxmin()
    selected_row = validation_table.loc[chosen_index]
    selection = {
        "baseline": str(selected_row["baseline"]),
        "alpha": (
            None if pd.isna(selected_row["alpha"]) else float(selected_row["alpha"])
        ),
        "validation_qlike": float(selected_row["qlike"]),
        "selection_metric": "qlike",
        "selection_split": "validation",
        "test_was_used_for_selection": False,
    }
    chosen_mask = (
        validation_table["baseline"].eq(selection["baseline"])
        & validation_table["alpha"].fillna(-1.0).eq(
            -1.0 if selection["alpha"] is None else selection["alpha"]
        )
    )
    validation_table.loc[chosen_mask, "selected"] = True
    selected_key = (selection["baseline"], selection["alpha"])

    refit = bool(config["baseline"].get("refit_test_with_train_validation", False))
    test_model = _fit_selected_for_test(selection, train, validation, refit)
    test_prediction = predict_baseline(test_model, test)
    test_rows = _result_rows(
        selection["baseline"],
        selection["alpha"],
        "test",
        test,
        test_prediction,
        selected=True,
    )
    test_row = test_rows[0]
    results = pd.concat(
        [validation_table, pd.DataFrame(test_rows)], ignore_index=True
    )

    validation_key_columns = [
        "ticker",
        "feature_date",
        "target_date",
        TARGET_COLUMN,
    ]
    validation_output = validation[validation_key_columns].copy()
    validation_output["split"] = "validation"
    validation_output["baseline_prediction"] = validation_predictions[selected_key]
    test_output = test[validation_key_columns].copy()
    test_output["split"] = "test"
    test_output["baseline_prediction"] = test_prediction
    predictions = pd.concat([validation_output, test_output], ignore_index=True)
    predictions["baseline_residual"] = (
        predictions[TARGET_COLUMN] - predictions["baseline_prediction"]
    )

    tables = project_path(config, "outputs", "tables")
    models = project_path(config, "outputs", "models")
    processed = project_path(config, "data", "processed")
    results_path = atomic_write_csv(results, tables / "baseline_results.csv")
    selection_path = write_json(selection, models / "baseline_selection.json")
    train_model_path = atomic_joblib_dump(
        train_models[selected_key], models / "baseline_train_only.joblib"
    )
    test_model_path = atomic_joblib_dump(
        test_model, models / "baseline_for_test.joblib"
    )
    predictions_path = write_table(
        predictions, processed / "baseline_holdout_predictions.parquet"
    )
    logger.info(
        "Locked baseline %s alpha=%s on validation QLIKE %.6f; test QLIKE %.6f",
        selection["baseline"],
        selection["alpha"],
        selection["validation_qlike"],
        test_row["qlike"],
    )
    return {
        "baseline_results": results_path,
        "baseline_selection": selection_path,
        "baseline_train_model": train_model_path,
        "baseline_test_model": test_model_path,
        "holdout_predictions": predictions_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to config YAML")
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
