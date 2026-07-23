"""Generate leakage-safe expanding-window baseline residuals and targets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.build_baseline import (
    HAR_COLUMNS,
    TARGET_COLUMN,
    fit_baseline,
    load_market_supervised,
    predict_baseline,
)
from src.utils import (
    atomic_write_csv,
    ensure_directories,
    get_logger,
    load_config,
    project_path,
    read_json,
    set_global_seed,
    write_json,
    write_table,
)


def _expanding_train_predictions(
    train: pd.DataFrame,
    method: str,
    alpha: float,
    min_history_days: int,
    initial_fraction: float,
    step_days: int,
    logger: Any,
) -> pd.DataFrame:
    dates = np.sort(pd.to_datetime(train["target_date"].unique()))
    initial_index = max(
        min_history_days,
        int(np.ceil(len(dates) * initial_fraction)),
    )
    if initial_index >= len(dates):
        raise ValueError(
            "Training span is too short for expanding residual generation: "
            f"{len(dates)} dates, initial_index={initial_index}"
        )
    pieces: list[pd.DataFrame] = []
    for block_start in range(initial_index, len(dates), step_days):
        history_dates = dates[:block_start]
        forecast_dates = dates[block_start : block_start + step_days]
        history = train.loc[train["target_date"].isin(history_dates)]
        forecast = train.loc[train["target_date"].isin(forecast_dates)].copy()
        if forecast.empty:
            continue
        if pd.Timestamp(history["target_date"].max()) >= pd.Timestamp(
            forecast["target_date"].min()
        ):
            raise AssertionError("Expanding-window residual block leaks future dates")
        model = fit_baseline(history, method, alpha)
        forecast["baseline_prediction"] = predict_baseline(model, forecast)
        forecast["residual_origin"] = "train_expanding_oof"
        pieces.append(forecast)
        logger.info(
            "OOF block history<=%s forecast=[%s,%s] n=%d",
            pd.Timestamp(history_dates[-1]).date(),
            pd.Timestamp(forecast_dates[0]).date(),
            pd.Timestamp(forecast_dates[-1]).date(),
            len(forecast),
        )
    if not pieces:
        raise ValueError("No expanding-window training residuals were created")
    return pd.concat(pieces, ignore_index=True)


def _holdout_predictions(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    method: str,
    alpha: float,
    refit_test: bool,
) -> pd.DataFrame:
    if refit_test:
        raise ValueError(
            "baseline.refit_test_with_train_validation=true violates this "
            "experiment's strict train-fit/holdout-transform contract"
        )
    train_model = fit_baseline(train, method, alpha)
    validation_output = validation.copy()
    validation_output["baseline_prediction"] = predict_baseline(
        train_model, validation_output
    )
    validation_output["residual_origin"] = "validation_train_only"

    test_model = fit_baseline(train, method, alpha)
    test_output = test.copy()
    test_output["baseline_prediction"] = predict_baseline(test_model, test_output)
    test_output["residual_origin"] = "test_train_only"
    return pd.concat([validation_output, test_output], ignore_index=True)


def _fit_threshold_artifact(
    train_residuals: pd.DataFrame,
    epsilon: float,
) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "ticker": {},
        "pooled_standardized": {},
        "epsilon": epsilon,
        "fit_split": "train_expanding_oof",
    }
    standardized_parts: list[np.ndarray] = []
    for ticker, group in train_residuals.groupby("ticker", sort=True):
        values = group["signed_residual"].to_numpy(dtype=float)
        absolute = np.abs(values)
        center = float(np.mean(values))
        scale = float(np.std(values, ddof=1))
        scale = max(scale, epsilon)
        artifact["ticker"][str(ticker)] = {
            "q50": float(np.quantile(absolute, 0.50)),
            "q90": float(np.quantile(absolute, 0.90)),
            "q95": float(np.quantile(absolute, 0.95)),
            "center": center,
            "scale": scale,
            "n": int(len(group)),
        }
        standardized_parts.append(np.abs((values - center) / scale))
    pooled = np.concatenate(standardized_parts)
    artifact["pooled_standardized"] = {
        "q50": float(np.quantile(pooled, 0.50)),
        "q90": float(np.quantile(pooled, 0.90)),
        "q95": float(np.quantile(pooled, 0.95)),
        "n": int(len(pooled)),
    }
    return artifact


def _attach_targets(
    frame: pd.DataFrame,
    thresholds: dict[str, Any],
    epsilon: float,
) -> pd.DataFrame:
    output = frame.copy()
    output["signed_residual"] = (
        output[TARGET_COLUMN] - output["baseline_prediction"]
    )
    output["residual_magnitude"] = output["signed_residual"].abs()
    output["squared_residual"] = output["signed_residual"].pow(2)
    output["log_squared_residual"] = np.log(
        output["squared_residual"].to_numpy(dtype=float) + epsilon
    )

    ticker_q50 = output["ticker"].map(
        {ticker: values["q50"] for ticker, values in thresholds["ticker"].items()}
    )
    ticker_q90 = output["ticker"].map(
        {ticker: values["q90"] for ticker, values in thresholds["ticker"].items()}
    )
    ticker_q95 = output["ticker"].map(
        {ticker: values["q95"] for ticker, values in thresholds["ticker"].items()}
    )
    if ticker_q50.isna().any() or ticker_q90.isna().any() or ticker_q95.isna().any():
        missing = sorted(
            output.loc[
                ticker_q50.isna() | ticker_q90.isna() | ticker_q95.isna(),
                "ticker",
            ]
            .astype(str)
            .unique()
        )
        raise ValueError(
            "Residual thresholds were not fitted for holdout ticker(s): "
            f"{missing}"
        )
    output["spike_q90"] = (output["residual_magnitude"] > ticker_q90).astype(int)
    output["spike_q95"] = (output["residual_magnitude"] > ticker_q95).astype(int)
    output["regime"] = np.select(
        [
            output["residual_magnitude"] <= ticker_q50,
            output["residual_magnitude"] <= ticker_q90,
        ],
        [0, 1],
        default=2,
    ).astype(int)

    centers = output["ticker"].map(
        {ticker: values["center"] for ticker, values in thresholds["ticker"].items()}
    )
    scales = output["ticker"].map(
        {ticker: values["scale"] for ticker, values in thresholds["ticker"].items()}
    )
    if centers.isna().any() or scales.isna().any():
        raise AssertionError("Ticker residual standardization parameters are incomplete")
    output["standardized_residual"] = (
        output["signed_residual"] - centers
    ) / scales.clip(lower=epsilon)
    pooled = thresholds["pooled_standardized"]
    output["spike_q90_pooled_standardized"] = (
        output["standardized_residual"].abs() > pooled["q90"]
    ).astype(int)
    output["spike_q95_pooled_standardized"] = (
        output["standardized_residual"].abs() > pooled["q95"]
    ).astype(int)
    output["regime_pooled_standardized"] = np.select(
        [
            output["standardized_residual"].abs() <= pooled["q50"],
            output["standardized_residual"].abs() <= pooled["q90"],
        ],
        [0, 1],
        default=2,
    ).astype(int)
    return output


def _summary(frame: pd.DataFrame, thresholds: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (split, ticker), group in frame.groupby(["split", "ticker"], sort=True):
        residual = group["signed_residual"]
        rows.append(
            {
                "record_type": "distribution",
                "split": split,
                "ticker": ticker,
                "n": len(group),
                "mean": residual.mean(),
                "std": residual.std(),
                "mae": residual.abs().mean(),
                "rmse": np.sqrt(np.mean(residual.pow(2))),
                "q01": residual.quantile(0.01),
                "q50": residual.quantile(0.50),
                "q99": residual.quantile(0.99),
                "spike_q90_rate": group["spike_q90"].mean(),
                "spike_q95_rate": group["spike_q95"].mean(),
                "spike_q90_pooled_standardized_rate": group[
                    "spike_q90_pooled_standardized"
                ].mean(),
                "spike_q95_pooled_standardized_rate": group[
                    "spike_q95_pooled_standardized"
                ].mean(),
            }
        )
    for ticker, values in thresholds["ticker"].items():
        rows.append(
            {
                "record_type": "train_threshold",
                "split": "train",
                "ticker": ticker,
                "n": values["n"],
                "mean": values["center"],
                "std": values["scale"],
                "threshold_q50": values["q50"],
                "threshold_q90": values["q90"],
                "threshold_q95": values["q95"],
            }
        )
    return pd.DataFrame(rows)


def run(config: dict[str, Any]) -> dict[str, Path]:
    ensure_directories(config)
    seed = int(config["project"]["seed"])
    set_global_seed(seed, bool(config["project"].get("deterministic", True)))
    logger = get_logger(
        __name__,
        config,
        project_path(config, "outputs", "logs", "residuals.log"),
    )

    selection = read_json(
        project_path(config, "outputs", "models", "baseline_selection.json")
    )
    method = str(selection["baseline"])
    alpha = float(selection.get("alpha") or 0.0)
    frame = load_market_supervised(config)
    train = frame.loc[frame["split"] == "train"].copy()
    validation = frame.loc[frame["split"] == "validation"].copy()
    test = frame.loc[frame["split"] == "test"].copy()

    baseline_config = config["baseline"]
    oof = _expanding_train_predictions(
        train,
        method,
        alpha,
        int(baseline_config["min_history_days"]),
        float(baseline_config["oof_initial_fraction"]),
        int(baseline_config["oof_step_days"]),
        logger,
    )
    holdout = _holdout_predictions(
        train,
        validation,
        test,
        method,
        alpha,
        bool(baseline_config.get("refit_test_with_train_validation", False)),
    )
    all_predictions = pd.concat([oof, holdout], ignore_index=True)
    epsilon = float(config["targets"]["residual_epsilon"])
    preliminary = all_predictions.copy()
    preliminary["signed_residual"] = (
        preliminary[TARGET_COLUMN] - preliminary["baseline_prediction"]
    )
    thresholds = _fit_threshold_artifact(
        preliminary.loc[preliminary["split"] == "train"], epsilon
    )
    residuals = _attach_targets(all_predictions, thresholds, epsilon)
    residuals = residuals.sort_values(["target_date", "ticker"]).reset_index(drop=True)

    processed = project_path(config, "data", "processed")
    tables = project_path(config, "outputs", "tables")
    models = project_path(config, "outputs", "models")
    residual_path = write_table(
        residuals, processed / "residual_targets.parquet", index=False
    )
    threshold_path = write_json(
        thresholds, models / "residual_thresholds_train_only.json"
    )
    summary_path = atomic_write_csv(
        _summary(residuals, thresholds), tables / "residual_summary.csv"
    )
    logger.info(
        "Generated %d residual targets; %d are expanding-window train OOF",
        len(residuals),
        len(oof),
    )
    return {
        "residual_targets": residual_path,
        "residual_thresholds": threshold_path,
        "residual_summary": summary_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to config YAML")
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
