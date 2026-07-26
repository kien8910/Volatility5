from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, mean_squared_error, r2_score


def huber_values(
    actual: np.ndarray,
    prediction: np.ndarray,
    delta: float = 1.0,
) -> np.ndarray:
    residual = np.asarray(actual, dtype=float) - np.asarray(prediction, dtype=float)
    absolute = np.abs(residual)
    return np.where(
        absolute <= delta,
        0.5 * residual**2,
        delta * (absolute - 0.5 * delta),
    )


def safe_correlation(
    actual: np.ndarray,
    prediction: np.ndarray,
    *,
    method: str,
) -> float:
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if len(actual) < 3 or np.std(actual) == 0 or np.std(prediction) == 0:
        return float("nan")
    if method == "pearson":
        return float(pearsonr(actual, prediction).statistic)
    if method == "spearman":
        return float(spearmanr(actual, prediction).statistic)
    raise ValueError(f"Unknown correlation method: {method}")


def regression_metrics(
    frame: pd.DataFrame,
    *,
    actual_column: str = "actual_t1",
    prediction_column: str = "prediction",
    huber_delta: float = 1.0,
) -> dict[str, float]:
    actual = frame[actual_column].to_numpy(dtype=float)
    prediction = frame[prediction_column].to_numpy(dtype=float)
    if len(actual) == 0:
        raise ValueError("Cannot evaluate an empty prediction frame")
    actual_sign = (actual > 0).astype(int)
    predicted_sign = (prediction > 0).astype(int)
    result = {
        "n_samples": int(len(frame)),
        "n_dates": int(frame["date"].nunique()) if "date" in frame else np.nan,
        "mae": float(mean_absolute_error(actual, prediction)),
        "rmse": float(math.sqrt(mean_squared_error(actual, prediction))),
        "huber": float(huber_values(actual, prediction, huber_delta).mean()),
        "r2_oos": float(r2_score(actual, prediction)),
        "pearson": safe_correlation(actual, prediction, method="pearson"),
        "spearman": safe_correlation(actual, prediction, method="spearman"),
        "sign_accuracy": float((actual_sign == predicted_sign).mean()),
    }
    if len(np.unique(actual_sign)) == 2:
        result["balanced_sign_accuracy"] = float(
            balanced_accuracy_score(actual_sign, predicted_sign)
        )
    else:
        result["balanced_sign_accuracy"] = float("nan")
    return result


def daily_ic_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["split", "model_name", "seed", "date"]
    for (split, model, seed, date), group in predictions.groupby(
        keys, sort=True, dropna=False
    ):
        if len(group) < 3:
            ic = float("nan")
        else:
            ic = safe_correlation(
                group["actual_t1"].to_numpy(),
                group["prediction"].to_numpy(),
                method="spearman",
            )
        actual_rank = group["actual_t1"].rank(method="average", ascending=False)
        predicted_rank = group["prediction"].rank(method="average", ascending=False)
        top_actual = set(group.loc[actual_rank <= 3, "ticker"])
        top_predicted = set(group.loc[predicted_rank <= 3, "ticker"])
        bottom_actual = set(group.loc[actual_rank > len(group) - 3, "ticker"])
        bottom_predicted = set(group.loc[predicted_rank > len(group) - 3, "ticker"])
        rows.append(
            {
                "split": split,
                "model_name": model,
                "seed": int(seed),
                "date": date,
                "offset": int(group["offset"].iloc[0]),
                "n_tickers": int(len(group)),
                "daily_ic": ic,
                "top3_accuracy": len(top_actual & top_predicted) / 3.0,
                "bottom3_accuracy": len(bottom_actual & bottom_predicted) / 3.0,
                "daily_sign_accuracy": float(
                    (
                        (group["actual_t1"].to_numpy() > 0)
                        == (group["prediction"].to_numpy() > 0)
                    ).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_daily_ic(frame: pd.DataFrame) -> dict[str, float]:
    values = frame["daily_ic"].dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return {
            "mean_daily_ic": np.nan,
            "median_daily_ic": np.nan,
            "daily_ic_std": np.nan,
            "daily_ic_ir": np.nan,
            "positive_daily_ic_rate": np.nan,
        }
    standard_deviation = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
    return {
        "mean_daily_ic": float(np.mean(values)),
        "median_daily_ic": float(np.median(values)),
        "daily_ic_std": standard_deviation,
        "daily_ic_ir": float(np.mean(values) / standard_deviation)
        if standard_deviation and np.isfinite(standard_deviation)
        else np.nan,
        "positive_daily_ic_rate": float((values > 0).mean()),
    }
