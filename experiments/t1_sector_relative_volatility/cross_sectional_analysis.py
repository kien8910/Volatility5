from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf

from .config import ExperimentConfig
from .block_bootstrap import moving_block_indices
from .utils import progress


def _pairwise_summary(matrix: pd.DataFrame) -> dict[str, float]:
    correlation = matrix.corr()
    values = correlation.to_numpy()
    upper = values[np.triu_indices_from(values, k=1)]
    return {
        "mean_pairwise_correlation": float(np.nanmean(upper)),
        "median_pairwise_correlation": float(np.nanmedian(upper)),
    }


def correlation_and_effective_sample(
    target: pd.DataFrame,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_matrix = target.pivot(
        index="date", columns="ticker", values="forward_mean_5d"
    )
    t1_matrix = target.pivot(index="date", columns="ticker", values="t1_target")
    before = base_matrix.corr()
    after = t1_matrix.corr()
    rows = []
    for label, matrix in [
        ("forward_mean_5d_before_demean", base_matrix),
        ("t1_after_leave_one_out_demean", t1_matrix),
    ]:
        summary = _pairwise_summary(matrix)
        n_tickers = int(matrix.shape[1])
        n_dates = int(matrix.dropna(how="all").shape[0])
        rho = summary["mean_pairwise_correlation"]
        denominator = 1.0 + (n_tickers - 1) * rho
        effective = (
            n_tickers * n_dates / denominator if denominator > 0 else np.nan
        )
        nominal = n_tickers * n_dates
        rows.append(
            {
                "target": label,
                "n_tickers": n_tickers,
                "n_dates": n_dates,
                **summary,
                "effective_sample_size_approximation": float(effective),
                "effective_sample_size_capped_at_nominal": float(
                    min(effective, nominal) if np.isfinite(effective) else np.nan
                ),
                "note": (
                    "Cross-sectional approximation; temporal dependence remains. "
                    "Negative average correlation can make the raw formula exceed "
                    "N*T, so a capped value is also reported."
                ),
            }
        )
    return before, after, pd.DataFrame(rows)


def _mean_ticker_acf(
    target: pd.DataFrame,
    value_column: str,
    max_lag: int,
) -> np.ndarray:
    values = []
    for _, group in target.groupby("ticker", sort=True):
        series = group.sort_values("date")[value_column].dropna().to_numpy(dtype=float)
        values.append(acf(series, nlags=max_lag, fft=True, missing="drop"))
    minimum = min(len(item) for item in values)
    return np.nanmean(np.vstack([item[:minimum] for item in values]), axis=0)


def target_diagnostics(
    target: pd.DataFrame,
    paired_daily: pd.DataFrame | None = None,
    max_lag: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    series_map = {
        "base_target_t_plus_1": "base_target_t_plus_1",
        "forward_mean_5d": "forward_mean_5d",
        "t1_target": "t1_target",
    }
    distribution_rows = []
    acf_rows = []
    for label, column in progress(
        series_map.items(),
        total=len(series_map),
        description="[Cross-sectional analysis] target diagnostics",
    ):
        values = target[column].dropna().to_numpy(dtype=float)
        distribution_rows.append(
            {
                "target": label,
                "count": len(values),
                "mean": float(np.mean(values)),
                "variance": float(np.var(values, ddof=1)),
                "standard_deviation": float(np.std(values, ddof=1)),
                "minimum": float(np.min(values)),
                "median": float(np.median(values)),
                "maximum": float(np.max(values)),
            }
        )
        mean_acf = _mean_ticker_acf(target, column, max_lag)
        acf_rows.extend(
            {"series": label, "lag": lag, "acf": float(value)}
            for lag, value in enumerate(mean_acf)
        )
    if paired_daily is not None and not paired_daily.empty:
        for (comparison, seed), group in paired_daily.groupby(
            ["comparison", "seed"], sort=True
        ):
            series = (
                group[group["split"].eq("test")]
                .sort_values("date")["absolute_loss_difference"]
                .to_numpy(dtype=float)
            )
            if len(series) > max_lag + 1:
                values = acf(series, nlags=max_lag, fft=True, missing="drop")
                acf_rows.extend(
                    {
                        "series": f"paired_loss_{comparison}_seed{seed}",
                        "lag": lag,
                        "acf": float(value),
                    }
                    for lag, value in enumerate(values)
                )
    return pd.DataFrame(distribution_rows), pd.DataFrame(acf_rows)


def ranking_portfolio(
    predictions: pd.DataFrame,
    config: ExperimentConfig,
) -> pd.DataFrame:
    test = predictions[predictions["split"].eq("test")].copy()
    rows: list[dict[str, Any]] = []
    for (model, seed, date), group in test.groupby(
        ["model_name", "seed", "date"], sort=True
    ):
        ranked = group.sort_values("prediction", ascending=False)
        if len(ranked) < 6:
            continue
        top = ranked.head(3)
        bottom = ranked.tail(3)
        rows.append(
            {
                "model_name": model,
                "seed": int(seed),
                "date": date,
                "offset": int(group["offset"].iloc[0]),
                "top3_realized_t1": float(top["actual_t1"].mean()),
                "bottom3_realized_t1": float(bottom["actual_t1"].mean()),
                "realized_spread": float(
                    top["actual_t1"].mean() - bottom["actual_t1"].mean()
                ),
                "top3_tickers": ",".join(top["ticker"].astype(str)),
                "bottom3_tickers": ",".join(bottom["ticker"].astype(str)),
            }
        )
    return pd.DataFrame(rows)


def ranking_portfolio_summary(
    daily: pd.DataFrame,
    config: ExperimentConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    alpha = (1.0 - config.bootstrap_confidence) / 2.0
    for (model, seed), group in daily.groupby(["model_name", "seed"], sort=True):
        group = group.sort_values("date")
        values = group["realized_spread"].to_numpy(dtype=float)
        for block_length in config.bootstrap_block_lengths:
            rng = np.random.default_rng(int(seed) + 7919 * block_length)
            draws = np.asarray(
                [
                    values[
                        moving_block_indices(len(values), block_length, rng)
                    ].mean()
                    for _ in range(config.bootstrap_repetitions)
                ]
            )
            rows.append(
                {
                    "model_name": model,
                    "seed": int(seed),
                    "block_length": block_length,
                    "n_dates": len(values),
                    "mean_realized_spread": float(values.mean()),
                    "bootstrap_mean": float(draws.mean()),
                    "bootstrap_standard_error": float(draws.std(ddof=1)),
                    "ci_lower": float(np.quantile(draws, alpha)),
                    "ci_upper": float(np.quantile(draws, 1.0 - alpha)),
                    "probability_spread_gt_zero": float((draws > 0).mean()),
                }
            )
    return pd.DataFrame(rows)
