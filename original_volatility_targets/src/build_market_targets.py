"""Create leakage-safe stock-level targets directly from original volatility."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import (
    assert_split_contract,
    atomic_write_csv,
    atomic_write_json,
    ensure_directories,
    get_logger,
    load_shared_market,
    price_feature_columns,
    project_path,
    validate_columns,
    write_table,
)


def _ticker_thresholds(
    frame: pd.DataFrame,
    quantiles: list[float],
) -> pd.DataFrame:
    train = frame.loc[frame["split"] == "train"]
    rows = []
    for ticker, group in train.groupby("ticker", sort=True, observed=True):
        values = pd.to_numeric(group["volatility_level"], errors="raise")
        row: dict[str, Any] = {
            "ticker": str(ticker),
            "train_mean": float(values.mean()),
            "train_std": float(values.std(ddof=0)),
            "n_train": int(len(values)),
        }
        for quantile in quantiles:
            row[f"q{int(round(100 * quantile)):02d}"] = float(
                values.quantile(quantile)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_targets(frame: pd.DataFrame, figures: Path) -> list[Path]:
    figures.mkdir(parents=True, exist_ok=True)
    paths = [
        figures / "volatility_distribution.png",
        figures / "volatility_by_ticker.png",
        figures / "volatility_spike_rate.png",
        figures / "volatility_regime_timeline.png",
    ]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for split, group in frame.groupby("split", sort=False, observed=True):
        axis.hist(
            group["volatility_level"],
            bins=60,
            alpha=0.45,
            density=True,
            label=str(split),
        )
    axis.set(
        title="Original next-day log volatility distribution",
        xlabel="log Garman–Klass variance",
        ylabel="Density",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(paths[0], dpi=150)
    plt.close(figure)

    pivot = frame.pivot(
        index="target_date", columns="ticker", values="volatility_level"
    )
    figure, axis = plt.subplots(figsize=(13, 6))
    pivot.plot(ax=axis, linewidth=0.65, alpha=0.75)
    axis.set(title="Original volatility by ticker", ylabel="log variance")
    axis.legend(ncol=4, fontsize=8)
    figure.tight_layout()
    figure.savefig(paths[1], dpi=150)
    plt.close(figure)

    rates = (
        frame.groupby(["split", "ticker"], observed=True)[
            ["vol_spike_q90_ticker", "vol_spike_q95_ticker"]
        ]
        .mean()
        .reset_index()
    )
    figure, axis = plt.subplots(figsize=(11, 4.8))
    x = np.arange(len(rates))
    axis.bar(x - 0.2, rates["vol_spike_q90_ticker"], 0.4, label="q90")
    axis.bar(x + 0.2, rates["vol_spike_q95_ticker"], 0.4, label="q95")
    axis.set_xticks(x, [f"{s}\n{t}" for s, t in zip(rates["split"], rates["ticker"])])
    axis.tick_params(axis="x", rotation=90, labelsize=7)
    axis.set(title="Original volatility spike rate", ylabel="Rate")
    axis.legend()
    figure.tight_layout()
    figure.savefig(paths[2], dpi=150)
    plt.close(figure)

    daily_regime = (
        frame.groupby("target_date", observed=True)["vol_regime_q50_q90"]
        .mean()
        .sort_index()
    )
    figure, axis = plt.subplots(figsize=(13, 4.5))
    axis.plot(daily_regime.index, daily_regime.values, linewidth=0.8)
    axis.set(
        title="Mean stock-level volatility regime over time",
        ylabel="Mean regime (0/1/2)",
    )
    figure.tight_layout()
    figure.savefig(paths[3], dpi=150)
    plt.close(figure)
    return paths


def run(config: Mapping[str, Any]) -> dict[str, Path]:
    ensure_directories(config)
    logger = get_logger("prepare.market_targets", config)
    market, market_path = load_shared_market(config)
    tickers = list(map(str, config["universe"]["tickers"]))
    market = market.loc[market["ticker"].isin(tickers)].copy()
    assert_split_contract(market, tickers)
    validate_columns(
        market,
        ("target_log_variance", "target_gk_variance", "log_variance"),
        "market panel",
    )
    if not np.isfinite(market["target_log_variance"]).all():
        raise ValueError("Original volatility target contains non-finite values")
    if (market["target_gk_variance"] <= 0).any():
        raise ValueError("Original Garman–Klass target must be strictly positive")

    market["volatility_level"] = pd.to_numeric(
        market["target_log_variance"], errors="raise"
    )
    market["volatility_variance"] = pd.to_numeric(
        market["target_gk_variance"], errors="raise"
    )
    quantiles = sorted(
        set(
            [
                *map(float, config["targets"]["ticker_spike_quantiles"]),
                *map(float, config["targets"]["regime_primary_quantiles"]),
                *map(float, config["targets"]["regime_alternative_quantiles"]),
            ]
        )
    )
    thresholds = _ticker_thresholds(market, quantiles)
    if set(thresholds["ticker"]) != set(tickers):
        raise AssertionError("Train thresholds are unavailable for one or more tickers")
    market = market.merge(
        thresholds,
        on="ticker",
        how="left",
        validate="many_to_one",
    )
    epsilon = float(config["targets"]["epsilon"])
    market["volatility_standardized"] = (
        market["volatility_level"] - market["train_mean"]
    ) / (market["train_std"] + epsilon)
    for quantile in (0.90, 0.95):
        suffix = f"q{int(round(100 * quantile)):02d}"
        market[f"vol_spike_{suffix}_ticker"] = (
            market["volatility_level"] > market[suffix]
        ).astype(np.int8)
        pooled_threshold = float(
            market.loc[market["split"] == "train", "volatility_standardized"].quantile(
                quantile
            )
        )
        market[f"pooled_standardized_{suffix}"] = pooled_threshold
        market[f"vol_spike_{suffix}_pooled_standardized"] = (
            market["volatility_standardized"] > pooled_threshold
        ).astype(np.int8)
    market["vol_regime_q50_q90"] = np.select(
        [
            market["volatility_level"] <= market["q50"],
            market["volatility_level"] <= market["q90"],
        ],
        [0, 1],
        default=2,
    ).astype(np.int8)
    market["vol_regime_q33_q67"] = np.select(
        [
            market["volatility_level"] <= market["q33"],
            market["volatility_level"] <= market["q67"],
        ],
        [0, 1],
        default=2,
    ).astype(np.int8)
    market = market.sort_values(["feature_date", "ticker"], kind="mergesort")

    processed = project_path(config, "data", "processed")
    tables = project_path(config, "outputs", "tables")
    figures = project_path(config, "outputs", "figures")
    target_path = write_table(
        market, processed / "original_market_targets.parquet", index=False
    )
    threshold_path = atomic_write_csv(
        thresholds, tables / "original_target_thresholds.csv", index=False
    )
    price_columns = price_feature_columns(market, config)
    contract = {
        "source_market_path": str(market_path),
        "source_market_size": market_path.stat().st_size,
        "source_market_modified_ns": market_path.stat().st_mtime_ns,
        "target_definition": "target_log_variance from shared Garman-Klass panel",
        "feature_target_alignment": "feature_date t predicts target_date t+1",
        "price_feature_columns": price_columns,
        "threshold_fit_split": "train_only",
        "residual_columns_used": [],
    }
    contract_path = atomic_write_json(
        contract, processed / "original_target_contract.json"
    )

    summary_rows: list[dict[str, Any]] = []
    for (split, ticker), group in market.groupby(
        ["split", "ticker"], sort=True, observed=True
    ):
        summary_rows.append(
            {
                "record_type": "ticker_split",
                "split": split,
                "ticker": ticker,
                "start_feature_date": group["feature_date"].min(),
                "end_feature_date": group["feature_date"].max(),
                "start_target_date": group["target_date"].min(),
                "end_target_date": group["target_date"].max(),
                "n_samples": len(group),
                "n_feature_days": group["feature_date"].nunique(),
                "mean_log_volatility": group["volatility_level"].mean(),
                "std_log_volatility": group["volatility_level"].std(),
                "minimum_log_volatility": group["volatility_level"].min(),
                "maximum_log_volatility": group["volatility_level"].max(),
                "spike_q90_count": group["vol_spike_q90_ticker"].sum(),
                "spike_q95_count": group["vol_spike_q95_ticker"].sum(),
                "regime_low_count": (group["vol_regime_q50_q90"] == 0).sum(),
                "regime_medium_count": (group["vol_regime_q50_q90"] == 1).sum(),
                "regime_high_count": (group["vol_regime_q50_q90"] == 2).sum(),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary_path = atomic_write_csv(
        summary, tables / "original_target_summary.csv", index=False
    )
    figure_paths = _plot_targets(market, figures)
    for split, group in market.groupby("split", sort=True, observed=True):
        ticker_counts = group.groupby("ticker", observed=True).size()
        logger.info(
            "TARGET SPLIT %s | feature=%s..%s | target=%s..%s | "
            "days=%d | samples=%d | samples_per_ticker=%d..%d | "
            "missing_target=%d | q90_spikes=%d | q95_spikes=%d | "
            "regimes=%s",
            split,
            group["feature_date"].min().date(),
            group["feature_date"].max().date(),
            group["target_date"].min().date(),
            group["target_date"].max().date(),
            group["feature_date"].nunique(),
            len(group),
            int(ticker_counts.min()),
            int(ticker_counts.max()),
            int(group["volatility_level"].isna().sum()),
            int(group["vol_spike_q90_ticker"].sum()),
            int(group["vol_spike_q95_ticker"].sum()),
            group["vol_regime_q50_q90"]
            .value_counts()
            .sort_index()
            .to_dict(),
        )
    return {
        "market_targets": target_path,
        "thresholds": threshold_path,
        "contract": contract_path,
        "summary": summary_path,
        **{f"figure_{index}": path for index, path in enumerate(figure_paths)},
    }
