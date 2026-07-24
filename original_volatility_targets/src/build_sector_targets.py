"""Construct train-normalized semiconductor-wide volatility targets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import (
    atomic_write_csv,
    get_logger,
    price_feature_columns,
    project_path,
    read_table,
    write_table,
)


def _plot_sector(frame: pd.DataFrame, figures: Path) -> list[Path]:
    breadth_path = figures / "sector_volatility_breadth.png"
    regime_path = figures / "sector_regime.png"
    figure, axis = plt.subplots(figsize=(13, 4.5))
    axis.plot(
        frame["target_date"],
        frame["sector_spike_breadth_q90"],
        linewidth=0.8,
    )
    axis.axhline(3 / 11, linestyle="--", color="gray", label="3/11")
    axis.axhline(5 / 11, linestyle=":", color="black", label="5/11")
    axis.set(
        title="Semiconductor volatility spike breadth",
        ylabel="Fraction of stocks above ticker q90",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(breadth_path, dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(13, 4.5))
    axis.step(
        frame["target_date"],
        frame["sector_regime_q50_q90"],
        where="mid",
        linewidth=0.8,
    )
    axis.set(
        title="Semiconductor sector volatility regime",
        ylabel="0 normal / 1 elevated / 2 stress",
    )
    figure.tight_layout()
    figure.savefig(regime_path, dpi=150)
    plt.close(figure)
    return [breadth_path, regime_path]


def run(config: Mapping[str, Any]) -> dict[str, Path]:
    logger = get_logger("prepare.sector_targets", config)
    market_path = project_path(
        config, "data", "processed", "original_market_targets.parquet"
    )
    market = read_table(market_path)
    price_columns = price_feature_columns(market, config)
    aggregation: dict[str, tuple[str, str]] = {
        "target_date": ("target_date", "first"),
        "split": ("split", "first"),
        "n_stocks": ("ticker", "nunique"),
        "sector_mean_volatility": ("volatility_standardized", "mean"),
        "sector_raw_mean_log_volatility": ("volatility_level", "mean"),
        "sector_spike_breadth_q90": ("vol_spike_q90_ticker", "mean"),
        "sector_spike_breadth_q95": ("vol_spike_q95_ticker", "mean"),
        "sector_spike_count_q90": ("vol_spike_q90_ticker", "sum"),
        "sector_spike_count_q95": ("vol_spike_q95_ticker", "sum"),
    }
    for column in price_columns:
        aggregation[f"sector_price_mean__{column}"] = (column, "mean")
        aggregation[f"sector_price_std__{column}"] = (column, "std")
    sector = (
        market.groupby("feature_date", sort=True, observed=True)
        .agg(**aggregation)
        .reset_index()
    )
    expected_stocks = len(config["universe"]["tickers"])
    if (sector["n_stocks"] != expected_stocks).any():
        examples = sector.loc[
            sector["n_stocks"] != expected_stocks,
            ["feature_date", "n_stocks"],
        ].head(5)
        raise AssertionError(
            "Sector targets require the full ticker cross-section: "
            f"{examples.to_dict(orient='records')}"
        )
    definitions = config["targets"]["sector_spike_definitions"]
    for name, minimum_count in definitions.items():
        sector[f"sector_spike_{name}"] = (
            sector["sector_spike_count_q90"] >= int(minimum_count)
        ).astype(np.int8)
    train = sector.loc[sector["split"] == "train"]
    primary = list(map(float, config["targets"]["regime_primary_quantiles"]))
    alternative = list(map(float, config["targets"]["regime_alternative_quantiles"]))
    q50, q90 = train["sector_mean_volatility"].quantile(primary).to_numpy()
    q33, q67 = train["sector_mean_volatility"].quantile(alternative).to_numpy()
    sector["sector_regime_q50_q90"] = np.select(
        [
            sector["sector_mean_volatility"] <= q50,
            sector["sector_mean_volatility"] <= q90,
        ],
        [0, 1],
        default=2,
    ).astype(np.int8)
    sector["sector_regime_q33_q67"] = np.select(
        [
            sector["sector_mean_volatility"] <= q33,
            sector["sector_mean_volatility"] <= q67,
        ],
        [0, 1],
        default=2,
    ).astype(np.int8)
    sector["sector_regime_q50"] = float(q50)
    sector["sector_regime_q90"] = float(q90)
    sector["sector_regime_q33"] = float(q33)
    sector["sector_regime_q67"] = float(q67)
    output_path = write_table(
        sector,
        project_path(config, "data", "processed", "sector_targets.parquet"),
        index=False,
    )
    summary_rows = []
    for split, group in sector.groupby("split", sort=True, observed=True):
        row: dict[str, Any] = {
            "split": split,
            "start": group["feature_date"].min(),
            "end": group["feature_date"].max(),
            "n_days": len(group),
            "mean_sector_volatility": group["sector_mean_volatility"].mean(),
            "mean_q90_breadth": group["sector_spike_breadth_q90"].mean(),
        }
        for name in definitions:
            row[f"positive_{name}"] = int(group[f"sector_spike_{name}"].sum())
        for regime in (0, 1, 2):
            row[f"regime_{regime}_count"] = int(
                (group["sector_regime_q50_q90"] == regime).sum()
            )
        summary_rows.append(row)
    summary_path = atomic_write_csv(
        pd.DataFrame(summary_rows),
        project_path(config, "outputs", "tables", "sector_target_summary.csv"),
        index=False,
    )
    figures = _plot_sector(
        sector, project_path(config, "outputs", "figures")
    )
    for row in summary_rows:
        logger.info(
            "SECTOR SPLIT %s | dates=%s..%s | days=%d | "
            "mean_q90_breadth=%.4f | positives=%s | regimes=%s",
            row["split"],
            pd.Timestamp(row["start"]).date(),
            pd.Timestamp(row["end"]).date(),
            int(row["n_days"]),
            float(row["mean_q90_breadth"]),
            {
                name: row[f"positive_{name}"]
                for name in definitions
            },
            {
                regime: row[f"regime_{regime}_count"]
                for regime in (0, 1, 2)
            },
        )
    return {
        "sector_targets": output_path,
        "sector_summary": summary_path,
        "breadth_figure": figures[0],
        "regime_figure": figures[1],
    }
