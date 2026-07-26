from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _heatmap(matrix: pd.DataFrame, title: str, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    axis.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    axis.set_yticks(range(len(matrix.index)), matrix.index)
    axis.set_title(title)
    fig.colorbar(image, ax=axis, label="Correlation")
    _save(fig, path)


def generate_figures(
    *,
    target: pd.DataFrame,
    predictions: pd.DataFrame,
    overlapping: pd.DataFrame,
    offsets: pd.DataFrame,
    daily_ic: pd.DataFrame,
    paired_daily: pd.DataFrame,
    bootstrap: pd.DataFrame,
    acf_table: pd.DataFrame,
    correlation_before: pd.DataFrame,
    correlation_after: pd.DataFrame,
    output_directory: Path,
) -> list[Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.hist(target["base_target_t_plus_1"], bins=50, alpha=0.55, label="One-day base")
    axis.hist(target["t1_target"], bins=50, alpha=0.55, label="T1 sector-relative")
    axis.set_title("Base and T1 target distributions")
    axis.legend()
    path = output_directory / "target_distributions.png"
    _save(fig, path)
    paths.append(path)

    for series, filename, title in [
        ("base_target_t_plus_1", "acf_base_target.png", "ACF: one-day base target"),
        ("forward_mean_5d", "acf_forward_5d.png", "ACF: five-day forward mean"),
        ("t1_target", "acf_t1_target.png", "ACF: sector-relative T1"),
    ]:
        subset = acf_table[acf_table["series"].eq(series)]
        fig, axis = plt.subplots(figsize=(8, 4))
        axis.bar(subset["lag"], subset["acf"])
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(title)
        axis.set_xlabel("Lag")
        path = output_directory / filename
        _save(fig, path)
        paths.append(path)

    before_path = output_directory / "correlation_before_demean.png"
    after_path = output_directory / "correlation_after_demean.png"
    _heatmap(correlation_before, "Correlation before sector neutralization", before_path)
    _heatmap(correlation_after, "Correlation after sector neutralization", after_path)
    paths.extend([before_path, after_path])

    test_metrics = overlapping[overlapping["split"].eq("test")]
    summary = test_metrics.groupby("model_name")[["mae", "rmse"]].mean()
    fig, axis = plt.subplots(figsize=(9, 5))
    summary.plot(kind="bar", ax=axis)
    axis.set_title("Test MAE and RMSE by model")
    axis.tick_params(axis="x", rotation=25)
    path = output_directory / "model_mae_rmse.png"
    _save(fig, path)
    paths.append(path)

    test = predictions[predictions["split"].eq("test")]
    ticker_rows = []
    if {"M0_PRICE", "M2_PRICE_SEMANTIC"}.issubset(set(test["model_name"])):
        for (seed, ticker), group in test.groupby(["seed", "ticker"]):
            base = group[group["model_name"].eq("M0_PRICE")]
            text = group[group["model_name"].eq("M2_PRICE_SEMANTIC")]
            merged = base.merge(text, on=["date", "ticker", "seed"])
            ticker_rows.append(
                {
                    "ticker": ticker,
                    "delta_mae": np.abs(
                        merged["actual_t1_x"] - merged["prediction_x"]
                    ).mean()
                    - np.abs(
                        merged["actual_t1_y"] - merged["prediction_y"]
                    ).mean(),
                }
            )
    ticker_frame = pd.DataFrame(ticker_rows)
    fig, axis = plt.subplots(figsize=(9, 5))
    if not ticker_frame.empty:
        ticker_frame.groupby("ticker")["delta_mae"].mean().plot(kind="bar", ax=axis)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_title("Semantic increment by ticker (positive is better)")
    path = output_directory / "semantic_increment_by_ticker.png"
    _save(fig, path)
    paths.append(path)

    fig, axis = plt.subplots(figsize=(10, 5))
    for model, group in daily_ic[daily_ic["split"].eq("test")].groupby("model_name"):
        curve = group.groupby("date")["daily_ic"].mean()
        axis.plot(curve.index, curve.rolling(10, min_periods=1).mean(), label=model)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_title("Daily cross-sectional IC (10-date rolling mean)")
    axis.legend(fontsize=7)
    path = output_directory / "daily_cross_sectional_ic.png"
    _save(fig, path)
    paths.append(path)

    fig, axis = plt.subplots(figsize=(10, 5))
    for comparison, group in paired_daily[paired_daily["split"].eq("test")].groupby(
        "comparison"
    ):
        curve = group.groupby("date")["absolute_loss_difference"].mean().cumsum()
        axis.plot(curve.index, curve, label=comparison)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_title("Cumulative daily paired absolute-loss difference")
    axis.legend()
    path = output_directory / "cumulative_paired_loss_difference.png"
    _save(fig, path)
    paths.append(path)

    fig, axis = plt.subplots(figsize=(9, 5))
    boot = bootstrap[bootstrap["metric"].eq("delta_mae")].copy()
    if not boot.empty:
        positions = np.arange(len(boot))
        lower = boot["bootstrap_mean"] - boot["ci_lower"]
        upper = boot["ci_upper"] - boot["bootstrap_mean"]
        axis.errorbar(
            positions,
            boot["bootstrap_mean"],
            yerr=np.vstack([lower, upper]),
            fmt="o",
        )
        axis.set_xticks(
            positions,
            [
                f"{row.comparison}\ns{row.seed}/b{row.block_length}"
                for row in boot.itertuples()
            ],
            rotation=45,
            ha="right",
        )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_title("Block-bootstrap ΔMAE summaries")
    path = output_directory / "bootstrap_distribution.png"
    _save(fig, path)
    paths.append(path)

    fig, axis = plt.subplots(figsize=(9, 5))
    test_offsets = offsets[offsets["split"].eq("test")]
    for model, group in test_offsets.groupby("model_name"):
        curve = group.groupby("offset")["mae"].mean()
        axis.plot(curve.index, curve, marker="o", label=model)
    axis.set_title("Non-overlapping MAE across five offsets")
    axis.set_xticks(sorted(test_offsets["offset"].unique()))
    axis.legend(fontsize=7)
    path = output_directory / "results_by_offset.png"
    _save(fig, path)
    paths.append(path)

    sample = test.sort_values(["date", "ticker"]).head(220)
    fig, axis = plt.subplots(figsize=(10, 5))
    if not sample.empty:
        model = sorted(sample["model_name"].unique())[-1]
        one = sample[sample["model_name"].eq(model)].head(80)
        axis.plot(range(len(one)), one["actual_t1"], label="Actual")
        axis.plot(range(len(one)), one["prediction"], label=model)
    axis.set_title("Actual and predicted T1 on sample observations")
    axis.legend()
    path = output_directory / "actual_vs_prediction_sample.png"
    _save(fig, path)
    paths.append(path)

    counts = target.groupby("date")["ticker"].nunique()
    fig, axis = plt.subplots(figsize=(10, 4))
    axis.plot(counts.index, counts.values)
    axis.set_title("Valid ticker count by anchor date")
    axis.set_ylim(0, max(12, counts.max() + 1))
    path = output_directory / "valid_tickers_by_date.png"
    _save(fig, path)
    paths.append(path)
    return paths
