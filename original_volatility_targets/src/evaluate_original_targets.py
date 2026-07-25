"""Aggregate locked predictions, build diagnostics, and print the final report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve

from src.compare_representations import (
    PROTOTYPES,
    metric_gain,
    metric_spec,
    write_comparison_outputs,
)
from src.utils import (
    atomic_write_csv,
    binary_metrics,
    load_representation_catalog,
    load_representation_frame,
    multiclass_metrics,
    project_path,
    read_table,
    regression_metrics,
    representation_feature_columns,
    uncertainty_metrics,
)


METADATA_COLUMNS = (
    "task_id",
    "stage",
    "target_family",
    "target",
    "representation",
    "representation_variant",
    "representation_variant_family",
    "input_variant",
    "model",
    "alpha",
    "fold",
    "seed",
    "news_scope",
    "threshold_mode",
    "quantile",
    "regime_definition",
    "distribution",
    "mean_design",
    "representation_fit_scope",
    "qualifies_for_robustness",
    "experiment_profile",
    "text_news_levels",
    "training_cohort",
    "evaluation_news_level",
    "evaluation_gate_representation",
    "primary_evaluation_cohort",
    "required_pooling",
    "text_feature_prefixes",
    "random_prototype_seed",
)


def evaluation_outputs(config: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    tables = "outputs/tables"
    figures = "outputs/figures"
    return {
        "aggregate": tuple(
            f"{tables}/{name}"
            for name in (
                "volatility_level_results_validation.csv",
                "volatility_level_results_test.csv",
                "volatility_spike_results_validation.csv",
                "volatility_spike_results_test.csv",
                "volatility_regime_results_validation.csv",
                "volatility_regime_results_test.csv",
                "volatility_uncertainty_results_validation.csv",
                "volatility_uncertainty_results_test.csv",
                "sector_target_results.csv",
                "ticker_level_results.csv",
                "news_level_results.csv",
            )
        ),
        "compare": tuple(
            f"{tables}/{name}"
            for name in (
                "placebo_results.csv",
                "robustness_results.csv",
                "comparison_with_residual_targets.csv",
                "final_original_target_decision.csv",
            )
        ),
        "figures": tuple(
            f"{figures}/{name}"
            for name in config["outputs"]["required_figures"]
            if name
            not in {
                "volatility_distribution.png",
                "volatility_by_ticker.png",
                "volatility_spike_rate.png",
                "volatility_regime_timeline.png",
                "sector_volatility_breadth.png",
                "sector_regime.png",
            }
        ),
        "report": tuple(
            f"{tables}/{name}" for name in config["outputs"]["required_tables"]
        ),
    }


def _prediction_files(config: Mapping[str, Any]) -> list[Path]:
    directory = project_path(config, "outputs", "checkpoints", "tasks")
    return sorted(directory.glob("*.parquet"))


def load_predictions(
    config: Mapping[str, Any], mode: str | None = None
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in _prediction_files(config):
        frame = read_table(path)
        if {
            "task_id",
            "target_family",
            "target",
            "evaluation_split",
            "y_true",
        }.issubset(frame.columns):
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(
            "No valid prediction checkpoints found in outputs/checkpoints/tasks"
        )
    output = pd.concat(frames, ignore_index=True, sort=False)
    if mode not in {None, "quick", "full"}:
        if "experiment_profile" not in output.columns:
            raise FileNotFoundError(
                f"No {mode} prediction checkpoints were found."
            )
        output = output.loc[
            output["experiment_profile"].astype(str).eq(mode)
        ].copy()
        if output.empty:
            raise FileNotFoundError(
                f"No prediction checkpoints belong to mode={mode}"
            )
    elif mode is not None and "quick" in output.columns:
        expected = mode == "quick"
        quick_values = output["quick"].astype(str).str.lower().map(
            {"true": True, "false": False, "1": True, "0": False}
        )
        output = output.loc[quick_values == expected].copy()
        if output.empty:
            raise FileNotFoundError(
                f"No prediction checkpoints belong to mode={mode}"
            )
    for column in ("feature_date", "target_date"):
        if column in output.columns:
            output[column] = pd.to_datetime(
                output[column],
                format="mixed",
                errors="raise",
            )
    return output


def _probability_matrix(group: pd.DataFrame) -> np.ndarray:
    columns = sorted(
        (
            column
            for column in group.columns
            if column.startswith("probability_class_")
        ),
        key=lambda value: int(value.rsplit("_", 1)[1]),
    )
    if not columns:
        raise KeyError("Multiclass predictions lack probability_class_* columns")
    return group[columns].to_numpy(dtype=float)


def _task_metrics(group: pd.DataFrame) -> dict[str, Any]:
    family = str(group["target_family"].iloc[0])
    y_true = group["y_true"].to_numpy()
    if family in {"level", "sector_level"}:
        metrics = regression_metrics(
            y_true.astype(float), group["prediction"].to_numpy(dtype=float)
        )
    elif family in {"spike", "sector_spike"}:
        metrics = binary_metrics(
            y_true.astype(int), group["probability"].to_numpy(dtype=float)
        )
    elif family in {"regime", "sector_regime"}:
        probabilities = _probability_matrix(group)
        metrics = multiclass_metrics(y_true.astype(int), probabilities)
        metrics.update(
            _transition_metrics(
                group,
                y_true.astype(int),
                np.argmax(probabilities, axis=1),
                probabilities.shape[1],
            )
        )
    elif family == "uncertainty":
        distribution = str(group["distribution"].iloc[0])
        metrics = uncertainty_metrics(
            y_true.astype(float),
            group["distribution_mean"].to_numpy(dtype=float),
            group["distribution_scale"].to_numpy(dtype=float),
            distribution,
            group["distribution_df"].to_numpy(dtype=float),
        )
    else:
        raise ValueError(f"Unsupported target family: {family}")
    row: dict[str, Any] = {
        column: group[column].iloc[0]
        for column in METADATA_COLUMNS
        if column in group.columns
    }
    row["evaluation_split"] = str(group["evaluation_split"].iloc[0])
    row.update(metrics)
    primary_metric, larger = metric_spec(family)
    row["primary_metric"] = primary_metric
    row["primary_value"] = metrics.get(primary_metric, np.nan)
    row["larger_is_better"] = larger
    row["first_target_date"] = (
        group["target_date"].min() if "target_date" in group else pd.NaT
    )
    row["last_target_date"] = (
        group["target_date"].max() if "target_date" in group else pd.NaT
    )
    return row


def _transition_metrics(
    frame: pd.DataFrame,
    actual: np.ndarray,
    predicted: np.ndarray,
    class_count: int,
) -> dict[str, str]:
    working = frame.reset_index(drop=True).copy()
    working["__actual"] = actual
    working["__predicted"] = predicted
    sort_columns = [
        column for column in ("ticker", "target_date") if column in working
    ]
    working = working.sort_values(sort_columns, kind="mergesort")
    actual_matrix = np.zeros((class_count, class_count), dtype=int)
    predicted_matrix = np.zeros((class_count, class_count), dtype=int)
    groups = (
        working.groupby("ticker", sort=False, observed=True)
        if "ticker" in working
        else [(None, working)]
    )
    for _, group in groups:
        true_values = group["__actual"].to_numpy(dtype=int)
        predicted_values = group["__predicted"].to_numpy(dtype=int)
        for left, right in zip(true_values[:-1], true_values[1:]):
            actual_matrix[left, right] += 1
        for left, right in zip(predicted_values[:-1], predicted_values[1:]):
            predicted_matrix[left, right] += 1
    return {
        "regime_transition_matrix_true": json.dumps(actual_matrix.tolist()),
        "regime_transition_matrix_predicted": json.dumps(
            predicted_matrix.tolist()
        ),
    }


def aggregate_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = [
        _task_metrics(group)
        for _, group in predictions.groupby(
            ["task_id", "evaluation_split"], sort=True, observed=True
        )
    ]
    metrics = pd.DataFrame(rows)
    if metrics.empty:
        return metrics
    metrics["seed"] = pd.to_numeric(metrics["seed"], errors="coerce")
    for index, row in metrics.iterrows():
        reference = metrics.loc[
            (metrics["evaluation_split"] == row["evaluation_split"])
            & (metrics["target"].astype(str) == str(row["target"]))
            & (metrics["fold"].astype(str) == str(row["fold"]))
            & (pd.to_numeric(metrics["seed"], errors="coerce") == row["seed"])
            & (metrics["representation"].astype(str) == "R0")
        ]
        if reference.empty:
            gain = np.nan
        else:
            reference = reference.sort_values(
                ["primary_value", "task_id"],
                ascending=[not bool(row["larger_is_better"]), True],
                kind="mergesort",
            ).iloc[0]
            gain = metric_gain(
                float(row["primary_value"]),
                float(reference["primary_value"]),
                bool(row["larger_is_better"]),
            )
        metrics.loc[index, "gain_vs_R0"] = gain
    return metrics


def _split_result_tables(
    metrics: pd.DataFrame, config: Mapping[str, Any]
) -> dict[str, Path]:
    tables = project_path(config, "outputs", "tables")
    mapping = {
        "level": "volatility_level_results",
        "spike": "volatility_spike_results",
        "regime": "volatility_regime_results",
        "uncertainty": "volatility_uncertainty_results",
    }
    paths: dict[str, Path] = {}
    for family, basename in mapping.items():
        for split in ("validation", "test"):
            subset = metrics.loc[
                (metrics["target_family"].astype(str) == family)
                & (metrics["evaluation_split"].astype(str) == split)
            ].copy()
            path = tables / f"{basename}_{split}.csv"
            paths[f"{family}_{split}"] = atomic_write_csv(
                subset, path, index=False
            )
    sector = metrics.loc[
        metrics["target_family"].astype(str).str.startswith("sector_")
    ].copy()
    paths["sector"] = atomic_write_csv(
        sector, tables / "sector_target_results.csv", index=False
    )
    return paths


def _best_validation_tasks(metrics: pd.DataFrame) -> pd.DataFrame:
    validation = metrics.loc[
        (metrics["evaluation_split"].astype(str) == "validation")
        & (metrics["fold"].astype(str) == "holdout")
    ].copy()
    rows = []
    for _, group in validation.groupby(
        ["target", "representation"], sort=True, observed=True
    ):
        larger = bool(group["larger_is_better"].iloc[0])
        rows.append(
            group.sort_values(
                ["primary_value", "task_id"],
                ascending=[not larger, True],
                kind="mergesort",
            ).iloc[0]
        )
    return pd.DataFrame(rows)


def _ticker_results(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    config: Mapping[str, Any],
) -> Path:
    chosen = _best_validation_tasks(metrics)
    chosen_ids = set(chosen["task_id"].astype(str)) if not chosen.empty else set()
    subset = predictions.loc[
        predictions["task_id"].astype(str).isin(chosen_ids)
        & predictions["ticker"].astype(str).isin(config["universe"]["tickers"])
    ].copy()
    market = read_table(
        project_path(
            config, "data", "processed", "original_market_targets.parquet"
        )
    )
    subset = subset.merge(
        market[
            [
                "ticker",
                "feature_date",
                "vol_spike_q90_ticker",
            ]
        ],
        on=["ticker", "feature_date"],
        how="left",
        validate="many_to_one",
    )
    rows = []
    for _, group in subset.groupby(
        ["task_id", "evaluation_split", "ticker"],
        sort=True,
        observed=True,
    ):
        row = _task_metrics(group)
        row["ticker"] = str(group["ticker"].iloc[0])
        row["condition"] = "all"
        rows.append(row)
    level = subset.loc[subset["target_family"].astype(str) == "level"].copy()
    level["condition"] = np.where(
        level["vol_spike_q90_ticker"].fillna(0).astype(bool),
        "spike_q90_day",
        "normal_day",
    )
    for _, group in level.groupby(
        ["task_id", "evaluation_split", "ticker", "condition"],
        sort=True,
        observed=True,
    ):
        if len(group) < 5:
            continue
        row = _task_metrics(group)
        row["ticker"] = str(group["ticker"].iloc[0])
        row["condition"] = str(group["condition"].iloc[0])
        rows.append(row)
    return atomic_write_csv(
        pd.DataFrame(rows),
        project_path(config, "outputs", "tables", "ticker_level_results.csv"),
        index=False,
    )


def _selected_r7_row(config: Mapping[str, Any]) -> pd.Series | None:
    catalog = load_representation_catalog(config)
    subset = catalog.loc[
        (catalog["representation"].astype(str) == "R7")
        & catalog["selected"].fillna(False).astype(bool)
    ]
    return None if subset.empty else subset.iloc[0]


def _news_level_results(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    config: Mapping[str, Any],
) -> Path:
    r7 = _selected_r7_row(config)
    rows: list[dict[str, Any]] = []
    if r7 is not None:
        features = load_representation_frame(
            config, r7, seed=int(config["project"]["seed"])
        )
        feature_columns = representation_feature_columns(features)
        stock_predictions = predictions.loc[
            predictions["representation"].astype(str) == "R7"
        ].copy()
        joined = stock_predictions.merge(
            features[["ticker", "feature_date", *feature_columns]],
            on=["ticker", "feature_date"],
            how="left",
            validate="many_to_one",
        )
        for level in ("macro", "sector", "related", "target"):
            level_columns = [
                column
                for column in feature_columns
                if f"__{level}__" in column
                or column.startswith(f"meta__{level}__")
            ]
            count_columns = [
                column
                for column in level_columns
                if "count" in column.lower() or "mask" in column.lower()
            ]
            signal_columns = count_columns or level_columns
            if not signal_columns:
                continue
            joined[f"__has_{level}"] = (
                joined[signal_columns].fillna(0.0).abs().sum(axis=1) > 0
            )
            for _, group in joined.groupby(
                ["task_id", "evaluation_split", f"__has_{level}"],
                sort=True,
                observed=True,
            ):
                if len(group) < 5:
                    continue
                row = _task_metrics(group)
                row.update(
                    {
                        "news_level": level,
                        "has_news": bool(group[f"__has_{level}"].iloc[0]),
                        "sample_count": len(group),
                    }
                )
                rows.append(row)
    return atomic_write_csv(
        pd.DataFrame(rows),
        project_path(config, "outputs", "tables", "news_level_results.csv"),
        index=False,
    )


def run_aggregate(
    config: Mapping[str, Any], mode: str
) -> dict[str, Path]:
    predictions = load_predictions(config, mode)
    metrics = aggregate_metrics(predictions)
    paths = _split_result_tables(metrics, config)
    paths["ticker"] = _ticker_results(predictions, metrics, config)
    paths["news"] = _news_level_results(predictions, metrics, config)
    return paths


def _combined_metric_tables(config: Mapping[str, Any]) -> pd.DataFrame:
    tables = project_path(config, "outputs", "tables")
    frames = []
    for name in (
        "volatility_level_results_validation.csv",
        "volatility_level_results_test.csv",
        "volatility_spike_results_validation.csv",
        "volatility_spike_results_test.csv",
        "volatility_regime_results_validation.csv",
        "volatility_regime_results_test.csv",
        "volatility_uncertainty_results_validation.csv",
        "volatility_uncertainty_results_test.csv",
        "sector_target_results.csv",
    ):
        path = tables / name
        if path.exists() and path.stat().st_size:
            frame = read_table(path)
            if not frame.empty:
                frames.append(frame)
    if not frames:
        raise FileNotFoundError("No aggregate metric tables are available")
    return pd.concat(frames, ignore_index=True, sort=False)


def run_compare(config: Mapping[str, Any], mode: str) -> dict[str, Path]:
    metrics = _combined_metric_tables(config)
    validation = metrics.loc[
        metrics["evaluation_split"].astype(str) == "validation"
    ].copy()
    test = metrics.loc[
        metrics["evaluation_split"].astype(str) == "test"
    ].copy()
    return write_comparison_outputs(validation, test, config, mode)


def _save_or_placeholder(
    path: Path,
    title: str,
    draw: Any | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8, 5))
    if draw is None:
        axis.text(
            0.5,
            0.5,
            "No eligible completed task",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_axis_off()
    else:
        draw(axis)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _best_prediction_group(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    family: str,
    split: str = "test",
) -> pd.DataFrame:
    validation = metrics.loc[
        (metrics["target_family"].astype(str) == family)
        & (metrics["evaluation_split"].astype(str) == "validation")
        & (metrics["fold"].astype(str) == "holdout")
        & metrics["representation"].astype(str).isin(PROTOTYPES)
    ]
    if validation.empty:
        return pd.DataFrame()
    larger = bool(validation["larger_is_better"].iloc[0])
    selected = validation.sort_values(
        ["primary_value", "task_id"],
        ascending=[not larger, True],
        kind="mergesort",
    ).iloc[0]
    return predictions.loc[
        (predictions["task_id"].astype(str) == str(selected["task_id"]))
        & (predictions["evaluation_split"].astype(str) == split)
    ].copy()


def run_figures(
    config: Mapping[str, Any], mode: str
) -> dict[str, Path]:
    predictions = load_predictions(config, mode)
    metrics = aggregate_metrics(predictions)
    figures = project_path(config, "outputs", "figures")
    paths: dict[str, Path] = {}

    spike = _best_prediction_group(predictions, metrics, "spike")
    paths["spike_pr_curve"] = _save_or_placeholder(
        figures / "spike_pr_curve.png",
        "Locked test precision-recall curve",
        None
        if spike.empty or spike["y_true"].nunique() < 2
        else lambda ax: _draw_pr(ax, spike),
    )
    paths["calibration"] = _save_or_placeholder(
        figures / "calibration_curve.png",
        "Locked test spike calibration",
        None
        if spike.empty or spike["y_true"].nunique() < 2
        else lambda ax: _draw_calibration(ax, spike),
    )
    regime = _best_prediction_group(predictions, metrics, "regime")
    paths["confusion"] = _save_or_placeholder(
        figures / "confusion_matrix.png",
        "Locked test regime confusion matrix",
        None if regime.empty else lambda ax: _draw_confusion(ax, regime),
    )
    uncertainty = _best_prediction_group(predictions, metrics, "uncertainty")
    paths["pit"] = _save_or_placeholder(
        figures / "pit_histogram.png",
        "Locked test PIT histogram",
        None if uncertainty.empty else lambda ax: _draw_pit(ax, uncertainty),
    )
    paths["coverage"] = _save_or_placeholder(
        figures / "prediction_interval_coverage.png",
        "Locked test prediction interval coverage",
        None
        if uncertainty.empty
        else lambda ax: _draw_coverage(ax, uncertainty),
    )
    paths["representations"] = _save_or_placeholder(
        figures / "price_vs_prototype.png",
        "Validation performance: price versus prototype",
        None if metrics.empty else lambda ax: _draw_representation(ax, metrics),
    )
    paths["shuffled"] = _save_or_placeholder(
        figures / "true_vs_shuffled_news.png",
        "Validation gain: true versus shuffled news",
        _comparison_drawer(
            project_path(config, "outputs", "tables", "placebo_results.csv"),
            "reference_representation",
            "validation_gain",
        ),
    )
    paths["residual"] = _save_or_placeholder(
        figures / "original_vs_residual.png",
        "Original-target versus residual-target gain",
        _residual_drawer(
            project_path(
                config,
                "outputs",
                "tables",
                "comparison_with_residual_targets.csv",
            )
        ),
    )
    paths["progress"] = _save_or_placeholder(
        figures / "progress_history.png",
        "Pipeline completion history",
        _progress_drawer(
            project_path(config, str(config["progress"]["history_file"]))
        ),
    )
    return paths


def _draw_pr(axis: Any, frame: pd.DataFrame) -> None:
    precision, recall, _ = precision_recall_curve(
        frame["y_true"].astype(int), frame["probability"].astype(float)
    )
    axis.plot(recall, precision)
    axis.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
    axis.grid(alpha=0.3)


def _draw_calibration(axis: Any, frame: pd.DataFrame) -> None:
    observed, predicted = calibration_curve(
        frame["y_true"].astype(int),
        frame["probability"].astype(float),
        n_bins=8,
        strategy="quantile",
    )
    axis.plot(predicted, observed, marker="o", label="model")
    axis.plot([0, 1], [0, 1], linestyle="--", color="black", label="ideal")
    axis.set(xlabel="Mean predicted probability", ylabel="Observed frequency")
    axis.legend()
    axis.grid(alpha=0.3)


def _draw_confusion(axis: Any, frame: pd.DataFrame) -> None:
    probabilities = _probability_matrix(frame)
    actual = frame["y_true"].astype(int).to_numpy()
    predicted = np.argmax(probabilities, axis=1)
    size = probabilities.shape[1]
    matrix = np.zeros((size, size), dtype=int)
    for left, right in zip(actual, predicted):
        matrix[left, right] += 1
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(size):
        for column in range(size):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    axis.set(xlabel="Predicted regime", ylabel="True regime")
    plt.colorbar(image, ax=axis, fraction=0.046)


def _pit_values(frame: pd.DataFrame) -> np.ndarray:
    from scipy import stats

    standardized = (
        frame["y_true"].to_numpy(dtype=float)
        - frame["distribution_mean"].to_numpy(dtype=float)
    ) / frame["distribution_scale"].to_numpy(dtype=float)
    if str(frame["distribution"].iloc[0]) == "student_t":
        return stats.t.cdf(
            standardized, df=frame["distribution_df"].to_numpy(dtype=float)
        )
    return stats.norm.cdf(standardized)


def _draw_pit(axis: Any, frame: pd.DataFrame) -> None:
    axis.hist(_pit_values(frame), bins=10, range=(0, 1), density=True, alpha=0.8)
    axis.axhline(1.0, color="black", linestyle="--")
    axis.set(xlabel="PIT", ylabel="Density")


def _draw_coverage(axis: Any, frame: pd.DataFrame) -> None:
    from scipy import stats

    y = frame["y_true"].to_numpy(dtype=float)
    mean = frame["distribution_mean"].to_numpy(dtype=float)
    scale = frame["distribution_scale"].to_numpy(dtype=float)
    student = str(frame["distribution"].iloc[0]) == "student_t"
    levels = np.asarray([0.80, 0.90, 0.95, 0.99])
    observed = []
    for level in levels:
        alpha = (1 - level) / 2
        if student:
            df = frame["distribution_df"].to_numpy(dtype=float)
            lower = mean + scale * stats.t.ppf(alpha, df=df)
            upper = mean + scale * stats.t.ppf(1 - alpha, df=df)
        else:
            lower = mean + scale * stats.norm.ppf(alpha)
            upper = mean + scale * stats.norm.ppf(1 - alpha)
        observed.append(np.mean((y >= lower) & (y <= upper)))
    axis.plot(levels, observed, marker="o", label="observed")
    axis.plot(levels, levels, linestyle="--", color="black", label="nominal")
    axis.set(xlabel="Nominal coverage", ylabel="Observed coverage", ylim=(0, 1))
    axis.legend()
    axis.grid(alpha=0.3)


def _draw_representation(axis: Any, metrics: pd.DataFrame) -> None:
    subset = metrics.loc[
        (metrics["evaluation_split"].astype(str) == "validation")
        & (metrics["fold"].astype(str) == "holdout")
        & metrics["representation"].astype(str).isin(["R0", *PROTOTYPES])
    ]
    grouped = subset.groupby("representation", observed=True)["gain_vs_R0"].max()
    if grouped.empty:
        axis.text(0.5, 0.5, "No eligible comparison", ha="center", va="center")
        axis.set_axis_off()
        return
    grouped.plot.bar(ax=axis)
    axis.axhline(0, color="black", linewidth=1)
    axis.set(xlabel="Representation", ylabel="Best relative gain vs R0")


def _comparison_drawer(path: Path, key: str, value: str) -> Any | None:
    if not path.exists() or not path.stat().st_size:
        return None
    frame = read_table(path)
    if frame.empty or not {key, value}.issubset(frame.columns):
        return None

    def draw(axis: Any) -> None:
        grouped = frame.groupby(key, observed=True)[value].mean().sort_values()
        grouped.plot.barh(ax=axis)
        axis.axvline(0, color="black", linewidth=1)
        axis.set(xlabel=value, ylabel=key)

    return draw


def _residual_drawer(path: Path) -> Any | None:
    if not path.exists() or not path.stat().st_size:
        return None
    frame = read_table(path)
    if frame.empty:
        return None

    def draw(axis: Any) -> None:
        positions = np.arange(len(frame))
        axis.bar(positions - 0.18, frame["original_gain"], 0.36, label="original")
        axis.bar(positions + 0.18, frame["residual_gain"], 0.36, label="residual")
        axis.set_xticks(positions, frame["original_target"], rotation=30, ha="right")
        axis.axhline(0, color="black", linewidth=1)
        axis.legend()
        axis.set_ylabel("Validation gain")

    return draw


def _progress_drawer(path: Path) -> Any | None:
    if not path.exists() or not path.stat().st_size:
        return None
    frame = read_table(path)
    if frame.empty or "overall_percent" not in frame:
        return None

    def draw(axis: Any) -> None:
        elapsed = pd.to_numeric(frame["elapsed_seconds"], errors="coerce") / 60.0
        percent = pd.to_numeric(frame["overall_percent"], errors="coerce")
        axis.plot(elapsed, percent)
        axis.set(xlabel="Elapsed minutes", ylabel="Overall completion (%)", ylim=(0, 105))
        axis.grid(alpha=0.3)

    return draw


def run_report(config: Mapping[str, Any], mode: str) -> dict[str, Path]:
    tables = project_path(config, "outputs", "tables")
    figures = project_path(config, "outputs", "figures")
    missing_tables = [
        name
        for name in config["outputs"]["required_tables"]
        if not (tables / name).is_file() or (tables / name).stat().st_size == 0
    ]
    missing_figures = [
        name
        for name in config["outputs"]["required_figures"]
        if not (figures / name).is_file() or (figures / name).stat().st_size == 0
    ]
    if missing_tables or missing_figures:
        raise FileNotFoundError(
            f"Missing required outputs; tables={missing_tables}, figures={missing_figures}"
        )
    decision = read_table(tables / "final_original_target_decision.csv")
    final = decision.loc[decision["record_type"].astype(str) == "final"]
    if len(final) != 1:
        raise AssertionError("Decision table must contain exactly one final row")
    row = final.iloc[0]
    summary = {
        "mode": mode,
        "selected_target": row.get("target", "none"),
        "decision": row.get("decision", "NO-GO"),
        "prototype_better_than_price_only": row.get(
            "prototype_better_than_price_only", False
        ),
        "prototype_better_than_raw_embedding": row.get(
            "prototype_better_than_raw_embedding", False
        ),
        "prototype_better_than_pca_random_projection": row.get(
            "prototype_better_than_pca_random_projection", False
        ),
        "true_news_better_than_shuffled": row.get(
            "true_news_better_than_shuffled", False
        ),
        "stable_across_folds_seeds": row.get(
            "stable_across_folds_seeds", False
        ),
        "note": (
            "Quick mode is a smoke test and is not eligible for a GO decision."
            if mode == "quick"
            else "Full mode applies validation, locked-test, placebo and robustness gates."
        ),
    }
    report_path = tables / "final_report.json"
    report_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print("\nFINAL ORIGINAL-VOLATILITY TARGET REPORT")
    print(f"1. Mode: {mode}")
    print(f"2. Best-evidence target: {summary['selected_target']}")
    print(
        "3. Prototype better than price-only: "
        f"{summary['prototype_better_than_price_only']}"
    )
    print(
        "4. Prototype better than raw/PCA/random: "
        f"{summary['prototype_better_than_raw_embedding']} / "
        f"{summary['prototype_better_than_pca_random_projection']}"
    )
    print(
        "5. True news better than shuffled: "
        f"{summary['true_news_better_than_shuffled']}"
    )
    print(
        "6. Stable across folds/seeds: "
        f"{summary['stable_across_folds_seeds']}"
    )
    print(f"7. Final decision: {summary['decision']}")
    print(f"8. Note: {summary['note']}")
    return {"report": report_path}


def run_action(
    action: str,
    config: Mapping[str, Any],
    *,
    mode: str,
) -> dict[str, Path]:
    if action == "aggregate":
        return run_aggregate(config, mode)
    if action == "compare":
        return run_compare(config, mode)
    if action == "figures":
        return run_figures(config, mode)
    if action == "report":
        return run_report(config, mode)
    raise ValueError(f"Unknown evaluation action: {action}")
