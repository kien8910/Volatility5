"""Evaluate locked target models and make a validation-led research decision.

The training manifest stores scalar scores for every validation candidate.
Row-level prediction artifacts contain only configurations locked by validation,
and test rows are used only to report their already-locked performance.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import special, stats
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    ndcg_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.utils import (
    atomic_write_csv,
    ensure_directories,
    expected_calibration_error,
    get_logger,
    load_config,
    project_path,
    qlike,
    regression_metrics,
    safe_read_table,
    validate_required_columns,
)

PROTOTYPE_REPRESENTATIONS = ("R5", "R6", "R7", "R8")
PLACEBO_REPRESENTATIONS = ("R9", "R10", "R11")
REFERENCE_REPRESENTATIONS = ("R0", "R2", "R3", "R4")
CONFIGURATION_COLUMNS = (
    "target",
    "representation",
    "representation_variant",
    "representation_variant_family",
    "prototype_seed",
    "input_variant",
    "model",
    "config_id",
    "seed",
)
TARGET_TABLE_COLUMNS = (
    "evaluation_split",
    "target",
    "representation",
    "input_variant",
    "model",
    "config_id",
    "seed",
    "fold",
    "selected_on_validation",
    "n",
    "primary_metric",
    "primary_value",
)
TICKER_TABLE_COLUMNS = (*TARGET_TABLE_COLUMNS, "ticker")
NEWS_TABLE_COLUMNS = (*TARGET_TABLE_COLUMNS, "news_level", "news_day_state")


def primary_metric_spec(target: str) -> tuple[str, bool]:
    """Return the primary metric name and whether larger values are better."""
    if target == "signed":
        return "final_qlike", False
    if target in {"magnitude", "squared"}:
        return "spearman", True
    if target.startswith("spike"):
        return "pr_auc", True
    if target.startswith("regime"):
        return "macro_f1", True
    if target == "uncertainty":
        return "nll", False
    raise ValueError(f"Unsupported target: {target}")


def configuration_columns(frame: pd.DataFrame) -> list[str]:
    """Columns that uniquely identify a fitted/evaluated configuration."""
    columns = [column for column in CONFIGURATION_COLUMNS if column in frame.columns]
    if "fold" in frame.columns:
        columns.append("fold")
    return columns


def attach_variant_metadata(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    logger: Any | None = None,
) -> pd.DataFrame:
    """Attach prototype family/seed metadata without treating seed as a choice."""
    output = frame.copy()
    if {
        "representation_variant_family",
        "prototype_seed",
    }.issubset(output.columns):
        return output
    manifest_path = _first_existing(
        (
            project_path(config, "data", "processed", "representation_manifest.csv"),
            project_path(config, "outputs", "tables", "representation_manifest.csv"),
        )
    )
    mapping = pd.DataFrame()
    if manifest_path is not None:
        manifest = safe_read_table(manifest_path)
        if {"representation", "representation_variant"}.issubset(manifest.columns):
            columns = [
                "representation",
                "representation_variant",
                *[
                    column
                    for column in (
                        "representation_variant_family",
                        "prototype_seed",
                    )
                    if column in manifest.columns
                ],
            ]
            mapping = manifest[columns].drop_duplicates(
                ["representation", "representation_variant"], keep="first"
            )
    if not mapping.empty:
        output = output.merge(
            mapping,
            on=["representation", "representation_variant"],
            how="left",
            validate="many_to_one",
            suffixes=("", "_manifest"),
        )
    if "representation_variant_family" not in output.columns:
        output["representation_variant_family"] = np.nan
    if "prototype_seed" not in output.columns:
        output["prototype_seed"] = np.nan
    variant = output["representation_variant"].fillna("selected_default").astype(str)
    parsed_family = variant.str.replace(r"_seed-?\d+$", "", regex=True)
    parsed_seed = pd.to_numeric(
        variant.str.extract(r"_seed(-?\d+)$", expand=False), errors="coerce"
    )
    output["representation_variant_family"] = output[
        "representation_variant_family"
    ].fillna(parsed_family)
    output["prototype_seed"] = pd.to_numeric(
        output["prototype_seed"], errors="coerce"
    ).fillna(parsed_seed)
    default_seed = int(config["project"]["seed"])
    default_mask = output["representation_variant_family"].eq("selected_default")
    output.loc[default_mask, "prototype_seed"] = output.loc[
        default_mask, "prototype_seed"
    ].fillna(default_seed)
    unresolved = output["prototype_seed"].isna() & output[
        "representation"
    ].isin(PROTOTYPE_REPRESENTATIONS)
    if unresolved.any() and logger is not None:
        logger.warning(
            "%d prototype prediction rows lack prototype_seed metadata",
            int(unresolved.sum()),
        )
    return output


def _finite_pair(
    truth: Sequence[float], prediction: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(truth, dtype=float)
    y_pred = np.asarray(prediction, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[valid], y_pred[valid]


def _directional_accuracy(
    truth: Sequence[float], prediction: Sequence[float]
) -> float:
    y_true, y_pred = _finite_pair(truth, prediction)
    if not len(y_true):
        return float("nan")
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def _ranking_metrics(
    truth: Sequence[float],
    prediction: Sequence[float],
    top_fraction: float,
) -> dict[str, float]:
    y_true, y_pred = _finite_pair(truth, prediction)
    if not len(y_true):
        return {"top_fraction_recall": np.nan, "ndcg": np.nan}
    count = max(1, int(math.ceil(len(y_true) * top_fraction)))
    true_top = set(np.argpartition(y_true, -count)[-count:].tolist())
    predicted_top = set(np.argpartition(y_pred, -count)[-count:].tolist())
    top_recall = len(true_top.intersection(predicted_top)) / count
    # NDCG requires non-negative relevance. Ranking targets are monotone under
    # shifting, so this does not change the desired order.
    relevance = y_true - np.min(y_true)
    if np.allclose(relevance, 0.0):
        ranking = np.nan
    else:
        ranking = float(ndcg_score(relevance[None, :], y_pred[None, :]))
    return {"top_fraction_recall": float(top_recall), "ndcg": ranking}


def _binary_metrics(
    truth: Sequence[int],
    probability: Sequence[float],
    fixed_precision_levels: Sequence[float],
) -> dict[str, float]:
    labels = np.asarray(truth, dtype=int)
    probabilities = np.asarray(probability, dtype=float)
    valid = np.isfinite(probabilities) & np.isfinite(labels)
    labels = labels[valid]
    probabilities = np.clip(probabilities[valid], 0.0, 1.0)
    if not len(labels):
        return {
            "pr_auc": np.nan,
            "roc_auc": np.nan,
            "macro_f1": np.nan,
            "balanced_accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "brier": np.nan,
            "ece": np.nan,
        }
    predictions = (probabilities >= 0.5).astype(int)
    two_classes = np.unique(labels).size == 2
    result = {
        "pr_auc": (
            float(average_precision_score(labels, probabilities))
            if two_classes
            else np.nan
        ),
        "roc_auc": (
            float(roc_auc_score(labels, probabilities)) if two_classes else np.nan
        ),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(
            precision_recall_fscore_support(
                labels, predictions, average="binary", zero_division=0
            )[0]
        ),
        "recall": float(
            precision_recall_fscore_support(
                labels, predictions, average="binary", zero_division=0
            )[1]
        ),
        "brier": float(brier_score_loss(labels, probabilities)),
        "ece": expected_calibration_error(labels, probabilities),
    }
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    result.update(
        {
            "confusion_tn": int(matrix[0, 0]),
            "confusion_fp": int(matrix[0, 1]),
            "confusion_fn": int(matrix[1, 0]),
            "confusion_tp": int(matrix[1, 1]),
        }
    )
    if two_classes:
        precision_curve, recall_curve, _ = precision_recall_curve(
            labels, probabilities
        )
        for level in fixed_precision_levels:
            eligible = precision_curve >= float(level)
            key = f"recall_at_precision_{int(round(float(level) * 100)):02d}"
            result[key] = (
                float(np.max(recall_curve[eligible])) if eligible.any() else 0.0
            )
    else:
        for level in fixed_precision_levels:
            key = f"recall_at_precision_{int(round(float(level) * 100)):02d}"
            result[key] = np.nan
    return result


def _multiclass_ece(labels: np.ndarray, probabilities: np.ndarray) -> float:
    confidence = probabilities.max(axis=1)
    correctness = (probabilities.argmax(axis=1) == labels).astype(int)
    return expected_calibration_error(correctness, confidence)


def _regime_metrics(frame: pd.DataFrame) -> dict[str, float]:
    probability_columns = [f"prob_{class_id}" for class_id in range(3)]
    validate_required_columns(frame, ["y_true", *probability_columns], "regime predictions")
    labels = pd.to_numeric(frame["y_true"], errors="coerce").to_numpy(dtype=float)
    probabilities = frame[probability_columns].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)
    valid = np.isfinite(labels) & np.isfinite(probabilities).all(axis=1)
    labels = labels[valid].astype(int)
    probabilities = probabilities[valid]
    if not len(labels):
        return {
            "macro_f1": np.nan,
            "balanced_accuracy": np.nan,
            "multiclass_brier": np.nan,
            "multiclass_ece": np.nan,
        }
    row_sums = probabilities.sum(axis=1, keepdims=True)
    probabilities = probabilities / np.clip(row_sums, 1.0e-12, None)
    predictions = probabilities.argmax(axis=1)
    precision, recall, _, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=[0, 1, 2],
        zero_division=0,
    )
    one_hot = np.eye(3, dtype=float)[labels]
    matrix = confusion_matrix(labels, predictions, labels=[0, 1, 2])
    result: dict[str, float] = {
        "macro_f1": float(
            f1_score(labels, predictions, labels=[0, 1, 2], average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "multiclass_brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "multiclass_ece": _multiclass_ece(labels, probabilities),
    }
    for class_id in range(3):
        result[f"class_{class_id}_precision"] = float(precision[class_id])
        result[f"class_{class_id}_recall"] = float(recall[class_id])
        result[f"class_{class_id}_support"] = int(support[class_id])
        for predicted_id in range(3):
            result[f"confusion_{class_id}_{predicted_id}"] = int(
                matrix[class_id, predicted_id]
            )
    return result


def _gaussian_crps(z: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return scale * (
        z * (2.0 * stats.norm.cdf(z) - 1.0)
        + 2.0 * stats.norm.pdf(z)
        - 1.0 / np.sqrt(np.pi)
    )


def _student_t_crps(
    z: np.ndarray, scale: np.ndarray, degrees_of_freedom: np.ndarray
) -> np.ndarray:
    """Closed-form CRPS for scipy's location-scale Student-t parameterization."""
    nu = degrees_of_freedom
    first = z * (2.0 * stats.t.cdf(z, df=nu) - 1.0)
    second = (
        2.0
        * stats.t.pdf(z, df=nu)
        * (nu + np.square(z))
        / np.clip(nu - 1.0, 1.0e-12, None)
    )
    beta_ratio = np.exp(
        special.betaln(0.5, nu - 0.5)
        - 2.0 * special.betaln(0.5, nu / 2.0)
    )
    constant = (
        2.0
        * np.sqrt(nu)
        * beta_ratio
        / np.clip(nu - 1.0, 1.0e-12, None)
    )
    return scale * (first + second - constant)


def _xlog_probability(count: int, probability: float) -> float:
    if count == 0:
        return 0.0
    if probability <= 0.0:
        return -np.inf
    return float(count * np.log(probability))


def kupiec_test(
    exceedances: Sequence[bool], expected_failure_probability: float
) -> tuple[float, float]:
    failures = np.asarray(exceedances, dtype=bool)
    count = len(failures)
    if count == 0 or not 0.0 < expected_failure_probability < 1.0:
        return np.nan, np.nan
    observed = int(failures.sum())
    observed_probability = observed / count
    null_log_likelihood = _xlog_probability(
        observed, expected_failure_probability
    ) + _xlog_probability(count - observed, 1.0 - expected_failure_probability)
    alternative_log_likelihood = _xlog_probability(
        observed, observed_probability
    ) + _xlog_probability(count - observed, 1.0 - observed_probability)
    statistic = max(0.0, -2.0 * (null_log_likelihood - alternative_log_likelihood))
    return float(statistic), float(stats.chi2.sf(statistic, df=1))


def christoffersen_test(
    exceedances: Sequence[bool], expected_failure_probability: float
) -> dict[str, float]:
    failures = np.asarray(exceedances, dtype=int)
    if len(failures) < 3:
        return {
            "independence_lr": np.nan,
            "independence_pvalue": np.nan,
            "conditional_coverage_lr": np.nan,
            "conditional_coverage_pvalue": np.nan,
        }
    previous, current = failures[:-1], failures[1:]
    n00 = int(np.sum((previous == 0) & (current == 0)))
    n01 = int(np.sum((previous == 0) & (current == 1)))
    n10 = int(np.sum((previous == 1) & (current == 0)))
    n11 = int(np.sum((previous == 1) & (current == 1)))
    pooled = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    p01 = n01 / max(n00 + n01, 1)
    p11 = n11 / max(n10 + n11, 1)
    independent_ll = _xlog_probability(n01 + n11, pooled) + _xlog_probability(
        n00 + n10, 1.0 - pooled
    )
    markov_ll = (
        _xlog_probability(n01, p01)
        + _xlog_probability(n00, 1.0 - p01)
        + _xlog_probability(n11, p11)
        + _xlog_probability(n10, 1.0 - p11)
    )
    independence_lr = max(0.0, -2.0 * (independent_ll - markov_ll))
    kupiec_lr, _ = kupiec_test(failures, expected_failure_probability)
    conditional_lr = independence_lr + kupiec_lr
    return {
        "independence_lr": float(independence_lr),
        "independence_pvalue": float(stats.chi2.sf(independence_lr, df=1)),
        "conditional_coverage_lr": float(conditional_lr),
        "conditional_coverage_pvalue": float(stats.chi2.sf(conditional_lr, df=2)),
    }


def _uncertainty_arrays(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    validate_required_columns(
        frame, ["y_true", "baseline_prediction", "scale"], "uncertainty predictions"
    )
    truth = pd.to_numeric(frame["y_true"], errors="coerce").to_numpy(dtype=float)
    mean = pd.to_numeric(
        frame["baseline_prediction"], errors="coerce"
    ).to_numpy(dtype=float)
    scale = pd.to_numeric(frame["scale"], errors="coerce").to_numpy(dtype=float)
    degrees: np.ndarray | None = None
    if "degrees_of_freedom" in frame.columns and frame[
        "degrees_of_freedom"
    ].notna().any():
        degrees = pd.to_numeric(
            frame["degrees_of_freedom"], errors="coerce"
        ).to_numpy(dtype=float)
        valid = (
            np.isfinite(truth)
            & np.isfinite(mean)
            & np.isfinite(scale)
            & (scale > 0)
            & np.isfinite(degrees)
            & (degrees > 1)
        )
        degrees = degrees[valid]
    else:
        valid = (
            np.isfinite(truth)
            & np.isfinite(mean)
            & np.isfinite(scale)
            & (scale > 0)
        )
    return truth[valid], mean[valid], scale[valid], degrees


def _uncertainty_metrics(
    frame: pd.DataFrame, config: Mapping[str, Any]
) -> dict[str, float]:
    truth, mean, scale, degrees = _uncertainty_arrays(frame)
    if not len(truth):
        return {
            "nll": np.nan,
            "crps": np.nan,
            "pit_ks_statistic": np.nan,
            "pit_ks_pvalue": np.nan,
            "mean_scale": np.nan,
        }
    standardized = (truth - mean) / scale
    if degrees is None:
        nll_values = (
            0.5 * np.log(2.0 * np.pi)
            + np.log(scale)
            + 0.5 * np.square(standardized)
        )
        crps_values = _gaussian_crps(standardized, scale)
        pit = stats.norm.cdf(standardized)
        quantile = lambda probability: mean + scale * stats.norm.ppf(probability)
        distribution = "gaussian"
    else:
        nll_values = -stats.t.logpdf(standardized, df=degrees) + np.log(scale)
        crps_values = np.maximum(
            _student_t_crps(standardized, scale, degrees), 0.0
        )
        pit = stats.t.cdf(standardized, df=degrees)
        quantile = (
            lambda probability: mean
            + scale * stats.t.ppf(probability, df=degrees)
        )
        distribution = "student_t"
    pit = np.clip(pit, 0.0, 1.0)
    ks = stats.kstest(pit, "uniform")
    result: dict[str, float] = {
        "distribution": distribution,
        "nll": float(np.mean(nll_values)),
        "crps": float(np.mean(crps_values)),
        "pit_ks_statistic": float(ks.statistic),
        "pit_ks_pvalue": float(ks.pvalue),
        "pit_mean": float(np.mean(pit)),
        "pit_variance": float(np.var(pit)),
        "mean_scale": float(np.mean(scale)),
        "median_scale": float(np.median(scale)),
    }
    coverage_errors: list[float] = []
    for level in config["uncertainty"]["interval_levels"]:
        nominal = float(level)
        tail = (1.0 - nominal) / 2.0
        lower = quantile(tail)
        upper = quantile(1.0 - tail)
        covered = (truth >= lower) & (truth <= upper)
        token = int(round(100 * nominal))
        empirical = float(np.mean(covered))
        result[f"coverage_{token}"] = empirical
        result[f"interval_width_{token}"] = float(np.mean(upper - lower))
        result[f"coverage_error_{token}"] = abs(empirical - nominal)
        coverage_errors.append(abs(empirical - nominal))
    result["mean_interval_calibration_error"] = float(np.mean(coverage_errors))
    for level in config["uncertainty"]["var_levels"]:
        nominal = float(level)
        upper_quantile = quantile(nominal)
        exceedance = truth > upper_quantile
        token = int(round(100 * nominal))
        kupiec_lr, kupiec_pvalue = kupiec_test(exceedance, 1.0 - nominal)
        christoffersen = christoffersen_test(exceedance, 1.0 - nominal)
        result[f"var_{token}_mean"] = float(np.mean(upper_quantile))
        result[f"var_{token}_exceedance_rate"] = float(np.mean(exceedance))
        result[f"var_{token}_kupiec_lr"] = kupiec_lr
        result[f"var_{token}_kupiec_pvalue"] = kupiec_pvalue
        result[f"var_{token}_christoffersen_independence_lr"] = christoffersen[
            "independence_lr"
        ]
        result[f"var_{token}_christoffersen_independence_pvalue"] = christoffersen[
            "independence_pvalue"
        ]
        result[f"var_{token}_christoffersen_conditional_lr"] = christoffersen[
            "conditional_coverage_lr"
        ]
        result[f"var_{token}_christoffersen_conditional_pvalue"] = christoffersen[
            "conditional_coverage_pvalue"
        ]
    return result


def evaluate_prediction_group(
    frame: pd.DataFrame, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate one homogeneous target/configuration/subgroup."""
    targets = frame["target"].dropna().astype(str).unique()
    if len(targets) != 1:
        raise ValueError(f"Expected one target per group, found {targets.tolist()}")
    target = targets[0]
    validate_required_columns(frame, ["y_true"], f"{target} predictions")
    result: dict[str, Any] = {"n": int(len(frame))}
    if target == "signed":
        validate_required_columns(
            frame,
            ["prediction", "target_log_variance", "baseline_prediction"],
            "signed predictions",
        )
        result.update(regression_metrics(frame["y_true"], frame["prediction"]))
        result["directional_accuracy"] = _directional_accuracy(
            frame["y_true"], frame["prediction"]
        )
        final_prediction = (
            pd.to_numeric(frame["baseline_prediction"], errors="coerce")
            + pd.to_numeric(frame["prediction"], errors="coerce")
        )
        result["baseline_qlike"] = qlike(
            frame["target_log_variance"], frame["baseline_prediction"]
        )
        result["final_qlike"] = qlike(
            frame["target_log_variance"], final_prediction
        )
        denominator = max(abs(result["baseline_qlike"]), 1.0e-12)
        result["relative_qlike_gain"] = (
            result["baseline_qlike"] - result["final_qlike"]
        ) / denominator
    elif target in {"magnitude", "squared"}:
        validate_required_columns(frame, ["prediction"], f"{target} predictions")
        result.update(regression_metrics(frame["y_true"], frame["prediction"]))
        result.update(
            _ranking_metrics(
                frame["y_true"],
                frame["prediction"],
                float(config["targets"]["top_fraction"]),
            )
        )
    elif target.startswith("spike"):
        validate_required_columns(frame, ["probability"], f"{target} predictions")
        result.update(
            _binary_metrics(
                frame["y_true"],
                frame["probability"],
                config["targets"]["fixed_precision_levels"],
            )
        )
    elif target.startswith("regime"):
        result.update(_regime_metrics(frame))
    elif target == "uncertainty":
        result.update(_uncertainty_metrics(frame, config))
    else:
        raise ValueError(f"Unsupported target in predictions: {target}")
    metric_name, _ = primary_metric_spec(target)
    result["primary_metric"] = metric_name
    result["primary_value"] = result.get(metric_name, np.nan)
    return result


def _validate_prediction_lock(predictions: pd.DataFrame) -> None:
    validation = predictions.loc[
        predictions["evaluation_split"].astype(str) == "validation"
    ]
    test = predictions.loc[predictions["evaluation_split"].astype(str) == "test"]
    if test.empty:
        raise ValueError("target_predictions.parquet contains no test predictions")
    if not test["selected_on_validation"].fillna(False).astype(bool).all():
        raise AssertionError("Every test prediction must be selected on validation")
    identity = [
        column
        for column in CONFIGURATION_COLUMNS
        if column in predictions.columns
    ]
    selected_validation = validation.loc[
        validation["selected_on_validation"].fillna(False).astype(bool), identity
    ].drop_duplicates()
    marked = test[identity].merge(
        selected_validation.assign(__locked=True),
        on=identity,
        how="left",
        validate="many_to_one",
    )
    if marked["__locked"].isna().any():
        examples = marked.loc[marked["__locked"].isna(), identity].head(3)
        raise AssertionError(
            "Test predictions contain configurations not locked on validation: "
            f"{examples.to_dict(orient='records')}"
        )


def _summarize_predictions(
    predictions: pd.DataFrame, config: Mapping[str, Any]
) -> pd.DataFrame:
    grouping = ["evaluation_split", *configuration_columns(predictions)]
    rows: list[dict[str, Any]] = []
    for keys, group in predictions.groupby(grouping, sort=True, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(grouping, key_values))
        row["selected_on_validation"] = bool(
            group["selected_on_validation"].fillna(False).astype(bool).all()
        )
        row.update(evaluate_prediction_group(group, config))
        rows.append(row)
    table = pd.DataFrame(rows)
    for column in TARGET_TABLE_COLUMNS:
        if column not in table.columns:
            table[column] = np.nan
    ordered = [*TARGET_TABLE_COLUMNS, *sorted(set(table.columns) - set(TARGET_TABLE_COLUMNS))]
    return table[ordered]


def _parse_json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON ticker list: {text[:80]}") from exc
        if not isinstance(loaded, list):
            raise ValueError(f"Expected JSON list, found {type(loaded).__name__}")
        return [str(item) for item in loaded]
    return [item.strip() for item in text.split(",") if item.strip()]


def _first_existing(paths: Iterable[Path]) -> Path | None:
    return next((path for path in paths if path.exists()), None)


def _news_exposure(
    config: Mapping[str, Any], logger: Any
) -> pd.DataFrame:
    events_config = config.get("events", {})
    events_variant = str(
        events_config.get("variant", "canonical")
        if isinstance(events_config, Mapping)
        else "canonical"
    ).lower()
    if events_variant not in {"canonical", "near", "exact", "raw"}:
        raise ValueError(
            "events.variant must be canonical, near, exact, or raw"
        )
    if events_variant in {"exact", "raw"}:
        event_candidates = (
            project_path(
                config,
                "data",
                "processed",
                f"canonical_events_{events_variant}.parquet",
            ),
        )
    else:
        event_candidates = (
            project_path(config, "data", "processed", "canonical_events.parquet"),
            project_path(config, "outputs", "tables", "canonical_events.csv"),
        )
    event_path = _first_existing(
        event_candidates
    )
    columns = ["ticker", "feature_date", "news_level", "news_event_count"]
    if event_path is None:
        logger.warning(
            "%s events are unavailable; news_level_results.csv will have no rows",
            events_variant,
        )
        return pd.DataFrame(columns=columns)
    events = safe_read_table(event_path)
    validate_required_columns(events, ["date", "news_level"], "canonical events")
    tickers = [str(value).upper() for value in config["data"]["tickers"]]
    rows: list[dict[str, Any]] = []
    for event in events.itertuples(index=False):
        level = str(getattr(event, "news_level"))
        available = (
            _parse_json_list(getattr(event, "available_to_tickers"))
            if hasattr(event, "available_to_tickers")
            else []
        )
        if level in {"macro", "sector"}:
            available = tickers
        elif level == "target" and not available and hasattr(event, "target_ticker"):
            target = getattr(event, "target_ticker")
            available = [] if pd.isna(target) else [str(target)]
        elif level == "related" and not available and hasattr(event, "related_tickers"):
            available = _parse_json_list(getattr(event, "related_tickers"))
        for ticker in sorted(set(available).intersection(tickers)):
            rows.append(
                {
                    "ticker": ticker,
                    "feature_date": pd.Timestamp(getattr(event, "date")).normalize(),
                    "news_level": level,
                    "event_id": getattr(event, "event_id", None),
                }
            )
    if not rows:
        return pd.DataFrame(columns=columns)
    expanded = pd.DataFrame(rows)
    return (
        expanded.groupby(
            ["ticker", "feature_date", "news_level"], sort=True, observed=True
        )["event_id"]
        .nunique()
        .rename("news_event_count")
        .reset_index()
    )


def _subgroup_results(
    predictions: pd.DataFrame,
    config: Mapping[str, Any],
    news_exposure: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = predictions.loc[
        predictions["selected_on_validation"].fillna(False).astype(bool)
    ].copy()
    grouping = ["evaluation_split", *configuration_columns(selected)]
    ticker_rows: list[dict[str, Any]] = []
    for keys, group in selected.groupby([*grouping, "ticker"], sort=True, dropna=False):
        values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip([*grouping, "ticker"], values))
        row["selected_on_validation"] = True
        row.update(evaluate_prediction_group(group, config))
        ticker_rows.append(row)
    ticker_table = pd.DataFrame(ticker_rows)
    for column in TICKER_TABLE_COLUMNS:
        if column not in ticker_table.columns:
            ticker_table[column] = np.nan

    news_rows: list[dict[str, Any]] = []
    if not news_exposure.empty:
        selected["feature_date"] = pd.to_datetime(selected["feature_date"]).dt.normalize()
        count_wide = news_exposure.pivot_table(
            index=["ticker", "feature_date"],
            columns="news_level",
            values="news_event_count",
            aggfunc="sum",
            fill_value=0,
        ).reset_index()
        enriched = selected.merge(
            count_wide,
            on=["ticker", "feature_date"],
            how="left",
            validate="many_to_one",
        )
        levels = ["macro", "sector", "related", "target"]
        for level in levels:
            if level not in enriched.columns:
                enriched[level] = 0
            enriched[level] = pd.to_numeric(
                enriched[level], errors="coerce"
            ).fillna(0)
            for state, mask in (
                ("has_news", enriched[level] > 0),
                ("no_news", enriched[level] <= 0),
            ):
                subset = enriched.loc[mask]
                for keys, group in subset.groupby(grouping, sort=True, dropna=False):
                    values = keys if isinstance(keys, tuple) else (keys,)
                    row = dict(zip(grouping, values))
                    row.update(
                        {
                            "selected_on_validation": True,
                            "news_level": level,
                            "news_day_state": state,
                            "mean_news_count": float(group[level].mean()),
                        }
                    )
                    row.update(evaluate_prediction_group(group, config))
                    news_rows.append(row)
    news_table = pd.DataFrame(news_rows)
    for column in NEWS_TABLE_COLUMNS:
        if column not in news_table.columns:
            news_table[column] = np.nan
    return ticker_table, news_table


def _metric_gain(candidate: float, reference: float, larger: bool) -> float:
    if not np.isfinite(candidate) or not np.isfinite(reference):
        return np.nan
    denominator = max(abs(reference), 1.0e-12)
    return (
        (candidate - reference) / denominator
        if larger
        else (reference - candidate) / denominator
    )


def _configuration_identity_columns(frame: pd.DataFrame) -> list[str]:
    variant_column = (
        "representation_variant_family"
        if "representation_variant_family" in frame.columns
        else "representation_variant"
    )
    return [
        column
        for column in (
            "target",
            "representation",
            variant_column,
            "input_variant",
        )
        if column in frame.columns
    ]


def _aggregate_configuration_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Average repetitions without allowing seed/fold to become hyperparameters."""
    if frame.empty:
        return frame.copy()
    identity = _configuration_identity_columns(frame)
    numeric_columns = [
        column
        for column in frame.columns
        if column not in {*identity, "seed", "fold", "selected_on_validation"}
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(identity, sort=True, dropna=False):
        values = keys if isinstance(keys, tuple) else (keys,)
        row: dict[str, Any] = dict(zip(identity, values))
        for column in numeric_columns:
            numeric = pd.to_numeric(group[column], errors="coerce")
            row[column] = float(numeric.mean()) if numeric.notna().any() else np.nan
        row["primary_metric"] = str(group["primary_metric"].iloc[0])
        row["selected_on_validation"] = bool(
            group["selected_on_validation"].fillna(False).astype(bool).all()
        )
        row["replicate_count"] = int(len(group))
        row["seed_count"] = (
            int(group["seed"].nunique()) if "seed" in group.columns else 1
        )
        row["fold_count"] = (
            int(group["fold"].dropna().nunique()) if "fold" in group.columns else 0
        )
        if "prototype_seed" in group.columns:
            row["prototype_seed_count"] = int(
                pd.to_numeric(group["prototype_seed"], errors="coerce")
                .dropna()
                .nunique()
            )
            deterministic = group.assign(
                __prototype_seed=pd.to_numeric(
                    group["prototype_seed"], errors="coerce"
                )
            ).sort_values(
                [
                    "__prototype_seed",
                    *(["seed"] if "seed" in group.columns else []),
                    "representation_variant",
                ],
                kind="mergesort",
                na_position="last",
            )
        else:
            row["prototype_seed_count"] = 0
            deterministic = group.sort_values(
                "representation_variant", kind="mergesort"
            )
        if "representation_variant" in deterministic.columns:
            row["representation_variant"] = str(
                deterministic.iloc[0]["representation_variant"]
            )
        if "prototype_seed" in deterministic.columns:
            row["prototype_seed"] = deterministic.iloc[0]["prototype_seed"]
        for column in ("model", "config_id"):
            if column in deterministic.columns:
                row[column] = deterministic.iloc[0][column]
        row["per_repeat_model_config_count"] = (
            int(
                deterministic[["model", "config_id"]]
                .drop_duplicates()
                .shape[0]
            )
            if {"model", "config_id"}.issubset(deterministic.columns)
            else 0
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _best_row(frame: pd.DataFrame, target: str, representations: Sequence[str]) -> pd.Series | None:
    subset = frame.loc[
        (frame["target"] == target)
        & frame["representation"].isin(representations)
    ].copy()
    if "selected_on_validation" in subset.columns:
        subset = subset.loc[
            subset["selected_on_validation"].fillna(False).astype(bool)
        ]
    if "experiment_group" in subset.columns:
        subset = subset.loc[subset["experiment_group"].astype(str).eq("semantic")]
    elif "representation_variant_family" in subset.columns:
        experimental = subset["representation_variant_family"].astype(str).str.lower()
        subset = subset.loc[
            ~experimental.str.startswith(("response_aware", "shuffled_response"))
        ]
    if subset.empty:
        return None
    family_columns = [
        column
        for column in (
            "target",
            "representation",
            "representation_variant_family",
            "input_variant",
        )
        if column in subset.columns
    ]
    coverage = (
        subset.groupby(family_columns, dropna=False, observed=True)
        .agg(
            expected_model_seed_count=("seed", "nunique"),
            expected_prototype_seed_count=(
                "prototype_seed",
                lambda values: pd.to_numeric(values, errors="coerce")
                .dropna()
                .nunique(),
            ),
        )
        .reset_index()
    )
    subset = _aggregate_configuration_rows(subset)
    subset = subset.merge(
        coverage,
        on=family_columns,
        how="left",
        validate="many_to_one",
    )
    complete = (
        subset["seed_count"].astype(int)
        >= subset["expected_model_seed_count"].astype(int)
    )
    complete &= (
        subset["prototype_seed_count"].astype(int)
        >= subset["expected_prototype_seed_count"].astype(int)
    )
    subset = subset.loc[complete]
    _, larger = primary_metric_spec(target)
    subset = subset.loc[np.isfinite(pd.to_numeric(subset["primary_value"], errors="coerce"))]
    if subset.empty:
        return None
    subset = subset.sort_values(
        ["primary_value", "representation", "config_id"],
        ascending=[not larger, True, True],
        kind="mergesort",
    )
    return subset.iloc[0]


def _matching_metric(frame: pd.DataFrame, selected: pd.Series | None) -> float:
    if selected is None or frame.empty:
        return np.nan
    mask = pd.Series(True, index=frame.index)
    for column in _configuration_identity_columns(frame):
        if column in frame.columns and column in selected.index and not pd.isna(selected[column]):
            mask &= frame[column].astype(str) == str(selected[column])
    values = pd.to_numeric(frame.loc[mask, "primary_value"], errors="coerce")
    return float(values.mean()) if values.notna().any() else np.nan


def _matching_row(
    frame: pd.DataFrame, selected: pd.Series | None
) -> pd.Series | None:
    if selected is None or frame.empty:
        return None
    mask = pd.Series(True, index=frame.index)
    for column in _configuration_identity_columns(frame):
        if column in selected.index and not pd.isna(selected[column]):
            mask &= frame[column].astype(str) == str(selected[column])
    aggregated = _aggregate_configuration_rows(frame.loc[mask])
    return aggregated.iloc[0] if len(aggregated) == 1 else None


def _read_best_baseline(config: Mapping[str, Any]) -> str:
    path = project_path(config, "outputs", "models", "baseline_selection.json")
    if not path.exists():
        return "unavailable"
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return str(payload.get("baseline", "unavailable"))


def _finite_improvement(
    candidate: pd.Series,
    reference: pd.Series,
    metric: str,
    larger_is_better: bool,
    tolerance: float = 0.0,
) -> bool:
    left = pd.to_numeric(pd.Series([candidate.get(metric, np.nan)]), errors="coerce").iloc[0]
    right = pd.to_numeric(pd.Series([reference.get(metric, np.nan)]), errors="coerce").iloc[0]
    if not np.isfinite(left) or not np.isfinite(right):
        return False
    if larger_is_better:
        return bool(left >= right - tolerance)
    return bool(left <= right + tolerance)


def _target_specific_gate(
    target: str, prototype: pd.Series, reference: pd.Series | None
) -> tuple[bool, dict[str, Any]]:
    """Require auxiliary evidence appropriate to the scientific target."""
    if reference is None:
        return False, {"auxiliary_gate_reason": "R0 auxiliary metrics unavailable"}
    evidence: dict[str, Any] = {}
    if target == "signed":
        passed = _finite_improvement(
            prototype, reference, "final_qlike", larger_is_better=False
        )
        evidence["signed_final_qlike_gate"] = passed
        return passed, evidence
    if target in {"magnitude", "squared"}:
        rank = _finite_improvement(
            prototype, reference, "spearman", larger_is_better=True
        )
        error = _finite_improvement(
            prototype, reference, "rmse", larger_is_better=False
        )
        top_recall = _finite_improvement(
            prototype,
            reference,
            "top_fraction_recall",
            larger_is_better=True,
        )
        evidence.update(
            {
                "rank_gate": rank,
                "error_gate": error,
                "top_recall_gate": top_recall,
            }
        )
        return bool(rank and error and top_recall), evidence
    if target.startswith("spike"):
        pr_auc = _finite_improvement(
            prototype, reference, "pr_auc", larger_is_better=True
        )
        brier = _finite_improvement(
            prototype, reference, "brier", larger_is_better=False
        )
        calibration = _finite_improvement(
            prototype, reference, "ece", larger_is_better=False
        )
        recall_columns = sorted(
            column
            for column in prototype.index
            if str(column).startswith("recall_at_precision_")
            and column in reference.index
        )
        recall = bool(
            recall_columns
            and all(
                _finite_improvement(
                    prototype, reference, column, larger_is_better=True
                )
                for column in recall_columns
            )
        )
        evidence.update(
            {
                "spike_pr_auc_gate": pr_auc,
                "spike_brier_gate": brier,
                "spike_calibration_gate": calibration,
                "spike_fixed_precision_recall_gate": recall,
            }
        )
        return bool(pr_auc and brier and calibration and recall), evidence
    if target.startswith("regime"):
        macro_f1 = _finite_improvement(
            prototype, reference, "macro_f1", larger_is_better=True
        )
        balanced = _finite_improvement(
            prototype, reference, "balanced_accuracy", larger_is_better=True
        )
        brier = _finite_improvement(
            prototype, reference, "multiclass_brier", larger_is_better=False
        )
        calibration = _finite_improvement(
            prototype, reference, "multiclass_ece", larger_is_better=False
        )
        evidence.update(
            {
                "regime_macro_f1_gate": macro_f1,
                "regime_balanced_accuracy_gate": balanced,
                "regime_brier_gate": brier,
                "regime_calibration_gate": calibration,
            }
        )
        return bool(macro_f1 and balanced and brier and calibration), evidence
    if target == "uncertainty":
        nll = _finite_improvement(
            prototype, reference, "nll", larger_is_better=False
        )
        crps = _finite_improvement(
            prototype, reference, "crps", larger_is_better=False
        )
        pit = _finite_improvement(
            prototype,
            reference,
            "pit_ks_statistic",
            larger_is_better=False,
        )
        coverage = _finite_improvement(
            prototype,
            reference,
            "mean_interval_calibration_error",
            larger_is_better=False,
        )
        width_columns = sorted(
            column
            for column in prototype.index
            if str(column).startswith("interval_width_") and column in reference.index
        )
        prototype_widths = np.asarray(
            [prototype.get(column, np.nan) for column in width_columns], dtype=float
        )
        reference_widths = np.asarray(
            [reference.get(column, np.nan) for column in width_columns], dtype=float
        )
        finite = np.isfinite(prototype_widths) & np.isfinite(reference_widths)
        width_not_inflated = bool(
            finite.any()
            and float(np.mean(prototype_widths[finite]))
            <= 1.05 * float(np.mean(reference_widths[finite]))
        )
        evidence.update(
            {
                "uncertainty_nll_gate": nll,
                "uncertainty_crps_gate": crps,
                "uncertainty_pit_gate": pit,
                "uncertainty_coverage_gate": coverage,
                "uncertainty_width_not_inflated_gate": width_not_inflated,
            }
        )
        return bool(nll and crps and pit and coverage and width_not_inflated), evidence
    return False, {"auxiliary_gate_reason": f"Unsupported target {target}"}


def _decision_table(
    config: Mapping[str, Any],
    validation: pd.DataFrame,
    test: pd.DataFrame,
    news_level: pd.DataFrame,
) -> pd.DataFrame:
    threshold = float(config["decision"]["min_relative_gain"])
    fold_threshold = float(config["decision"]["min_fold_win_rate"])
    target_rows: list[dict[str, Any]] = []
    targets = sorted(validation["target"].dropna().astype(str).unique())
    for target in targets:
        metric, larger = primary_metric_spec(target)
        prototype = _best_row(validation, target, PROTOTYPE_REPRESENTATIONS)
        if prototype is None:
            target_rows.append(
                {
                    "record_type": "target",
                    "target": target,
                    "primary_metric": metric,
                    "decision": "NO-GO",
                    "reason": "No selected prototype representation was available",
                }
            )
            continue
        row: dict[str, Any] = {
            "record_type": "target",
            "target": target,
            "primary_metric": metric,
            "representation": prototype["representation"],
            "representation_variant_family": prototype.get(
                "representation_variant_family", prototype.get("representation_variant", "")
            ),
            "representative_variant_for_artifacts": prototype.get(
                "representation_variant", ""
            ),
            "prototype_seed_count": prototype.get("prototype_seed_count", 0),
            "model_seed_count": prototype.get("seed_count", 0),
            "input_variant": prototype["input_variant"],
            "model": prototype["model"],
            "config_id": prototype["config_id"],
            "selection_split": "validation",
            "validation_primary_value": float(prototype["primary_value"]),
            "test_primary_value": _matching_metric(test, prototype),
        }
        available_gains: list[float] = []
        reference_rows: dict[str, pd.Series | None] = {}
        for representation in (*REFERENCE_REPRESENTATIONS, *PLACEBO_REPRESENTATIONS):
            reference = _best_row(validation, target, (representation,))
            reference_rows[representation] = reference
            reference_value = (
                float(reference["primary_value"]) if reference is not None else np.nan
            )
            gain = _metric_gain(float(prototype["primary_value"]), reference_value, larger)
            row[f"validation_gain_vs_{representation}"] = gain
            if np.isfinite(gain):
                available_gains.append(gain)
            reference_test = _matching_metric(test, reference)
            row[f"test_gain_vs_{representation}"] = _metric_gain(
                row["test_primary_value"], reference_test, larger
            )
        required_gains = [
            row[f"validation_gain_vs_{representation}"]
            for representation in ("R0", "R2", "R3", "R4")
        ]
        placebo_gains = [
            row[f"validation_gain_vs_{representation}"]
            for representation in PLACEBO_REPRESENTATIONS
        ]
        required_complete = bool(all(np.isfinite(value) for value in required_gains))
        placebo_complete = bool(all(np.isfinite(value) for value in placebo_gains))
        row["required_comparator_evidence_complete"] = required_complete
        row["placebo_evidence_complete"] = placebo_complete
        row["better_than_no_text"] = bool(
            np.isfinite(row["validation_gain_vs_R0"])
            and row["validation_gain_vs_R0"] >= threshold
        )
        row["better_than_raw_embedding"] = bool(
            np.isfinite(row["validation_gain_vs_R2"])
            and row["validation_gain_vs_R2"] >= threshold
        )
        row["better_than_pca"] = bool(
            np.isfinite(row["validation_gain_vs_R3"])
            and row["validation_gain_vs_R3"] >= threshold
        )
        row["better_than_random_projection"] = bool(
            np.isfinite(row["validation_gain_vs_R4"])
            and row["validation_gain_vs_R4"] >= threshold
        )
        row["better_than_shuffled"] = bool(
            placebo_complete and min(placebo_gains) >= threshold
        )
        row["minimum_validation_gain"] = (
            float(min(available_gains)) if available_gains else np.nan
        )
        test_confirmation_references = (
            "R0",
            "R2",
            "R3",
            "R4",
            *PLACEBO_REPRESENTATIONS,
        )
        test_confirmation_gains = [
            row[f"test_gain_vs_{representation}"]
            for representation in test_confirmation_references
        ]
        row["test_comparator_evidence_complete"] = bool(
            all(np.isfinite(value) for value in test_confirmation_gains)
        )
        row["test_confirms_locked_gain"] = bool(
            row["test_comparator_evidence_complete"]
            and all(value > 0 for value in test_confirmation_gains)
        )

        auxiliary_passed, auxiliary_evidence = _target_specific_gate(
            target, prototype, reference_rows["R0"]
        )
        row.update(auxiliary_evidence)
        row["target_specific_auxiliary_gate"] = auxiliary_passed
        prototype_test_row = _matching_row(test, prototype)
        reference_test_row = _matching_row(test, reference_rows["R0"])
        test_auxiliary_passed, test_auxiliary_evidence = (
            _target_specific_gate(target, prototype_test_row, reference_test_row)
            if prototype_test_row is not None
            else (False, {"auxiliary_gate_reason": "locked test metrics unavailable"})
        )
        row.update(
            {
                f"test_{key}": value
                for key, value in test_auxiliary_evidence.items()
            }
        )
        row["test_target_specific_auxiliary_gate"] = test_auxiliary_passed

        replicate_columns = [
            column
            for column in ("seed", "fold")
            if column in validation.columns and validation[column].notna().any()
        ]
        win_rate = np.nan
        if replicate_columns and reference_rows["R0"] is not None:
            proto_mask = pd.Series(True, index=validation.index)
            base_mask = pd.Series(True, index=validation.index)
            for column in _configuration_identity_columns(validation):
                if column in prototype.index and not pd.isna(prototype[column]):
                    proto_mask &= (
                        validation[column].astype(str) == str(prototype[column])
                    )
                reference_value = reference_rows["R0"].get(column, np.nan)  # type: ignore[union-attr]
                if not pd.isna(reference_value):
                    base_mask &= (
                        validation[column].astype(str) == str(reference_value)
                    )
            proto_subset = validation.loc[proto_mask]
            base_subset = validation.loc[base_mask]
            paired = proto_subset.merge(
                base_subset,
                on=replicate_columns,
                suffixes=("_prototype", "_baseline"),
            )
            if len(paired) > 1:
                gains = [
                    _metric_gain(left, right, larger)
                    for left, right in zip(
                        paired["primary_value_prototype"],
                        paired["primary_value_baseline"],
                    )
                ]
                win_rate = float(np.mean(np.asarray(gains) > 0))
        row["validation_replicate_win_rate"] = win_rate
        row["stable_across_validation_replicates"] = bool(
            np.isfinite(win_rate) and win_rate >= fold_threshold
        )

        strong_validation = bool(
            required_complete
            and placebo_complete
            and min(required_gains) >= threshold
            and min(placebo_gains) >= threshold
            and auxiliary_passed
        )
        weak_validation = bool(
            required_complete
            and placebo_complete
            and row["better_than_no_text"]
            and auxiliary_passed
        )
        row["validation_target_eligible"] = bool(
            strong_validation or weak_validation
        )
        if target == "signed":
            go_label = "GO-DIRECT"
        elif target in {"magnitude", "squared"}:
            go_label = "GO-MAGNITUDE"
        elif target.startswith("spike"):
            go_label = "GO-SPIKE"
        elif target == "uncertainty":
            go_label = "GO-UNCERTAINTY"
        elif target.startswith("regime"):
            go_label = "GO-REGIME"
        else:
            raise ValueError(f"Unsupported decision target: {target}")
        if (
            strong_validation
            and row["test_confirms_locked_gain"]
            and test_auxiliary_passed
        ):
            # This stage has no evidence that PCA, clustering, aggregation and
            # the downstream model were all refitted inside each chronological
            # fold.  A full GO label is therefore deliberately deferred to the
            # robustness stage even when repeated prediction rows look stable.
            row["decision"] = "WEAK-GO"
            row["provisional_go_label_after_refit_evidence"] = go_label
            row["reason"] = (
                "Validation-locked prototype beats available references and test confirms; "
                "full chronological fold refit evidence is still required"
            )
        elif (
            weak_validation
            and test_auxiliary_passed
            and row["test_confirms_locked_gain"]
        ):
            row["decision"] = "WEAK-GO"
            row["reason"] = (
                "Gain versus price-only survives test, but one or more representation/placebo/"
                "stability requirements are not established"
            )
        else:
            row["decision"] = "NO-GO"
            row["reason"] = (
                "The validation-led prototype gain is absent or is not confirmed on locked test"
            )
        target_rows.append(row)

    target_table = pd.DataFrame(target_rows)
    viable = target_table.loc[
        target_table.get(
            "validation_target_eligible",
            pd.Series(False, index=target_table.index),
        )
        .fillna(False)
        .astype(bool)
    ].copy()
    # The research target itself is selected by validation gain only.  Test can
    # confirm/downgrade its conclusion but never changes which target was chosen.
    if viable.empty:
        chosen_target = "none"
        conclusion = "NO-GO"
        chosen_configuration: pd.Series | None = None
    else:
        viable = viable.sort_values(
            ["validation_gain_vs_R0", "target"],
            ascending=[False, True],
            kind="mergesort",
        )
        chosen = viable.iloc[0]
        chosen_configuration = chosen
        chosen_target = str(chosen["target"])
        conclusion = str(chosen["decision"])

    best_news_level = "unavailable"
    if (
        not news_level.empty
        and chosen_target != "none"
        and chosen_configuration is not None
    ):
        subset = news_level.loc[
            (news_level["target"] == chosen_target)
            & (news_level["evaluation_split"] == "validation")
            & (news_level["news_day_state"] == "has_news")
            & (
                news_level["representation"]
                == chosen_configuration["representation"]
            )
        ].copy()
        if (
            "representation_variant_family" in subset.columns
            and "representation_variant_family" in chosen_configuration.index
        ):
            subset = subset.loc[
                subset["representation_variant_family"].astype(str)
                == str(chosen_configuration["representation_variant_family"])
            ]
        if not subset.empty:
            _, larger = primary_metric_spec(chosen_target)
            subset = (
                subset.groupby("news_level", sort=True, observed=True)[
                    "primary_value"
                ]
                .mean()
                .reset_index()
            )
            subset = subset.sort_values(
                ["primary_value", "news_level"],
                ascending=[not larger, True],
                kind="mergesort",
            )
            best_news_level = str(subset.iloc[0]["news_level"])

    overall = {
        "record_type": "final",
        "target": chosen_target,
        "decision": conclusion,
        "selection_split": "validation",
        "best_baseline": _read_best_baseline(config),
        "prototype_vs_raw": (
            bool(viable.iloc[0]["better_than_raw_embedding"])
            if not viable.empty
            else False
        ),
        "prototype_vs_pca_random_projection": (
            bool(
                viable.iloc[0]["better_than_pca"]
                and viable.iloc[0]["better_than_random_projection"]
            )
            if not viable.empty
            else False
        ),
        "true_vs_shuffled": (
            bool(viable.iloc[0]["better_than_shuffled"])
            if not viable.empty
            else False
        ),
        # Repeated predictions are only diagnostic here.  The stability field
        # is unlocked later only by executed chronological full-refit folds.
        "stable": False,
        "best_news_level": best_news_level,
        "cross_stock_common_effect": "pending analyze_prototypes.py",
        "best_target": chosen_target,
        "prototype_better_than_raw_embedding": (
            bool(viable.iloc[0]["better_than_raw_embedding"])
            if not viable.empty
            else False
        ),
        "prototype_better_than_pca_random_projection": (
            bool(
                viable.iloc[0]["better_than_pca"]
                and viable.iloc[0]["better_than_random_projection"]
            )
            if not viable.empty
            else False
        ),
        "real_news_better_than_shuffled": (
            bool(viable.iloc[0]["better_than_shuffled"])
            if not viable.empty
            else False
        ),
        "stable_across_seed_fold": False,
        "most_useful_news_level": best_news_level,
        "semiconductor_common_response": "pending analyze_prototypes.py",
        "selection_rule": (
            "target/configuration selected on validation only; locked test used for confirmation"
        ),
        "robustness_gate": (
            "GO labels are blocked until placebo_tests.py records sufficient executed "
            "chronological fold refits"
        ),
    }
    return pd.concat([target_table, pd.DataFrame([overall])], ignore_index=True)


def _save_placeholder(path: Path, title: str, message: str) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.axis("off")
    axis.set_title(title)
    axis.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _prediction_rows_for_selection(
    predictions: pd.DataFrame, selected: pd.Series | None, split: str
) -> pd.DataFrame:
    if selected is None:
        return predictions.iloc[0:0].copy()
    subset = predictions.loc[predictions["evaluation_split"] == split].copy()
    for column in [
        *_configuration_identity_columns(subset),
        *[
            value
            for value in ("model", "config_id")
            if value in subset.columns and value in selected.index
        ],
    ]:
        if column in subset.columns and column in selected.index and not pd.isna(selected[column]):
            subset = subset.loc[subset[column].astype(str) == str(selected[column])]
    return subset


def _plot_target_performance(
    validation: pd.DataFrame, test: pd.DataFrame, path: Path
) -> None:
    rows: list[dict[str, Any]] = []
    for target in sorted(validation["target"].dropna().astype(str).unique()):
        prototype = _best_row(validation, target, PROTOTYPE_REPRESENTATIONS)
        baseline = _best_row(validation, target, ("R0",))
        if prototype is None or baseline is None:
            continue
        _, larger = primary_metric_spec(target)
        rows.append(
            {
                "target": target,
                "validation": _metric_gain(
                    float(prototype["primary_value"]),
                    float(baseline["primary_value"]),
                    larger,
                ),
                "test": _metric_gain(
                    _matching_metric(test, prototype),
                    _matching_metric(test, baseline),
                    larger,
                ),
            }
        )
    if not rows:
        _save_placeholder(
            path,
            "Validation/test performance by target",
            "No locked prototype and R0 pairs were available.",
        )
        return
    plot = pd.DataFrame(rows).set_index("target")
    figure, axis = plt.subplots(figsize=(10, 5))
    plot[["validation", "test"]].plot(kind="bar", ax=axis)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylabel("Relative gain vs price-only (higher is better)")
    axis.set_xlabel("Target")
    axis.set_title("Validation-selected prototype performance")
    axis.legend(title="Evaluation split")
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _plot_calibration(
    predictions: pd.DataFrame, validation: pd.DataFrame, path: Path
) -> None:
    candidates: list[pd.Series] = []
    for target in (
        "spike_q90",
        "spike_q95",
        "spike_q90_pooled_standardized",
        "spike_q95_pooled_standardized",
    ):
        candidate = _best_row(validation, target, PROTOTYPE_REPRESENTATIONS)
        if candidate is not None:
            candidates.append(candidate)
    if not candidates:
        _save_placeholder(path, "Spike calibration", "No spike prototype prediction available.")
        return
    candidates.sort(key=lambda row: float(row["primary_value"]), reverse=True)
    selected = candidates[0]
    rows = _prediction_rows_for_selection(predictions, selected, "test")
    if rows.empty:
        rows = _prediction_rows_for_selection(predictions, selected, "validation")
    labels = rows["y_true"].to_numpy(dtype=int)
    probabilities = rows["probability"].to_numpy(dtype=float)
    valid = np.isfinite(probabilities)
    labels, probabilities = labels[valid], np.clip(probabilities[valid], 0.0, 1.0)
    if not len(labels):
        _save_placeholder(path, "Spike calibration", "No finite probabilities available.")
        return
    edges = np.linspace(0.0, 1.0, 11)
    bin_id = np.clip(np.digitize(probabilities, edges[1:-1]), 0, 9)
    predicted: list[float] = []
    observed: list[float] = []
    for index in range(10):
        mask = bin_id == index
        if mask.any():
            predicted.append(float(probabilities[mask].mean()))
            observed.append(float(labels[mask].mean()))
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    axis.plot(predicted, observed, marker="o", label=str(selected["representation"]))
    axis.set(xlabel="Mean predicted probability", ylabel="Observed spike rate")
    axis.set_title(f"Calibration: {selected['target']}")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _pit_values(frame: pd.DataFrame) -> np.ndarray:
    truth, mean, scale, degrees = _uncertainty_arrays(frame)
    if degrees is None:
        return stats.norm.cdf((truth - mean) / scale)
    return stats.t.cdf((truth - mean) / scale, df=degrees)


def _plot_pit_and_coverage(
    predictions: pd.DataFrame,
    validation: pd.DataFrame,
    config: Mapping[str, Any],
    pit_path: Path,
    coverage_path: Path,
) -> None:
    selected = _best_row(validation, "uncertainty", PROTOTYPE_REPRESENTATIONS)
    if selected is None:
        _save_placeholder(pit_path, "PIT histogram", "No uncertainty prototype available.")
        _save_placeholder(
            coverage_path, "Interval coverage", "No uncertainty prototype available."
        )
        return
    rows = _prediction_rows_for_selection(predictions, selected, "test")
    if rows.empty:
        rows = _prediction_rows_for_selection(predictions, selected, "validation")
    pit = _pit_values(rows)
    pit = pit[np.isfinite(pit)]
    if not len(pit):
        _save_placeholder(pit_path, "PIT histogram", "No finite PIT values available.")
    else:
        bins = int(config["uncertainty"]["pit_bins"])
        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.hist(pit, bins=bins, range=(0, 1), density=True, alpha=0.75)
        axis.axhline(1.0, linestyle="--", color="black", linewidth=1)
        axis.set(xlabel="PIT", ylabel="Density", title="PIT histogram (locked test)")
        figure.tight_layout()
        figure.savefig(pit_path, dpi=160, bbox_inches="tight")
        plt.close(figure)

    metrics = _uncertainty_metrics(rows, config)
    levels = [float(value) for value in config["uncertainty"]["interval_levels"]]
    empirical = [
        metrics.get(f"coverage_{int(round(level * 100))}", np.nan)
        for level in levels
    ]
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.plot(levels, levels, "--", color="gray", label="Nominal")
    axis.plot(levels, empirical, marker="o", label=str(selected["representation"]))
    axis.set(
        xlabel="Nominal coverage",
        ylabel="Empirical coverage",
        title="Prediction interval coverage (locked test)",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(coverage_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def run(config: dict[str, Any]) -> dict[str, Path]:
    ensure_directories(config)
    logger = get_logger(
        __name__, config, project_path(config, "outputs", "logs", "evaluation.log")
    )
    prediction_path = project_path(
        config, "data", "processed", "target_predictions.parquet"
    )
    predictions = safe_read_table(prediction_path)
    validate_required_columns(
        predictions,
        [
            "ticker",
            "feature_date",
            "target_date",
            "target",
            "representation",
            "representation_variant",
            "input_variant",
            "model",
            "config_id",
            "seed",
            "evaluation_split",
            "selected_on_validation",
            "y_true",
            "baseline_prediction",
        ],
        "target predictions",
    )
    predictions = predictions.copy()
    predictions["feature_date"] = pd.to_datetime(
        predictions["feature_date"], errors="raise"
    ).dt.normalize()
    predictions["target_date"] = pd.to_datetime(
        predictions["target_date"], errors="raise"
    ).dt.normalize()
    predictions = attach_variant_metadata(predictions, config, logger)
    _validate_prediction_lock(predictions)

    comparison = _summarize_predictions(predictions, config)
    validation = comparison.loc[
        comparison["evaluation_split"] == "validation"
    ].reset_index(drop=True)
    test = comparison.loc[
        comparison["evaluation_split"] == "test"
    ].reset_index(drop=True)
    exposure = _news_exposure(config, logger)
    ticker_results, news_results = _subgroup_results(
        predictions, config, exposure
    )
    decision = _decision_table(config, validation, test, news_results)

    tables = project_path(config, "outputs", "tables")
    figures = project_path(config, "outputs", "figures")
    validation_path = atomic_write_csv(
        validation, tables / "target_comparison_validation.csv", index=False
    )
    test_path = atomic_write_csv(
        test, tables / "target_comparison_test.csv", index=False
    )
    ticker_path = atomic_write_csv(
        ticker_results, tables / "ticker_level_results.csv", index=False
    )
    news_path = atomic_write_csv(
        news_results, tables / "news_level_results.csv", index=False
    )
    decision_path = atomic_write_csv(
        decision, tables / "final_decision.csv", index=False
    )
    performance_path = figures / "target_performance_validation_test.png"
    calibration_path = figures / "calibration_curve.png"
    pit_path = figures / "pit_histogram.png"
    coverage_path = figures / "interval_coverage.png"
    _plot_target_performance(validation, test, performance_path)
    _plot_calibration(predictions, validation, calibration_path)
    _plot_pit_and_coverage(
        predictions, validation, config, pit_path, coverage_path
    )

    overall = decision.loc[decision["record_type"] == "final"].iloc[0]
    logger.info("1. Best baseline: %s", overall["best_baseline"])
    logger.info(
        "2. Prototype better than raw embedding: %s",
        overall["prototype_vs_raw"],
    )
    logger.info(
        "3. Prototype better than PCA/random projection: %s",
        overall["prototype_vs_pca_random_projection"],
    )
    logger.info(
        "4. Real news better than shuffled news: %s",
        overall["true_vs_shuffled"],
    )
    logger.info(
        "5. Stable across seed/fold: %s", overall["stable"]
    )
    logger.info("6. Most useful news level: %s", overall["best_news_level"])
    logger.info(
        "7. Semiconductor common response: %s",
        overall["cross_stock_common_effect"],
    )
    logger.info("8. Best target: %s", overall["best_target"])
    logger.info("9. Decision: %s", overall["decision"])
    return {
        "target_comparison_validation": validation_path,
        "target_comparison_test": test_path,
        "ticker_level_results": ticker_path,
        "news_level_results": news_path,
        "final_decision": decision_path,
        "target_performance_figure": performance_path,
        "calibration_figure": calibration_path,
        "pit_figure": pit_path,
        "interval_coverage_figure": coverage_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to config YAML")
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
