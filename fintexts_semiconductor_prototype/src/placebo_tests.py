"""Compare semantic prototypes with placebos and run honest robustness checks.

Configuration identities are locked from validation.  Holdout slicing,
bootstrap intervals, and leave-one-group score recomputation are explicitly
labelled diagnostic-only.  The only rows allowed to satisfy the stability gate
are chronological folds that refit the baseline forecast, fold thresholds,
PCA, K-means prototypes, ticker-day aggregation, feature processor, and target
model using fold-train information.
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
from scipy import stats
from sklearn.decomposition import PCA

from src.aggregate_features import (
    NEWS_LEVELS,
    _aggregate_matrix,
    _aggregate_metadata,
    _build_edges,
    _instances,
    _join_feature_blocks,
)
from src.build_baseline import (
    fit_baseline,
    load_market_supervised,
    predict_baseline,
)
from src.build_prototypes import (
    _assign,
    _fit_projection,
    _fit_spherical_kmeans,
)
from src.evaluate_targets import (
    PROTOTYPE_REPRESENTATIONS,
    attach_variant_metadata,
    evaluate_prediction_group,
    primary_metric_spec,
)
from src.train_targets import (
    Candidate,
    _fit_candidate,
    _is_usable_numeric,
    _make_preprocessor,
    _predict_candidate,
    _primary_score,
)
from src.utils import (
    atomic_write_csv,
    ensure_directories,
    get_logger,
    l2_normalize,
    load_config,
    project_path,
    read_json,
    safe_read_table,
    set_global_seed,
    validate_required_columns,
)

PLACEBO_REPRESENTATIONS = ("R9", "R10", "R11")
IDENTITY_COLUMNS = (
    "target",
    "representation",
    "representation_variant_family",
    "input_variant",
)
ROBUSTNESS_COLUMNS = (
    "record_type",
    "evidence_type",
    "qualifies_for_stability",
    "executed",
    "not_executed_reason",
    "target",
    "representation",
    "comparison_representation",
    "seed",
    "fold",
    "excluded_ticker",
    "excluded_year",
    "metric",
    "larger_is_better",
    "prototype_value",
    "reference_value",
    "gain",
    "win",
    "ci_lower",
    "ci_upper",
    "pvalue",
    "n_train",
    "n_evaluation",
    "refit_components",
)


def _first_existing(paths: Iterable[Path]) -> Path | None:
    return next((path for path in paths if path.exists()), None)


def _events_variant(config: Mapping[str, Any]) -> str:
    events_config = config.get("events", {})
    variant = str(
        events_config.get("variant", "canonical")
        if isinstance(events_config, Mapping)
        else "canonical"
    ).lower()
    if variant not in {"canonical", "near", "exact", "raw"}:
        raise ValueError("events.variant must be canonical, near, exact, or raw")
    return variant


def _variant_suffix(variant: str) -> str:
    return "" if variant == "canonical" else f"_{variant}"


def _fold_event_artifact_paths(
    config: Mapping[str, Any],
) -> tuple[tuple[Path, ...], Path, Path]:
    variant = _events_variant(config)
    suffix = _variant_suffix(variant)
    if variant in {"exact", "raw"}:
        event_paths = (
            project_path(
                config,
                "data",
                "processed",
                f"canonical_events_{variant}.parquet",
            ),
        )
    else:
        event_paths = (
            project_path(
                config, "data", "processed", "canonical_events.parquet"
            ),
            project_path(
                config, "outputs", "tables", "canonical_events.csv"
            ),
        )
    metadata_path = project_path(
        config,
        "data",
        "embeddings",
        f"event_embedding_metadata{suffix}.csv",
    )
    embedding_path = project_path(
        config,
        "data",
        "embeddings",
        f"event_embeddings{suffix}.npy",
    )
    return event_paths, metadata_path, embedding_path


def _metric_gain(candidate: float, reference: float, larger: bool) -> float:
    if not np.isfinite(candidate) or not np.isfinite(reference):
        return np.nan
    denominator = max(abs(reference), 1.0e-12)
    return (
        (candidate - reference) / denominator
        if larger
        else (reference - candidate) / denominator
    )


def _identity_mask(frame: pd.DataFrame, selection: Mapping[str, Any]) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for column in IDENTITY_COLUMNS:
        if column in frame.columns and column in selection and not pd.isna(selection[column]):
            mask &= frame[column].astype(str) == str(selection[column])
    return mask


def _locked_configurations(validation: pd.DataFrame) -> pd.DataFrame:
    selected = validation.loc[
        validation["selected_on_validation"].fillna(False).astype(bool)
    ].copy()
    if selected.empty:
        return pd.DataFrame(
            columns=[
                *IDENTITY_COLUMNS,
                "representation_variant",
                "prototype_seed",
                "model",
                "config_id",
                "primary_metric",
                "primary_value",
                "n_replicates",
                "prototype_seed_count",
                "model_seed_count",
                "family_prototype_seed_count",
                "family_model_seed_count",
                "per_repeat_model_config_count",
            ]
        )
    family_columns = [
        column
        for column in (
            "target",
            "representation",
            "representation_variant_family",
            "input_variant",
        )
        if column in selected.columns
    ]
    family_coverage = (
        selected.groupby(family_columns, dropna=False, observed=True)
        .agg(
            family_model_seed_count=("seed", "nunique"),
            family_prototype_seed_count=(
                "prototype_seed",
                lambda values: pd.to_numeric(values, errors="coerce")
                .dropna()
                .nunique(),
            ),
        )
        .reset_index()
    )
    grouping = [column for column in IDENTITY_COLUMNS if column in selected.columns]
    rows: list[dict[str, Any]] = []
    for keys, group in selected.groupby(grouping, sort=True, dropna=False):
        values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(grouping, values))
        row["primary_metric"] = str(group["primary_metric"].iloc[0])
        row["primary_value"] = float(
            pd.to_numeric(group["primary_value"], errors="coerce").mean()
        )
        row["n_replicates"] = int(len(group))
        deterministic = group.assign(
            __prototype_seed=pd.to_numeric(
                group.get(
                    "prototype_seed",
                    pd.Series(index=group.index, dtype=float),
                ),
                errors="coerce",
            )
        ).sort_values(
            [
                "__prototype_seed",
                *(
                    ["seed"]
                    if "seed" in group.columns
                    else []
                ),
                "representation_variant",
            ],
            kind="mergesort",
            na_position="last",
        )
        row["representation_variant"] = str(
            deterministic.iloc[0].get("representation_variant", "selected_default")
        )
        row["prototype_seed"] = deterministic.iloc[0]["__prototype_seed"]
        row["model"] = deterministic.iloc[0]["model"]
        row["config_id"] = deterministic.iloc[0]["config_id"]
        row["per_repeat_model_config_count"] = int(
            deterministic[["model", "config_id"]].drop_duplicates().shape[0]
        )
        row["prototype_seed_count"] = int(
            deterministic["__prototype_seed"].dropna().nunique()
        )
        row["model_seed_count"] = (
            int(group["seed"].nunique()) if "seed" in group.columns else 1
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    return result.merge(
        family_coverage,
        on=family_columns,
        how="left",
        validate="many_to_one",
    )


def _best_locked(
    locked: pd.DataFrame, target: str, representations: Sequence[str]
) -> pd.Series | None:
    subset = locked.loc[
        (locked["target"] == target)
        & locked["representation"].isin(representations)
    ].copy()
    if "representation_variant_family" in subset.columns:
        experimental = subset["representation_variant_family"].astype(str).str.lower()
        subset = subset.loc[
            ~experimental.str.startswith(("response_aware", "shuffled_response"))
        ]
    if subset.empty:
        return None
    subset = subset.loc[
        (subset["model_seed_count"] >= subset["family_model_seed_count"])
        & (
            (subset["family_prototype_seed_count"] <= 0)
            | (
                subset["prototype_seed_count"]
                >= subset["family_prototype_seed_count"]
            )
        )
    ]
    if subset.empty:
        return None
    _, larger = primary_metric_spec(target)
    subset = subset.loc[
        np.isfinite(pd.to_numeric(subset["primary_value"], errors="coerce"))
    ]
    if subset.empty:
        return None
    subset = subset.sort_values(
        ["primary_value", "representation", "representation_variant", "config_id"],
        ascending=[not larger, True, True, True],
        kind="mergesort",
    )
    return subset.iloc[0]


def _test_metric(test: pd.DataFrame, selection: pd.Series | None) -> float:
    if selection is None:
        return np.nan
    subset = test.loc[_identity_mask(test, selection)]
    values = pd.to_numeric(subset["primary_value"], errors="coerce")
    return float(values.mean()) if values.notna().any() else np.nan


def _placebo_comparison(
    validation: pd.DataFrame, test: pd.DataFrame
) -> pd.DataFrame:
    locked = _locked_configurations(validation)
    rows: list[dict[str, Any]] = []
    for target in sorted(locked["target"].dropna().astype(str).unique()):
        _, larger = primary_metric_spec(target)
        for true_representation in PROTOTYPE_REPRESENTATIONS:
            true = _best_locked(locked, target, (true_representation,))
            for placebo_representation in PLACEBO_REPRESENTATIONS:
                placebo = _best_locked(locked, target, (placebo_representation,))
                if true is None or placebo is None:
                    rows.append(
                        {
                            "target": target,
                            "true_representation": true_representation,
                            "placebo_representation": placebo_representation,
                            "primary_metric": primary_metric_spec(target)[0],
                            "larger_is_better": larger,
                            "executed": False,
                            "not_executed_reason": (
                                "Missing validation-locked "
                                + (
                                    true_representation
                                    if true is None
                                    else placebo_representation
                                )
                                + " evidence"
                            ),
                            "configuration_selection_split": "validation",
                            "test_used_for_selection": False,
                        }
                    )
                    continue
                validation_gain = _metric_gain(
                    float(true["primary_value"]),
                    float(placebo["primary_value"]),
                    larger,
                )
                true_test = _test_metric(test, true)
                placebo_test = _test_metric(test, placebo)
                rows.append(
                    {
                        "target": target,
                        "true_representation": true_representation,
                        "true_representation_variant": true.get(
                            "representation_variant", ""
                        ),
                        "true_model": true["model"],
                        "true_config_id": true["config_id"],
                        "placebo_representation": placebo_representation,
                        "placebo_model": placebo["model"],
                        "placebo_config_id": placebo["config_id"],
                        "primary_metric": true["primary_metric"],
                        "larger_is_better": larger,
                        "true_validation_value": float(true["primary_value"]),
                        "placebo_validation_value": float(placebo["primary_value"]),
                        "validation_gain": validation_gain,
                        "better_on_validation": bool(validation_gain > 0),
                        "true_test_value": true_test,
                        "placebo_test_value": placebo_test,
                        "test_gain": _metric_gain(true_test, placebo_test, larger),
                        "executed": True,
                        "not_executed_reason": "",
                        "configuration_selection_split": "validation",
                        "test_used_for_selection": False,
                    }
                )
    columns = [
        "target",
        "true_representation",
        "true_representation_variant",
        "true_model",
        "true_config_id",
        "placebo_representation",
        "placebo_model",
        "placebo_config_id",
        "primary_metric",
        "larger_is_better",
        "true_validation_value",
        "placebo_validation_value",
        "validation_gain",
        "better_on_validation",
        "true_test_value",
        "placebo_test_value",
        "test_gain",
        "executed",
        "not_executed_reason",
        "configuration_selection_split",
        "test_used_for_selection",
    ]
    result = pd.DataFrame(rows)
    for column in columns:
        if column not in result.columns:
            result[column] = np.nan
    return result[columns]


def _prediction_subset(
    predictions: pd.DataFrame, selection: pd.Series, split: str
) -> pd.DataFrame:
    return predictions.loc[
        (predictions["evaluation_split"] == split)
        & _identity_mask(predictions, selection)
    ].copy()


def _subgroup_metric(frame: pd.DataFrame, config: Mapping[str, Any]) -> float:
    if frame.empty:
        return np.nan
    return float(evaluate_prediction_group(frame, config)["primary_value"])


def _diagnostic_slice_rows(
    predictions: pd.DataFrame,
    prototype: pd.Series,
    reference: pd.Series,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    target = str(prototype["target"])
    metric, larger = primary_metric_spec(target)
    prototype_rows = _prediction_subset(predictions, prototype, "test")
    reference_rows = _prediction_subset(predictions, reference, "test")
    rows: list[dict[str, Any]] = []
    common_tickers = sorted(
        set(prototype_rows["ticker"]).intersection(reference_rows["ticker"])
    )
    for ticker in common_tickers:
        left = prototype_rows.loc[prototype_rows["ticker"] != ticker]
        right = reference_rows.loc[reference_rows["ticker"] != ticker]
        left_value = _subgroup_metric(left, config)
        right_value = _subgroup_metric(right, config)
        gain = _metric_gain(left_value, right_value, larger)
        rows.append(
            {
                "record_type": "leave_one_ticker_score_recompute",
                "evidence_type": "diagnostic_holdout_slice_not_refit",
                "qualifies_for_stability": False,
                "executed": True,
                "not_executed_reason": "",
                "target": target,
                "representation": prototype["representation"],
                "comparison_representation": reference["representation"],
                "excluded_ticker": ticker,
                "metric": metric,
                "larger_is_better": larger,
                "prototype_value": left_value,
                "reference_value": right_value,
                "gain": gain,
                "win": bool(gain > 0),
                "n_evaluation": int(len(left)),
                "refit_components": "none; score recomputation only",
            }
        )
    years = sorted(pd.to_datetime(prototype_rows["target_date"]).dt.year.unique())
    for year in years:
        left = prototype_rows.loc[
            pd.to_datetime(prototype_rows["target_date"]).dt.year != year
        ]
        right = reference_rows.loc[
            pd.to_datetime(reference_rows["target_date"]).dt.year != year
        ]
        left_value = _subgroup_metric(left, config)
        right_value = _subgroup_metric(right, config)
        gain = _metric_gain(left_value, right_value, larger)
        rows.append(
            {
                "record_type": "leave_one_year_score_recompute",
                "evidence_type": "diagnostic_holdout_slice_not_refit",
                "qualifies_for_stability": False,
                "executed": True,
                "not_executed_reason": "",
                "target": target,
                "representation": prototype["representation"],
                "comparison_representation": reference["representation"],
                "excluded_year": int(year),
                "metric": metric,
                "larger_is_better": larger,
                "prototype_value": left_value,
                "reference_value": right_value,
                "gain": gain,
                "win": bool(gain > 0),
                "n_evaluation": int(len(left)),
                "refit_components": "none; score recomputation only",
            }
        )
    trim_fraction = float(config["robustness"]["trim_extreme_residual_fraction"])
    if 0.0 < trim_fraction < 0.5 and len(prototype_rows):
        if "signed_residual" in prototype_rows.columns:
            severity = prototype_rows["signed_residual"].abs()
        elif target == "signed":
            severity = prototype_rows["y_true"].abs()
        else:
            severity = pd.to_numeric(prototype_rows["y_true"], errors="coerce").abs()
        threshold = severity.quantile(1.0 - trim_fraction)
        keys = prototype_rows.loc[
            severity <= threshold, ["ticker", "target_date"]
        ].drop_duplicates()
        left = prototype_rows.merge(keys, on=["ticker", "target_date"], how="inner")
        right = reference_rows.merge(keys, on=["ticker", "target_date"], how="inner")
        left_value = _subgroup_metric(left, config)
        right_value = _subgroup_metric(right, config)
        gain = _metric_gain(left_value, right_value, larger)
        rows.append(
            {
                "record_type": "trim_extreme_score_recompute",
                "evidence_type": "diagnostic_holdout_slice_not_refit",
                "qualifies_for_stability": False,
                "executed": True,
                "not_executed_reason": "",
                "target": target,
                "representation": prototype["representation"],
                "comparison_representation": reference["representation"],
                "metric": metric,
                "larger_is_better": larger,
                "prototype_value": left_value,
                "reference_value": right_value,
                "gain": gain,
                "win": bool(gain > 0),
                "trim_fraction": trim_fraction,
                "n_evaluation": int(len(left)),
                "refit_components": "none; score recomputation only",
            }
        )
    return rows


def _per_observation_loss(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    target_values = frame["target"].dropna().astype(str).unique()
    if len(target_values) != 1:
        raise ValueError("Per-observation loss requires one target")
    target = target_values[0]
    output = frame[["ticker", "target_date"]].copy()
    if target == "signed":
        final = frame["baseline_prediction"].to_numpy(dtype=float) + frame[
            "prediction"
        ].to_numpy(dtype=float)
        log_ratio = np.clip(
            frame["target_log_variance"].to_numpy(dtype=float) - final, -50.0, 50.0
        )
        output["loss"] = np.exp(log_ratio) - log_ratio - 1.0
        loss_name = "observation_qlike"
    elif target in {"magnitude", "squared"}:
        output["loss"] = np.square(
            frame["y_true"].to_numpy(dtype=float)
            - frame["prediction"].to_numpy(dtype=float)
        )
        loss_name = "squared_error"
    elif target.startswith("spike"):
        output["loss"] = np.square(
            frame["y_true"].to_numpy(dtype=float)
            - frame["probability"].to_numpy(dtype=float)
        )
        loss_name = "brier_loss"
    elif target.startswith("regime"):
        probabilities = frame[[f"prob_{index}" for index in range(3)]].to_numpy(
            dtype=float
        )
        labels = frame["y_true"].to_numpy(dtype=int)
        one_hot = np.eye(3)[labels]
        output["loss"] = np.sum(np.square(probabilities - one_hot), axis=1)
        loss_name = "multiclass_brier_loss"
    elif target == "uncertainty":
        residual = (
            frame["y_true"].to_numpy(dtype=float)
            - frame["baseline_prediction"].to_numpy(dtype=float)
        )
        scale = frame["scale"].to_numpy(dtype=float)
        if "degrees_of_freedom" in frame.columns and frame[
            "degrees_of_freedom"
        ].notna().any():
            degrees = frame["degrees_of_freedom"].to_numpy(dtype=float)
            student = np.isfinite(degrees)
            loss = np.empty(len(frame), dtype=float)
            loss[student] = (
                -stats.t.logpdf(
                    residual[student] / scale[student], df=degrees[student]
                )
                + np.log(scale[student])
            )
            loss[~student] = (
                0.5 * np.log(2.0 * np.pi)
                + np.log(scale[~student])
                + 0.5 * np.square(residual[~student] / scale[~student])
            )
            output["loss"] = loss
        else:
            output["loss"] = (
                0.5 * np.log(2.0 * np.pi)
                + np.log(scale)
                + 0.5 * np.square(residual / scale)
            )
        loss_name = "negative_log_likelihood"
    else:
        raise ValueError(target)
    return output, loss_name


def _moving_blocks(dates: np.ndarray, block_length: int) -> list[np.ndarray]:
    if not len(dates):
        return []
    length = min(max(int(block_length), 1), len(dates))
    return [dates[start : start + length] for start in range(0, len(dates) - length + 1)]


def _bootstrap_and_dm_rows(
    predictions: pd.DataFrame,
    prototype: pd.Series,
    reference: pd.Series,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    target = str(prototype["target"])
    left = _prediction_subset(predictions, prototype, "test")
    right = _prediction_subset(predictions, reference, "test")
    common_seeds = sorted(set(left["seed"]).intersection(right["seed"]))
    if common_seeds:
        seed = int(common_seeds[0])
        left = left.loc[left["seed"] == seed]
        right = right.loc[right["seed"] == seed]
    left_loss, loss_name = _per_observation_loss(left)
    right_loss, _ = _per_observation_loss(right)
    left_loss = (
        left_loss.groupby(["ticker", "target_date"], sort=True, observed=True)[
            "loss"
        ]
        .mean()
        .reset_index()
    )
    right_loss = (
        right_loss.groupby(["ticker", "target_date"], sort=True, observed=True)[
            "loss"
        ]
        .mean()
        .reset_index()
    )
    paired = left_loss.merge(
        right_loss,
        on=["ticker", "target_date"],
        suffixes=("_prototype", "_reference"),
        validate="one_to_one",
    )
    paired["target_date"] = pd.to_datetime(paired["target_date"])
    paired["loss_gain"] = paired["loss_reference"] - paired["loss_prototype"]
    by_date = paired.groupby("target_date", sort=True)["loss_gain"].mean()
    rows: list[dict[str, Any]] = []
    if by_date.empty:
        return [
            {
                "record_type": "block_bootstrap",
                "evidence_type": "diagnostic_locked_test_not_refit",
                "qualifies_for_stability": False,
                "executed": False,
                "not_executed_reason": "No paired locked test predictions",
                "target": target,
                "representation": prototype["representation"],
                "comparison_representation": reference["representation"],
                "metric": loss_name,
                "refit_components": "none",
            }
        ]
    repetitions = int(config["robustness"]["block_bootstrap_repetitions"])
    block_length = int(config["robustness"]["block_length_days"])
    confidence = float(config["robustness"]["confidence_level"])
    dates = by_date.index.to_numpy()
    blocks = _moving_blocks(dates, block_length)
    rng = np.random.default_rng(int(config["project"]["seed"]) + 4409)
    bootstrap = np.empty(repetitions, dtype=float)
    blocks_needed = int(math.ceil(len(dates) / max(block_length, 1)))
    for repetition in range(repetitions):
        chosen = rng.integers(0, len(blocks), size=blocks_needed)
        sampled_dates = np.concatenate([blocks[index] for index in chosen])[: len(dates)]
        bootstrap[repetition] = float(by_date.reindex(sampled_dates).mean())
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(bootstrap, [tail, 1.0 - tail])
    rows.append(
        {
            "record_type": "block_bootstrap",
            "evidence_type": "diagnostic_locked_test_not_refit",
            "qualifies_for_stability": False,
            "executed": True,
            "not_executed_reason": "",
            "target": target,
            "representation": prototype["representation"],
            "comparison_representation": reference["representation"],
            "metric": f"reference_minus_prototype_{loss_name}",
            "larger_is_better": True,
            "gain": float(by_date.mean()),
            "win": bool(by_date.mean() > 0),
            "ci_lower": float(lower),
            "ci_upper": float(upper),
            "n_evaluation": int(len(paired)),
            "refit_components": "none; locked test loss resampling only",
        }
    )

    differential = by_date.to_numpy(dtype=float)
    if len(differential) < max(20, 2 * block_length):
        rows.append(
            {
                "record_type": "diebold_mariano",
                "evidence_type": "diagnostic_locked_test_not_refit",
                "qualifies_for_stability": False,
                "executed": False,
                "not_executed_reason": "Too few paired dates for HAC loss-difference test",
                "target": target,
                "representation": prototype["representation"],
                "comparison_representation": reference["representation"],
                "metric": loss_name,
                "n_evaluation": int(len(differential)),
                "refit_components": "none",
            }
        )
    else:
        centered = differential - differential.mean()
        lag = min(block_length - 1, len(centered) - 2)
        long_run_variance = float(np.dot(centered, centered) / len(centered))
        for offset in range(1, lag + 1):
            covariance = float(
                np.dot(centered[offset:], centered[:-offset]) / len(centered)
            )
            weight = 1.0 - offset / (lag + 1.0)
            long_run_variance += 2.0 * weight * covariance
        standard_error = math.sqrt(
            max(long_run_variance, 1.0e-16) / len(centered)
        )
        statistic = float(differential.mean() / standard_error)
        rows.append(
            {
                "record_type": "diebold_mariano",
                "evidence_type": "diagnostic_locked_test_not_refit",
                "qualifies_for_stability": False,
                "executed": True,
                "not_executed_reason": "",
                "target": target,
                "representation": prototype["representation"],
                "comparison_representation": reference["representation"],
                "metric": f"reference_minus_prototype_{loss_name}",
                "larger_is_better": True,
                "gain": float(differential.mean()),
                "dm_statistic": statistic,
                "pvalue": float(2.0 * stats.norm.sf(abs(statistic))),
                "n_evaluation": int(len(differential)),
                "refit_components": "none; locked test HAC comparison only",
            }
        )
    return rows


def _fold_schedule(
    dates: np.ndarray,
    n_folds: int,
    min_train_days: int,
    min_validation_days: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_folds < 1 or len(dates) < min_train_days + n_folds * min_validation_days:
        return []
    validation_size = max(
        min_validation_days, (len(dates) - min_train_days) // n_folds
    )
    initial_train = len(dates) - n_folds * validation_size
    if initial_train < min_train_days:
        validation_size = (len(dates) - min_train_days) // n_folds
        initial_train = len(dates) - n_folds * validation_size
    if validation_size < min_validation_days:
        return []
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in range(n_folds):
        train_end = initial_train + fold * validation_size
        validation_end = train_end + validation_size
        train_dates = dates[:train_end]
        validation_dates = dates[train_end:validation_end]
        if len(train_dates) < min_train_days or len(validation_dates) < min_validation_days:
            continue
        if pd.Timestamp(train_dates[-1]) >= pd.Timestamp(validation_dates[0]):
            raise AssertionError("Chronological fold boundary is not strict")
        folds.append((train_dates, validation_dates))
    return folds


def _selected_model_candidate(
    config: Mapping[str, Any], selection: pd.Series
) -> Candidate:
    path = project_path(config, "outputs", "tables", "target_training_manifest.csv")
    manifest = safe_read_table(path)
    validate_required_columns(
        manifest,
        [
            "target",
            "representation",
            "input_variant",
            "model",
            "config_id",
            "parameters",
        ],
        "target training manifest",
    )
    mask = _identity_mask(manifest, selection)
    for column in ("representation_variant", "model", "config_id"):
        if column in manifest.columns and column in selection and not pd.isna(
            selection[column]
        ):
            mask &= manifest[column].astype(str) == str(selection[column])
    rows = manifest.loc[mask]
    if rows.empty:
        raise ValueError(
            "Locked configuration is absent from target_training_manifest.csv: "
            f"{dict(selection)}"
        )
    parameters = json.loads(str(rows.iloc[0]["parameters"]))
    if not isinstance(parameters, dict):
        raise ValueError("Target model parameters must decode to a mapping")
    candidate = Candidate(str(selection["model"]), parameters)
    if candidate.identifier != str(selection["config_id"]):
        raise ValueError("Reconstructed target candidate does not match locked config_id")
    return candidate


def _locked_manifest_variant_and_pooling(
    config: Mapping[str, Any], selection: Mapping[str, Any]
) -> tuple[str, str]:
    """Recover the manifest variant and pooling encoded in a training identity."""

    locked_variant = str(selection.get("representation_variant", "selected_default"))
    marker = "__pool_"
    if marker in locked_variant:
        manifest_variant, pooling = locked_variant.rsplit(marker, 1)
        if not manifest_variant or not pooling:
            raise ValueError(
                f"Malformed locked representation variant: {locked_variant!r}"
            )
        return manifest_variant, pooling
    return locked_variant, str(config["prototype"]["pooling"])


def _prototype_level_configuration(
    config: Mapping[str, Any], selection: pd.Series
) -> dict[str, dict[str, Any]]:
    suffix = _variant_suffix(_events_variant(config))
    representation_manifest_path = _first_existing(
        (
            project_path(
                config,
                "data",
                "processed",
                f"representation_manifest{suffix}.csv",
            ),
            project_path(
                config,
                "outputs",
                "tables",
                f"representation_manifest{suffix}.csv",
            ),
            project_path(config, "data", "processed", "representation_manifest.csv"),
            project_path(config, "outputs", "tables", "representation_manifest.csv"),
        )
    )
    if representation_manifest_path is None:
        raise FileNotFoundError("representation_manifest.csv")
    representation_manifest = safe_read_table(representation_manifest_path)
    validate_required_columns(
        representation_manifest,
        [
            "representation",
            "representation_variant",
            "pooling",
            "prototype_candidate_id",
        ],
        "representation manifest",
    )
    manifest_variant, locked_pooling = _locked_manifest_variant_and_pooling(
        config, selection
    )
    mask = (
        representation_manifest["representation"].astype(str)
        == str(selection["representation"])
    )
    mask &= (
        representation_manifest["representation_variant"].astype(str)
        == manifest_variant
    )
    mask &= representation_manifest["pooling"].astype(str).eq(locked_pooling)
    rows = representation_manifest.loc[mask]
    if rows.empty:
        raise ValueError(
            "Locked prototype representation variant/pooling is absent from manifest: "
            f"representation={selection['representation']!r}, "
            f"variant={manifest_variant!r}, pooling={locked_pooling!r}"
        )
    candidate_payload = json.loads(str(rows.iloc[0]["prototype_candidate_id"]))
    if not isinstance(candidate_payload, dict):
        raise ValueError("prototype_candidate_id must be a level-to-candidate mapping")
    prototype_manifest = safe_read_table(
        project_path(
            config,
            "data",
            "processed",
            f"prototype_manifest{suffix}.csv",
        )
    )
    validate_required_columns(
        prototype_manifest,
        ["candidate_id", "news_level", "k", "pca_dim", "temperature"],
        "prototype manifest",
    )
    result: dict[str, dict[str, Any]] = {}
    for level in NEWS_LEVELS:
        candidate_id = candidate_payload.get(level)
        if candidate_id is None:
            raise ValueError(f"Locked representation has no {level} candidate")
        matches = prototype_manifest.loc[
            prototype_manifest["candidate_id"].astype(str) == str(candidate_id)
        ]
        if matches.empty:
            raise ValueError(f"Prototype candidate {candidate_id} is absent from manifest")
        row = matches.iloc[0]
        result[level] = {
            "candidate_id": str(candidate_id),
            "k": int(row["k"]),
            "pca_dim": None if pd.isna(row["pca_dim"]) else int(row["pca_dim"]),
            "temperature": float(row["temperature"]),
        }
    return result


def _fold_event_inputs(
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, np.ndarray]:
    event_paths, metadata_path, embedding_path = _fold_event_artifact_paths(
        config
    )
    event_path = _first_existing(event_paths)
    if event_path is None:
        raise FileNotFoundError(
            f"{_events_variant(config)} events: "
            + "; ".join(str(path) for path in event_paths)
        )
    events = safe_read_table(event_path)
    metadata = safe_read_table(metadata_path)
    validate_required_columns(
        events,
        ["event_id", "date", "news_level", "available_to_tickers"],
        f"{_events_variant(config)} events",
    )
    validate_required_columns(
        metadata, ["event_id", "embedding_index"], "embedding metadata"
    )
    merged = events.merge(
        metadata[["event_id", "embedding_index"]],
        on="event_id",
        how="left",
        validate="one_to_one",
    )
    if merged["embedding_index"].isna().any():
        raise ValueError("Some fold events have no embedding")
    embeddings_file = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
    indices = merged["embedding_index"].to_numpy(dtype=int)
    embeddings = l2_normalize(np.asarray(embeddings_file[indices], dtype=np.float64))
    merged["date"] = pd.to_datetime(merged["date"], errors="raise").dt.normalize()
    merged["news_level"] = merged["news_level"].astype(str).str.lower()
    return merged.reset_index(drop=True), embeddings


def _fit_fold_event_representation(
    events: pd.DataFrame,
    embeddings: np.ndarray,
    train_end: pd.Timestamp,
    validation_end: pd.Timestamp,
    level_config: Mapping[str, Mapping[str, Any]],
    seed: int,
    n_init: int,
    max_iter: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]] | str:
    selected_mask = events["date"] <= validation_end
    fold_events = events.loc[selected_mask].copy().reset_index(drop=True)
    fold_embeddings = embeddings[selected_mask.to_numpy()]
    fold_events["split"] = np.where(
        fold_events["date"] <= train_end, "train", "validation"
    )
    fold_events.insert(0, "event_pos", np.arange(len(fold_events), dtype=np.int64))
    soft_by_level: dict[str, np.ndarray] = {}
    hard_by_level: dict[str, np.ndarray] = {}
    pca_by_level: dict[str, np.ndarray] = {}
    for diagnostic in (
        "assignment_entropy",
        "novelty",
        "nearest_distance",
        "nearest_similarity",
        "effective_soft_prototypes",
    ):
        fold_events[diagnostic] = np.nan
    for level in NEWS_LEVELS:
        level_mask = fold_events["news_level"].to_numpy() == level
        train_mask = level_mask & (fold_events["split"].to_numpy() == "train")
        settings = level_config[level]
        k = int(settings["k"])
        pca_dim = settings["pca_dim"]
        if int(train_mask.sum()) < k:
            return f"{level}: train events {int(train_mask.sum())} < k={k}"
        if pca_dim is not None and int(pca_dim) > min(
            int(train_mask.sum()), fold_embeddings.shape[1]
        ):
            return (
                f"{level}: PCA dim {pca_dim} exceeds fold train rank "
                f"{min(int(train_mask.sum()), fold_embeddings.shape[1])}"
            )
        level_indices = np.flatnonzero(level_mask)
        level_values = fold_embeddings[level_mask]
        local_train = fold_events.loc[level_mask, "split"].to_numpy() == "train"
        _, transformed, _ = _fit_projection(
            level_values[local_train],
            level_values,
            pca_dim,
            seed,
        )
        centers = _fit_spherical_kmeans(
            transformed[local_train], k, seed, n_init, max_iter
        )
        assignment = _assign(
            transformed, centers, float(settings["temperature"])
        )
        soft_matrix = np.zeros((len(fold_events), k), dtype=np.float32)
        hard_matrix = np.zeros((len(fold_events), k), dtype=np.float32)
        soft_matrix[level_indices] = assignment["soft_assignment"]
        hard_matrix[
            level_indices, assignment["hard_cluster_id"].astype(int)
        ] = 1.0
        soft_by_level[level] = soft_matrix
        hard_by_level[level] = hard_matrix
        for diagnostic in (
            "assignment_entropy",
            "novelty",
            "nearest_distance",
            "nearest_similarity",
            "effective_soft_prototypes",
        ):
            fold_events.loc[level_mask, diagnostic] = assignment[diagnostic]

        comparator_dim = min(k, int(train_mask.sum()), fold_embeddings.shape[1])
        solver = (
            "randomized"
            if comparator_dim < min(int(train_mask.sum()), fold_embeddings.shape[1])
            else "full"
        )
        comparator = PCA(
            n_components=comparator_dim, svd_solver=solver, random_state=seed
        )
        comparator.fit(level_values[local_train])
        reduced = l2_normalize(comparator.transform(level_values))
        pca_matrix = np.zeros((len(fold_events), comparator_dim), dtype=np.float32)
        pca_matrix[level_indices] = reduced.astype(np.float32)
        pca_by_level[level] = pca_matrix
    return fold_events, soft_by_level, hard_by_level, pca_by_level


def _aggregate_fold_representation(
    config: Mapping[str, Any],
    market: pd.DataFrame,
    events: pd.DataFrame,
    soft: Mapping[str, np.ndarray],
    hard: Mapping[str, np.ndarray],
    pca: Mapping[str, np.ndarray],
    representation: str,
    seed: int,
    pooling: str,
) -> pd.DataFrame:
    tickers = [str(value).upper() for value in config["data"]["tickers"]]
    edges = _build_edges(events, tickers, "true", market, seed)
    if edges.empty:
        raise ValueError("No fold event-to-ticker edges were available")
    pooling = str(pooling)
    instances = _instances(
        market,
        edges,
        pooling,
        float(config["prototype"]["exponential_half_life_days"]),
        int(config["prototype"]["max_lag_days"]),
    )
    sentinel = float(config["prototype"].get("days_since_sentinel", 3650.0))
    soft_blocks = [
        _aggregate_matrix(
            instances, soft[level], level, len(market), pooling, "softproto"
        )
        for level in NEWS_LEVELS
    ]
    hard_blocks = [
        _aggregate_matrix(
            instances, hard[level], level, len(market), pooling, "hardproto"
        )
        for level in NEWS_LEVELS
    ]
    pca_blocks = [
        _aggregate_matrix(instances, pca[level], level, len(market), pooling, "pca")
        for level in NEWS_LEVELS
    ]
    metadata_blocks = [
        _aggregate_metadata(market, events, edges, instances, level, sentinel)
        for level in NEWS_LEVELS
    ]
    if representation == "R5":
        blocks = hard_blocks
    elif representation == "R6":
        blocks = soft_blocks
    elif representation == "R7":
        blocks = [*soft_blocks, *metadata_blocks]
    elif representation == "R8":
        blocks = [*pca_blocks, *soft_blocks, *metadata_blocks]
    else:
        raise ValueError(f"Fold refit does not support representation {representation}")
    return _join_feature_blocks(market, blocks)


def _relabel_fold_targets(
    train: pd.DataFrame, validation: pd.DataFrame, epsilon: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    validation = validation.copy()
    for frame in (train, validation):
        frame["residual_magnitude"] = frame["signed_residual"].abs()
        frame["squared_residual"] = frame["signed_residual"].pow(2)
        frame["log_squared_residual"] = np.log(
            frame["squared_residual"].to_numpy(dtype=float) + epsilon
        )
    thresholds: dict[str, tuple[float, float, float, float, float]] = {}
    for ticker, group in train.groupby("ticker", sort=True):
        values = group["signed_residual"].to_numpy(dtype=float)
        absolute = np.abs(values)
        center = float(np.mean(values))
        scale = max(float(np.std(values, ddof=1)), epsilon)
        thresholds[str(ticker)] = tuple(
            [
                *(
                    float(np.quantile(absolute, quantile))
                    for quantile in (0.50, 0.90, 0.95)
                ),
                center,
                scale,
            ]
        )
    train_centers = train["ticker"].map(
        {ticker: values[3] for ticker, values in thresholds.items()}
    )
    train_scales = train["ticker"].map(
        {ticker: values[4] for ticker, values in thresholds.items()}
    )
    train_standardized_absolute = (
        (train["signed_residual"] - train_centers) / train_scales.clip(lower=epsilon)
    ).abs()
    pooled_q50, pooled_q90, pooled_q95 = (
        float(train_standardized_absolute.quantile(quantile))
        for quantile in (0.50, 0.90, 0.95)
    )
    for frame in (train, validation):
        q50 = frame["ticker"].map(
            {ticker: values[0] for ticker, values in thresholds.items()}
        )
        q90 = frame["ticker"].map(
            {ticker: values[1] for ticker, values in thresholds.items()}
        )
        q95 = frame["ticker"].map(
            {ticker: values[2] for ticker, values in thresholds.items()}
        )
        if q50.isna().any() or q90.isna().any() or q95.isna().any():
            raise ValueError("Fold validation contains a ticker absent from fold train")
        frame["spike_q90"] = (frame["residual_magnitude"] > q90).astype(int)
        frame["spike_q95"] = (frame["residual_magnitude"] > q95).astype(int)
        frame["regime"] = np.select(
            [
                frame["residual_magnitude"] <= q50,
                frame["residual_magnitude"] <= q90,
            ],
            [0, 1],
            default=2,
        ).astype(int)
        centers = frame["ticker"].map(
            {ticker: values[3] for ticker, values in thresholds.items()}
        )
        scales = frame["ticker"].map(
            {ticker: values[4] for ticker, values in thresholds.items()}
        )
        frame["standardized_residual"] = (
            frame["signed_residual"] - centers
        ) / scales.clip(lower=epsilon)
        standardized_absolute = frame["standardized_residual"].abs()
        frame["spike_q90_pooled_standardized"] = (
            standardized_absolute > pooled_q90
        ).astype(int)
        frame["spike_q95_pooled_standardized"] = (
            standardized_absolute > pooled_q95
        ).astype(int)
        frame["regime_pooled_standardized"] = np.select(
            [
                standardized_absolute <= pooled_q50,
                standardized_absolute <= pooled_q90,
            ],
            [0, 1],
            default=2,
        ).astype(int)
    return train, validation


def _fit_locked_target_on_fold(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    selection: pd.Series,
    candidate: Candidate,
    text_features: pd.DataFrame | None,
    config: Mapping[str, Any],
    seed: int,
) -> tuple[float, str, bool, int]:
    joined_train = train.copy()
    joined_validation = validation.copy()
    text_columns: list[str] = []
    if text_features is not None:
        feature_columns = [
            column
            for column in text_features.columns
            if column not in {"ticker", "feature_date", "split"}
        ]
        joined_train = joined_train.merge(
            text_features[["ticker", "feature_date", *feature_columns]],
            on=["ticker", "feature_date"],
            how="left",
            validate="one_to_one",
        )
        joined_validation = joined_validation.merge(
            text_features[["ticker", "feature_date", *feature_columns]],
            on=["ticker", "feature_date"],
            how="left",
            validate="one_to_one",
        )
        text_columns = feature_columns
    price_columns = [
        column
        for column in train.columns
        if _is_usable_numeric(column, train[column])
    ]
    input_variant = str(selection["input_variant"])
    if input_variant == "text_only":
        features = text_columns
    elif input_variant == "price_only":
        features = price_columns
    else:
        features = list(dict.fromkeys([*price_columns, *text_columns]))
    if not features:
        raise ValueError("Locked fold model has no usable features")
    processor = _make_preprocessor(features)
    x_train = np.asarray(
        processor.fit_transform(joined_train[[*features, "ticker"]]), dtype=np.float32
    )
    x_validation = np.asarray(
        processor.transform(joined_validation[[*features, "ticker"]]), dtype=np.float32
    )
    target = str(selection["target"])
    model = _fit_candidate(
        candidate, x_train, joined_train, target, config, seed
    )
    prediction = _predict_candidate(candidate, model, x_validation, config)
    metric, value, larger = _primary_score(target, joined_validation, prediction)
    return float(value), metric, larger, int(x_train.shape[1])


def run_chronological_refit_folds(
    config: Mapping[str, Any],
    prototype_selection: pd.Series,
    reference_selection: pd.Series,
    logger: Any,
) -> pd.DataFrame:
    """Execute the only robustness evidence eligible for the stability gate."""
    if (
        int(prototype_selection.get("per_repeat_model_config_count", 1)) > 1
        or int(reference_selection.get("per_repeat_model_config_count", 1)) > 1
    ):
        return pd.DataFrame(
            [
                {
                    "record_type": "chronological_fold_refit",
                    "evidence_type": "chronological_full_refit",
                    "qualifies_for_stability": True,
                    "executed": False,
                    "not_executed_reason": (
                        "The locked prototype or R0 family contains multiple "
                        "per-repeat target model configurations; no seed-free "
                        "target model configuration is available for an honest "
                        "fold refit"
                    ),
                    "target": prototype_selection["target"],
                    "representation": prototype_selection["representation"],
                    "comparison_representation": reference_selection["representation"],
                    "refit_components": "none",
                }
            ]
        )
    event_paths, metadata_path, embedding_path = _fold_event_artifact_paths(
        config
    )
    event_path = _first_existing(event_paths) or event_paths[0]
    suffix = _variant_suffix(_events_variant(config))
    prerequisites = [
        project_path(config, "data", "processed", "residual_targets.parquet"),
        project_path(config, "data", "processed", "market_supervised.parquet"),
        embedding_path,
        metadata_path,
        event_path,
        project_path(
            config,
            "data",
            "processed",
            f"prototype_manifest{suffix}.csv",
        ),
        project_path(config, "outputs", "tables", "target_training_manifest.csv"),
    ]
    missing = [str(path) for path in prerequisites if not path.exists()]
    n_folds = int(config["robustness"]["chronological_folds"])
    if missing:
        return pd.DataFrame(
            [
                {
                    "record_type": "chronological_fold_refit",
                    "evidence_type": "chronological_full_refit",
                    "qualifies_for_stability": True,
                    "executed": False,
                    "not_executed_reason": "Missing artifacts: " + "; ".join(missing),
                    "target": prototype_selection["target"],
                    "representation": prototype_selection["representation"],
                    "comparison_representation": reference_selection["representation"],
                    "refit_components": (
                        "baseline, thresholds, PCA, KMeans, prototype assignment, "
                        "aggregation, scaler, target model"
                    ),
                }
            ]
        )
    residuals = safe_read_table(prerequisites[0])
    validate_required_columns(
        residuals,
        [
            "ticker",
            "feature_date",
            "target_date",
            "split",
            "target_log_variance",
            "baseline_prediction",
            "signed_residual",
        ],
        "residual targets for fold refit",
    )
    residuals = residuals.copy()
    residuals["feature_date"] = pd.to_datetime(
        residuals["feature_date"], errors="raise"
    ).dt.normalize()
    residuals["target_date"] = pd.to_datetime(
        residuals["target_date"], errors="raise"
    ).dt.normalize()
    # The main validation block selected the family/model.  It must not also
    # appear inside the fold evidence used to confirm that selection.  These
    # expanding refits therefore use original-train dates only.
    pretest = residuals.loc[residuals["split"].eq("train")].copy()
    dates = np.sort(pretest["target_date"].unique())
    folds = _fold_schedule(
        dates,
        n_folds,
        int(config["split"]["min_train_days"]),
        int(config["split"]["min_validation_days"]),
    )
    if len(folds) < n_folds:
        return pd.DataFrame(
            [
                {
                    "record_type": "chronological_fold_refit",
                    "evidence_type": "chronological_full_refit",
                    "qualifies_for_stability": True,
                    "executed": False,
                    "not_executed_reason": (
                        f"Only {len(folds)} valid folds could be formed; {n_folds} required"
                    ),
                    "target": prototype_selection["target"],
                    "representation": prototype_selection["representation"],
                    "comparison_representation": reference_selection["representation"],
                    "n_evaluation": int(len(dates)),
                    "refit_components": (
                        "baseline, thresholds, PCA, KMeans, prototype assignment, "
                        "aggregation, scaler, target model"
                    ),
                }
            ]
        )
    level_config = _prototype_level_configuration(config, prototype_selection)
    _, locked_pooling = _locked_manifest_variant_and_pooling(
        config, prototype_selection
    )
    events, embeddings = _fold_event_inputs(config)
    prototype_candidate = _selected_model_candidate(config, prototype_selection)
    reference_candidate = _selected_model_candidate(config, reference_selection)
    market_supervised = load_market_supervised(dict(config))
    baseline_selection = read_json(
        project_path(config, "outputs", "models", "baseline_selection.json")
    )
    baseline_method = str(baseline_selection["baseline"])
    baseline_alpha = float(baseline_selection.get("alpha") or 0.0)
    seeds = list(
        dict.fromkeys(
            [
                int(config["project"]["seed"]),
                *(int(value) for value in config["robustness"]["seeds"]),
            ]
        )
    )
    rows: list[dict[str, Any]] = []
    for fold_index, (train_dates, validation_dates) in enumerate(folds, start=1):
        train = pretest.loc[pretest["target_date"].isin(train_dates)].copy()
        validation = pretest.loc[
            pretest["target_date"].isin(validation_dates)
        ].copy()
        train_end = pd.Timestamp(train["feature_date"].max())
        validation_end = pd.Timestamp(validation["feature_date"].max())
        # Preserve every earlier market observation for the fold baseline, even
        # when early dates are absent from residual_targets because the OOF
        # residual generator needs an initial history window.
        market_history = market_supervised.loc[
            market_supervised["target_date"] <= pd.Timestamp(train_dates[-1])
        ].copy()
        market_validation = market_supervised.loc[
            market_supervised["target_date"].isin(validation_dates)
        ].copy()
        baseline_model = fit_baseline(
            market_history, baseline_method, baseline_alpha
        )
        fold_baseline = market_validation[
            ["ticker", "feature_date", "target_date", "target_log_variance"]
        ].copy()
        fold_baseline["baseline_prediction"] = predict_baseline(
            baseline_model, market_validation
        )
        validation = validation.drop(
            columns=["baseline_prediction", "target_log_variance"],
            errors="raise",
        ).merge(
            fold_baseline,
            on=["ticker", "feature_date", "target_date"],
            how="left",
            validate="one_to_one",
        )
        if validation[
            ["target_log_variance", "baseline_prediction"]
        ].isna().any().any():
            raise ValueError(
                f"Fold {fold_index} baseline predictions did not align to every "
                "validation ticker/date"
            )
        validation["signed_residual"] = (
            validation["target_log_variance"] - validation["baseline_prediction"]
        )
        train, validation = _relabel_fold_targets(
            train,
            validation,
            float(config["targets"]["residual_epsilon"]),
        )
        fold_market = pd.concat(
            [
                train[["ticker", "feature_date"]].assign(split="train"),
                validation[["ticker", "feature_date"]].assign(split="validation"),
            ],
            ignore_index=True,
        ).sort_values(["feature_date", "ticker"], kind="mergesort")
        fold_market = fold_market.reset_index(drop=True)
        fold_market.insert(0, "__row_id", np.arange(len(fold_market), dtype=np.int64))
        target_name = str(prototype_selection["target"])
        class_reason = ""
        if target_name.startswith("spike") and (
            train[target_name].nunique() < 2
            or validation[target_name].nunique() < 2
        ):
            class_reason = (
                f"{target_name} does not contain both classes in fold train "
                "and validation"
            )
        elif target_name.startswith("regime") and (
            train[target_name].nunique() < 3
            or validation[target_name].nunique() < 3
        ):
            class_reason = (
                f"{target_name} does not contain all three classes in fold "
                "train and validation"
            )
        if class_reason:
            for seed in seeds:
                rows.append(
                    {
                        "record_type": "chronological_fold_refit",
                        "evidence_type": "chronological_full_refit",
                        "qualifies_for_stability": True,
                        "executed": False,
                        "not_executed_reason": class_reason,
                        "target": target_name,
                        "representation": prototype_selection["representation"],
                        "comparison_representation": reference_selection[
                            "representation"
                        ],
                        "seed": seed,
                        "fold": fold_index,
                        "n_train": int(len(train)),
                        "n_evaluation": int(len(validation)),
                        "refit_components": (
                            "baseline and fold thresholds executed; target class "
                            "support prevented model refit"
                        ),
                    }
                )
            continue
        for seed in seeds:
            set_global_seed(
                seed, bool(config["project"].get("deterministic", True))
            )
            fitted = _fit_fold_event_representation(
                events,
                embeddings,
                train_end,
                validation_end,
                level_config,
                seed,
                int(config["prototype"]["n_init"]),
                int(config["prototype"]["max_iter"]),
            )
            if isinstance(fitted, str):
                rows.append(
                    {
                        "record_type": "chronological_fold_refit",
                        "evidence_type": "chronological_full_refit",
                        "qualifies_for_stability": True,
                        "executed": False,
                        "not_executed_reason": fitted,
                        "target": prototype_selection["target"],
                        "representation": prototype_selection["representation"],
                        "comparison_representation": reference_selection["representation"],
                        "seed": seed,
                        "fold": fold_index,
                        "n_train": int(len(train)),
                        "n_evaluation": int(len(validation)),
                        "refit_components": (
                            "baseline and thresholds executed; prototype rank/event "
                            "requirements prevented remaining refits"
                        ),
                    }
                )
                continue
            fold_events, soft, hard, pca = fitted
            text_features = _aggregate_fold_representation(
                config,
                fold_market,
                fold_events,
                soft,
                hard,
                pca,
                str(prototype_selection["representation"]),
                seed,
                locked_pooling,
            )
            prototype_value, metric, larger, prototype_features = (
                _fit_locked_target_on_fold(
                    train,
                    validation,
                    prototype_selection,
                    prototype_candidate,
                    text_features,
                    config,
                    seed,
                )
            )
            reference_value, reference_metric, reference_larger, reference_features = (
                _fit_locked_target_on_fold(
                    train,
                    validation,
                    reference_selection,
                    reference_candidate,
                    None,
                    config,
                    seed,
                )
            )
            if metric != reference_metric or larger != reference_larger:
                raise AssertionError("Fold prototype/reference primary metrics disagree")
            gain = _metric_gain(prototype_value, reference_value, larger)
            rows.append(
                {
                    "record_type": "chronological_fold_refit",
                    "evidence_type": "chronological_full_refit",
                    "qualifies_for_stability": True,
                    "executed": True,
                    "not_executed_reason": "",
                    "target": prototype_selection["target"],
                    "representation": prototype_selection["representation"],
                    "comparison_representation": reference_selection["representation"],
                    "seed": seed,
                    "fold": fold_index,
                    "train_start": pd.Timestamp(train_dates[0]),
                    "train_end": pd.Timestamp(train_dates[-1]),
                    "validation_start": pd.Timestamp(validation_dates[0]),
                    "validation_end": pd.Timestamp(validation_dates[-1]),
                    "metric": metric,
                    "larger_is_better": larger,
                    "prototype_value": prototype_value,
                    "reference_value": reference_value,
                    "gain": gain,
                    "win": bool(gain > 0),
                    "n_train": int(len(train)),
                    "n_evaluation": int(len(validation)),
                    "prototype_feature_count": prototype_features,
                    "reference_feature_count": reference_features,
                    "refit_components": (
                        "baseline validation forecast; train-only residual thresholds; "
                        "per-level PCA; KMeans centroids; soft/hard assignments; "
                        "ticker-day aggregation; imputer/scaler/ticker one-hot; target model"
                    ),
                }
            )
            logger.info(
                "Full refit fold=%d seed=%d target=%s gain=%.6f",
                fold_index,
                seed,
                prototype_selection["target"],
                gain,
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["fold_evaluation_scope"] = "original_train_only"
        result["configuration_selection_scope"] = (
            "disjoint_later_main_validation"
        )
        result["selection_fold_overlap"] = False
    return result


def _refit_summary_rows(
    full_refit: pd.DataFrame, config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    executed = full_refit.loc[
        full_refit.get(
            "executed", pd.Series(index=full_refit.index, dtype=bool)
        )
        .fillna(False)
        .astype(bool)
    ]
    if executed.empty:
        return []
    gains = pd.to_numeric(executed["gain"], errors="coerce").dropna().to_numpy()
    if not len(gains):
        return []
    confidence = float(config["robustness"]["confidence_level"])
    if len(gains) == 1:
        lower = upper = float(gains[0])
        standard_deviation = 0.0
    else:
        standard_deviation = float(np.std(gains, ddof=1))
        half_width = float(
            stats.t.ppf((1.0 + confidence) / 2.0, df=len(gains) - 1)
            * stats.sem(gains)
        )
        lower = float(np.mean(gains) - half_width)
        upper = float(np.mean(gains) + half_width)
    first = executed.iloc[0]
    return [
        {
            "record_type": "chronological_fold_refit_summary",
            "evidence_type": "summary_of_chronological_full_refits",
            "qualifies_for_stability": False,
            "executed": True,
            "not_executed_reason": "",
            "target": first["target"],
            "representation": first["representation"],
            "comparison_representation": first["comparison_representation"],
            "metric": first["metric"],
            "larger_is_better": True,
            "gain": float(np.mean(gains)),
            "gain_standard_deviation": standard_deviation,
            "win": bool(np.mean(gains > 0) >= 0.5),
            "win_rate": float(np.mean(gains > 0)),
            "ci_lower": lower,
            "ci_upper": upper,
            "n_evaluation": int(executed["n_evaluation"].sum()),
            "executed_seed_count": int(executed["seed"].nunique()),
            "executed_fold_count": int(executed["fold"].nunique()),
            "refit_components": "summary only; see qualifying fold rows",
        }
    ]


def _diagnostic_summary_rows(
    frame: pd.DataFrame, config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    confidence = float(config["robustness"]["confidence_level"])
    for record_type in (
        "leave_one_ticker_score_recompute",
        "leave_one_year_score_recompute",
        "trim_extreme_score_recompute",
    ):
        subset = frame.loc[
            (frame.get("record_type", "") == record_type)
            & frame.get(
                "executed", pd.Series(False, index=frame.index)
            ).fillna(False).astype(bool)
        ]
        gains = pd.to_numeric(subset.get("gain"), errors="coerce").dropna()
        if gains.empty:
            continue
        if len(gains) == 1:
            lower = upper = float(gains.iloc[0])
            standard_deviation = 0.0
        else:
            standard_deviation = float(gains.std(ddof=1))
            half_width = float(
                stats.t.ppf((1.0 + confidence) / 2.0, len(gains) - 1)
                * stats.sem(gains)
            )
            lower = float(gains.mean() - half_width)
            upper = float(gains.mean() + half_width)
        first = subset.iloc[0]
        rows.append(
            {
                "record_type": f"{record_type}_summary",
                "evidence_type": "summary_of_diagnostic_score_recomputations",
                "qualifies_for_stability": False,
                "executed": True,
                "not_executed_reason": "",
                "target": first["target"],
                "representation": first["representation"],
                "comparison_representation": first["comparison_representation"],
                "metric": first["metric"],
                "larger_is_better": True,
                "gain": float(gains.mean()),
                "gain_standard_deviation": standard_deviation,
                "win": bool((gains > 0).mean() >= 0.5),
                "win_rate": float((gains > 0).mean()),
                "ci_lower": lower,
                "ci_upper": upper,
                "n_evaluation": int(
                    pd.to_numeric(
                        subset["n_evaluation"], errors="coerce"
                    ).fillna(0).sum()
                ),
                "refit_components": (
                    "summary only; diagnostic score recomputation does not "
                    "qualify as refit stability"
                ),
            }
        )
    return rows


def _plot_placebos(
    placebo: pd.DataFrame, final_target: str, path: Path
) -> None:
    subset = placebo.loc[
        (placebo["target"] == final_target)
        & placebo["executed"].fillna(False).astype(bool)
    ].copy()
    if subset.empty:
        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.axis("off")
        axis.set_title("True versus shuffled news")
        axis.text(
            0.5,
            0.5,
            "No validation-locked true/placebo comparisons were available.",
            ha="center",
            va="center",
        )
    else:
        best_true = (
            subset.groupby("true_representation", sort=True)["validation_gain"]
            .mean()
            .sort_values(ascending=False)
            .index[0]
        )
        subset = subset.loc[subset["true_representation"] == best_true]
        labels = subset["placebo_representation"].astype(str)
        positions = np.arange(len(subset))
        width = 0.36
        figure, axis = plt.subplots(figsize=(9, 5))
        axis.bar(
            positions - width / 2,
            subset["true_validation_value"],
            width,
            label=f"True {best_true}",
        )
        axis.bar(
            positions + width / 2,
            subset["placebo_validation_value"],
            width,
            label="Placebo",
        )
        axis.set_xticks(positions)
        axis.set_xticklabels(labels)
        axis.set_ylabel(str(subset["primary_metric"].iloc[0]))
        axis.set_title(f"True versus shuffled/random news: {final_target}")
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _read_final_selection(
    config: Mapping[str, Any], locked: pd.DataFrame
) -> tuple[str, pd.Series | None, pd.Series | None]:
    decision_path = project_path(config, "outputs", "tables", "final_decision.csv")
    target = "none"
    final_was_available = False
    if decision_path.exists():
        decision = safe_read_table(decision_path)
        final = decision.loc[
            decision.get("record_type", pd.Series(index=decision.index, dtype=str))
            .astype(str)
            .eq("final")
        ]
        if not final.empty:
            final_was_available = True
            target = str(final.iloc[-1].get("best_target", final.iloc[-1].get("target", "none")))
    if final_was_available:
        return (
            target,
            _best_locked(locked, target, PROTOTYPE_REPRESENTATIONS),
            _best_locked(locked, target, ("R0",)),
        )
    if target == "none" or target not in set(locked["target"].astype(str)):
        candidates: list[tuple[float, str]] = []
        for candidate_target in sorted(locked["target"].astype(str).unique()):
            prototype = _best_locked(
                locked, candidate_target, PROTOTYPE_REPRESENTATIONS
            )
            reference = _best_locked(locked, candidate_target, ("R0",))
            if prototype is None or reference is None:
                continue
            _, larger = primary_metric_spec(candidate_target)
            candidates.append(
                (
                    _metric_gain(
                        float(prototype["primary_value"]),
                        float(reference["primary_value"]),
                        larger,
                    ),
                    candidate_target,
                )
            )
        if candidates:
            target = sorted(candidates, key=lambda item: (-item[0], item[1]))[0][1]
    return (
        target,
        _best_locked(locked, target, PROTOTYPE_REPRESENTATIONS),
        _best_locked(locked, target, ("R0",)),
    )


def _update_final_decision(
    config: Mapping[str, Any],
    placebo: pd.DataFrame,
    robustness: pd.DataFrame,
    target: str,
    prototype: pd.Series | None,
) -> Path | None:
    path = project_path(config, "outputs", "tables", "final_decision.csv")
    if not path.exists():
        return None
    decision = safe_read_table(path)
    final_mask = decision["record_type"].astype(str) == "final"
    if not final_mask.any():
        raise ValueError("final_decision.csv has no record_type=final row")
    threshold = float(config["decision"]["min_relative_gain"])
    relevant_placebo = placebo.loc[
        (placebo["target"] == target)
        & (
            placebo["true_representation"]
            == (prototype["representation"] if prototype is not None else "")
        )
    ]
    placebo_pass = bool(
        set(relevant_placebo["placebo_representation"].astype(str))
        == set(PLACEBO_REPRESENTATIONS)
        and relevant_placebo["validation_gain"].notna().all()
        and (relevant_placebo["validation_gain"] >= threshold).all()
        and relevant_placebo["test_gain"].notna().all()
        and (relevant_placebo["test_gain"] > 0).all()
    )
    qualifying = robustness.loc[
        robustness["qualifies_for_stability"].fillna(False).astype(bool)
        & robustness["executed"].fillna(False).astype(bool)
        & (robustness["target"].astype(str) == target)
    ]
    required_folds = int(config["robustness"]["chronological_folds"])
    required_seeds = len(
        set(
            [
                int(config["project"]["seed"]),
                *(int(value) for value in config["robustness"]["seeds"]),
            ]
        )
    )
    unique_folds = int(qualifying["fold"].dropna().nunique())
    unique_seeds = int(qualifying["seed"].dropna().nunique())
    win_rate = (
        float(qualifying["win"].astype(bool).mean()) if not qualifying.empty else np.nan
    )
    fold_stable = bool(
        unique_folds >= required_folds
        and unique_seeds >= required_seeds
        and len(qualifying) >= required_folds * required_seeds
        and np.isfinite(win_rate)
        and win_rate >= float(config["decision"]["min_fold_win_rate"])
    )
    ticker_diagnostics = robustness.loc[
        robustness["record_type"].astype(str).eq(
            "leave_one_ticker_score_recompute"
        )
        & robustness["executed"].fillna(False).astype(bool)
        & robustness["target"].astype(str).eq(target)
    ]
    required_tickers = len(config["data"]["tickers"])
    ticker_win_rate = (
        float(ticker_diagnostics["win"].fillna(False).astype(bool).mean())
        if not ticker_diagnostics.empty
        else np.nan
    )
    ticker_stable = bool(
        ticker_diagnostics["excluded_ticker"].dropna().astype(str).nunique()
        >= required_tickers
        and np.isfinite(ticker_win_rate)
        and ticker_win_rate
        >= float(config["decision"]["min_fold_win_rate"])
    )
    outlier_diagnostics = robustness.loc[
        robustness["record_type"].astype(str).eq(
            "trim_extreme_score_recompute"
        )
        & robustness["executed"].fillna(False).astype(bool)
        & robustness["target"].astype(str).eq(target)
    ]
    outlier_stable = bool(
        not outlier_diagnostics.empty
        and pd.to_numeric(
            outlier_diagnostics["gain"], errors="coerce"
        ).notna().all()
        and (
            pd.to_numeric(outlier_diagnostics["gain"], errors="coerce") > 0
        ).all()
    )
    stable = bool(fold_stable and ticker_stable and outlier_stable)
    decision.loc[final_mask, "true_vs_shuffled"] = placebo_pass
    decision.loc[final_mask, "real_news_better_than_shuffled"] = placebo_pass
    decision.loc[final_mask, "stable"] = stable
    decision.loc[final_mask, "stable_across_seed_fold"] = stable
    decision.loc[final_mask, "executed_refit_folds"] = unique_folds
    decision.loc[final_mask, "executed_refit_seeds"] = unique_seeds
    decision.loc[final_mask, "refit_fold_seed_win_rate"] = win_rate
    decision.loc[final_mask, "fold_seed_stable"] = fold_stable
    decision.loc[final_mask, "leave_one_ticker_win_rate"] = ticker_win_rate
    decision.loc[final_mask, "leave_one_ticker_stable"] = ticker_stable
    decision.loc[final_mask, "extreme_trim_stable"] = outlier_stable
    decision.loc[final_mask, "best_target"] = target
    current = str(decision.loc[final_mask, "decision"].iloc[-1])
    target_mask = (
        (decision["record_type"].astype(str) == "target")
        & (decision["target"].astype(str) == target)
    )
    provisional = (
        str(
            decision.loc[
                target_mask, "provisional_go_label_after_refit_evidence"
            ].dropna().iloc[-1]
        )
        if target_mask.any()
        and "provisional_go_label_after_refit_evidence" in decision.columns
        and decision.loc[
            target_mask, "provisional_go_label_after_refit_evidence"
        ].notna().any()
        else "WEAK-GO"
    )
    if stable and placebo_pass and current == "WEAK-GO":
        decision.loc[final_mask, "decision"] = provisional
        decision.loc[target_mask, "decision"] = provisional
        decision.loc[final_mask, "robustness_gate"] = (
            f"passed: {unique_folds} full refit folds x {unique_seeds} seeds, "
            f"fold_win_rate={win_rate:.3f}, "
            f"leave_one_ticker_win_rate={ticker_win_rate:.3f}, "
            "extreme-trim and validation/test placebo gates passed"
        )
    else:
        if current.startswith("GO-"):
            decision.loc[final_mask, "decision"] = "WEAK-GO"
        decision.loc[final_mask, "robustness_gate"] = (
            f"not passed: full_refit_folds={unique_folds}/{required_folds}, "
            f"full_refit_seeds={unique_seeds}/{required_seeds}, "
            f"fold_win_rate={win_rate}, "
            f"leave_one_ticker_win_rate={ticker_win_rate}, "
            f"extreme_trim_pass={outlier_stable}, "
            f"validation_test_placebo_pass={placebo_pass}"
        )
    atomic_write_csv(decision, path, index=False)
    return path


def run(config: dict[str, Any]) -> dict[str, Path]:
    ensure_directories(config)
    logger = get_logger(
        __name__, config, project_path(config, "outputs", "logs", "placebo_tests.log")
    )
    seed = int(config["project"]["seed"])
    set_global_seed(seed, bool(config["project"].get("deterministic", True)))
    tables = project_path(config, "outputs", "tables")
    validation = safe_read_table(tables / "target_comparison_validation.csv")
    test = safe_read_table(tables / "target_comparison_test.csv")
    predictions = safe_read_table(
        project_path(config, "data", "processed", "target_predictions.parquet")
    )
    validation = attach_variant_metadata(validation, config, logger)
    test = attach_variant_metadata(test, config, logger)
    predictions = attach_variant_metadata(predictions, config, logger)
    validate_required_columns(
        validation,
        [
            "target",
            "representation",
            "input_variant",
            "model",
            "config_id",
            "selected_on_validation",
            "primary_metric",
            "primary_value",
        ],
        "validation target comparison",
    )
    validate_required_columns(
        test,
        ["target", "representation", "model", "config_id", "primary_value"],
        "test target comparison",
    )
    placebo = _placebo_comparison(validation, test)
    locked = _locked_configurations(validation)
    target, prototype, reference = _read_final_selection(config, locked)
    robustness_rows: list[dict[str, Any]] = []
    if prototype is None or reference is None:
        robustness_rows.append(
            {
                "record_type": "chronological_fold_refit",
                "evidence_type": "chronological_full_refit",
                "qualifies_for_stability": True,
                "executed": False,
                "not_executed_reason": (
                    "No validation-locked prototype/R0 pair was available"
                ),
                "target": target,
                "refit_components": (
                    "baseline, thresholds, PCA, KMeans, aggregation, scaler, model"
                ),
            }
        )
    else:
        robustness_rows.extend(
            _diagnostic_slice_rows(
                predictions, prototype, reference, config
            )
        )
        robustness_rows.extend(
            [
                {
                    "record_type": "leave_one_ticker_refit",
                    "evidence_type": "structural_refit_not_executed",
                    "qualifies_for_stability": False,
                    "executed": False,
                    "not_executed_reason": (
                        "The reported leave-one-ticker rows are score-recomputation "
                        "proxies; pooled baseline/prototype/model leave-one-ticker "
                        "refits are not executed by this stage"
                    ),
                    "target": target,
                    "representation": prototype["representation"],
                    "comparison_representation": reference["representation"],
                    "refit_components": "none",
                },
                {
                    "record_type": "leave_one_year_refit",
                    "evidence_type": "structural_refit_not_executed",
                    "qualifies_for_stability": False,
                    "executed": False,
                    "not_executed_reason": (
                        "The reported leave-one-year rows are score-recomputation "
                        "proxies; all transforms and models were not refitted after "
                        "removing each year"
                    ),
                    "target": target,
                    "representation": prototype["representation"],
                    "comparison_representation": reference["representation"],
                    "refit_components": "none",
                },
            ]
        )
        robustness_rows.extend(
            _bootstrap_and_dm_rows(
                predictions, prototype, reference, config
            )
        )
        full_refit = run_chronological_refit_folds(
            config, prototype, reference, logger
        )
        robustness_rows.extend(full_refit.to_dict(orient="records"))
        robustness_rows.extend(_refit_summary_rows(full_refit, config))
    robustness = pd.DataFrame(robustness_rows)
    robustness_rows.extend(_diagnostic_summary_rows(robustness, config))
    robustness = pd.DataFrame(robustness_rows)
    for column in ROBUSTNESS_COLUMNS:
        if column not in robustness.columns:
            robustness[column] = np.nan
    ordered = [
        *ROBUSTNESS_COLUMNS,
        *sorted(set(robustness.columns).difference(ROBUSTNESS_COLUMNS)),
    ]
    robustness = robustness[ordered]
    placebo_path = atomic_write_csv(
        placebo, tables / "placebo_results.csv", index=False
    )
    robustness_path = atomic_write_csv(
        robustness, tables / "robustness_results.csv", index=False
    )
    figure_path = project_path(
        config, "outputs", "figures", "true_vs_shuffled_news.png"
    )
    _plot_placebos(placebo, target, figure_path)
    decision_path = _update_final_decision(
        config, placebo, robustness, target, prototype
    )
    executed_refits = robustness.loc[
        robustness["qualifies_for_stability"].fillna(False).astype(bool)
        & robustness["executed"].fillna(False).astype(bool)
    ]
    logger.info(
        "Placebo comparisons=%d; full refit rows executed=%d; target=%s",
        len(placebo),
        len(executed_refits),
        target,
    )
    outputs = {
        "placebo_results": placebo_path,
        "robustness_results": robustness_path,
        "true_vs_shuffled_news": figure_path,
    }
    if decision_path is not None:
        outputs["final_decision"] = decision_path
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to config YAML")
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
