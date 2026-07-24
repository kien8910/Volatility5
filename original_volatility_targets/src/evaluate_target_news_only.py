"""Evaluate the locked target-company-news-only volatility-level experiment."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluate_original_targets import aggregate_metrics, load_predictions
from src.evaluate_r6_confirmatory import (
    comparison_rows,
    comparison_summaries,
)
from src.utils import (
    atomic_write_csv,
    atomic_write_json,
    project_path,
    validate_columns,
)


OUTPUT_NAMES = (
    "target_news_only_fold_results.csv",
    "target_news_only_comparisons.csv",
    "target_news_only_summary.csv",
    "target_news_only_cohort_summary.csv",
    "target_news_only_decision.csv",
    "target_news_only_report.json",
)


def evaluation_outputs() -> tuple[str, ...]:
    return tuple(f"outputs/tables/{name}" for name in OUTPUT_NAMES)


def _profile(config: Mapping[str, Any]) -> Mapping[str, Any]:
    profile = config.get("target_news_only")
    if not isinstance(profile, Mapping):
        raise KeyError("Missing target_news_only configuration section.")
    required = {
        "representations",
        "comparison_representations",
        "representation_variant_family",
        "ridge_alphas",
        "folds",
        "seeds",
        "text_news_levels",
        "training_cohort",
        "evaluation_news_level",
        "primary_evaluation_cohort",
    }
    missing = sorted(required.difference(profile))
    if missing:
        raise KeyError(f"target_news_only configuration is missing: {missing}")
    if list(map(str, profile["text_news_levels"])) != ["target"]:
        raise ValueError(
            "The target-news-only experiment must lock text_news_levels=[target]."
        )
    if str(profile["training_cohort"]) != "all_days":
        raise ValueError(
            "The locked target-news-only design trains on all fold-train days."
        )
    if str(profile["evaluation_news_level"]) != "target":
        raise ValueError("The locked evaluation news level must be target.")
    if str(profile["primary_evaluation_cohort"]) != "target_news_days":
        raise ValueError(
            "The locked primary cohort must be target_news_days."
        )
    return profile


def _truthy_mask(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_numeric(values, errors="coerce").fillna(0).ne(0)
    return (
        values.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def _parse_level_scope(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            parsed = json.loads(stripped)
            return tuple(map(str, parsed))
        return (stripped,)
    if isinstance(value, Sequence):
        return tuple(map(str, value))
    raise TypeError(f"Unsupported text_news_levels value: {value!r}")


def _locked_predictions(config: Mapping[str, Any]) -> pd.DataFrame:
    profile = _profile(config)
    predictions = load_predictions(config, mode="target_news_only")
    validate_columns(
        predictions,
        (
            "task_id",
            "target_family",
            "target",
            "evaluation_split",
            "model",
            "alpha",
            "representation",
            "representation_variant_family",
            "input_variant",
            "fold",
            "seed",
            "text_news_levels",
            "training_cohort",
            "evaluation_news_level",
            "primary_evaluation_cohort",
            "has_target_news",
            "representation_fit_scope",
            "qualifies_for_robustness",
            "ticker",
            "feature_date",
            "target_date",
            "y_true",
            "prediction",
        ),
        "target-news-only predictions",
    )
    observed_splits = set(predictions["evaluation_split"].astype(str))
    if observed_splits != {"validation"}:
        raise AssertionError(
            "Target-news-only confirmation may contain validation predictions "
            f"only; observed={sorted(observed_splits)}."
        )
    alpha_values = tuple(
        float(value) for value in profile.get("ridge_alphas", [10.0])
    )
    if len(alpha_values) != 1:
        raise ValueError(
            "Target-news-only evaluation requires exactly one Ridge alpha."
        )
    alpha = alpha_values[0]
    selected = predictions.loc[
        predictions["target_family"].astype(str).eq("level")
        & predictions["target"].astype(str).eq("volatility_level")
        & predictions["model"].astype(str).eq("ridge")
        & pd.to_numeric(predictions["alpha"], errors="coerce").eq(alpha)
        & predictions["representation"].astype(str).isin(
            profile["representations"]
        )
    ].copy()
    family_matches = selected["representation"].astype(str).eq("R0") | (
        selected["representation_variant_family"].astype(str).eq(
            str(profile["representation_variant_family"])
        )
    )
    selected = selected.loc[family_matches].copy()
    expected_inputs = np.where(
        selected["representation"].astype(str).eq("R0"),
        "price_only",
        "price_plus_text",
    )
    selected = selected.loc[
        selected["input_variant"].astype(str).to_numpy() == expected_inputs
    ].copy()
    selected["fold"] = pd.to_numeric(
        selected["fold"], errors="raise"
    ).astype(int)
    selected["seed"] = pd.to_numeric(
        selected["seed"], errors="raise"
    ).astype(int)
    selected = selected.loc[
        selected["fold"].isin(profile["folds"])
        & selected["seed"].isin(profile["seeds"])
    ].copy()
    level_scopes = {
        _parse_level_scope(value)
        for value in selected["text_news_levels"].drop_duplicates()
    }
    if level_scopes != {("target",)}:
        raise AssertionError(
            f"Unexpected text news-level scopes: {sorted(level_scopes)}"
        )
    if set(selected["training_cohort"].astype(str)) != {"all_days"}:
        raise AssertionError("A task did not use the locked all-days training cohort.")
    if set(selected["evaluation_news_level"].astype(str)) != {"target"}:
        raise AssertionError("A task did not use the locked target-news gate.")
    if set(selected["primary_evaluation_cohort"].astype(str)) != {
        "target_news_days"
    }:
        raise AssertionError("A task changed the locked primary cohort.")
    if not _truthy_mask(selected["qualifies_for_robustness"]).all():
        invalid = selected.loc[
            ~_truthy_mask(selected["qualifies_for_robustness"]),
            [
                "fold",
                "seed",
                "representation",
                "representation_fit_scope",
            ],
        ]
        raise AssertionError(
            "Non-fold-safe artifact entered target-news-only confirmation: "
            f"{invalid.drop_duplicates().to_dict(orient='records')}"
        )
    task_grid = selected[
        ["fold", "seed", "representation", "task_id"]
    ].drop_duplicates()
    expected = {
        (int(fold), int(seed), str(representation))
        for fold in profile["folds"]
        for seed in profile["seeds"]
        for representation in profile["representations"]
    }
    observed_counts = (
        task_grid.groupby(
            ["fold", "seed", "representation"],
            sort=True,
            observed=True,
        )
        .size()
        .to_dict()
    )
    missing = sorted(expected.difference(observed_counts))
    duplicates = {
        key: count
        for key, count in observed_counts.items()
        if key in expected and count != 1
    }
    unexpected = sorted(set(observed_counts).difference(expected))
    if missing or duplicates or unexpected:
        raise ValueError(
            "Target-news-only checkpoint grid is not exact: "
            f"missing={missing}; duplicates={duplicates}; "
            f"unexpected={unexpected}"
        )
    row_keys = ["task_id", "ticker", "feature_date", "target_date"]
    if selected.duplicated(row_keys).any():
        raise ValueError("Target-news-only predictions contain duplicate row keys.")
    selected["has_target_news"] = _truthy_mask(selected["has_target_news"])
    within_seed_keys = [
        "fold",
        "seed",
        "ticker",
        "feature_date",
        "target_date",
    ]
    if (
        selected.groupby(within_seed_keys, observed=True)["has_target_news"]
        .nunique()
        .gt(1)
        .any()
    ):
        raise AssertionError(
            "True target-news gate differs across representations."
        )
    across_seed_keys = ["fold", "ticker", "feature_date", "target_date"]
    if (
        selected.groupby(across_seed_keys, observed=True)["has_target_news"]
        .nunique()
        .gt(1)
        .any()
    ):
        raise AssertionError("True target-news gate differs across seeds.")
    return selected.sort_values(
        ["fold", "seed", "representation", "target_date", "ticker"],
        kind="mergesort",
    ).reset_index(drop=True)


def _cohorts(profile: Mapping[str, Any]) -> tuple[str, ...]:
    requested = [
        str(profile["primary_evaluation_cohort"]),
        *map(str, profile.get("secondary_evaluation_cohorts", [])),
    ]
    cohorts = tuple(dict.fromkeys(requested))
    allowed = {"target_news_days", "no_target_news_days", "all_days"}
    unknown = sorted(set(cohorts).difference(allowed))
    if unknown:
        raise ValueError(f"Unknown target-news evaluation cohorts: {unknown}")
    return cohorts


def _cohort_predictions(
    predictions: pd.DataFrame,
    cohort: str,
) -> pd.DataFrame:
    if cohort == "target_news_days":
        return predictions.loc[predictions["has_target_news"]].copy()
    if cohort == "no_target_news_days":
        return predictions.loc[~predictions["has_target_news"]].copy()
    if cohort == "all_days":
        return predictions.copy()
    raise ValueError(f"Unknown evaluation cohort: {cohort}")


def _cohort_metrics(
    predictions: pd.DataFrame,
    profile: Mapping[str, Any],
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    minimum_primary = int(profile.get("minimum_primary_samples_per_fold", 20))
    for cohort in _cohorts(profile):
        subset = _cohort_predictions(predictions, cohort)
        if subset.empty:
            raise ValueError(f"Evaluation cohort {cohort!r} is empty.")
        metrics = aggregate_metrics(subset)
        metrics.insert(
            metrics.columns.get_loc("evaluation_split") + 1,
            "evaluation_cohort",
            cohort,
        )
        expected_rows = (
            len(profile["folds"])
            * len(profile["seeds"])
            * len(profile["representations"])
        )
        if len(metrics) != expected_rows:
            raise ValueError(
                f"Cohort {cohort!r} produced {len(metrics)} metric rows; "
                f"expected {expected_rows}."
            )
        if cohort == str(profile["primary_evaluation_cohort"]):
            counts = pd.to_numeric(metrics["n"], errors="raise")
            if counts.min() < minimum_primary:
                raise ValueError(
                    f"Primary cohort contains fewer than {minimum_primary} "
                    "samples in a fold/seed/representation cell."
                )
            count_spread = (
                metrics.groupby(["fold", "seed"], observed=True)["n"]
                .nunique()
            )
            if count_spread.gt(1).any():
                raise AssertionError(
                    "Primary-cohort sample counts differ across comparators."
                )
        pieces.append(metrics)
    return pd.concat(pieces, ignore_index=True, sort=False)


def _cohort_summary(
    metrics: pd.DataFrame,
    config: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    primary = str(profile["primary_evaluation_cohort"])
    for cohort in _cohorts(profile):
        selected = metrics.loc[
            metrics["evaluation_cohort"].astype(str).eq(cohort)
        ].copy()
        comparisons = comparison_rows(selected, profile)
        summary = comparison_summaries(comparisons, config, profile)
        summary.insert(0, "evaluation_cohort", cohort)
        summary.insert(1, "decision_eligible", cohort == primary)
        if cohort != primary:
            summary["comparison_passed"] = pd.NA
        pieces.append(summary)
    return pd.concat(pieces, ignore_index=True, sort=False)


def run(config: Mapping[str, Any]) -> dict[str, Path]:
    profile = _profile(config)
    predictions = _locked_predictions(config)
    metrics = _cohort_metrics(predictions, profile)
    primary_cohort = str(profile["primary_evaluation_cohort"])
    primary_metrics = metrics.loc[
        metrics["evaluation_cohort"].astype(str).eq(primary_cohort)
    ].copy()
    comparisons = comparison_rows(primary_metrics, profile)
    summaries = comparison_summaries(comparisons, config, profile)
    cohort_summary = _cohort_summary(metrics, config, profile)
    expected_references = set(map(str, profile["comparison_representations"]))
    observed_references = set(summaries["reference"].astype(str))
    complete = bool(
        observed_references == expected_references
        and summaries["evidence_complete"].fillna(False).astype(bool).all()
    )
    passed = bool(
        complete
        and summaries["comparison_passed"].fillna(False).astype(bool).all()
    )
    decision = (
        "TARGET-NEWS-ONLY-PASS"
        if passed
        else "TARGET-NEWS-ONLY-FAIL"
    )
    reason = (
        "Target-only R6 beat every locked price/embedding/placebo comparator "
        "on true target-news validation days across all folds and paired seeds."
        if passed
        else "At least one locked target-news-day comparator, fold-stability, "
        "gain or win-rate gate failed."
    )
    primary_counts = (
        primary_metrics.groupby("fold", observed=True)["n"]
        .first()
        .astype(int)
        .to_dict()
    )
    decision_frame = pd.DataFrame(
        [
            {
                "target": "volatility_level",
                "representation": "R6",
                "text_news_levels": json.dumps(["target"]),
                "training_cohort": "all_days",
                "primary_evaluation_cohort": primary_cohort,
                "representation_variant_family": str(
                    profile["representation_variant_family"]
                ),
                "fold_count": primary_metrics["fold"].nunique(),
                "prototype_seed_count": primary_metrics["seed"].nunique(),
                "model_seed_count": primary_metrics["seed"].nunique(),
                "primary_sample_counts_by_fold": json.dumps(
                    {str(key): value for key, value in primary_counts.items()},
                    sort_keys=True,
                ),
                "comparison_count": len(comparisons),
                "all_expected_evidence_present": complete,
                "all_comparisons_passed": passed,
                "decision": decision,
                "reason": reason,
                "locked_test_used": False,
                "next_step": (
                    "Lock target-only features and the target-news-day cohort, "
                    "then run one final untouched holdout evaluation."
                    if passed
                    else "Do not promote target-only R6 to the locked holdout."
                ),
            }
        ]
    )
    table_dir = project_path(config, "outputs", "tables")
    paths = {
        "fold_results": table_dir / OUTPUT_NAMES[0],
        "comparisons": table_dir / OUTPUT_NAMES[1],
        "summary": table_dir / OUTPUT_NAMES[2],
        "cohort_summary": table_dir / OUTPUT_NAMES[3],
        "decision": table_dir / OUTPUT_NAMES[4],
        "report": table_dir / OUTPUT_NAMES[5],
    }
    atomic_write_csv(metrics, paths["fold_results"], index=False)
    atomic_write_csv(comparisons, paths["comparisons"], index=False)
    atomic_write_csv(summaries, paths["summary"], index=False)
    atomic_write_csv(cohort_summary, paths["cohort_summary"], index=False)
    atomic_write_csv(decision_frame, paths["decision"], index=False)
    atomic_write_json(
        {
            "experiment_profile": "target_news_only",
            "target": "volatility_level",
            "representation": "R6",
            "text_news_levels": ["target"],
            "training_cohort": "all_days",
            "primary_evaluation_cohort": primary_cohort,
            "secondary_evaluation_cohorts": list(
                map(str, profile.get("secondary_evaluation_cohorts", []))
            ),
            "family": profile["representation_variant_family"],
            "folds": list(map(int, profile["folds"])),
            "prototype_model_seeds": list(map(int, profile["seeds"])),
            "comparators": list(
                map(str, profile["comparison_representations"])
            ),
            "primary_sample_counts_by_fold": {
                str(key): value for key, value in primary_counts.items()
            },
            "decision": decision,
            "reason": reason,
            "locked_test_used": False,
        },
        paths["report"],
    )
    return paths
