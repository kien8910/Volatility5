"""Evaluate the locked 3-fold x 5-seed R6 confirmatory experiment."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.compare_representations import metric_gain
from src.evaluate_original_targets import aggregate_metrics, load_predictions
from src.utils import (
    atomic_write_csv,
    atomic_write_json,
    confidence_interval,
    project_path,
)


OUTPUT_NAMES = (
    "r6_confirmatory_fold_results.csv",
    "r6_confirmatory_comparisons.csv",
    "r6_confirmatory_summary.csv",
    "r6_confirmatory_decision.csv",
    "r6_confirmatory_report.json",
)


def evaluation_outputs() -> tuple[str, ...]:
    return tuple(f"outputs/tables/{name}" for name in OUTPUT_NAMES)


def _profile(config: Mapping[str, Any]) -> Mapping[str, Any]:
    profile = config.get("r6_confirmatory")
    if not isinstance(profile, Mapping):
        raise KeyError("Missing r6_confirmatory configuration section.")
    return profile


def _locked_metrics(config: Mapping[str, Any]) -> pd.DataFrame:
    profile = _profile(config)
    predictions = load_predictions(config, mode="r6_confirmatory")
    metrics = aggregate_metrics(predictions)
    locked_alpha_values = tuple(
        float(value) for value in profile.get("ridge_alphas", [10.0])
    )
    if len(locked_alpha_values) != 1:
        raise ValueError("R6 confirmatory evaluation requires one Ridge alpha.")
    locked_alpha = locked_alpha_values[0]
    selected = metrics.loc[
        metrics["target_family"].astype(str).eq("level")
        & metrics["target"].astype(str).eq("volatility_level")
        & metrics["evaluation_split"].astype(str).eq("validation")
        & metrics["model"].astype(str).eq("ridge")
        & pd.to_numeric(metrics["alpha"], errors="coerce").eq(locked_alpha)
        & metrics["representation"].astype(str).isin(
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
    selected["fold"] = pd.to_numeric(selected["fold"], errors="raise").astype(
        int
    )
    selected["seed"] = pd.to_numeric(selected["seed"], errors="raise").astype(
        int
    )
    selected = selected.loc[
        selected["fold"].isin(profile["folds"])
        & selected["seed"].isin(profile["seeds"])
    ].copy()
    expected = {
        (int(fold), int(seed), str(representation))
        for fold in profile["folds"]
        for seed in profile["seeds"]
        for representation in profile["representations"]
    }
    observed_counts = (
        selected.groupby(
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
            "R6 confirmatory checkpoint grid is not exact: "
            f"missing={missing}; duplicates={duplicates}; "
            f"unexpected={unexpected}"
        )
    if not selected["qualifies_for_robustness"].fillna(False).astype(bool).all():
        invalid = selected.loc[
            ~selected["qualifies_for_robustness"]
            .fillna(False)
            .astype(bool),
            [
                "fold",
                "seed",
                "representation",
                "representation_fit_scope",
            ],
        ]
        raise AssertionError(
            "Non-fold-safe artifact entered the confirmatory grid: "
            f"{invalid.to_dict(orient='records')}"
        )
    if not selected["primary_metric"].astype(str).eq("qlike").all():
        raise AssertionError("R6 confirmatory primary metric must be QLIKE.")
    if selected["larger_is_better"].fillna(True).astype(bool).any():
        raise AssertionError("QLIKE must be treated as lower-is-better.")
    return selected.sort_values(
        ["fold", "seed", "representation"], kind="mergesort"
    ).reset_index(drop=True)


def _comparison_rows(
    metrics: pd.DataFrame,
    profile: Mapping[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in profile["folds"]:
        for seed in profile["seeds"]:
            cell = metrics.loc[
                metrics["fold"].eq(int(fold))
                & metrics["seed"].eq(int(seed))
            ]
            r6 = cell.loc[cell["representation"].astype(str).eq("R6")]
            if len(r6) != 1:
                raise ValueError(
                    f"Expected one R6 row for fold={fold}, seed={seed}."
                )
            candidate = r6.iloc[0]
            for reference_name in profile["comparison_representations"]:
                reference = cell.loc[
                    cell["representation"].astype(str).eq(
                        str(reference_name)
                    )
                ]
                if len(reference) != 1:
                    raise ValueError(
                        f"Expected one {reference_name} row for fold={fold}, "
                        f"seed={seed}."
                    )
                baseline = reference.iloc[0]
                candidate_value = float(candidate["primary_value"])
                reference_value = float(baseline["primary_value"])
                gain = metric_gain(
                    candidate_value,
                    reference_value,
                    larger=False,
                )
                rows.append(
                    {
                        "target": "volatility_level",
                        "primary_metric": "qlike",
                        "fold": int(fold),
                        "prototype_seed": int(seed),
                        "model_seed": int(seed),
                        "representation": "R6",
                        "reference": str(reference_name),
                        "r6_qlike": candidate_value,
                        "reference_qlike": reference_value,
                        "absolute_qlike_reduction": (
                            reference_value - candidate_value
                        ),
                        "relative_gain": gain,
                        "win": bool(np.isfinite(gain) and gain > 0.0),
                        "r6_task_id": str(candidate["task_id"]),
                        "reference_task_id": str(baseline["task_id"]),
                        "representation_variant_family": str(
                            candidate["representation_variant_family"]
                        ),
                        "r6_fit_scope": str(
                            candidate["representation_fit_scope"]
                        ),
                        "reference_fit_scope": str(
                            baseline["representation_fit_scope"]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _summaries(
    comparisons: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    threshold = float(config["decision"]["minimum_relative_gain"])
    minimum_win_rate = float(config["decision"]["minimum_fold_win_rate"])
    rows: list[dict[str, Any]] = []
    for reference, group in comparisons.groupby(
        "reference", sort=True, observed=True
    ):
        gains = pd.to_numeric(group["relative_gain"], errors="coerce").dropna()
        lower, upper = confidence_interval(gains)
        fold_means = (
            group.groupby("fold", sort=True, observed=True)["relative_gain"]
            .mean()
            .to_dict()
        )
        fold_wins = sum(float(value) > 0.0 for value in fold_means.values())
        win_rate = float((gains > 0.0).mean()) if len(gains) else np.nan
        complete = bool(
            len(gains) == 15
            and group["fold"].nunique() == 3
            and group["prototype_seed"].nunique() == 5
        )
        passed = bool(
            complete
            and float(gains.mean()) >= threshold
            and win_rate >= minimum_win_rate
            and fold_wins == 3
        )
        rows.append(
            {
                "representation": "R6",
                "reference": str(reference),
                "n_comparisons": len(gains),
                "fold_count": group["fold"].nunique(),
                "prototype_seed_count": group["prototype_seed"].nunique(),
                "model_seed_count": group["model_seed"].nunique(),
                "mean_relative_gain": (
                    float(gains.mean()) if len(gains) else np.nan
                ),
                "median_relative_gain": (
                    float(gains.median()) if len(gains) else np.nan
                ),
                "standard_deviation": (
                    float(gains.std(ddof=1)) if len(gains) > 1 else 0.0
                ),
                "confidence_interval_lower": lower,
                "confidence_interval_upper": upper,
                "win_rate": win_rate,
                "folds_with_positive_mean_gain": fold_wins,
                "fold_mean_gains": json.dumps(
                    {str(key): float(value) for key, value in fold_means.items()},
                    sort_keys=True,
                ),
                "minimum_required_mean_gain": threshold,
                "minimum_required_win_rate": minimum_win_rate,
                "evidence_complete": complete,
                "comparison_passed": passed,
            }
        )
    return pd.DataFrame(rows)


def run(config: Mapping[str, Any]) -> dict[str, Path]:
    profile = _profile(config)
    metrics = _locked_metrics(config)
    comparisons = _comparison_rows(metrics, profile)
    summaries = _summaries(comparisons, config)
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
    decision = "CONFIRMATORY-PASS" if passed else "CONFIRMATORY-FAIL"
    reason = (
        "R6 beat every locked price/embedding/placebo comparator across all "
        "3 folds and 5 paired prototype/model seeds."
        if passed
        else "At least one locked comparator, fold-stability, gain or win-rate "
        "gate failed."
    )
    decision_frame = pd.DataFrame(
        [
            {
                "target": "volatility_level",
                "representation": "R6",
                "representation_variant_family": str(
                    profile["representation_variant_family"]
                ),
                "fold_count": metrics["fold"].nunique(),
                "prototype_seed_count": metrics["seed"].nunique(),
                "model_seed_count": metrics["seed"].nunique(),
                "comparison_count": len(comparisons),
                "all_expected_evidence_present": complete,
                "all_comparisons_passed": passed,
                "decision": decision,
                "reason": reason,
                "locked_test_used": False,
                "next_step": (
                    "Lock this family/model and run one final untouched holdout "
                    "test; this fold experiment alone cannot issue a GO label."
                    if passed
                    else "Do not promote R6 to the locked holdout test."
                ),
            }
        ]
    )
    table_dir = project_path(config, "outputs", "tables")
    paths = {
        "fold_results": table_dir / OUTPUT_NAMES[0],
        "comparisons": table_dir / OUTPUT_NAMES[1],
        "summary": table_dir / OUTPUT_NAMES[2],
        "decision": table_dir / OUTPUT_NAMES[3],
        "report": table_dir / OUTPUT_NAMES[4],
    }
    atomic_write_csv(metrics, paths["fold_results"], index=False)
    atomic_write_csv(comparisons, paths["comparisons"], index=False)
    atomic_write_csv(summaries, paths["summary"], index=False)
    atomic_write_csv(decision_frame, paths["decision"], index=False)
    atomic_write_json(
        {
            "experiment_profile": "r6_confirmatory",
            "target": "volatility_level",
            "representation": "R6",
            "family": profile["representation_variant_family"],
            "folds": list(map(int, profile["folds"])),
            "prototype_model_seeds": list(map(int, profile["seeds"])),
            "comparators": list(
                map(str, profile["comparison_representations"])
            ),
            "decision": decision,
            "reason": reason,
            "locked_test_used": False,
        },
        paths["report"],
    )
    return paths
