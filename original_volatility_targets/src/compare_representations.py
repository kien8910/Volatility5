"""Validation-led representation, placebo, robustness, and residual comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.utils import (
    atomic_write_csv,
    confidence_interval,
    project_path,
    read_table,
    resolve_shared_file,
)


PROTOTYPES = ("R5", "R6", "R7", "R8")
REFERENCES = ("R0", "R2", "R3", "R4")
PLACEBOS = ("R9", "R10", "R11", "P_LAGGED", "P_PERMUTED")
DECISION_REFERENCES = (*REFERENCES, "R1", *PLACEBOS)


def metric_spec(target_family: str) -> tuple[str, bool]:
    if target_family == "level":
        return "qlike", False
    if target_family in {"spike", "sector_spike"}:
        return "pr_auc", True
    if target_family in {"regime", "sector_regime"}:
        return "macro_f1", True
    if target_family == "uncertainty":
        return "nll", False
    if target_family == "sector_level":
        return "rmse", False
    raise ValueError(f"Unsupported target family: {target_family}")


def metric_gain(candidate: float, reference: float, larger: bool) -> float:
    if not np.isfinite(candidate) or not np.isfinite(reference):
        return np.nan
    denominator = max(abs(reference), 1.0e-12)
    return (
        (candidate - reference) / denominator
        if larger
        else (reference - candidate) / denominator
    )


def best_row(
    frame: pd.DataFrame,
    target: str,
    representations: tuple[str, ...],
    *,
    fold: str = "holdout",
    seed: int | None = None,
) -> pd.Series | None:
    subset = frame.loc[
        (frame["target"].astype(str) == target)
        & frame["representation"].astype(str).isin(representations)
        & (frame["fold"].astype(str) == str(fold))
    ].copy()
    if seed is not None:
        subset = subset.loc[pd.to_numeric(subset["seed"]) == int(seed)]
    if subset.empty:
        return None
    larger_values = subset["larger_is_better"].dropna().astype(bool).unique()
    if len(larger_values) != 1:
        raise AssertionError(f"Metric direction is ambiguous for {target}")
    return subset.sort_values(
        ["primary_value", "task_id"],
        ascending=[not bool(larger_values[0]), True],
        kind="mergesort",
    ).iloc[0]


def build_placebo_results(
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for target in sorted(validation["target"].dropna().astype(str).unique()):
        prototype = best_row(validation, target, PROTOTYPES)
        if prototype is None:
            continue
        test_prototype = test.loc[test["task_id"] == prototype["task_id"]]
        prototype_test_value = (
            float(test_prototype["primary_value"].iloc[0])
            if len(test_prototype) == 1
            else np.nan
        )
        larger = bool(prototype["larger_is_better"])
        for reference_name in (*REFERENCES, *PLACEBOS, "R1"):
            reference = best_row(validation, target, (reference_name,))
            reference_test_value = np.nan
            if reference is not None:
                reference_test = test.loc[test["task_id"] == reference["task_id"]]
                if len(reference_test) == 1:
                    reference_test_value = float(
                        reference_test["primary_value"].iloc[0]
                    )
            rows.append(
                {
                    "target": target,
                    "target_family": prototype["target_family"],
                    "true_representation": prototype["representation"],
                    "reference_representation": reference_name,
                    "primary_metric": prototype["primary_metric"],
                    "larger_is_better": larger,
                    "prototype_validation_value": prototype["primary_value"],
                    "reference_validation_value": (
                        np.nan if reference is None else reference["primary_value"]
                    ),
                    "validation_gain": metric_gain(
                        float(prototype["primary_value"]),
                        (
                            np.nan
                            if reference is None
                            else float(reference["primary_value"])
                        ),
                        larger,
                    ),
                    "prototype_test_value": prototype_test_value,
                    "reference_test_value": reference_test_value,
                    "test_gain": metric_gain(
                        prototype_test_value, reference_test_value, larger
                    ),
                    "validation_locked_prototype_task": prototype["task_id"],
                    "validation_locked_reference_task": (
                        "" if reference is None else reference["task_id"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_robustness_results(
    validation: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    fold_rows = validation.loc[
        validation["fold"].astype(str) != "holdout"
    ].copy()
    for row in fold_rows.loc[
        fold_rows["representation"].astype(str).isin(PROTOTYPES)
    ].itertuples(index=False):
        references = fold_rows.loc[
            (fold_rows["target"].astype(str) == str(row.target))
            & (fold_rows["representation"].astype(str) == "R0")
            & (fold_rows["fold"].astype(str) == str(row.fold))
            & (pd.to_numeric(fold_rows["seed"]) == int(row.seed))
        ]
        if references.empty:
            reference_value = np.nan
            reference_task = ""
        else:
            reference = references.sort_values(
                ["primary_value", "task_id"],
                ascending=[not bool(row.larger_is_better), True],
                kind="mergesort",
            ).iloc[0]
            reference_value = float(reference["primary_value"])
            reference_task = str(reference["task_id"])
        gain = metric_gain(
            float(row.primary_value),
            reference_value,
            bool(row.larger_is_better),
        )
        rows.append(
            {
                "record_type": "fold_seed",
                "target": row.target,
                "target_family": row.target_family,
                "representation": row.representation,
                "representation_variant": row.representation_variant,
                "model": row.model,
                "fold": row.fold,
                "seed": row.seed,
                "primary_metric": row.primary_metric,
                "primary_value": row.primary_value,
                "reference_value": reference_value,
                "gain_vs_price_only": gain,
                "win": bool(np.isfinite(gain) and gain > 0),
                "qualifies_for_robustness": bool(
                    row.qualifies_for_robustness
                ),
                "representation_fit_scope": row.representation_fit_scope,
                "task_id": row.task_id,
                "reference_task_id": reference_task,
            }
        )
    details = pd.DataFrame(rows)
    summaries = []
    if not details.empty:
        for keys, group in details.groupby(
            ["target", "representation"], sort=True, observed=True
        ):
            target, representation = keys
            qualifying = group.loc[
                group["qualifies_for_robustness"].astype(bool)
            ]
            values = pd.to_numeric(
                qualifying["gain_vs_price_only"], errors="coerce"
            ).dropna()
            lower, upper = confidence_interval(values)
            summaries.append(
                {
                    "record_type": "summary",
                    "target": target,
                    "representation": representation,
                    "n_qualifying": len(values),
                    "fold_count": qualifying["fold"].nunique(),
                    "seed_count": qualifying["seed"].nunique(),
                    "mean_gain": values.mean() if len(values) else np.nan,
                    "standard_deviation": (
                        values.std(ddof=1) if len(values) > 1 else 0.0
                    ),
                    "confidence_interval_lower": lower,
                    "confidence_interval_upper": upper,
                    "win_rate": (
                        float((values > 0).mean()) if len(values) else np.nan
                    ),
                    "folds_better_than_price_only": int(
                        qualifying.loc[
                            pd.to_numeric(
                                qualifying["gain_vs_price_only"], errors="coerce"
                            )
                            > 0,
                            "fold",
                        ].nunique()
                    ),
                }
            )
    output = pd.concat(
        [details, pd.DataFrame(summaries)], ignore_index=True, sort=False
    )
    if output.empty:
        output = pd.DataFrame(
            columns=[
                "record_type",
                "target",
                "target_family",
                "representation",
                "representation_variant",
                "model",
                "fold",
                "seed",
                "primary_metric",
                "primary_value",
                "reference_value",
                "gain_vs_price_only",
                "win",
                "qualifies_for_robustness",
                "representation_fit_scope",
                "task_id",
                "reference_task_id",
                "n_qualifying",
                "fold_count",
                "seed_count",
                "mean_gain",
                "standard_deviation",
                "confidence_interval_lower",
                "confidence_interval_upper",
                "win_rate",
                "folds_better_than_price_only",
            ]
        )
    return output


def compare_with_residual(
    original_validation: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    mapping = [
        ("volatility_level", "signed"),
        ("volatility_spike", "spike"),
        ("volatility_regime", "regime"),
        ("volatility_uncertainty", "uncertainty"),
        ("sector_stress", "sector_residual_failure_breadth"),
    ]
    residual_path = resolve_shared_file(
        config,
        str(config["shared"]["residual_validation_results"]),
        kinds=("tables",),
        required=False,
    )
    residual = (
        read_table(residual_path)
        if residual_path is not None
        else pd.DataFrame()
    )
    rows = []
    for original_family, residual_family in mapping:
        original_subset = original_validation.loc[
            original_validation["target_family"].astype(str).str.contains(
                original_family.replace("volatility_", ""),
                regex=False,
            )
        ]
        original_best = (
            original_subset.sort_values(
                "primary_value",
                ascending=not bool(
                    original_subset["larger_is_better"].iloc[0]
                ),
            ).iloc[0]
            if not original_subset.empty
            else None
        )
        residual_subset = (
            residual.loc[
                residual.get("target", pd.Series(dtype=str))
                .astype(str)
                .str.contains(residual_family, regex=False)
            ]
            if not residual.empty
            else pd.DataFrame()
        )
        residual_best = (
            residual_subset.sort_values(
                "primary_value",
                ascending=False,
            ).iloc[0]
            if not residual_subset.empty
            else None
        )
        original_gain = (
            np.nan
            if original_best is None
            else float(original_best.get("gain_vs_R0", np.nan))
        )
        residual_gain = (
            np.nan
            if residual_best is None
            else float(residual_best.get("validation_gain_vs_R0", np.nan))
        )
        if np.isfinite(original_gain) and original_gain > 0 and np.isfinite(residual_gain) and residual_gain > 0:
            interpretation = "better_for_both"
        elif np.isfinite(original_gain) and original_gain > 0:
            interpretation = "better_for_original_only"
        elif np.isfinite(residual_gain) and residual_gain > 0:
            interpretation = "better_for_residual_only"
        else:
            interpretation = "better_for_neither_or_evidence_missing"
        rows.append(
            {
                "representation": (
                    "" if original_best is None else original_best["representation"]
                ),
                "original_target": original_family,
                "original_gain": original_gain,
                "residual_target": residual_family,
                "residual_gain": residual_gain,
                "interpretation": interpretation,
                "residual_results_path": (
                    "" if residual_path is None else str(residual_path)
                ),
            }
        )
    return pd.DataFrame(rows)


def build_final_decision(
    validation: pd.DataFrame,
    test: pd.DataFrame,
    placebo: pd.DataFrame,
    robustness: pd.DataFrame,
    config: Mapping[str, Any],
    mode: str,
) -> pd.DataFrame:
    threshold = float(config["decision"]["minimum_relative_gain"])
    rows: list[dict[str, Any]] = []
    for target in sorted(validation["target"].dropna().astype(str).unique()):
        prototype = best_row(validation, target, PROTOTYPES)
        if prototype is None:
            continue
        larger = bool(prototype["larger_is_better"])
        row: dict[str, Any] = {
            "record_type": "target",
            "target": target,
            "target_family": prototype["target_family"],
            "representation": prototype["representation"],
            "model": prototype["model"],
            "primary_metric": prototype["primary_metric"],
            "validation_primary_value": prototype["primary_value"],
            "validation_task_id": prototype["task_id"],
            "experiment_mode": mode,
        }
        comparator_gains = []
        for reference_name in DECISION_REFERENCES:
            reference = best_row(validation, target, (reference_name,))
            gain = metric_gain(
                float(prototype["primary_value"]),
                np.nan if reference is None else float(reference["primary_value"]),
                larger,
            )
            row[f"validation_gain_vs_{reference_name}"] = gain
            comparator_gains.append(gain)
        prototype_test = test.loc[test["task_id"] == prototype["task_id"]]
        test_value = (
            float(prototype_test["primary_value"].iloc[0])
            if len(prototype_test) == 1
            else np.nan
        )
        row["test_primary_value"] = test_value
        test_gains = []
        for reference_name in DECISION_REFERENCES:
            reference = best_row(validation, target, (reference_name,))
            reference_test = (
                test.loc[test["task_id"] == reference["task_id"]]
                if reference is not None
                else pd.DataFrame()
            )
            reference_value = (
                float(reference_test["primary_value"].iloc[0])
                if len(reference_test) == 1
                else np.nan
            )
            gain = metric_gain(test_value, reference_value, larger)
            row[f"test_gain_vs_{reference_name}"] = gain
            test_gains.append(gain)
        evidence_complete = all(
            np.isfinite(value) for value in [*comparator_gains, *test_gains]
        )
        validation_pass = evidence_complete and all(
            value >= threshold for value in comparator_gains
        )
        test_pass = evidence_complete and all(value > 0 for value in test_gains)
        price_validation = best_row(validation, target, ("R0",))
        price_test = (
            test.loc[test["task_id"] == price_validation["task_id"]]
            if price_validation is not None
            else pd.DataFrame()
        )
        secondary_validation = _secondary_gate(
            prototype,
            price_validation,
            str(prototype["target_family"]),
        )
        secondary_test = _secondary_gate(
            prototype_test.iloc[0] if len(prototype_test) == 1 else None,
            price_test.iloc[0] if len(price_test) == 1 else None,
            str(prototype["target_family"]),
        )
        if robustness.empty or not {
            "record_type",
            "target",
            "representation",
        }.issubset(robustness.columns):
            robustness_summary = pd.DataFrame()
        else:
            robustness_summary = robustness.loc[
                (robustness["record_type"].astype(str) == "summary")
                & (robustness["target"].astype(str) == target)
                & (
                    robustness["representation"].astype(str)
                    == str(prototype["representation"])
                )
            ]
        stable = False
        if not robustness_summary.empty:
            summary = robustness_summary.iloc[0]
            stable = bool(
                int(summary.get("fold_count", 0)) >= 3
                and int(summary.get("seed_count", 0))
                >= int(config["decision"]["minimum_seed_count"])
                and float(summary.get("win_rate", np.nan))
                >= float(config["decision"]["minimum_fold_win_rate"])
            )
        row["comparator_evidence_complete"] = evidence_complete
        row["validation_gate_passed"] = validation_pass
        row["test_gate_passed"] = test_pass
        row["secondary_validation_gate_passed"] = secondary_validation
        row["secondary_test_gate_passed"] = secondary_test
        row["stable_across_folds_seeds"] = stable
        row["decision_eligible"] = bool(
            validation_pass
            and test_pass
            and secondary_validation
            and secondary_test
            and stable
            and mode == "full"
        )
        row["gain_vs_price_only"] = row.get("validation_gain_vs_R0", np.nan)
        rows.append(row)
    table = pd.DataFrame(rows)
    eligible = (
        table.loc[table["decision_eligible"].fillna(False).astype(bool)]
        if "decision_eligible" in table.columns
        else pd.DataFrame()
    )
    if eligible.empty:
        final_target = (
            str(
                table.sort_values(
                    "gain_vs_price_only", ascending=False, na_position="last"
                ).iloc[0]["target"]
            )
            if not table.empty
            else "none"
        )
        decision = "NO-GO"
    else:
        chosen = eligible.sort_values(
            "gain_vs_price_only", ascending=False
        ).iloc[0]
        final_target = str(chosen["target"])
        family = str(chosen["target_family"])
        decision = {
            "level": "GO-VOLATILITY-LEVEL",
            "spike": "GO-VOLATILITY-SPIKE",
            "regime": "GO-VOLATILITY-REGIME",
            "uncertainty": "GO-VOLATILITY-UNCERTAINTY",
            "sector_level": "GO-SECTOR-REGIME",
            "sector_spike": "GO-SECTOR-REGIME",
            "sector_regime": "GO-SECTOR-REGIME",
        }[family]
    final_row = {
        "record_type": "final",
        "target": final_target,
        "experiment_mode": mode,
        "decision": decision,
        "reason": (
            "All validation/test/placebo/fold/seed gates passed"
            if decision != "NO-GO"
            else "One or more price/comparator/placebo/test/full-refit stability gates failed"
        ),
        "prototype_better_than_price_only": _any_row_passes(
            table, ("R0",), threshold
        ),
        "prototype_better_than_raw_embedding": _any_row_passes(
            table, ("R2",), threshold
        ),
        "prototype_better_than_pca_random_projection": _any_row_passes(
            table, ("R3", "R4"), threshold
        ),
        "true_news_better_than_shuffled": _any_row_passes(
            table, ("R9", "R10", "R11"), threshold
        ),
        "stable_across_folds_seeds": bool(
            not table.empty
            and table["stable_across_folds_seeds"].fillna(False).any()
        ),
    }
    return pd.concat([table, pd.DataFrame([final_row])], ignore_index=True, sort=False)


def _metric_gate(
    candidate: pd.Series | None,
    reference: pd.Series | None,
    metric: str,
    *,
    larger: bool,
    tolerance: float = 0.0,
) -> bool:
    if candidate is None or reference is None:
        return False
    left = float(candidate.get(metric, np.nan))
    right = float(reference.get(metric, np.nan))
    if not np.isfinite(left) or not np.isfinite(right):
        return False
    return left >= right - tolerance if larger else left <= right + tolerance


def _secondary_gate(
    candidate: pd.Series | None,
    reference: pd.Series | None,
    family: str,
) -> bool:
    if family in {"level", "sector_level"}:
        return (
            _metric_gate(candidate, reference, "mae", larger=False)
            and _metric_gate(candidate, reference, "rmse", larger=False)
            and (
                family == "sector_level"
                or _metric_gate(candidate, reference, "spearman", larger=True)
            )
        )
    if family in {"spike", "sector_spike"}:
        return (
            _metric_gate(candidate, reference, "brier", larger=False)
            and _metric_gate(candidate, reference, "ece", larger=False)
            and _metric_gate(candidate, reference, "recall", larger=True)
        )
    if family in {"regime", "sector_regime"}:
        return (
            _metric_gate(
                candidate, reference, "balanced_accuracy", larger=True
            )
            and _metric_gate(
                candidate, reference, "high_volatility_recall", larger=True
            )
            and _metric_gate(
                candidate, reference, "multiclass_brier", larger=False
            )
        )
    if family == "uncertainty":
        return (
            _metric_gate(candidate, reference, "crps", larger=False)
            and _metric_gate(candidate, reference, "pit_ks", larger=False)
            and _metric_gate(
                candidate,
                reference,
                "coverage_calibration_error",
                larger=False,
            )
            and _metric_gate(
                candidate,
                reference,
                "width_95",
                larger=False,
                tolerance=0.10 * float(reference.get("width_95", 0.0))
                if reference is not None
                else 0.0,
            )
        )
    return False


def _any_row_passes(
    table: pd.DataFrame,
    references: tuple[str, ...],
    threshold: float,
) -> bool:
    if table.empty:
        return False
    mask = pd.Series(True, index=table.index)
    for representation in references:
        column = f"validation_gain_vs_{representation}"
        if column not in table:
            return False
        mask &= pd.to_numeric(table[column], errors="coerce") >= threshold
    return bool(mask.any())


def write_comparison_outputs(
    validation: pd.DataFrame,
    test: pd.DataFrame,
    config: Mapping[str, Any],
    mode: str,
) -> dict[str, Path]:
    tables = project_path(config, "outputs", "tables")
    placebo = build_placebo_results(validation, test)
    robustness = build_robustness_results(validation, config)
    residual = compare_with_residual(validation, config)
    decision = build_final_decision(
        validation, test, placebo, robustness, config, mode
    )
    return {
        "placebo": atomic_write_csv(
            placebo, tables / "placebo_results.csv", index=False
        ),
        "robustness": atomic_write_csv(
            robustness, tables / "robustness_results.csv", index=False
        ),
        "residual_comparison": atomic_write_csv(
            residual,
            tables / "comparison_with_residual_targets.csv",
            index=False,
        ),
        "decision": atomic_write_csv(
            decision,
            tables / "final_original_target_decision.csv",
            index=False,
        ),
    }
