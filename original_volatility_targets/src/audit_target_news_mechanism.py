"""Post-hoc mechanism audit for target-news volatility-level results.

The audit stays on chronological validation folds.  It compares the locked
semantic R6 predictions with target-news metadata only and with an empirical
null distribution built from independently randomized prototype assignments.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
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
    validate_columns,
)


OUTPUT_NAMES = (
    "target_mechanism_fold_results.csv",
    "target_mechanism_metadata_comparisons.csv",
    "target_mechanism_random_seed_results.csv",
    "target_mechanism_summary.csv",
    "target_mechanism_decision.csv",
    "target_mechanism_report.json",
)


def audit_outputs() -> tuple[str, ...]:
    return tuple(f"outputs/tables/{name}" for name in OUTPUT_NAMES)


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


def _parse_sequence(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            return tuple(map(str, json.loads(stripped)))
        return (stripped,)
    if isinstance(value, Sequence):
        return tuple(map(str, value))
    raise TypeError(f"Unsupported sequence value: {value!r}")


def _profile(config: Mapping[str, Any]) -> dict[str, Any]:
    section = config.get("target_mechanism_audit")
    if not isinstance(section, Mapping):
        raise KeyError("Missing target_mechanism_audit configuration section.")
    required = {
        "representation_variant_family",
        "random_representation_prefix",
        "random_prototype_seeds",
        "ridge_alphas",
        "seeds",
        "folds",
        "text_news_levels",
        "training_cohort",
        "primary_evaluation_cohort",
        "base_experiment_profile",
        "semantic_representation",
        "metadata_representation",
        "price_representation",
    }
    missing = sorted(required.difference(section))
    if missing:
        raise KeyError(f"Target mechanism configuration is missing: {missing}")
    folds = tuple(dict.fromkeys(int(value) for value in section["folds"]))
    seeds = tuple(dict.fromkeys(int(value) for value in section["seeds"]))
    random_seeds = tuple(
        dict.fromkeys(int(value) for value in section["random_prototype_seeds"])
    )
    minimum_random = int(section.get("minimum_random_prototype_seeds", 20))
    if len(folds) != 3 or len(seeds) != 5:
        raise ValueError(
            "Target mechanism audit requires 3 folds x 5 paired "
            f"prototype/model seeds; got folds={folds}, seeds={seeds}."
        )
    if len(random_seeds) < minimum_random:
        raise ValueError(
            f"At least {minimum_random} random-prototype seeds are required; "
            f"got {len(random_seeds)}."
        )
    if list(map(str, section["text_news_levels"])) != ["target"]:
        raise ValueError("Mechanism audit requires text_news_levels=[target].")
    if str(section["training_cohort"]) != "all_days":
        raise ValueError("Mechanism audit must train on all fold-train days.")
    if str(section["primary_evaluation_cohort"]) != "target_news_days":
        raise ValueError("Mechanism audit must evaluate on target-news days.")
    alphas = tuple(float(value) for value in section["ridge_alphas"])
    if len(alphas) != 1:
        raise ValueError("Mechanism audit requires exactly one locked Ridge alpha.")
    profile = dict(section)
    profile.update(
        {
            "folds": folds,
            "seeds": seeds,
            "random_prototype_seeds": random_seeds,
            "alpha": alphas[0],
        }
    )
    return profile


def _random_representations(profile: Mapping[str, Any]) -> tuple[str, ...]:
    prefix = str(profile["random_representation_prefix"])
    return tuple(
        f"{prefix}_{int(seed)}"
        for seed in profile["random_prototype_seeds"]
    )


def _prior_decision(config: Mapping[str, Any]) -> str:
    path = project_path(
        config,
        "outputs",
        "tables",
        "target_news_only_decision.csv",
    )
    if not path.is_file():
        raise FileNotFoundError(
            "target_news_only_decision.csv is missing. Complete the locked "
            "--target-news-only evaluation before this post-hoc audit."
        )
    frame = pd.read_csv(path)
    validate_columns(
        frame,
        ("decision", "locked_test_used"),
        "target-news-only decision",
    )
    if len(frame) != 1:
        raise ValueError("Expected exactly one target-news-only decision row.")
    if _truthy_mask(frame["locked_test_used"]).any():
        raise AssertionError(
            "The mechanism audit must be completed before using locked test."
        )
    return str(frame.iloc[0]["decision"])


def _select_prediction_grid(
    predictions: pd.DataFrame,
    *,
    representations: Sequence[str],
    profile: Mapping[str, Any],
    label: str,
) -> pd.DataFrame:
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
        label,
    )
    requested = tuple(map(str, representations))
    selected = predictions.loc[
        predictions["target_family"].astype(str).eq("level")
        & predictions["target"].astype(str).eq("volatility_level")
        & predictions["evaluation_split"].astype(str).eq("validation")
        & predictions["model"].astype(str).eq("ridge")
        & pd.to_numeric(predictions["alpha"], errors="coerce").eq(
            float(profile["alpha"])
        )
        & predictions["representation"].astype(str).isin(requested)
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
    non_price = ~selected["representation"].astype(str).eq(
        str(profile["price_representation"])
    )
    family_ok = (
        selected["representation_variant_family"].astype(str).eq(
            str(profile["representation_variant_family"])
        )
    )
    selected = selected.loc[~non_price | family_ok].copy()
    expected_input = np.where(non_price.loc[selected.index], "price_plus_text", "price_only")
    selected = selected.loc[
        selected["input_variant"].astype(str).to_numpy() == expected_input
    ].copy()
    if selected.empty:
        raise FileNotFoundError(f"No locked {label} were found.")
    scopes = {
        _parse_sequence(value)
        for value in selected["text_news_levels"].dropna().drop_duplicates()
    }
    if scopes != {("target",)}:
        raise AssertionError(f"{label} has unexpected text scopes: {scopes}")
    if set(selected["training_cohort"].astype(str)) != {"all_days"}:
        raise AssertionError(f"{label} changed the all-days training cohort.")
    if set(selected["primary_evaluation_cohort"].astype(str)) != {
        "target_news_days"
    }:
        raise AssertionError(f"{label} changed the primary evaluation cohort.")
    if not _truthy_mask(selected["qualifies_for_robustness"]).all():
        raise AssertionError(f"{label} contains a non-fold-safe artifact.")
    task_grid = selected[
        ["fold", "seed", "representation", "task_id"]
    ].drop_duplicates()
    expected = {
        (fold, seed, representation)
        for fold in profile["folds"]
        for seed in profile["seeds"]
        for representation in requested
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
            f"{label} grid is not exact: missing={missing}; "
            f"duplicates={duplicates}; unexpected={unexpected}"
        )
    row_keys = ["task_id", "ticker", "feature_date", "target_date"]
    if selected.duplicated(row_keys).any():
        raise ValueError(f"{label} contains duplicate prediction rows.")
    selected["has_target_news"] = _truthy_mask(selected["has_target_news"])
    return selected


def _locked_predictions(
    config: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> pd.DataFrame:
    price = str(profile["price_representation"])
    semantic = str(profile["semantic_representation"])
    metadata = str(profile["metadata_representation"])
    random_representations = _random_representations(profile)
    base = _select_prediction_grid(
        load_predictions(
            config,
            mode=str(profile["base_experiment_profile"]),
        ),
        representations=(price, semantic),
        profile=profile,
        label="base target-news predictions",
    )
    audit = _select_prediction_grid(
        load_predictions(config, mode="target_mechanism_audit"),
        representations=(metadata, *random_representations),
        profile=profile,
        label="target mechanism predictions",
    )
    prefixes = {
        _parse_sequence(value)
        for value in audit.loc[
            audit["representation"].astype(str).eq(metadata),
            "text_feature_prefixes",
        ]
        .dropna()
        .drop_duplicates()
    }
    if prefixes != {("meta__",)}:
        raise AssertionError(
            f"Metadata-only R7 did not lock text_feature_prefixes: {prefixes}"
        )
    combined = pd.concat([base, audit], ignore_index=True, sort=False)
    expected_representations = {
        price,
        semantic,
        metadata,
        *random_representations,
    }
    expected_per_observation = len(expected_representations)
    observation_keys = [
        "fold",
        "seed",
        "ticker",
        "feature_date",
        "target_date",
    ]
    consistency = combined.groupby(
        observation_keys,
        sort=False,
        observed=True,
    ).agg(
        representation_count=("representation", "nunique"),
        y_true_count=("y_true", "nunique"),
        news_gate_count=("has_target_news", "nunique"),
    )
    invalid = consistency.loc[
        consistency["representation_count"].ne(expected_per_observation)
        | consistency["y_true_count"].ne(1)
        | consistency["news_gate_count"].ne(1)
    ]
    if not invalid.empty:
        raise AssertionError(
            "Prediction rows or true target-news gates are not paired across "
            f"all mechanisms; invalid observation groups={len(invalid)}."
        )
    primary = combined.loc[combined["has_target_news"]].copy()
    if primary.empty:
        raise ValueError("The target-news-day audit cohort is empty.")
    minimum = int(profile.get("minimum_primary_samples_per_fold", 20))
    counts = primary.groupby(
        ["fold", "seed", "representation"],
        observed=True,
    ).size()
    if counts.min() < minimum:
        raise ValueError(
            f"An audit cell has fewer than {minimum} target-news samples."
        )
    if (
        counts.groupby(["fold", "seed"], observed=True)
        .nunique()
        .gt(1)
        .any()
    ):
        raise AssertionError("Audit sample counts differ across mechanisms.")
    return primary.sort_values(
        ["fold", "seed", "representation", "target_date", "ticker"],
        kind="mergesort",
    ).reset_index(drop=True)


def _metric_lookup(metrics: pd.DataFrame) -> dict[tuple[int, int, str], pd.Series]:
    return {
        (int(row.fold), int(row.seed), str(row.representation)): row
        for row in metrics.itertuples(index=False)
    }


def _comparison_row(
    candidate: pd.Series | Any,
    reference: pd.Series | Any,
    *,
    comparison: str,
) -> dict[str, Any]:
    candidate_value = float(candidate.primary_value)
    reference_value = float(reference.primary_value)
    gain = metric_gain(candidate_value, reference_value, larger=False)
    return {
        "comparison": comparison,
        "fold": int(candidate.fold),
        "prototype_seed": int(candidate.seed),
        "model_seed": int(candidate.seed),
        "candidate": str(candidate.representation),
        "reference": str(reference.representation),
        "candidate_qlike": candidate_value,
        "reference_qlike": reference_value,
        "absolute_qlike_reduction": reference_value - candidate_value,
        "relative_gain": gain,
        "win": bool(np.isfinite(gain) and gain > 0.0),
        "candidate_task_id": str(candidate.task_id),
        "reference_task_id": str(reference.task_id),
    }


def _comparison_summary(
    comparisons: pd.DataFrame,
    config: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> pd.DataFrame:
    minimum_gain = float(config["decision"]["minimum_relative_gain"])
    minimum_win_rate = float(config["decision"]["minimum_fold_win_rate"])
    expected = len(profile["folds"]) * len(profile["seeds"])
    rows: list[dict[str, Any]] = []
    for name, group in comparisons.groupby(
        "comparison", sort=True, observed=True
    ):
        gains = pd.to_numeric(group["relative_gain"], errors="coerce").dropna()
        lower, upper = confidence_interval(gains)
        fold_means = (
            group.groupby("fold", sort=True, observed=True)["relative_gain"]
            .mean()
            .to_dict()
        )
        positive_folds = sum(float(value) > 0.0 for value in fold_means.values())
        win_rate = float((gains > 0.0).mean()) if len(gains) else np.nan
        complete = bool(
            len(gains) == expected
            and group["fold"].nunique() == len(profile["folds"])
            and group["prototype_seed"].nunique() == len(profile["seeds"])
        )
        passed = bool(
            complete
            and float(gains.mean()) >= minimum_gain
            and win_rate >= minimum_win_rate
            and positive_folds == len(profile["folds"])
        )
        rows.append(
            {
                "record_type": "metadata_comparison",
                "comparison": str(name),
                "n_comparisons": len(gains),
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
                "folds_with_positive_mean_gain": positive_folds,
                "fold_mean_gains": json.dumps(
                    {
                        str(key): float(value)
                        for key, value in fold_means.items()
                    },
                    sort_keys=True,
                ),
                "minimum_required_mean_gain": minimum_gain,
                "minimum_required_win_rate": minimum_win_rate,
                "evidence_complete": complete,
                "comparison_passed": passed,
            }
        )
    return pd.DataFrame(rows)


def _build_comparisons(
    metrics: pd.DataFrame,
    config: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lookup = _metric_lookup(metrics)
    price = str(profile["price_representation"])
    semantic = str(profile["semantic_representation"])
    metadata = str(profile["metadata_representation"])
    prefix = str(profile["random_representation_prefix"])
    metadata_rows: list[dict[str, Any]] = []
    random_rows: list[dict[str, Any]] = []
    for fold in profile["folds"]:
        for seed in profile["seeds"]:
            r0 = lookup[(fold, seed, price)]
            r6 = lookup[(fold, seed, semantic)]
            r7 = lookup[(fold, seed, metadata)]
            metadata_rows.extend(
                (
                    _comparison_row(
                        r6,
                        r0,
                        comparison="R6_semantic_vs_R0_price",
                    ),
                    _comparison_row(
                        r7,
                        r0,
                        comparison="R7_metadata_vs_R0_price",
                    ),
                    _comparison_row(
                        r6,
                        r7,
                        comparison="R6_semantic_vs_R7_metadata",
                    ),
                )
            )
            semantic_gain = metric_gain(
                float(r6.primary_value),
                float(r0.primary_value),
                larger=False,
            )
            for random_seed in profile["random_prototype_seeds"]:
                random_name = f"{prefix}_{random_seed}"
                random_row = lookup[(fold, seed, random_name)]
                random_rows.append(
                    {
                        "random_prototype_seed": int(random_seed),
                        "fold": int(fold),
                        "prototype_seed": int(seed),
                        "model_seed": int(seed),
                        "semantic_qlike": float(r6.primary_value),
                        "random_qlike": float(random_row.primary_value),
                        "price_qlike": float(r0.primary_value),
                        "semantic_gain_vs_price": semantic_gain,
                        "random_gain_vs_price": metric_gain(
                            float(random_row.primary_value),
                            float(r0.primary_value),
                            larger=False,
                        ),
                        "semantic_advantage_vs_random": metric_gain(
                            float(r6.primary_value),
                            float(random_row.primary_value),
                            larger=False,
                        ),
                        "semantic_beats_random": bool(
                            float(r6.primary_value)
                            < float(random_row.primary_value)
                        ),
                        "semantic_task_id": str(r6.task_id),
                        "random_task_id": str(random_row.task_id),
                    }
                )
    metadata_comparisons = pd.DataFrame(metadata_rows)
    random_cells = pd.DataFrame(random_rows)
    summaries = _comparison_summary(
        metadata_comparisons,
        config,
        profile,
    )
    return metadata_comparisons, random_cells, summaries


def _random_seed_summary(
    random_cells: pd.DataFrame,
    profile: Mapping[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    expected = len(profile["folds"]) * len(profile["seeds"])
    for random_seed, group in random_cells.groupby(
        "random_prototype_seed", sort=True, observed=True
    ):
        advantages = pd.to_numeric(
            group["semantic_advantage_vs_random"], errors="coerce"
        ).dropna()
        null_gains = pd.to_numeric(
            group["random_gain_vs_price"], errors="coerce"
        ).dropna()
        lower, upper = confidence_interval(advantages)
        fold_advantages = (
            group.groupby("fold", sort=True, observed=True)[
                "semantic_advantage_vs_random"
            ]
            .mean()
            .to_dict()
        )
        rows.append(
            {
                "random_prototype_seed": int(random_seed),
                "n_cells": len(group),
                "expected_cells": expected,
                "evidence_complete": len(group) == expected,
                "mean_random_gain_vs_price": float(null_gains.mean()),
                "mean_semantic_advantage_vs_random": float(
                    advantages.mean()
                ),
                "median_semantic_advantage_vs_random": float(
                    advantages.median()
                ),
                "semantic_advantage_ci_lower": lower,
                "semantic_advantage_ci_upper": upper,
                "semantic_win_rate": float(
                    group["semantic_beats_random"].astype(bool).mean()
                ),
                "fold_mean_semantic_advantages": json.dumps(
                    {
                        str(key): float(value)
                        for key, value in fold_advantages.items()
                    },
                    sort_keys=True,
                ),
            }
        )
    return pd.DataFrame(rows)


def _null_distribution_summary(
    random_cells: pd.DataFrame,
    random_summary: pd.DataFrame,
    config: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    semantic_gain = float(
        pd.to_numeric(
            random_cells["semantic_gain_vs_price"], errors="coerce"
        ).mean()
    )
    null_gains = pd.to_numeric(
        random_summary["mean_random_gain_vs_price"], errors="coerce"
    ).dropna()
    percentile = float((null_gains < semantic_gain).mean())
    empirical_p = float(
        (1 + int((null_gains >= semantic_gain).sum()))
        / (len(null_gains) + 1)
    )
    fold_advantages = (
        random_cells.groupby("fold", sort=True, observed=True)[
            "semantic_advantage_vs_random"
        ]
        .mean()
        .to_dict()
    )
    positive_folds = sum(float(value) > 0.0 for value in fold_advantages.values())
    advantages = pd.to_numeric(
        random_cells["semantic_advantage_vs_random"], errors="coerce"
    ).dropna()
    semantic_win_rate = float(
        random_cells["semantic_beats_random"].astype(bool).mean()
    )
    minimum_gain = float(config["decision"]["minimum_relative_gain"])
    minimum_win_rate = float(config["decision"]["minimum_fold_win_rate"])
    complete = bool(
        len(null_gains) == len(profile["random_prototype_seeds"])
        and random_summary["evidence_complete"].fillna(False).astype(bool).all()
    )
    passed = bool(
        complete
        and percentile >= float(profile.get("minimum_semantic_percentile", 0.95))
        and empirical_p <= float(profile.get("maximum_empirical_p_value", 0.05))
        and float(advantages.mean()) >= minimum_gain
        and semantic_win_rate >= minimum_win_rate
        and positive_folds == len(profile["folds"])
    )
    lower, upper = confidence_interval(advantages)
    return {
        "record_type": "random_prototype_null",
        "comparison": "R6_semantic_vs_random_prototype_distribution",
        "random_seed_count": len(null_gains),
        "cell_count": len(random_cells),
        "mean_semantic_gain_vs_price": semantic_gain,
        "mean_random_gain_vs_price": float(null_gains.mean()),
        "mean_semantic_advantage_vs_random": float(advantages.mean()),
        "semantic_advantage_ci_lower": lower,
        "semantic_advantage_ci_upper": upper,
        "semantic_win_rate": semantic_win_rate,
        "semantic_percentile_among_random": percentile,
        "empirical_one_sided_p_value": empirical_p,
        "folds_with_positive_mean_advantage": positive_folds,
        "fold_mean_advantages": json.dumps(
            {
                str(key): float(value)
                for key, value in fold_advantages.items()
            },
            sort_keys=True,
        ),
        "minimum_required_percentile": float(
            profile.get("minimum_semantic_percentile", 0.95)
        ),
        "maximum_allowed_empirical_p_value": float(
            profile.get("maximum_empirical_p_value", 0.05)
        ),
        "minimum_required_mean_gain": minimum_gain,
        "minimum_required_win_rate": minimum_win_rate,
        "evidence_complete": complete,
        "comparison_passed": passed,
    }


def run(config: Mapping[str, Any]) -> dict[str, Path]:
    profile = _profile(config)
    prior_decision = _prior_decision(config)
    predictions = _locked_predictions(config, profile)
    metrics = aggregate_metrics(predictions)
    expected_metric_rows = (
        len(profile["folds"])
        * len(profile["seeds"])
        * (3 + len(profile["random_prototype_seeds"]))
    )
    if len(metrics) != expected_metric_rows:
        raise ValueError(
            f"Mechanism audit produced {len(metrics)} metric rows; "
            f"expected {expected_metric_rows}."
        )
    metadata_comparisons, random_cells, metadata_summary = _build_comparisons(
        metrics,
        config,
        profile,
    )
    random_summary = _random_seed_summary(random_cells, profile)
    null_summary = _null_distribution_summary(
        random_cells,
        random_summary,
        config,
        profile,
    )
    summary = pd.concat(
        [metadata_summary, pd.DataFrame([null_summary])],
        ignore_index=True,
        sort=False,
    )
    comparison_pass = {
        str(row.comparison): bool(row.comparison_passed)
        for row in summary.itertuples(index=False)
    }
    metadata_signal = comparison_pass.get(
        "R7_metadata_vs_R0_price", False
    )
    semantic_beyond_metadata = comparison_pass.get(
        "R6_semantic_vs_R7_metadata", False
    )
    semantic_beyond_random = comparison_pass.get(
        "R6_semantic_vs_random_prototype_distribution", False
    )
    if semantic_beyond_metadata and semantic_beyond_random:
        decision = "SEMANTIC-SUPPORTED"
        reason = (
            "R6 beats metadata-only and lies above the locked empirical "
            "random-prototype null distribution on all chronological folds."
        )
    elif metadata_signal and not semantic_beyond_metadata:
        decision = "METADATA-ONLY"
        reason = (
            "Target-news metadata carries signal, but semantic prototypes do "
            "not add stable value beyond that metadata."
        )
    elif not semantic_beyond_random:
        decision = "RANDOM-PARTITION-COMPATIBLE"
        reason = (
            "R6 is not extreme relative to independently randomized prototype "
            "partitions; semantic meaning has not been isolated."
        )
    else:
        decision = "MECHANISM-INCONCLUSIVE"
        reason = (
            "The audit does not cleanly support a semantic, metadata-only or "
            "random-partition mechanism."
        )
    primary_counts = (
        metrics.groupby("fold", observed=True)["n"]
        .first()
        .astype(int)
        .to_dict()
    )
    decision_frame = pd.DataFrame(
        [
            {
                "target": "volatility_level",
                "primary_evaluation_cohort": "target_news_days",
                "fold_count": len(profile["folds"]),
                "prototype_model_seed_count": len(profile["seeds"]),
                "random_prototype_seed_count": len(
                    profile["random_prototype_seeds"]
                ),
                "primary_sample_counts_by_fold": json.dumps(
                    {
                        str(key): value
                        for key, value in primary_counts.items()
                    },
                    sort_keys=True,
                ),
                "metadata_signal_passed": metadata_signal,
                "semantic_beyond_metadata_passed": semantic_beyond_metadata,
                "semantic_beyond_random_passed": semantic_beyond_random,
                "decision": decision,
                "reason": reason,
                "locked_test_used": False,
                "confirmatory_decision_remains": prior_decision,
                "next_step": (
                    "Treat this as post-hoc mechanism evidence only. Do not "
                    "open the locked holdout; pre-register a fresh validation "
                    "period before any promotion decision."
                ),
            }
        ]
    )
    table_dir = project_path(config, "outputs", "tables")
    paths = {
        "fold_results": table_dir / OUTPUT_NAMES[0],
        "metadata_comparisons": table_dir / OUTPUT_NAMES[1],
        "random_seed_results": table_dir / OUTPUT_NAMES[2],
        "summary": table_dir / OUTPUT_NAMES[3],
        "decision": table_dir / OUTPUT_NAMES[4],
        "report": table_dir / OUTPUT_NAMES[5],
    }
    atomic_write_csv(metrics, paths["fold_results"], index=False)
    atomic_write_csv(
        metadata_comparisons,
        paths["metadata_comparisons"],
        index=False,
    )
    atomic_write_csv(random_summary, paths["random_seed_results"], index=False)
    atomic_write_csv(summary, paths["summary"], index=False)
    atomic_write_csv(decision_frame, paths["decision"], index=False)
    atomic_write_json(
        {
            "experiment_profile": "target_mechanism_audit",
            "audit_type": "post_hoc_validation_only",
            "target": "volatility_level",
            "text_news_levels": ["target"],
            "primary_evaluation_cohort": "target_news_days",
            "semantic_representation": profile["semantic_representation"],
            "metadata_representation": profile["metadata_representation"],
            "metadata_feature_prefixes": ["meta__"],
            "price_representation": profile["price_representation"],
            "random_representation_prefix": profile[
                "random_representation_prefix"
            ],
            "folds": list(profile["folds"]),
            "prototype_model_seeds": list(profile["seeds"]),
            "random_prototype_seeds": list(
                profile["random_prototype_seeds"]
            ),
            "null_distribution": null_summary,
            "decision": decision,
            "reason": reason,
            "locked_test_used": False,
            "confirmatory_decision_remains": prior_decision,
        },
        paths["report"],
    )
    return paths
