"""Ablate pure target-news metadata from prototype-derived diagnostics."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from src.audit_target_news_mechanism import (
    _build_comparisons,
    _comparison_row,
    _comparison_summary,
    _metric_lookup,
    _null_distribution_summary,
    _parse_sequence,
    _random_representations,
    _random_seed_summary,
    _select_prediction_grid,
    _truthy_mask,
)
from src.evaluate_original_targets import aggregate_metrics, load_predictions
from src.utils import (
    atomic_write_csv,
    atomic_write_json,
    project_path,
    validate_columns,
)


OUTPUT_NAMES = (
    "target_component_fold_results.csv",
    "target_component_comparisons.csv",
    "target_component_random_reference.csv",
    "target_component_summary.csv",
    "target_component_decision.csv",
    "target_component_report.json",
)


def audit_outputs() -> tuple[str, ...]:
    return tuple(f"outputs/tables/{name}" for name in OUTPUT_NAMES)


def _profile(config: Mapping[str, Any]) -> dict[str, Any]:
    section = config.get("target_component_audit")
    if not isinstance(section, Mapping):
        raise KeyError("Missing target_component_audit configuration section.")
    required = {
        "representations",
        "representation_source_map",
        "representation_feature_names",
        "representation_variant_family",
        "ridge_alphas",
        "folds",
        "seeds",
        "text_news_levels",
        "training_cohort",
        "primary_evaluation_cohort",
        "base_experiment_profile",
        "mechanism_experiment_profile",
        "price_representation",
        "semantic_representation",
        "combined_metadata_representation",
        "basic_metadata_representation",
        "prototype_diagnostics_representation",
        "random_representation_prefix",
        "reference_random_prototype_seeds",
    }
    missing = sorted(required.difference(section))
    if missing:
        raise KeyError(f"Target component configuration is missing: {missing}")
    folds = tuple(dict.fromkeys(int(value) for value in section["folds"]))
    seeds = tuple(dict.fromkeys(int(value) for value in section["seeds"]))
    random_seeds = tuple(
        dict.fromkeys(
            int(value)
            for value in section["reference_random_prototype_seeds"]
        )
    )
    minimum_random = int(section.get("minimum_random_prototype_seeds", 20))
    if len(folds) != 3 or len(seeds) != 5:
        raise ValueError(
            "Target component audit requires 3 folds x 5 paired seeds; "
            f"got folds={folds}, seeds={seeds}."
        )
    if len(random_seeds) < minimum_random:
        raise ValueError(
            f"At least {minimum_random} random references are required; "
            f"got {len(random_seeds)}."
        )
    alphas = tuple(float(value) for value in section["ridge_alphas"])
    if len(alphas) != 1:
        raise ValueError("Component audit requires exactly one Ridge alpha.")
    if list(map(str, section["text_news_levels"])) != ["target"]:
        raise ValueError("Component audit requires text_news_levels=[target].")
    if str(section["training_cohort"]) != "all_days":
        raise ValueError("Component audit must train on all fold-train days.")
    if str(section["primary_evaluation_cohort"]) != "target_news_days":
        raise ValueError("Component audit must evaluate on target-news days.")
    basic = str(section["basic_metadata_representation"])
    diagnostics = str(section["prototype_diagnostics_representation"])
    if set(map(str, section["representations"])) != {basic, diagnostics}:
        raise ValueError(
            "Component training grid must contain exactly the basic metadata "
            "and prototype diagnostics aliases."
        )
    source_map = {
        str(key): str(value)
        for key, value in section["representation_source_map"].items()
    }
    combined = str(section["combined_metadata_representation"])
    if source_map.get(basic) != combined or source_map.get(diagnostics) != combined:
        raise ValueError("Both component aliases must read the locked R7 artifact.")
    feature_map = section["representation_feature_names"]
    basic_names = tuple(map(str, feature_map[basic]))
    diagnostic_names = tuple(map(str, feature_map[diagnostics]))
    if not basic_names or not diagnostic_names:
        raise ValueError("Both component feature lists must be non-empty.")
    if set(basic_names).intersection(diagnostic_names):
        raise ValueError("Basic and diagnostic feature lists must be disjoint.")
    forbidden_basic_tokens = ("entropy", "novelty", "distance")
    contaminated = [
        name
        for name in basic_names
        if any(token in name.lower() for token in forbidden_basic_tokens)
    ]
    if contaminated:
        raise ValueError(
            f"Pure metadata contains prototype-derived diagnostics: {contaminated}"
        )
    required_diagnostic_tokens = ("entropy", "novelty", "distance")
    if not all(
        any(token in name.lower() for name in diagnostic_names)
        for token in required_diagnostic_tokens
    ):
        raise ValueError(
            "Prototype diagnostics must cover entropy, novelty and distance."
        )
    profile = dict(section)
    profile.update(
        {
            "folds": folds,
            "seeds": seeds,
            "random_prototype_seeds": random_seeds,
            "alpha": alphas[0],
            # Reused helpers call this field R7 metadata.
            "metadata_representation": combined,
        }
    )
    return profile


def _prior_mechanism_decision(config: Mapping[str, Any]) -> str:
    path = project_path(
        config,
        "outputs",
        "tables",
        "target_mechanism_decision.csv",
    )
    if not path.is_file():
        raise FileNotFoundError(
            "target_mechanism_decision.csv is missing. Complete "
            "--target-mechanism-audit before the component ablation."
        )
    frame = pd.read_csv(path)
    validate_columns(
        frame,
        ("decision", "locked_test_used"),
        "target mechanism decision",
    )
    if len(frame) != 1:
        raise ValueError("Expected exactly one target mechanism decision row.")
    if _truthy_mask(frame["locked_test_used"]).any():
        raise AssertionError(
            "The component audit must be completed before locked test."
        )
    return str(frame.iloc[0]["decision"])


def _validate_component_selectors(
    predictions: pd.DataFrame,
    profile: Mapping[str, Any],
) -> None:
    validate_columns(
        predictions,
        ("representation", "representation_source", "text_feature_names"),
        "target component predictions",
    )
    feature_map = profile["representation_feature_names"]
    source_map = profile["representation_source_map"]
    for representation in profile["representations"]:
        selected = predictions.loc[
            predictions["representation"].astype(str).eq(str(representation))
        ]
        observed_sources = set(
            selected["representation_source"].dropna().astype(str)
        )
        expected_source = str(source_map[representation])
        if observed_sources != {expected_source}:
            raise AssertionError(
                f"{representation} source mismatch: {observed_sources}"
            )
        observed_names = {
            _parse_sequence(value)
            for value in selected["text_feature_names"]
            .dropna()
            .drop_duplicates()
        }
        expected_names = tuple(map(str, feature_map[representation]))
        if observed_names != {expected_names}:
            raise AssertionError(
                f"{representation} feature selector mismatch: {observed_names}"
            )


def _locked_predictions(
    config: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> pd.DataFrame:
    price = str(profile["price_representation"])
    semantic = str(profile["semantic_representation"])
    combined = str(profile["combined_metadata_representation"])
    basic = str(profile["basic_metadata_representation"])
    diagnostics = str(profile["prototype_diagnostics_representation"])
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
    mechanism = _select_prediction_grid(
        load_predictions(
            config,
            mode=str(profile["mechanism_experiment_profile"]),
        ),
        representations=(combined, *random_representations),
        profile=profile,
        label="target mechanism reference predictions",
    )
    components = _select_prediction_grid(
        load_predictions(config, mode="target_component_audit"),
        representations=(basic, diagnostics),
        profile=profile,
        label="target component predictions",
    )
    _validate_component_selectors(components, profile)
    combined_predictions = pd.concat(
        [base, mechanism, components],
        ignore_index=True,
        sort=False,
    )
    expected_representations = {
        price,
        semantic,
        combined,
        basic,
        diagnostics,
        *random_representations,
    }
    observation_keys = [
        "fold",
        "seed",
        "ticker",
        "feature_date",
        "target_date",
    ]
    consistency = combined_predictions.groupby(
        observation_keys,
        sort=False,
        observed=True,
    ).agg(
        representation_count=("representation", "nunique"),
        y_true_count=("y_true", "nunique"),
        news_gate_count=("has_target_news", "nunique"),
    )
    invalid = consistency.loc[
        consistency["representation_count"].ne(len(expected_representations))
        | consistency["y_true_count"].ne(1)
        | consistency["news_gate_count"].ne(1)
    ]
    if not invalid.empty:
        raise AssertionError(
            "Component predictions are not paired with all locked references; "
            f"invalid observation groups={len(invalid)}."
        )
    primary = combined_predictions.loc[
        combined_predictions["has_target_news"]
    ].copy()
    minimum = int(profile.get("minimum_primary_samples_per_fold", 20))
    counts = primary.groupby(
        ["fold", "seed", "representation"],
        observed=True,
    ).size()
    if counts.empty or counts.min() < minimum:
        raise ValueError(
            f"A component cell has fewer than {minimum} target-news samples."
        )
    if (
        counts.groupby(["fold", "seed"], observed=True)
        .nunique()
        .gt(1)
        .any()
    ):
        raise AssertionError("Component sample counts differ across references.")
    return primary.sort_values(
        ["fold", "seed", "representation", "target_date", "ticker"],
        kind="mergesort",
    ).reset_index(drop=True)


def _component_comparisons(
    metrics: pd.DataFrame,
    config: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lookup = _metric_lookup(metrics)
    price = str(profile["price_representation"])
    semantic = str(profile["semantic_representation"])
    combined = str(profile["combined_metadata_representation"])
    basic = str(profile["basic_metadata_representation"])
    diagnostics = str(profile["prototype_diagnostics_representation"])
    pairs = (
        (basic, price, "META_BASIC_vs_R0_price"),
        (diagnostics, price, "PROTO_DIAG_vs_R0_price"),
        (combined, price, "R7_combined_vs_R0_price"),
        (diagnostics, basic, "PROTO_DIAG_vs_META_BASIC"),
        (combined, basic, "R7_combined_vs_META_BASIC"),
        (combined, diagnostics, "R7_combined_vs_PROTO_DIAG"),
        (semantic, basic, "R6_semantic_vs_META_BASIC"),
        (semantic, diagnostics, "R6_semantic_vs_PROTO_DIAG"),
        (semantic, combined, "R6_semantic_vs_R7_combined"),
    )
    rows: list[dict[str, Any]] = []
    for fold in profile["folds"]:
        for seed in profile["seeds"]:
            for candidate_name, reference_name, label in pairs:
                rows.append(
                    _comparison_row(
                        lookup[(fold, seed, candidate_name)],
                        lookup[(fold, seed, reference_name)],
                        comparison=label,
                    )
                )
    comparisons = pd.DataFrame(rows)
    return comparisons, _comparison_summary(comparisons, config, profile)


def run(config: Mapping[str, Any]) -> dict[str, Path]:
    profile = _profile(config)
    prior_decision = _prior_mechanism_decision(config)
    predictions = _locked_predictions(config, profile)
    metrics = aggregate_metrics(predictions)
    expected_metric_rows = (
        len(profile["folds"])
        * len(profile["seeds"])
        * (5 + len(profile["random_prototype_seeds"]))
    )
    if len(metrics) != expected_metric_rows:
        raise ValueError(
            f"Component audit produced {len(metrics)} metric rows; "
            f"expected {expected_metric_rows}."
        )
    comparisons, component_summary = _component_comparisons(
        metrics,
        config,
        profile,
    )
    _, random_cells, _ = _build_comparisons(metrics, config, profile)
    random_summary = _random_seed_summary(random_cells, profile)
    null_summary = _null_distribution_summary(
        random_cells,
        random_summary,
        config,
        profile,
    )
    summary = pd.concat(
        [component_summary, pd.DataFrame([null_summary])],
        ignore_index=True,
        sort=False,
    )
    passed = {
        str(row.comparison): bool(row.comparison_passed)
        for row in summary.itertuples(index=False)
    }
    basic_signal = passed.get("META_BASIC_vs_R0_price", False)
    diagnostic_signal = passed.get("PROTO_DIAG_vs_R0_price", False)
    combined_signal = passed.get("R7_combined_vs_R0_price", False)
    diagnostic_increment = passed.get(
        "R7_combined_vs_META_BASIC", False
    )
    basic_increment = passed.get(
        "R7_combined_vs_PROTO_DIAG", False
    )
    semantic_increment = passed.get(
        "R6_semantic_vs_R7_combined", False
    )
    semantic_random = passed.get(
        "R6_semantic_vs_random_prototype_distribution", False
    )
    if semantic_increment and semantic_random:
        decision = "SEMANTIC-ASSIGNMENT-SUPPORTED"
        reason = (
            "Soft semantic assignments add stable value beyond the combined "
            "metadata block and beat the empirical random-prototype null."
        )
    elif basic_signal and diagnostic_signal:
        if diagnostic_increment and basic_increment:
            decision = "BOTH-COMPONENTS-COMPLEMENTARY"
            reason = (
                "Pure metadata and prototype diagnostics both carry signal "
                "and each adds stable value to the other."
            )
        elif diagnostic_increment:
            decision = "PROTOTYPE-DIAGNOSTICS-INCREMENTAL"
            reason = (
                "Both components carry signal, with prototype diagnostics "
                "adding stable value beyond pure metadata."
            )
        elif basic_increment:
            decision = "PURE-METADATA-INCREMENTAL"
            reason = (
                "Both components carry signal, with pure metadata adding "
                "stable value beyond prototype diagnostics."
            )
        else:
            decision = "REDUNDANT-COMPONENT-SIGNAL"
            reason = (
                "Both components beat price-only, but neither adds the locked "
                "minimum gain when combined with the other."
            )
    elif basic_signal:
        decision = "PURE-METADATA-SUPPORTED"
        reason = (
            "Count, lag and presence features carry stable signal without "
            "prototype-derived diagnostics."
        )
    elif diagnostic_signal:
        decision = "PROTOTYPE-DIAGNOSTICS-SUPPORTED"
        reason = (
            "Entropy, novelty and distance carry signal while pure news "
            "metadata does not pass the locked gates."
        )
    elif combined_signal:
        decision = "COMBINATION-ONLY"
        reason = (
            "Only the combined R7 metadata block passes; neither isolated "
            "component is sufficient under the locked gates."
        )
    else:
        decision = "NO-COMPONENT-SUPPORTED"
        reason = (
            "Neither pure metadata nor prototype diagnostics provides stable "
            "incremental validation signal."
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
                "new_training_task_count": (
                    len(profile["representations"])
                    * len(profile["folds"])
                    * len(profile["seeds"])
                ),
                "primary_sample_counts_by_fold": json.dumps(
                    {
                        str(key): value
                        for key, value in primary_counts.items()
                    },
                    sort_keys=True,
                ),
                "pure_metadata_signal_passed": basic_signal,
                "prototype_diagnostics_signal_passed": diagnostic_signal,
                "combined_metadata_signal_passed": combined_signal,
                "diagnostics_incremental_to_basic_passed": (
                    diagnostic_increment
                ),
                "basic_incremental_to_diagnostics_passed": basic_increment,
                "semantic_incremental_to_combined_passed": semantic_increment,
                "semantic_beyond_random_passed": semantic_random,
                "decision": decision,
                "reason": reason,
                "locked_test_used": False,
                "prior_mechanism_decision_remains": prior_decision,
                "next_step": (
                    "Treat this as post-hoc component evidence. Keep the "
                    "locked holdout closed and pre-register any new temporal "
                    "confirmation."
                ),
            }
        ]
    )
    table_dir = project_path(config, "outputs", "tables")
    paths = {
        "fold_results": table_dir / OUTPUT_NAMES[0],
        "comparisons": table_dir / OUTPUT_NAMES[1],
        "random_reference": table_dir / OUTPUT_NAMES[2],
        "summary": table_dir / OUTPUT_NAMES[3],
        "decision": table_dir / OUTPUT_NAMES[4],
        "report": table_dir / OUTPUT_NAMES[5],
    }
    atomic_write_csv(metrics, paths["fold_results"], index=False)
    atomic_write_csv(comparisons, paths["comparisons"], index=False)
    atomic_write_csv(
        random_summary,
        paths["random_reference"],
        index=False,
    )
    atomic_write_csv(summary, paths["summary"], index=False)
    atomic_write_csv(decision_frame, paths["decision"], index=False)
    atomic_write_json(
        {
            "experiment_profile": "target_component_audit",
            "audit_type": "post_hoc_validation_only",
            "target": "volatility_level",
            "text_news_levels": ["target"],
            "primary_evaluation_cohort": "target_news_days",
            "representations": {
                "price": profile["price_representation"],
                "semantic": profile["semantic_representation"],
                "combined_metadata": profile[
                    "combined_metadata_representation"
                ],
                "pure_metadata": profile[
                    "basic_metadata_representation"
                ],
                "prototype_diagnostics": profile[
                    "prototype_diagnostics_representation"
                ],
            },
            "feature_lists": profile["representation_feature_names"],
            "folds": list(profile["folds"]),
            "prototype_model_seeds": list(profile["seeds"]),
            "random_prototype_seeds": list(
                profile["random_prototype_seeds"]
            ),
            "null_distribution": null_summary,
            "decision": decision,
            "reason": reason,
            "locked_test_used": False,
            "prior_mechanism_decision_remains": prior_decision,
        },
        paths["report"],
    )
    return paths
