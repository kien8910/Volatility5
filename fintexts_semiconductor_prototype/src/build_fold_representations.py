"""Build the locked R6 confirmatory fold/seed representation panel."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from src import build_prototypes
from src.aggregate_features import NEWS_LEVELS, aggregate_fold_features
from src.utils import (
    atomic_write_csv,
    ensure_directories,
    get_logger,
    project_path,
    safe_read_table,
    set_global_seed,
    validate_required_columns,
)

LOGGER = get_logger(__name__)


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


def _settings(config: Mapping[str, Any]) -> dict[str, Any]:
    section = config.get("r6_confirmatory")
    if not isinstance(section, Mapping):
        raise KeyError("Missing r6_confirmatory configuration section.")
    folds = tuple(dict.fromkeys(int(value) for value in section["folds"]))
    seeds = tuple(dict.fromkeys(int(value) for value in section["seeds"]))
    representations = tuple(
        dict.fromkeys(str(value) for value in section["representations"])
    )
    if len(folds) != 3:
        raise ValueError(
            f"R6 confirmatory run requires exactly 3 folds; got {folds}."
        )
    if len(seeds) != 5:
        raise ValueError(
            f"R6 confirmatory run requires exactly 5 seeds; got {seeds}."
        )
    required = {
        "R3",
        "R4",
        "R6",
        "R9",
        "R10",
        "R11",
        "P_LAGGED",
        "P_PERMUTED",
    }
    missing = sorted(required.difference(representations))
    if missing:
        raise ValueError(
            "R6 confirmatory fold artifacts are missing required comparators: "
            f"{missing}"
        )
    return {
        "folds": folds,
        "seeds": seeds,
        "representations": representations,
        "family": str(section["representation_variant_family"]),
        "pooling": str(section.get("pooling", "mean")),
    }


def _prototype_parameters(config: Mapping[str, Any]) -> dict[str, Any]:
    section = config.get("prototype")
    if not isinstance(section, Mapping):
        raise TypeError("prototype configuration must be a mapping.")
    levels = tuple(section.get("levels", NEWS_LEVELS))
    unknown_levels = sorted(set(levels).difference(NEWS_LEVELS))
    if unknown_levels:
        raise ValueError(f"Unknown prototype levels: {unknown_levels}")
    k_values = sorted({int(value) for value in section["k_values"]})
    pca_dims = build_prototypes._normalize_pca_dims(section["pca_dims"])
    temperatures = sorted({float(value) for value in section["temperatures"]})
    if len(k_values) != 1 or len(pca_dims) != 1 or len(temperatures) != 1:
        raise ValueError(
            "R6 confirmatory codebooks must lock exactly one K, PCA dimension "
            "and temperature before folds are built."
        )
    return {
        "levels": levels,
        "k_values": k_values,
        "pca_dims": pca_dims,
        "temperatures": temperatures,
        "n_init": int(section.get("n_init", 20)),
        "max_iter": int(section.get("max_iter", 500)),
        "min_cluster_events": int(
            section.get("min_events_per_cluster", 5)
        ),
        "max_small_fraction": float(
            section.get("max_small_cluster_fraction", 0.5)
        ),
        "min_effective_fraction": float(
            section.get("min_effective_fraction", 0.35)
        ),
        "silhouette_sample_size": int(
            section.get("silhouette_sample_size", 5000)
        ),
    }


def _validate_codebook_grid(
    manifest: pd.DataFrame,
    settings: Mapping[str, Any],
) -> None:
    validate_required_columns(
        manifest,
        (
            "fold_id",
            "news_level",
            "representation_variant_family",
            "prototype_seed",
            "eligible",
            "fit_scope",
            "assignment_path",
        ),
        "R6 confirmatory prototype fold manifest",
    )
    selected = manifest.loc[
        manifest["representation_variant_family"].astype(str).eq(
            str(settings["family"])
        )
        & pd.to_numeric(manifest["fold_id"], errors="coerce").isin(
            settings["folds"]
        )
        & pd.to_numeric(manifest["prototype_seed"], errors="coerce").isin(
            settings["seeds"]
        )
    ].copy()
    expected = {
        (fold, seed, level)
        for fold in settings["folds"]
        for seed in settings["seeds"]
        for level in NEWS_LEVELS
    }
    eligible_selected = selected.loc[
        _truthy_mask(selected["eligible"])
        & selected["fit_scope"].astype(str).eq("fold_train_only")
        & selected["assignment_path"].fillna("").astype(str).map(
            lambda value: Path(value).is_file()
        )
    ]
    observed = {
        (int(row.fold_id), int(row.prototype_seed), str(row.news_level))
        for row in eligible_selected.itertuples(index=False)
    }
    missing = sorted(expected.difference(observed))
    if missing:
        rejected = selected.loc[
            ~_truthy_mask(selected["eligible"]),
            [
                column
                for column in (
                    "fold_id",
                    "prototype_seed",
                    "news_level",
                    "rejection_reason",
                )
                if column in selected
            ],
        ]
        raise ValueError(
            "The locked fold codebook grid is incomplete. Missing eligible "
            f"fold/seed/level cells: {missing}. Rejections: "
            f"{rejected.to_dict(orient='records')}"
        )
    if len(eligible_selected) != len(expected):
        raise ValueError(
            "The fold codebook manifest has duplicate eligible cells for the "
            "locked family."
        )


def _validate_feature_grid(
    manifest: pd.DataFrame,
    settings: Mapping[str, Any],
) -> pd.DataFrame:
    validate_required_columns(
        manifest,
        (
            "fold_id",
            "representation",
            "representation_variant_family",
            "prototype_seed",
            "path",
            "fit_scope",
        ),
        "R6 confirmatory fold representation manifest",
    )
    selected = manifest.loc[
        manifest["representation_variant_family"].astype(str).eq(
            str(settings["family"])
        )
        & pd.to_numeric(manifest["fold_id"], errors="coerce").isin(
            settings["folds"]
        )
        & pd.to_numeric(manifest["prototype_seed"], errors="coerce").isin(
            settings["seeds"]
        )
        & manifest["representation"].astype(str).isin(
            settings["representations"]
        )
    ].copy()
    expected = {
        (fold, seed, representation)
        for fold in settings["folds"]
        for seed in settings["seeds"]
        for representation in settings["representations"]
    }
    observed = {
        (int(row.fold_id), int(row.prototype_seed), str(row.representation))
        for row in selected.itertuples(index=False)
        if str(row.fit_scope) == "fold_train_only"
        and Path(str(row.path)).is_file()
    }
    missing = sorted(expected.difference(observed))
    if missing:
        raise ValueError(
            "The locked fold representation grid is incomplete. Missing "
            f"fold/seed/representation cells: {missing}"
        )
    if len(selected) != len(expected):
        raise ValueError(
            "The fold representation manifest has duplicate cells for the "
            "locked confirmatory grid."
        )
    return selected.sort_values(
        ["fold_id", "prototype_seed", "representation"],
        kind="mergesort",
    ).reset_index(drop=True)


def run(config: dict) -> dict[str, Path]:
    """Fit 3 x 5 fold codebooks and materialize matched R6 comparators."""

    settings = _settings(config)
    prototype = _prototype_parameters(config)
    if not bool(config["prototype"].get("build_fold_codebooks", False)):
        raise ValueError(
            "prototype.build_fold_codebooks must be true for this stage."
        )
    ensure_directories(config)
    set_global_seed(
        int(settings["seeds"][0]),
        deterministic=bool(
            config.get("project", {}).get("deterministic", True)
        ),
    )
    events, embeddings = build_prototypes._load_inputs(config)
    build_prototypes._assert_chronology(events)

    events_variant = build_prototypes._events_variant(config)
    suffix = build_prototypes._variant_suffix(events_variant)
    fold_manifest_path = project_path(
        config, f"data/processed/prototype_fold_manifest{suffix}.csv"
    )
    reuse_codebooks = False
    if fold_manifest_path.is_file():
        existing_manifest = safe_read_table(fold_manifest_path)
        try:
            _validate_codebook_grid(existing_manifest, settings)
        except (KeyError, ValueError):
            reuse_codebooks = False
        else:
            reuse_codebooks = True
            LOGGER.info(
                "R6 confirmatory codebooks already complete; reusing the "
                "existing 3-fold x 5-seed assignment grid."
            )
    if not reuse_codebooks:
        for index, seed in enumerate(settings["seeds"]):
            LOGGER.info(
                "R6 confirmatory codebooks | seed %d/%d=%d | folds=%s | "
                "family=%s",
                index + 1,
                len(settings["seeds"]),
                seed,
                settings["folds"],
                settings["family"],
            )
            fold_manifest_path = build_prototypes._build_fold_codebooks(
                config=config,
                events=events,
                embeddings=embeddings,
                seed=int(seed),
                # Preserve any other fold families already materialized;
                # identical locked cells are replaced by the current run.
                append=True,
                **prototype,
            )
        codebook_manifest = safe_read_table(fold_manifest_path)
        _validate_codebook_grid(codebook_manifest, settings)

    total = len(settings["folds"]) * len(settings["seeds"])
    completed = 0
    representation_manifest_path: Path | None = None
    for fold in settings["folds"]:
        for seed in settings["seeds"]:
            completed += 1
            LOGGER.info(
                "R6 confirmatory features | cell %d/%d | fold=%d | "
                "prototype/model seed=%d",
                completed,
                total,
                fold,
                seed,
            )
            outputs = aggregate_fold_features(
                config,
                fold_id=int(fold),
                representation_variant_family=str(settings["family"]),
                prototype_seed=int(seed),
                pooling=str(settings["pooling"]),
                representations=settings["representations"],
            )
            representation_manifest_path = outputs[
                "fold_representation_manifest"
            ]
    if representation_manifest_path is None:
        raise AssertionError("No confirmatory fold feature cell was executed.")
    representation_manifest = safe_read_table(
        representation_manifest_path
    )
    selected = _validate_feature_grid(representation_manifest, settings)
    summary_path = project_path(
        config,
        "outputs",
        "tables",
        "r6_confirmatory_artifact_summary.csv",
    )
    summary = (
        selected.groupby(
            ["representation", "fit_scope"],
            sort=True,
            observed=True,
        )
        .agg(
            fold_count=("fold_id", "nunique"),
            seed_count=("prototype_seed", "nunique"),
            artifact_count=("path", "size"),
        )
        .reset_index()
    )
    summary["expected_artifact_count"] = total
    summary["complete"] = summary["artifact_count"].eq(total)
    atomic_write_csv(summary, summary_path, index=False)
    LOGGER.info(
        "R6 confirmatory fold artifacts complete | %d folds x %d seeds x "
        "%d representations",
        len(settings["folds"]),
        len(settings["seeds"]),
        len(settings["representations"]),
    )
    return {
        "prototype_fold_manifest": fold_manifest_path,
        "fold_representation_manifest": representation_manifest_path,
        "r6_confirmatory_artifact_summary": summary_path,
    }
