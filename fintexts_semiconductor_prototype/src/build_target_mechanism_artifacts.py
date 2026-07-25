"""Materialize fold-safe metadata and random-prototype mechanism artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from src.aggregate_features import (
    _events_variant,
    _variant_suffix,
    aggregate_fold_features,
)
from src.build_fold_representations import _validate_codebook_grid
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


def _settings(config: Mapping[str, Any]) -> dict[str, Any]:
    section = config.get("target_mechanism_audit")
    if not isinstance(section, Mapping):
        raise KeyError("Missing target_mechanism_audit configuration section.")
    folds = tuple(dict.fromkeys(int(value) for value in section["folds"]))
    prototype_seeds = tuple(
        dict.fromkeys(int(value) for value in section["prototype_seeds"])
    )
    random_seeds = tuple(
        dict.fromkeys(int(value) for value in section["random_prototype_seeds"])
    )
    if len(folds) != 3:
        raise ValueError(
            f"Mechanism artifacts require exactly 3 folds; got {folds}."
        )
    if len(prototype_seeds) != 5:
        raise ValueError(
            "Mechanism artifacts require exactly 5 prototype seeds; "
            f"got {prototype_seeds}."
        )
    minimum_random = int(section.get("minimum_random_prototype_seeds", 20))
    if len(random_seeds) < minimum_random:
        raise ValueError(
            f"At least {minimum_random} random-prototype seeds are required; "
            f"got {len(random_seeds)}."
        )
    if any(value < 0 for value in random_seeds):
        raise ValueError("Random-prototype seeds must be non-negative.")
    if set(prototype_seeds).intersection(random_seeds):
        raise ValueError(
            "Random-prototype seeds must be disjoint from prototype seeds."
        )
    return {
        "folds": folds,
        "prototype_seeds": prototype_seeds,
        "random_seeds": random_seeds,
        "family": str(section["representation_variant_family"]),
        "pooling": str(section.get("pooling", "mean")),
        "metadata_representation": str(
            section.get("metadata_representation", "R7")
        ),
        "random_prefix": str(
            section.get("random_representation_prefix", "R9_NULL")
        ),
    }


def _validate_artifacts(
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
            "random_prototype_seed",
            "path",
            "fit_scope",
        ),
        "target mechanism fold representation manifest",
    )
    random_representations = tuple(
        f"{settings['random_prefix']}_{seed}"
        for seed in settings["random_seeds"]
    )
    representations = (
        str(settings["metadata_representation"]),
        *random_representations,
    )
    selected = manifest.loc[
        manifest["representation_variant_family"].astype(str).eq(
            str(settings["family"])
        )
        & pd.to_numeric(manifest["fold_id"], errors="coerce").isin(
            settings["folds"]
        )
        & pd.to_numeric(manifest["prototype_seed"], errors="coerce").isin(
            settings["prototype_seeds"]
        )
        & manifest["representation"].astype(str).isin(representations)
    ].copy()
    expected = {
        (fold, prototype_seed, representation)
        for fold in settings["folds"]
        for prototype_seed in settings["prototype_seeds"]
        for representation in representations
    }
    observed = {
        (int(row.fold_id), int(row.prototype_seed), str(row.representation))
        for row in selected.itertuples(index=False)
        if str(row.fit_scope) == "fold_train_only"
        and Path(str(row.path)).is_file()
    }
    missing = sorted(expected.difference(observed))
    unexpected = sorted(observed.difference(expected))
    if missing or unexpected:
        raise ValueError(
            "Target mechanism artifact grid is incomplete: "
            f"missing={missing}; unexpected={unexpected}"
        )
    counts = (
        selected.groupby(
            ["fold_id", "prototype_seed", "representation"],
            sort=True,
            observed=True,
        )
        .size()
    )
    if counts.ne(1).any() or len(selected) != len(expected):
        raise ValueError(
            "Target mechanism artifact grid contains duplicate cells."
        )
    random_rows = selected.loc[
        selected["representation"].astype(str).str.startswith(
            f"{settings['random_prefix']}_"
        )
    ].copy()
    encoded_random_seed = pd.to_numeric(
        random_rows["representation"].astype(str).str.rsplit("_", n=1).str[-1],
        errors="raise",
    ).astype(int)
    manifest_random_seed = pd.to_numeric(
        random_rows["random_prototype_seed"],
        errors="raise",
    ).astype(int)
    if not encoded_random_seed.equals(manifest_random_seed):
        raise AssertionError(
            "Random-prototype representation name and manifest seed disagree."
        )
    return selected.sort_values(
        ["fold_id", "prototype_seed", "representation"],
        kind="mergesort",
    ).reset_index(drop=True)


def run(config: dict) -> dict[str, Path]:
    """Reuse locked fold codebooks and materialize mechanism-audit features."""

    settings = _settings(config)
    ensure_directories(config)
    set_global_seed(
        int(settings["prototype_seeds"][0]),
        deterministic=bool(
            config.get("project", {}).get("deterministic", True)
        ),
    )
    events_variant = _events_variant(config)
    suffix = _variant_suffix(events_variant)
    fold_manifest_path = project_path(
        config,
        f"data/processed/prototype_fold_manifest{suffix}.csv",
    )
    if not fold_manifest_path.is_file():
        raise FileNotFoundError(
            "Fold codebooks are missing. Run the r6-confirmatory artifact "
            "stage before target-mechanism-artifacts."
        )
    codebook_manifest = safe_read_table(fold_manifest_path)
    _validate_codebook_grid(
        codebook_manifest,
        {
            "folds": settings["folds"],
            "seeds": settings["prototype_seeds"],
            "family": settings["family"],
        },
    )
    total = len(settings["folds"]) * len(settings["prototype_seeds"])
    completed = 0
    representation_manifest_path: Path | None = None
    for fold in settings["folds"]:
        for prototype_seed in settings["prototype_seeds"]:
            completed += 1
            LOGGER.info(
                "Target mechanism artifacts | cell %d/%d | fold=%d | "
                "prototype_seed=%d | random_seeds=%d",
                completed,
                total,
                fold,
                prototype_seed,
                len(settings["random_seeds"]),
            )
            outputs = aggregate_fold_features(
                config,
                fold_id=int(fold),
                representation_variant_family=str(settings["family"]),
                prototype_seed=int(prototype_seed),
                pooling=str(settings["pooling"]),
                representations=(str(settings["metadata_representation"]),),
                random_placebo_seeds=settings["random_seeds"],
                random_representation_prefix=str(settings["random_prefix"]),
            )
            representation_manifest_path = outputs[
                "fold_representation_manifest"
            ]
    if representation_manifest_path is None:
        raise AssertionError("No target mechanism artifact cell was executed.")
    selected = _validate_artifacts(
        safe_read_table(representation_manifest_path),
        settings,
    )
    summary_path = project_path(
        config,
        "outputs/tables/target_mechanism_artifact_summary.csv",
    )
    selected["artifact_kind"] = (
        selected["representation"]
        .astype(str)
        .str.startswith(f"{settings['random_prefix']}_")
        .map({True: "random_prototype_null", False: "target_metadata"})
    )
    summary = (
        selected.groupby(
            ["artifact_kind", "fit_scope"],
            sort=True,
            observed=True,
        )
        .agg(
            fold_count=("fold_id", "nunique"),
            prototype_seed_count=("prototype_seed", "nunique"),
            random_seed_count=("random_prototype_seed", "nunique"),
            representation_count=("representation", "nunique"),
            artifact_count=("path", "size"),
        )
        .reset_index()
    )
    summary["complete"] = True
    atomic_write_csv(summary, summary_path, index=False)
    LOGGER.info(
        "Target mechanism artifacts complete | folds=%d | prototype_seeds=%d "
        "| random_seeds=%d | artifacts=%d",
        len(settings["folds"]),
        len(settings["prototype_seeds"]),
        len(settings["random_seeds"]),
        len(selected),
    )
    return {
        "fold_representation_manifest": representation_manifest_path,
        "target_mechanism_artifact_summary": summary_path,
    }
