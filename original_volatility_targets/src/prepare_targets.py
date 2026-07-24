"""Preparation facade for market targets, sector targets, and shared artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src import build_market_targets, build_sector_targets
from src.utils import (
    atomic_write_csv,
    atomic_write_json,
    load_representation_catalog,
    project_path,
    resolve_shared_file,
)


def verify_shared_artifacts(config: Mapping[str, Any]) -> dict[str, Path]:
    catalog = load_representation_catalog(config)
    required = {f"R{index}" for index in range(12)}
    available = set(catalog["representation"].astype(str))
    missing = sorted(required.difference(available))
    if missing:
        raise FileNotFoundError(f"Missing shared representations: {missing}")
    rows = []
    for row in catalog.itertuples(index=False):
        resolved = Path(str(row.resolved_path))
        rows.append(
            {
                "representation": row.representation,
                "representation_variant": row.representation_variant,
                "selected": bool(row.selected),
                "resolved_path": str(resolved),
                "exists": resolved.is_file(),
                "size_bytes": resolved.stat().st_size if resolved.is_file() else 0,
                "fit_split": getattr(row, "fit_split", ""),
                "placebo": getattr(row, "placebo", False),
            }
        )
    audit = pd.DataFrame(rows)
    if not audit["exists"].all():
        raise FileNotFoundError(
            "Some representation manifest paths could not be resolved"
        )
    audit_path = atomic_write_csv(
        audit,
        project_path(
            config, "outputs", "tables", "shared_artifact_compatibility.csv"
        ),
        index=False,
    )
    optional = {}
    for key, kinds in (
        ("canonical_events", ("processed", "tables")),
        ("prototype_assignments", ("processed",)),
    ):
        path = resolve_shared_file(
            config,
            str(config["shared"][key]),
            kinds=kinds,
            required=False,
        )
        optional[key] = None if path is None else str(path)
    metadata_path = atomic_write_json(
        {
            "residual_project_root": str(
                project_path(config, config["shared"]["residual_project_root"])
            ),
            "representation_manifest": str(catalog["manifest_path"].iloc[0]),
            "representations": sorted(available),
            "canonical_and_prototype_cache": optional,
            "embedding_or_prototype_recomputed": False,
            "shared_artifacts_are_read_only": True,
        },
        project_path(config, "data", "processed", "shared_artifact_contract.json"),
    )
    return {"compatibility": audit_path, "contract": metadata_path}


def run_action(action: str, config: Mapping[str, Any]) -> dict[str, Path]:
    if action == "market_targets":
        return build_market_targets.run(config)
    if action == "sector_targets":
        return build_sector_targets.run(config)
    if action == "verify_shared":
        return verify_shared_artifacts(config)
    raise ValueError(f"Unsupported prepare action: {action}")
