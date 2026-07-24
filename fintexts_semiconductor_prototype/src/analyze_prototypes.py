"""Enrich prototype diagnostics and analyse cross-stock semiconductor responses.

All response summaries align an event available on feature date ``t`` with the
already leakage-safe baseline residual for target date ``t+1``.  The analysis
is descriptive; it never re-selects a prototype configuration using test
responses and it does not promote descriptive common effects to forecasting
evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from src.utils import (
    atomic_write_csv,
    ensure_directories,
    get_logger,
    load_config,
    project_path,
    safe_read_table,
    validate_required_columns,
)

NEWS_LEVELS = ("macro", "sector", "related", "target")
PROTOTYPE_DETAIL_COLUMNS = (
    "record_type",
    "candidate_id",
    "news_level",
    "prototype_id",
    "split",
    "n_events",
    "event_fraction",
    "ticker_distribution",
    "year_distribution",
    "quarter_distribution",
    "mean_residual",
    "mean_absolute_residual",
    "spike_q90_rate",
    "spike_q95_rate",
    "regime_distribution",
    "mean_cross_stock_breadth",
    "within_prototype_response_variance",
    "mean_assignment_entropy",
    "mean_novelty",
    "mean_nearest_distance",
    "effective_number_of_prototypes",
    "dead_prototype_fraction",
    "small_prototype_fraction",
    "ticker_concentration",
    "time_concentration",
)
EXAMPLE_COLUMNS = (
    "record_type",
    "candidate_id",
    "news_level",
    "prototype_id",
    "rank",
    "event_id",
    "date",
    "text",
    "cosine_similarity",
    "mean_event_residual",
    "mean_event_absolute_residual",
    "event_spike_q90_rate",
    "event_spike_q95_rate",
)
STABILITY_COLUMNS = (
    "record_type",
    "candidate_id",
    "news_level",
    "prototype_id",
    "reference_split",
    "comparison_split",
    "adjusted_rand_index",
    "matched_centroid_cosine",
    "usage_js_distance",
    "active_prototype_fraction",
    "mean_assignment_entropy",
    "n_events",
    "evidence_scope",
)
CROSS_STOCK_COLUMNS = (
    "record_type",
    "date",
    "candidate_id",
    "news_level",
    "prototype_id",
    "target_ticker",
    "n_events",
    "n_stocks",
    "residual_breadth",
    "same_sign_fraction",
    "mean_residual",
    "mean_absolute_residual",
    "cross_stock_residual_correlation",
    "n_stocks_above_q90",
    "concentration_index",
    "common_component",
    "firm_specific_component",
    "target_residual",
    "non_target_mean_residual",
    "target_vs_nontarget_effect_size",
    "spillover_mean_absolute_residual",
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


def _event_paths(config: Mapping[str, Any]) -> tuple[Path, ...]:
    variant = _events_variant(config)
    if variant in {"exact", "raw"}:
        return (
            project_path(
                config,
                "data",
                "processed",
                f"canonical_events_{variant}.parquet",
            ),
        )
    return (
        project_path(config, "data", "processed", "canonical_events.parquet"),
        project_path(config, "outputs", "tables", "canonical_events.csv"),
    )


def _empty_with_columns(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _load_optional(
    paths: Iterable[Path],
    required: Sequence[str],
    context: str,
    logger: Any,
) -> pd.DataFrame:
    path = _first_existing(paths)
    if path is None:
        logger.warning("%s artifact is unavailable; dependent outputs will be empty", context)
        return _empty_with_columns(required)
    frame = safe_read_table(path)
    validate_required_columns(frame, required, context)
    return frame


def _json_list(value: Any) -> list[str]:
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
            raise ValueError(f"Invalid JSON list: {text[:100]}") from exc
        if not isinstance(loaded, list):
            raise ValueError(f"Expected a JSON list, found {type(loaded).__name__}")
        return [str(item) for item in loaded]
    return [part.strip() for part in text.split(",") if part.strip()]


def _json_distribution(values: pd.Series) -> str:
    counts = values.dropna().astype(str).value_counts(normalize=True, sort=False)
    payload = {key: float(value) for key, value in sorted(counts.items())}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _hhi(values: pd.Series) -> float:
    counts = values.dropna().astype(str).value_counts(normalize=True)
    if counts.empty:
        return np.nan
    return float(np.square(counts.to_numpy(dtype=float)).sum())


def _preserve_and_append(
    existing: pd.DataFrame,
    additions: pd.DataFrame,
    existing_record_type: str,
    keys: Sequence[str],
) -> pd.DataFrame:
    left = existing.copy()
    if not left.empty and "record_type" not in left.columns:
        left.insert(0, "record_type", existing_record_type)
    if additions.empty:
        return left
    combined = pd.concat([left, additions], ignore_index=True, sort=False)
    usable_keys = [column for column in keys if column in combined.columns]
    if usable_keys:
        combined = combined.drop_duplicates(usable_keys, keep="first")
    return combined


def _canonical_event_tickers(
    events: pd.DataFrame, tickers: Sequence[str]
) -> pd.DataFrame:
    ticker_set = set(tickers)
    rows: list[dict[str, Any]] = []
    for event in events.itertuples(index=False):
        level = str(getattr(event, "news_level"))
        available = (
            _json_list(getattr(event, "available_to_tickers"))
            if hasattr(event, "available_to_tickers")
            else []
        )
        if level in {"macro", "sector"}:
            available = list(tickers)
        elif level == "target" and not available and hasattr(event, "target_ticker"):
            value = getattr(event, "target_ticker")
            available = [] if pd.isna(value) else [str(value)]
        elif level == "related" and not available and hasattr(event, "related_tickers"):
            available = _json_list(getattr(event, "related_tickers"))
        for ticker in sorted(set(available).intersection(ticker_set)):
            rows.append(
                {
                    "event_id": str(getattr(event, "event_id")),
                    "ticker": ticker,
                    "feature_date": pd.Timestamp(getattr(event, "date")).normalize(),
                    "news_level": level,
                }
            )
    return (
        pd.DataFrame(rows)
        if rows
        else _empty_with_columns(["event_id", "ticker", "feature_date", "news_level"])
    )


def _split_manifest_variant_and_pooling(
    config: Mapping[str, Any], value: Any
) -> tuple[str, str]:
    locked = str(value)
    marker = "__pool_"
    if marker in locked:
        variant, pooling = locked.rsplit(marker, 1)
        if not variant or not pooling:
            raise ValueError(f"Malformed locked representation variant: {locked!r}")
        return variant, pooling
    prototype_config = config.get("prototype", {})
    pooling = (
        str(prototype_config.get("pooling", "mean"))
        if isinstance(prototype_config, Mapping)
        else "mean"
    )
    return locked, pooling


def _reconstruct_candidate_assignments(
    config: Mapping[str, Any],
    events: pd.DataFrame,
    prototype_manifest: pd.DataFrame,
    candidate_ids: Mapping[str, Any],
) -> pd.DataFrame:
    validate_required_columns(
        prototype_manifest,
        [
            "candidate_id",
            "news_level",
            "seed",
            "pca_dim",
            "k",
            "temperature",
            "assignment_path",
        ],
        "prototype manifest for locked analysis assignments",
    )
    frames: list[pd.DataFrame] = []
    for level in NEWS_LEVELS:
        candidate_id = candidate_ids.get(level)
        if candidate_id in (None, ""):
            raise ValueError(
                f"Validation-locked prototype family has no {level} candidate"
            )
        matches = prototype_manifest.loc[
            prototype_manifest["candidate_id"].astype(str).eq(str(candidate_id))
            & prototype_manifest["news_level"].astype(str).eq(level)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one manifest row for locked {level} candidate "
                f"{candidate_id!r}, found {len(matches)}"
            )
        manifest_row = matches.iloc[0]
        assignment_path = Path(str(manifest_row["assignment_path"]))
        if not assignment_path.is_absolute():
            assignment_path = project_path(config, assignment_path)
        if not assignment_path.is_file():
            raise FileNotFoundError(
                f"Locked prototype assignment does not exist: {assignment_path}"
            )
        with np.load(assignment_path, allow_pickle=False) as arrays:
            required_arrays = {
                "event_indices",
                "event_ids",
                "hard_cluster_id",
                "soft_assignment",
                "nearest_similarity",
                "nearest_distance",
                "novelty",
                "assignment_entropy",
                "effective_soft_prototypes",
            }
            missing_arrays = sorted(required_arrays.difference(arrays.files))
            if missing_arrays:
                raise ValueError(
                    f"Locked assignment {assignment_path} is missing arrays: "
                    f"{missing_arrays}"
                )
            event_indices = arrays["event_indices"].astype(int)
            if event_indices.size and (
                int(event_indices.min()) < 0
                or int(event_indices.max()) >= len(events)
            ):
                raise ValueError(
                    f"Locked assignment {assignment_path} has invalid event indices"
                )
            level_events = events.iloc[event_indices].reset_index(drop=True)
            stored_event_ids = arrays["event_ids"].astype(str)
            expected_event_ids = level_events["event_id"].astype(str).to_numpy()
            if not np.array_equal(stored_event_ids, expected_event_ids):
                raise ValueError(
                    f"Locked assignment {assignment_path} has stale event ordering"
                )
            if not level_events["news_level"].astype(str).eq(level).all():
                raise ValueError(
                    f"Locked assignment {assignment_path} contains another news level"
                )
            hard = arrays["hard_cluster_id"].astype(int)
            soft = arrays["soft_assignment"].astype(float)
            if len(hard) != len(level_events) or len(soft) != len(level_events):
                raise ValueError(
                    f"Locked assignment {assignment_path} has inconsistent lengths"
                )
            frame = level_events[
                ["event_id", "date", "split", "news_level"]
            ].copy()
            frame["candidate_id"] = str(candidate_id)
            frame["k"] = int(manifest_row["k"])
            frame["pca_dim"] = manifest_row["pca_dim"]
            frame["temperature"] = float(manifest_row["temperature"])
            frame["seed"] = int(manifest_row["seed"])
            frame["hard_cluster_id"] = hard
            frame["soft_assignment"] = [
                json.dumps(vector.tolist(), separators=(",", ":"))
                for vector in soft
            ]
            for column in (
                "nearest_similarity",
                "nearest_distance",
                "novelty",
                "assignment_entropy",
                "effective_soft_prototypes",
            ):
                frame[column] = arrays[column].astype(float)
            frames.append(frame)
    assignments = pd.concat(frames, ignore_index=True)
    if assignments["event_id"].duplicated().any():
        raise ValueError(
            "Validation-locked candidate reconstruction assigned an event more than once"
        )
    return assignments.sort_values(
        ["date", "news_level", "event_id"], kind="mergesort"
    ).reset_index(drop=True)


def _analysis_assignments(
    config: Mapping[str, Any],
    events: pd.DataFrame,
    logger: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the validation-locked family, or record an explicit no-target fallback."""

    events_variant = _events_variant(config)
    suffix = _variant_suffix(events_variant)
    decision_path = project_path(config, "outputs", "tables", "final_decision.csv")
    fallback_reason = ""
    decision_label = "unavailable"
    best_target = "none"
    decision = pd.DataFrame()
    target_rows = pd.DataFrame()
    if decision_path.is_file():
        decision = safe_read_table(decision_path)
        validate_required_columns(decision, ["record_type"], "final decision")
        final_rows = decision.loc[
            decision["record_type"].astype(str).eq("final")
        ]
        if final_rows.empty:
            fallback_reason = "final_decision_has_no_final_row"
        else:
            final_row = final_rows.iloc[-1]
            decision_label = str(final_row.get("decision", "NO-GO")).upper()
            best_target = str(
                final_row.get("best_target", final_row.get("target", "none"))
            )
            if best_target.lower() in {
                "",
                "nan",
                "none",
            }:
                fallback_reason = "final_decision_has_no_best_target"
    else:
        fallback_reason = "final_decision_unavailable"

    if not fallback_reason:
        if "target" not in decision.columns:
            fallback_reason = "locked_target_column_unavailable"
        else:
            target_rows = decision.loc[
                decision["record_type"].astype(str).eq("target")
                & decision["target"].astype(str).eq(best_target)
            ]
            if target_rows.empty:
                fallback_reason = "locked_target_unavailable"
            elif len(target_rows) > 1:
                raise ValueError(
                    f"Expected one validation-locked decision row for target "
                    f"{best_target!r}, found {len(target_rows)}"
                )

    if fallback_reason:
        assignment_path = project_path(
            config,
            "data",
            "processed",
            f"prototype_assignments{suffix}.parquet",
        )
        assignments = _load_optional(
            (assignment_path,),
            [
                "event_id",
                "date",
                "split",
                "news_level",
                "candidate_id",
                "hard_cluster_id",
                "assignment_entropy",
                "novelty",
                "nearest_distance",
            ],
            "default prototype assignments used for explicit analysis fallback",
            logger,
        )
        candidate_ids = (
            {
                str(level): sorted(
                    assignments.loc[
                        assignments["news_level"].astype(str).eq(str(level)),
                        "candidate_id",
                    ]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )
                for level in NEWS_LEVELS
            }
            if not assignments.empty
            else {}
        )
        logger.warning(
            "Prototype response analysis uses the configured default codebook "
            "instead of a validation-locked winner: %s",
            fallback_reason,
        )
        return assignments, {
            "analysis_assignment_source": "default_fallback",
            "analysis_fallback": True,
            "analysis_fallback_reason": fallback_reason,
            "decision_at_analysis": decision_label,
            "target_at_analysis": best_target,
            "representation_at_analysis": "",
            "representation_variant_family_at_analysis": "",
            "representation_variant_at_analysis": "selected_default",
            "pooling_at_analysis": (
                str(config.get("prototype", {}).get("pooling", "mean"))
                if isinstance(config.get("prototype", {}), Mapping)
                else "mean"
            ),
            "candidate_ids": json.dumps(candidate_ids, sort_keys=True),
            "events_variant": events_variant,
            "representation_manifest_path": "",
            "prototype_manifest_path": "",
            "assignment_artifact_path": str(assignment_path),
        }

    target_row = target_rows.iloc[0]
    representation = str(target_row.get("representation", ""))
    if representation not in {"R5", "R6", "R7", "R8"}:
        raise ValueError(
            "The validation-locked decision does not identify a semantic "
            f"prototype representation: {representation!r}"
        )
    locked_family = str(
        target_row.get("representation_variant_family", "")
    )
    locked_variant_value = target_row.get(
        "representative_variant_for_artifacts",
        target_row.get("representation_variant", ""),
    )
    if pd.isna(locked_variant_value) or not str(locked_variant_value).strip():
        raise ValueError(
            "The validation-locked decision has no representative artifact variant"
        )
    manifest_variant, locked_pooling = _split_manifest_variant_and_pooling(
        config, locked_variant_value
    )
    manifest_family, family_pooling = _split_manifest_variant_and_pooling(
        config, locked_family
    )
    if locked_pooling != family_pooling:
        raise ValueError(
            "Locked representation variant and family encode different pooling "
            f"methods: {locked_pooling!r} versus {family_pooling!r}"
        )

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
            project_path(
                config, "data", "processed", "representation_manifest.csv"
            ),
        )
    )
    if representation_manifest_path is None:
        raise FileNotFoundError(
            f"representation_manifest{suffix}.csv is unavailable"
        )
    representation_manifest = safe_read_table(representation_manifest_path)
    validate_required_columns(
        representation_manifest,
        [
            "representation",
            "representation_variant",
            "representation_variant_family",
            "pooling",
            "prototype_candidate_id",
        ],
        "representation manifest for locked prototype analysis",
    )
    manifest_mask = (
        representation_manifest["representation"].astype(str).eq(representation)
        & representation_manifest["representation_variant"]
        .astype(str)
        .eq(manifest_variant)
        & representation_manifest["representation_variant_family"]
        .astype(str)
        .eq(manifest_family)
        & representation_manifest["pooling"].astype(str).eq(locked_pooling)
    )
    if "events_variant" in representation_manifest.columns:
        manifest_mask &= representation_manifest["events_variant"].astype(str).eq(
            events_variant
        )
    representation_rows = representation_manifest.loc[manifest_mask]
    if len(representation_rows) != 1:
        raise ValueError(
            "Expected one representation-manifest row for the validation-locked "
            f"family, found {len(representation_rows)}"
        )
    candidate_ids = json.loads(
        str(representation_rows.iloc[0]["prototype_candidate_id"])
    )
    if not isinstance(candidate_ids, dict):
        raise ValueError(
            "Locked prototype_candidate_id must decode to a level-to-candidate mapping"
        )

    prototype_manifest_path = project_path(
        config,
        "data",
        "processed",
        f"prototype_manifest{suffix}.csv",
    )
    prototype_manifest = safe_read_table(prototype_manifest_path)
    if "events_variant" in prototype_manifest.columns:
        prototype_manifest = prototype_manifest.loc[
            prototype_manifest["events_variant"].astype(str).eq(events_variant)
        ].copy()
    assignments = _reconstruct_candidate_assignments(
        config, events, prototype_manifest, candidate_ids
    )
    logger.info(
        "Prototype response analysis reconstructed validation-locked %s/%s "
        "family %s using %s pooling.",
        best_target,
        representation,
        locked_family,
        locked_pooling,
    )
    return assignments, {
        "analysis_assignment_source": "validation_locked_family",
        "analysis_fallback": False,
        "analysis_fallback_reason": "",
        "decision_at_analysis": decision_label,
        "target_at_analysis": best_target,
        "representation_at_analysis": representation,
        "representation_variant_family_at_analysis": locked_family,
        "representation_variant_at_analysis": str(locked_variant_value),
        "pooling_at_analysis": locked_pooling,
        "candidate_ids": json.dumps(candidate_ids, sort_keys=True),
        "events_variant": events_variant,
        "representation_manifest_path": str(representation_manifest_path),
        "prototype_manifest_path": str(prototype_manifest_path),
        "assignment_artifact_path": "reconstructed_from_candidate_npz",
    }


def _prepare_inputs(
    config: Mapping[str, Any], logger: Any
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    events = _load_optional(
        _event_paths(config),
        [
            "event_id",
            "date",
            "split",
            "news_level",
            "text",
            "available_to_tickers",
        ],
        f"{_events_variant(config)} events",
        logger,
    )
    assignments, selection_audit = _analysis_assignments(
        config, events, logger
    )
    residuals = _load_optional(
        (project_path(config, "data", "processed", "residual_targets.parquet"),),
        [
            "ticker",
            "feature_date",
            "target_date",
            "split",
            "signed_residual",
            "residual_magnitude",
            "spike_q90",
            "spike_q95",
            "regime",
        ],
        "residual targets",
        logger,
    )
    if not events.empty:
        events = events.copy()
        events["event_id"] = events["event_id"].astype(str)
        events["date"] = pd.to_datetime(events["date"], errors="raise").dt.normalize()
        if events["event_id"].duplicated().any():
            raise ValueError(
                f"{_events_variant(config)} events must have unique event_id values"
            )
    if not assignments.empty:
        assignments = assignments.copy()
        assignments["event_id"] = assignments["event_id"].astype(str)
        assignments["date"] = pd.to_datetime(
            assignments["date"], errors="raise"
        ).dt.normalize()
        assignments["prototype_id"] = pd.to_numeric(
            assignments["hard_cluster_id"], errors="raise"
        ).astype(int)
        if assignments.duplicated(["event_id", "candidate_id"]).any():
            raise ValueError("prototype assignments contain duplicate event/candidate rows")
    if not residuals.empty:
        residuals = residuals.copy()
        residuals["ticker"] = residuals["ticker"].astype(str).str.upper()
        residuals["feature_date"] = pd.to_datetime(
            residuals["feature_date"], errors="raise"
        ).dt.normalize()
        residuals["target_date"] = pd.to_datetime(
            residuals["target_date"], errors="raise"
        ).dt.normalize()
        if residuals.duplicated(["ticker", "feature_date"]).any():
            raise ValueError("residual targets must be unique by ticker/feature_date")
    return events, assignments, residuals, selection_audit


def _event_response_table(
    assignments: pd.DataFrame,
    events: pd.DataFrame,
    residuals: pd.DataFrame,
    tickers: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if assignments.empty or events.empty:
        return (
            _empty_with_columns(
                [
                    "event_id",
                    "candidate_id",
                    "news_level",
                    "prototype_id",
                    "ticker",
                    "feature_date",
                ]
            ),
            _empty_with_columns(
                [
                    "event_id",
                    "mean_event_residual",
                    "mean_event_absolute_residual",
                    "event_spike_q90_rate",
                    "event_spike_q95_rate",
                ]
            ),
        )
    exposure = _canonical_event_tickers(events, tickers)
    assigned = assignments.merge(
        exposure[["event_id", "ticker", "feature_date"]],
        on="event_id",
        how="left",
        validate="one_to_many",
    )
    if residuals.empty:
        return assigned, _empty_with_columns(
            [
                "event_id",
                "mean_event_residual",
                "mean_event_absolute_residual",
                "event_spike_q90_rate",
                "event_spike_q95_rate",
            ]
        )
    response_columns = [
        "ticker",
        "feature_date",
        "target_date",
        "signed_residual",
        "residual_magnitude",
        "spike_q90",
        "spike_q95",
        "regime",
    ]
    assigned = assigned.merge(
        residuals[response_columns],
        on=["ticker", "feature_date"],
        how="left",
        validate="many_to_one",
    )
    event_response = (
        assigned.groupby("event_id", sort=True, observed=True)
        .agg(
            mean_event_residual=("signed_residual", "mean"),
            mean_event_absolute_residual=("residual_magnitude", "mean"),
            event_spike_q90_rate=("spike_q90", "mean"),
            event_spike_q95_rate=("spike_q95", "mean"),
        )
        .reset_index()
    )
    return assigned, event_response


def _prototype_details(
    assignments: pd.DataFrame,
    events: pd.DataFrame,
    event_ticker_response: pd.DataFrame,
    cross_aggregate: pd.DataFrame,
    min_events_per_cluster: int,
) -> pd.DataFrame:
    if assignments.empty:
        return _empty_with_columns(PROTOTYPE_DETAIL_COLUMNS)
    event_columns = [
        column
        for column in (
            "event_id",
            "target_ticker",
            "available_to_tickers",
            "source_tickers",
        )
        if column in events.columns
    ]
    enriched = assignments.merge(
        events[event_columns],
        on="event_id",
        how="left",
        validate="many_to_one",
    )
    enriched["year"] = enriched["date"].dt.year.astype(str)
    enriched["quarter"] = enriched["date"].dt.to_period("Q").astype(str)
    response = event_ticker_response.copy()
    level_totals = enriched.groupby(
        ["candidate_id", "news_level", "split"], observed=True
    )["event_id"].transform("nunique")
    enriched["__level_total"] = level_totals
    rows: list[dict[str, Any]] = []
    group_columns = [
        "candidate_id",
        "news_level",
        "prototype_id",
        "split",
    ]
    for keys, group in enriched.groupby(group_columns, sort=True, observed=True):
        candidate, level, prototype_id, split = keys
        event_ids = group["event_id"].astype(str).unique()
        response_group = response.loc[response["event_id"].isin(event_ids)]
        cluster_response = (
            event_ticker_response.loc[
                event_ticker_response["event_id"].isin(event_ids)
            ]
            if not event_ticker_response.empty
            else event_ticker_response
        )
        ticker_values = response_group["ticker"] if "ticker" in response_group else pd.Series(dtype=str)
        residual_values = (
            pd.to_numeric(response_group["signed_residual"], errors="coerce")
            if "signed_residual" in response_group
            else pd.Series(dtype=float)
        )
        abs_values = (
            pd.to_numeric(response_group["residual_magnitude"], errors="coerce")
            if "residual_magnitude" in response_group
            else pd.Series(dtype=float)
        )
        regimes = (
            response_group["regime"] if "regime" in response_group else pd.Series(dtype=float)
        )
        rows.append(
            {
                "record_type": "selected_prototype",
                "candidate_id": candidate,
                "news_level": level,
                "prototype_id": int(prototype_id),
                "split": split,
                "n_events": int(len(event_ids)),
                "event_fraction": float(len(event_ids) / max(group["__level_total"].iloc[0], 1)),
                "ticker_distribution": _json_distribution(ticker_values),
                "year_distribution": _json_distribution(group["year"]),
                "quarter_distribution": _json_distribution(group["quarter"]),
                "mean_residual": float(residual_values.mean()) if residual_values.notna().any() else np.nan,
                "mean_absolute_residual": float(abs_values.mean()) if abs_values.notna().any() else np.nan,
                "spike_q90_rate": (
                    float(pd.to_numeric(response_group["spike_q90"], errors="coerce").mean())
                    if "spike_q90" in response_group
                    else np.nan
                ),
                "spike_q95_rate": (
                    float(pd.to_numeric(response_group["spike_q95"], errors="coerce").mean())
                    if "spike_q95" in response_group
                    else np.nan
                ),
                "regime_distribution": _json_distribution(regimes),
                "within_prototype_response_variance": (
                    float(residual_values.var(ddof=1))
                    if residual_values.notna().sum() > 1
                    else np.nan
                ),
                "mean_assignment_entropy": float(group["assignment_entropy"].mean()),
                "mean_novelty": float(group["novelty"].mean()),
                "mean_nearest_distance": float(group["nearest_distance"].mean()),
                "ticker_concentration": _hhi(ticker_values),
                "time_concentration": _hhi(group["quarter"]),
                "mean_event_residual": (
                    float(cluster_response["mean_event_residual"].mean())
                    if "mean_event_residual" in cluster_response
                    else np.nan
                ),
            }
        )
    details = pd.DataFrame(rows)
    train = details.loc[details["split"] == "train"].copy()
    for (candidate, level), group in train.groupby(
        ["candidate_id", "news_level"], sort=True, observed=True
    ):
        assignment_group = assignments.loc[
            (assignments["candidate_id"] == candidate)
            & (assignments["news_level"] == level)
        ]
        configured_k = (
            int(pd.to_numeric(assignment_group["k"], errors="raise").iloc[0])
            if "k" in assignment_group.columns and not assignment_group.empty
            else int(group["prototype_id"].max()) + 1
        )
        proportions = group["n_events"].to_numpy(dtype=float)
        proportions /= max(proportions.sum(), 1.0)
        positive = proportions[proportions > 0]
        effective = float(np.exp(-np.sum(positive * np.log(positive))))
        missing_clusters = max(configured_k - len(group), 0)
        dead = float(missing_clusters / configured_k)
        small_count = int(
            np.sum(group["n_events"].to_numpy(dtype=float) < min_events_per_cluster)
        ) + missing_clusters
        small = float(small_count / configured_k)
        mask = (details["candidate_id"] == candidate) & (details["news_level"] == level)
        details.loc[mask, "effective_number_of_prototypes"] = effective
        details.loc[mask, "dead_prototype_fraction"] = dead
        details.loc[mask, "small_prototype_fraction"] = small
    if not cross_aggregate.empty:
        breadth = cross_aggregate.loc[
            cross_aggregate["record_type"] == "prototype_aggregate",
            [
                "candidate_id",
                "news_level",
                "prototype_id",
                "mean_residual_breadth",
            ],
        ].rename(columns={"mean_residual_breadth": "mean_cross_stock_breadth"})
        details = details.merge(
            breadth,
            on=["candidate_id", "news_level", "prototype_id"],
            how="left",
            validate="many_to_one",
            suffixes=("", "_cross"),
        )
        if "mean_cross_stock_breadth_cross" in details:
            details["mean_cross_stock_breadth"] = details[
                "mean_cross_stock_breadth_cross"
            ]
            details = details.drop(columns="mean_cross_stock_breadth_cross")
    for column in PROTOTYPE_DETAIL_COLUMNS:
        if column not in details.columns:
            details[column] = np.nan
    return details


def _concentration(values: np.ndarray) -> float:
    absolute = np.abs(values[np.isfinite(values)])
    total = absolute.sum()
    if total <= 0:
        return np.nan
    shares = absolute / total
    return float(np.square(shares).sum())


def _daily_cross_stock(
    assignments: pd.DataFrame,
    events: pd.DataFrame,
    residuals: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if assignments.empty or residuals.empty:
        empty = _empty_with_columns(CROSS_STOCK_COLUMNS)
        return empty, empty
    if "target_ticker" not in events.columns:
        events = events.assign(target_ticker=None)
    event_targets = events[
        [
            column
            for column in ("event_id", "target_ticker")
            if column in events.columns
        ]
    ]
    keys = assignments[
        ["event_id", "date", "candidate_id", "news_level", "prototype_id"]
    ].merge(event_targets, on="event_id", how="left", validate="many_to_one")
    day_keys = (
        keys.groupby(
            ["date", "candidate_id", "news_level", "prototype_id"],
            sort=True,
            observed=True,
        )
        .agg(
            n_events=("event_id", "nunique"),
            target_tickers=(
                "target_ticker",
                lambda values: sorted(
                    {str(value) for value in values.dropna() if str(value).strip()}
                ),
            ),
        )
        .reset_index()
    )
    response = residuals.copy()
    response["date"] = response["feature_date"]
    day_key_columns = [
        "date",
        "candidate_id",
        "news_level",
        "prototype_id",
    ]
    if day_keys.duplicated(day_key_columns).any():
        examples = day_keys.loc[
            day_keys.duplicated(day_key_columns, keep=False),
            day_key_columns,
        ].head(5)
        raise AssertionError(
            "Prototype day keys are not unique: "
            f"{examples.to_dict(orient='records')}"
        )
    response_key_columns = ["date", "ticker"]
    if response.duplicated(response_key_columns).any():
        examples = response.loc[
            response.duplicated(response_key_columns, keep=False),
            response_key_columns,
        ].head(5)
        raise AssertionError(
            "Cross-stock residuals are not unique by date/ticker: "
            f"{examples.to_dict(orient='records')}"
        )
    # A date can activate several prototype/level combinations, and every
    # combination intentionally expands to the ticker cross-section.
    merged = day_keys.merge(
        response,
        on="date",
        how="left",
        validate="many_to_many",
    )
    rows: list[dict[str, Any]] = []
    group_columns = day_key_columns
    for keys_value, group in merged.groupby(group_columns, sort=True, observed=True):
        date, candidate, level, prototype = keys_value
        residual = pd.to_numeric(group["signed_residual"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(residual)
        residual = residual[valid]
        if not len(residual):
            continue
        common = float(np.mean(residual))
        sign_share = max(float(np.mean(residual >= 0)), float(np.mean(residual < 0)))
        rows.append(
            {
                "record_type": "day_prototype",
                "date": date,
                "candidate_id": candidate,
                "news_level": level,
                "prototype_id": int(prototype),
                "n_events": int(group["n_events"].iloc[0]),
                "n_stocks": int(len(residual)),
                "residual_breadth": float(
                    pd.to_numeric(group.loc[valid, "spike_q90"], errors="coerce").mean()
                ),
                "same_sign_fraction": sign_share,
                "mean_residual": common,
                "mean_absolute_residual": float(np.mean(np.abs(residual))),
                "n_stocks_above_q90": int(
                    pd.to_numeric(group.loc[valid, "spike_q90"], errors="coerce").sum()
                ),
                "concentration_index": _concentration(residual),
                "common_component": common,
                "firm_specific_component": float(np.mean(np.abs(residual - common))),
            }
        )
    daily = pd.DataFrame(rows)
    if daily.empty:
        return _empty_with_columns(CROSS_STOCK_COLUMNS), daily
    aggregate_rows: list[dict[str, Any]] = []
    for keys_value, days in daily.groupby(
        ["candidate_id", "news_level", "prototype_id"],
        sort=True,
        observed=True,
    ):
        candidate, level, prototype = keys_value
        selected_dates = days["date"].drop_duplicates()
        response_group = response.loc[response["date"].isin(selected_dates)]
        pivot = response_group.pivot(index="date", columns="ticker", values="signed_residual")
        correlation = pivot.corr(min_periods=3).to_numpy(dtype=float)
        if correlation.shape[0] > 1:
            off_diagonal = correlation[~np.eye(correlation.shape[0], dtype=bool)]
            cross_correlation = (
                float(np.nanmean(off_diagonal))
                if np.isfinite(off_diagonal).any()
                else np.nan
            )
        else:
            cross_correlation = np.nan
        aggregate_rows.append(
            {
                "record_type": "prototype_aggregate",
                "candidate_id": candidate,
                "news_level": level,
                "prototype_id": int(prototype),
                "n_events": int(days["n_events"].sum()),
                "n_days": int(days["date"].nunique()),
                "n_stocks": int(days["n_stocks"].max()),
                "mean_residual_breadth": float(days["residual_breadth"].mean()),
                "residual_breadth": float(days["residual_breadth"].mean()),
                "same_sign_fraction": float(days["same_sign_fraction"].mean()),
                "mean_residual": float(days["mean_residual"].mean()),
                "mean_absolute_residual": float(days["mean_absolute_residual"].mean()),
                "cross_stock_residual_correlation": cross_correlation,
                "n_stocks_above_q90": float(days["n_stocks_above_q90"].mean()),
                "concentration_index": float(days["concentration_index"].mean()),
                "common_component": float(days["common_component"].abs().mean()),
                "firm_specific_component": float(days["firm_specific_component"].mean()),
            }
        )
    aggregate = pd.DataFrame(aggregate_rows)
    return pd.concat([daily, aggregate], ignore_index=True, sort=False), aggregate


def _target_vs_rest(
    assignments: pd.DataFrame,
    events: pd.DataFrame,
    residuals: pd.DataFrame,
) -> pd.DataFrame:
    if assignments.empty or events.empty or residuals.empty or "target_ticker" not in events.columns:
        return _empty_with_columns(CROSS_STOCK_COLUMNS)
    target_events = assignments.loc[
        assignments["news_level"] == "target",
        ["event_id", "date", "candidate_id", "news_level", "prototype_id"],
    ].merge(
        events[["event_id", "target_ticker"]],
        on="event_id",
        how="left",
        validate="many_to_one",
    )
    target_events = target_events.dropna(subset=["target_ticker"])
    if target_events.empty:
        return _empty_with_columns(CROSS_STOCK_COLUMNS)
    response = residuals.copy()
    response["date"] = response["feature_date"]
    response = response[
        ["date", "ticker", "signed_residual", "residual_magnitude", "spike_q90"]
    ]
    joined = target_events.merge(response, on="date", how="left", validate="many_to_many")
    joined["is_target"] = joined["ticker"] == joined["target_ticker"]
    event_rows: list[dict[str, Any]] = []
    group_columns = [
        "event_id",
        "date",
        "candidate_id",
        "news_level",
        "prototype_id",
        "target_ticker",
    ]
    for keys_value, group in joined.groupby(group_columns, sort=True, observed=True):
        event_id, date, candidate, level, prototype, target_ticker = keys_value
        target_values = pd.to_numeric(
            group.loc[group["is_target"], "signed_residual"], errors="coerce"
        ).dropna()
        rest_values = pd.to_numeric(
            group.loc[~group["is_target"], "signed_residual"], errors="coerce"
        ).dropna()
        if target_values.empty or rest_values.empty:
            continue
        event_rows.append(
            {
                "event_id": event_id,
                "date": date,
                "candidate_id": candidate,
                "news_level": level,
                "prototype_id": int(prototype),
                "target_ticker": str(target_ticker),
                "target_residual": float(target_values.mean()),
                "non_target_mean_residual": float(rest_values.mean()),
                "spillover_mean_absolute_residual": float(rest_values.abs().mean()),
            }
        )
    event_frame = pd.DataFrame(event_rows)
    if event_frame.empty:
        return _empty_with_columns(CROSS_STOCK_COLUMNS)
    rows: list[dict[str, Any]] = []
    for keys_value, group in event_frame.groupby(
        ["candidate_id", "news_level", "prototype_id", "target_ticker"],
        sort=True,
        observed=True,
    ):
        candidate, level, prototype, ticker = keys_value
        difference = group["target_residual"] - group["non_target_mean_residual"]
        scale = float(difference.std(ddof=1))
        effect = float(difference.mean() / scale) if np.isfinite(scale) and scale > 0 else np.nan
        rows.append(
            {
                "record_type": "target_vs_rest",
                "candidate_id": candidate,
                "news_level": level,
                "prototype_id": int(prototype),
                "target_ticker": ticker,
                "n_events": int(len(group)),
                "target_residual": float(group["target_residual"].mean()),
                "non_target_mean_residual": float(group["non_target_mean_residual"].mean()),
                "target_vs_nontarget_effect_size": effect,
                "spillover_mean_absolute_residual": float(
                    group["spillover_mean_absolute_residual"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _target_ticker_coverage(
    target_effect: pd.DataFrame, tickers: Sequence[str]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        group = (
            target_effect.loc[target_effect["target_ticker"] == ticker]
            if not target_effect.empty
            else target_effect
        )
        rows.append(
            {
                "record_type": "target_ticker_coverage",
                "news_level": "target",
                "target_ticker": ticker,
                "n_events": (
                    int(pd.to_numeric(group["n_events"], errors="coerce").sum())
                    if not group.empty
                    else 0
                ),
                "target_residual": (
                    float(pd.to_numeric(group["target_residual"], errors="coerce").mean())
                    if not group.empty
                    else np.nan
                ),
                "non_target_mean_residual": (
                    float(
                        pd.to_numeric(
                            group["non_target_mean_residual"], errors="coerce"
                        ).mean()
                    )
                    if not group.empty
                    else np.nan
                ),
                "target_vs_nontarget_effect_size": (
                    float(
                        pd.to_numeric(
                            group["target_vs_nontarget_effect_size"],
                            errors="coerce",
                        ).mean()
                    )
                    if not group.empty
                    else np.nan
                ),
                "spillover_mean_absolute_residual": (
                    float(
                        pd.to_numeric(
                            group["spillover_mean_absolute_residual"],
                            errors="coerce",
                        ).mean()
                    )
                    if not group.empty
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _prototype_examples(
    assignments: pd.DataFrame,
    events: pd.DataFrame,
    event_response: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    if assignments.empty or events.empty:
        return _empty_with_columns(EXAMPLE_COLUMNS)
    merged = assignments.merge(
        events[["event_id", "text"]],
        on="event_id",
        how="left",
        validate="many_to_one",
    ).merge(event_response, on="event_id", how="left", validate="many_to_one")
    merged = merged.loc[merged["split"] == "train"].copy()
    merged = merged.sort_values(
        ["candidate_id", "news_level", "prototype_id", "nearest_similarity", "event_id"],
        ascending=[True, True, True, False, True],
        kind="mergesort",
    )
    selected = merged.groupby(
        ["candidate_id", "news_level", "prototype_id"],
        sort=True,
        observed=True,
    ).head(top_n)
    selected = selected.copy()
    selected["rank"] = selected.groupby(
        ["candidate_id", "news_level", "prototype_id"], observed=True
    ).cumcount() + 1
    selected["record_type"] = "selected_prototype_example"
    selected["cosine_similarity"] = selected["nearest_similarity"]
    for column in EXAMPLE_COLUMNS:
        if column not in selected.columns:
            selected[column] = np.nan
    return selected[list(EXAMPLE_COLUMNS)]


def _usage_stability(assignments: pd.DataFrame) -> pd.DataFrame:
    if assignments.empty:
        return _empty_with_columns(STABILITY_COLUMNS)
    rows: list[dict[str, Any]] = []
    for (candidate, level), group in assignments.groupby(
        ["candidate_id", "news_level"], sort=True, observed=True
    ):
        k = int(group["prototype_id"].max()) + 1
        train = group.loc[group["split"] == "train"]
        if train.empty:
            continue
        train_counts = np.bincount(
            train["prototype_id"].to_numpy(dtype=int), minlength=k
        ).astype(float)
        train_probability = (train_counts + 1.0e-12) / (
            train_counts.sum() + k * 1.0e-12
        )
        for split in ("validation", "test"):
            comparison = group.loc[group["split"] == split]
            if comparison.empty:
                continue
            counts = np.bincount(
                comparison["prototype_id"].to_numpy(dtype=int), minlength=k
            ).astype(float)
            probability = (counts + 1.0e-12) / (counts.sum() + k * 1.0e-12)
            rows.append(
                {
                    "record_type": "split_usage_diagnostic",
                    "candidate_id": candidate,
                    "news_level": level,
                    "reference_split": "train",
                    "comparison_split": split,
                    "usage_js_distance": float(
                        jensenshannon(train_probability, probability, base=2.0)
                    ),
                    "active_prototype_fraction": float(np.mean(counts > 0)),
                    "mean_assignment_entropy": float(
                        comparison["assignment_entropy"].mean()
                    ),
                    "n_events": int(len(comparison)),
                    "evidence_scope": (
                        "assignment-distribution diagnostic only; not a forecasting refit fold"
                    ),
                }
            )
    table = pd.DataFrame(rows)
    for column in STABILITY_COLUMNS:
        if column not in table.columns:
            table[column] = np.nan
    return table


def _save_placeholder(path: Path, title: str, message: str) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.axis("off")
    axis.set_title(title)
    axis.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _bar_plot(
    frame: pd.DataFrame,
    x: str,
    y: str,
    path: Path,
    title: str,
    ylabel: str,
) -> None:
    usable = frame.dropna(subset=[x, y]).copy() if {x, y}.issubset(frame.columns) else frame.iloc[0:0]
    if usable.empty:
        _save_placeholder(path, title, "Required prototype observations are unavailable.")
        return
    labels = (
        usable["news_level"].astype(str)
        + ":"
        + usable[x].astype(int).astype(str)
    )
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.bar(np.arange(len(usable)), usable[y].to_numpy(dtype=float))
    axis.set_xticks(np.arange(len(usable)))
    axis.set_xticklabels(labels, rotation=90)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _plot_outputs(
    figures: Path,
    assignments: pd.DataFrame,
    details: pd.DataFrame,
    cross: pd.DataFrame,
    residuals: pd.DataFrame,
    events: pd.DataFrame,
) -> dict[str, Path]:
    paths = {
        "prototype_cluster_size": figures / "prototype_cluster_size.png",
        "assignment_entropy": figures / "assignment_entropy.png",
        "prototype_usage_over_time": figures / "prototype_usage_over_time.png",
        "prototype_ticker_concentration": figures / "prototype_ticker_concentration.png",
        "spike_rate_by_prototype": figures / "spike_rate_by_prototype.png",
        "response_breadth_by_prototype": figures / "response_breadth_by_prototype.png",
        "residual_distribution": figures / "residual_distribution.png",
        "residual_correlation_matrix": figures / "residual_correlation_matrix.png",
        "news_count_by_day": figures / "news_count_by_day.png",
    }
    selected = details.loc[
        (details.get("record_type", "") == "selected_prototype")
        & (details.get("split", "") == "train")
    ].copy()
    _bar_plot(
        selected,
        "prototype_id",
        "n_events",
        paths["prototype_cluster_size"],
        "Selected prototype cluster sizes (train)",
        "Events",
    )
    if assignments.empty:
        _save_placeholder(
            paths["assignment_entropy"],
            "Assignment entropy",
            "Prototype assignments are unavailable.",
        )
        _save_placeholder(
            paths["prototype_usage_over_time"],
            "Prototype usage over time",
            "Prototype assignments are unavailable.",
        )
    else:
        figure, axis = plt.subplots(figsize=(9, 5))
        for level, group in assignments.groupby("news_level", sort=True):
            axis.hist(
                group["assignment_entropy"].dropna(),
                bins=30,
                alpha=0.45,
                label=str(level),
                density=True,
            )
        axis.set(xlabel="Soft-assignment entropy", ylabel="Density", title="Assignment entropy")
        axis.legend()
        figure.tight_layout()
        figure.savefig(paths["assignment_entropy"], dpi=160, bbox_inches="tight")
        plt.close(figure)

        usage = (
            assignments.assign(month=assignments["date"].dt.to_period("M").dt.to_timestamp())
            .groupby(["month", "news_level", "prototype_id"], observed=True)
            .size()
            .rename("events")
            .reset_index()
        )
        figure, axes = plt.subplots(
            len(NEWS_LEVELS), 1, figsize=(12, 10), sharex=True, squeeze=False
        )
        for axis, level in zip(axes[:, 0], NEWS_LEVELS):
            level_data = usage.loc[usage["news_level"] == level]
            for prototype, group in level_data.groupby("prototype_id", sort=True):
                axis.plot(group["month"], group["events"], linewidth=0.9, label=str(prototype))
            axis.set_ylabel(level)
        axes[0, 0].set_title("Prototype usage over time")
        axes[-1, 0].set_xlabel("Month")
        figure.tight_layout()
        figure.savefig(paths["prototype_usage_over_time"], dpi=160, bbox_inches="tight")
        plt.close(figure)

    _bar_plot(
        selected,
        "prototype_id",
        "ticker_concentration",
        paths["prototype_ticker_concentration"],
        "Prototype–ticker concentration (train)",
        "Ticker HHI",
    )
    _bar_plot(
        selected,
        "prototype_id",
        "spike_q90_rate",
        paths["spike_rate_by_prototype"],
        "Spike q90 rate by prototype (train)",
        "Spike rate",
    )
    cross_aggregate = cross.loc[
        cross.get("record_type", "") == "prototype_aggregate"
    ].copy()
    _bar_plot(
        cross_aggregate,
        "prototype_id",
        "residual_breadth",
        paths["response_breadth_by_prototype"],
        "Cross-stock response breadth by prototype",
        "Fraction above ticker q90",
    )

    if residuals.empty:
        _save_placeholder(
            paths["residual_distribution"], "Residual distribution", "Residual targets unavailable."
        )
        _save_placeholder(
            paths["residual_correlation_matrix"],
            "Residual correlation matrix",
            "Residual targets unavailable.",
        )
    else:
        figure, axis = plt.subplots(figsize=(9, 5))
        for ticker, group in residuals.groupby("ticker", sort=True):
            axis.hist(
                group["signed_residual"].dropna(),
                bins=50,
                density=True,
                histtype="step",
                linewidth=1,
                label=str(ticker),
            )
        axis.set(xlabel="Signed residual", ylabel="Density", title="Residual distribution")
        axis.legend(ncol=3, fontsize=8)
        figure.tight_layout()
        figure.savefig(paths["residual_distribution"], dpi=160, bbox_inches="tight")
        plt.close(figure)

        pivot = residuals.pivot(
            index="target_date", columns="ticker", values="signed_residual"
        )
        correlation = pivot.corr()
        figure, axis = plt.subplots(figsize=(8, 7))
        image = axis.imshow(correlation.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
        axis.set_xticks(np.arange(len(correlation.columns)))
        axis.set_yticks(np.arange(len(correlation.index)))
        axis.set_xticklabels(correlation.columns, rotation=90)
        axis.set_yticklabels(correlation.index)
        axis.set_title("Residual correlation matrix")
        figure.colorbar(image, ax=axis, shrink=0.8)
        figure.tight_layout()
        figure.savefig(paths["residual_correlation_matrix"], dpi=160, bbox_inches="tight")
        plt.close(figure)

    if events.empty:
        _save_placeholder(paths["news_count_by_day"], "News count by day", "Canonical events unavailable.")
    else:
        counts = (
            events.groupby(["date", "news_level"], observed=True)
            .size()
            .unstack(fill_value=0)
            .sort_index()
        )
        figure, axis = plt.subplots(figsize=(12, 5))
        counts.plot(ax=axis, linewidth=0.9)
        axis.set(xlabel="Date", ylabel="Canonical events", title="News count by day")
        figure.tight_layout()
        figure.savefig(paths["news_count_by_day"], dpi=160, bbox_inches="tight")
        plt.close(figure)
    return paths


def _update_cross_stock_decision(
    config: Mapping[str, Any], cross: pd.DataFrame, logger: Any
) -> Path | None:
    path = project_path(config, "outputs", "tables", "final_decision.csv")
    if not path.exists():
        logger.warning("final_decision.csv is unavailable; cross-stock answer was not attached")
        return None
    decision = safe_read_table(path)
    if "record_type" not in decision.columns:
        raise ValueError("final_decision.csv is missing record_type")
    aggregate = cross.loc[
        (cross.get("record_type", "") == "prototype_aggregate")
        & cross.get("news_level", pd.Series(index=cross.index, dtype=str)).isin(["macro", "sector"])
    ]
    if aggregate.empty:
        answer = "not_available"
    else:
        breadth = float(pd.to_numeric(aggregate["residual_breadth"], errors="coerce").mean())
        common = float(pd.to_numeric(aggregate["common_component"], errors="coerce").mean())
        firm = float(pd.to_numeric(aggregate["firm_specific_component"], errors="coerce").mean())
        descriptive = bool(breadth > 0.10 or (np.isfinite(common) and np.isfinite(firm) and common > firm))
        answer = (
            f"descriptive_only={descriptive}; mean_q90_breadth={breadth:.4f}; "
            "forecasting evidence must come from locked out-of-sample target tests"
        )
    final_mask = decision["record_type"].astype(str) == "final"
    decision.loc[final_mask, "cross_stock_common_effect"] = answer
    decision.loc[final_mask, "semiconductor_common_response"] = answer
    atomic_write_csv(decision, path, index=False)
    return path


def run(config: dict[str, Any]) -> dict[str, Path]:
    ensure_directories(config)
    logger = get_logger(
        __name__, config, project_path(config, "outputs", "logs", "prototype_analysis.log")
    )
    events, assignments, residuals, selection_audit = _prepare_inputs(
        config, logger
    )
    tickers = [str(value).upper() for value in config["data"]["tickers"]]
    event_ticker_response, event_response = _event_response_table(
        assignments, events, residuals, tickers
    )
    cross, cross_aggregate = _daily_cross_stock(assignments, events, residuals)
    target_effect = _target_vs_rest(assignments, events, residuals)
    ticker_coverage = _target_ticker_coverage(target_effect, tickers)
    cross = pd.concat(
        [cross, target_effect, ticker_coverage], ignore_index=True, sort=False
    )
    for column in CROSS_STOCK_COLUMNS:
        if column not in cross.columns:
            cross[column] = np.nan

    details = _prototype_details(
        assignments,
        events,
        event_ticker_response,
        cross_aggregate,
        int(config["prototype"]["min_events_per_cluster"]),
    )
    examples = _prototype_examples(assignments, events, event_response)
    usage_stability = _usage_stability(assignments)

    tables = project_path(config, "outputs", "tables")
    suffix = _variant_suffix(_events_variant(config))
    source_summary_path = tables / f"prototype_summary{suffix}.csv"
    source_examples_path = tables / f"prototype_examples{suffix}.csv"
    source_stability_path = tables / f"prototype_stability{suffix}.csv"
    existing_summary = (
        safe_read_table(source_summary_path)
        if source_summary_path.exists()
        else pd.DataFrame()
    )
    existing_examples = (
        safe_read_table(source_examples_path)
        if source_examples_path.exists()
        else pd.DataFrame()
    )
    existing_stability = (
        safe_read_table(source_stability_path)
        if source_stability_path.exists()
        else pd.DataFrame()
    )
    summary = _preserve_and_append(
        existing_summary,
        details,
        "candidate_configuration",
        ["record_type", "candidate_id", "news_level", "prototype_id", "split"],
    )
    example_table = _preserve_and_append(
        existing_examples,
        examples,
        "centroid_example",
        ["record_type", "candidate_id", "base_candidate_id", "event_id", "prototype_id"],
    )
    stability = _preserve_and_append(
        existing_stability,
        usage_stability,
        "seed_refit",
        [
            "record_type",
            "candidate_id",
            "news_level",
            "prototype_id",
            "reference_seed",
            "comparison_seed",
            "comparison_split",
        ],
    )
    if "record_type" in summary.columns:
        summary = summary.loc[
            ~summary["record_type"].astype(str).eq("analysis_selection")
        ].copy()
    selection_row = pd.DataFrame(
        [{"record_type": "analysis_selection", **selection_audit}]
    )
    summary = pd.concat(
        [summary, selection_row], ignore_index=True, sort=False
    )
    for table, columns in (
        (summary, PROTOTYPE_DETAIL_COLUMNS),
        (example_table, EXAMPLE_COLUMNS),
        (stability, STABILITY_COLUMNS),
    ):
        for column in columns:
            if column not in table.columns:
                table[column] = np.nan
    summary_path = atomic_write_csv(summary, tables / "prototype_summary.csv", index=False)
    examples_path = atomic_write_csv(
        example_table, tables / "prototype_examples.csv", index=False
    )
    stability_path = atomic_write_csv(
        stability, tables / "prototype_stability.csv", index=False
    )
    selection_path = atomic_write_csv(
        selection_row,
        tables / f"prototype_analysis_selection{suffix}.csv",
        index=False,
    )
    if suffix:
        atomic_write_csv(summary, source_summary_path, index=False)
        atomic_write_csv(example_table, source_examples_path, index=False)
        atomic_write_csv(stability, source_stability_path, index=False)
    cross_path = atomic_write_csv(cross, tables / "cross_stock_results.csv", index=False)
    figure_paths = _plot_outputs(
        project_path(config, "outputs", "figures"),
        assignments,
        details,
        cross,
        residuals,
        events,
    )
    decision_path = _update_cross_stock_decision(config, cross, logger)
    logger.info(
        "Analysed %d %s assignments, %d cross-stock rows and %d target-vs-rest groups",
        len(assignments),
        selection_audit["analysis_assignment_source"],
        len(cross),
        len(target_effect),
    )
    outputs: dict[str, Path] = {
        "prototype_summary": summary_path,
        "prototype_examples": examples_path,
        "prototype_stability": stability_path,
        "prototype_analysis_selection": selection_path,
        "cross_stock_results": cross_path,
        **figure_paths,
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
