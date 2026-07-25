"""Fast leakage-safe feasibility test for LLM-extracted structured events."""

from __future__ import annotations

import json
import math
import re
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm.auto import tqdm

from src.structured_event_extractor import extract_events
from src.structured_event_schema import ontology_from_config
from src.utils import (
    atomic_write_json,
    binary_metrics,
    load_shared_market,
    price_feature_columns,
    project_path,
    read_table,
    resolve_shared_file,
    stable_id,
    validate_columns,
    write_table,
)


PILOT_SECTION = "structured_event_pilot"
TRUE_REPRESENTATIONS = (
    "F0_PRICE",
    "F1_META_BASIC",
    "F1_EVENT_META",
    "F2_EVENT_TYPE",
    "F3_SLOTS",
    "F4_INTERACTIONS",
)
PLACEBO_KINDS = ("shuffled_event_type", "shuffled_date", "random_structured_vector")
BASIC_META_COLUMNS = (
    "meta__news_count",
    "meta__has_news",
    "meta__no_news",
    "meta__days_since_news",
    "meta__candidate_news_count",
    "meta__has_candidate_news",
)
META_COLUMNS = (
    *BASIC_META_COLUMNS,
    "meta__structured_event_count",
    "meta__has_structured_event",
)


def _profile(config: Mapping[str, Any]) -> dict[str, Any]:
    raw = config.get(PILOT_SECTION)
    if not isinstance(raw, Mapping):
        raise KeyError(f"Missing {PILOT_SECTION!r} configuration section.")
    profile = dict(raw)
    required = {
        "output_directory",
        "news_levels",
        "initial_train_dates",
        "validation_block_dates",
        "fold_count",
        "lookbacks",
        "decay_rate",
        "event_types",
        "directions",
        "magnitudes",
        "certainties",
        "time_horizons",
        "entity_roles",
        "extractor",
        "logistic_c_values",
        "placebo_seeds",
    }
    missing = sorted(required.difference(profile))
    if missing:
        raise KeyError(f"{PILOT_SECTION} is missing configuration keys: {missing}")
    if int(profile["fold_count"]) < 2:
        raise ValueError("The pilot requires at least two chronological folds.")
    if int(profile["validation_block_dates"]) < 40:
        raise ValueError("Each validation block must contain at least 40 trading dates.")
    if int(profile["initial_train_dates"]) < 126:
        raise ValueError("Pilot initial training history is too short.")
    if tuple(profile["news_levels"]) != ("target",):
        raise ValueError("The fast pilot is intentionally locked to target news only.")
    profile["lookbacks"] = tuple(
        dict.fromkeys(int(value) for value in profile["lookbacks"])
    )
    if not profile["lookbacks"] or min(profile["lookbacks"]) < 1:
        raise ValueError("lookbacks must contain positive integers.")
    profile["placebo_seeds"] = tuple(
        dict.fromkeys(int(value) for value in profile["placebo_seeds"])
    )
    return profile


def output_root(config: Mapping[str, Any], profile: Mapping[str, Any]) -> Path:
    path = project_path(config, str(profile["output_directory"]))
    for child in (
        "cache",
        "checkpoints",
        "data",
        "figures",
        "logs",
        "predictions",
        "tables",
    ):
        (path / child).mkdir(parents=True, exist_ok=True)
    return path


def _paths(root: Path) -> dict[str, Path]:
    return {
        "plan": root / "data" / "pilot_plan.json",
        "folds": root / "tables" / "pilot_folds.csv",
        "event_manifest": root / "data" / "pilot_event_manifest.parquet",
        "market": root / "data" / "pilot_market.parquet",
        "coverage": root / "tables" / "pilot_data_coverage.csv",
        "structured_events": root / "data" / "structured_events.parquet",
        "feature_panel": root / "data" / "structured_event_feature_panel.parquet",
        "audit_sample": root / "tables" / "structured_event_audit_sample.csv",
        "results": root / "tables" / "structured_event_pilot_results.csv",
        "ticker_results": root / "tables" / "structured_event_pilot_ticker_results.csv",
        "comparisons": root / "tables" / "structured_event_pilot_comparisons.csv",
        "placebos": root / "tables" / "structured_event_pilot_placebos.csv",
        "decision": root / "tables" / "structured_event_pilot_decision.csv",
        "report": root / "tables" / "structured_event_pilot_report.json",
        "progress": root / "logs" / "progress_state.json",
    }


def _json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    parsed = json.loads(str(value))
    if not isinstance(parsed, list):
        raise TypeError(f"Expected JSON list, received {type(parsed).__name__}.")
    return [str(item) for item in parsed]


def _map_news_to_feature_dates(
    events: pd.DataFrame,
    market: pd.DataFrame,
) -> pd.DataFrame:
    """Map weekend/non-trading news to the next observable feature date."""

    mapped_parts: list[pd.DataFrame] = []
    for ticker, group in events.groupby("target_ticker", sort=True, observed=True):
        trading_dates = np.sort(
            market.loc[market["ticker"].eq(str(ticker)), "feature_date"]
            .dropna()
            .unique()
        )
        if not len(trading_dates):
            continue
        values = group["date"].to_numpy(dtype="datetime64[ns]")
        positions = np.searchsorted(trading_dates, values, side="left")
        valid = positions < len(trading_dates)
        selected = group.loc[valid].copy()
        selected["feature_date"] = pd.to_datetime(trading_dates[positions[valid]])
        mapped_parts.append(selected)
    if not mapped_parts:
        return events.iloc[0:0].assign(feature_date=pd.NaT)
    mapped = pd.concat(mapped_parts, ignore_index=True)
    if (mapped["feature_date"] < mapped["date"]).any():
        raise AssertionError("News was mapped to a feature date before availability.")
    return mapped


def _candidate_mask(
    events: pd.DataFrame,
    profile: Mapping[str, Any],
) -> pd.Series:
    candidate_config = profile.get("candidate_filter", {})
    if not isinstance(candidate_config, Mapping):
        raise TypeError("candidate_filter must be a mapping.")
    if not bool(candidate_config.get("enabled", True)):
        return pd.Series(True, index=events.index, dtype=bool)
    patterns = tuple(
        str(value).strip()
        for value in candidate_config.get("patterns", ())
        if str(value).strip()
    )
    if not patterns:
        raise ValueError("Enabled candidate_filter has no patterns.")
    expression = "(?:" + "|".join(patterns) + ")"
    # Compile first so malformed configuration fails before any LLM call.
    re.compile(expression, flags=re.IGNORECASE)
    return events["text"].fillna("").astype(str).str.contains(
        expression,
        case=False,
        regex=True,
        na=False,
    )


def run_plan(
    config: Mapping[str, Any],
    *,
    logger: Any,
    force: bool = False,
) -> dict[str, Path]:
    """Lock a compact development-only window and two expanding mini-folds."""

    profile = _profile(config)
    root = output_root(config, profile)
    paths = _paths(root)
    if (
        not force
        and paths["plan"].is_file()
        and paths["event_manifest"].is_file()
        and paths["market"].is_file()
    ):
        logger.info("Using existing locked pilot plan at %s", paths["plan"])
        return paths
    market, market_source = load_shared_market(config)
    tickers = tuple(map(str, config["universe"]["tickers"]))
    market = market.loc[
        market["ticker"].isin(tickers)
        & market["model_ready"].fillna(False).astype(bool)
    ].copy()
    development = market.loc[market["split"].astype(str) != "test"].copy()
    test = market.loc[market["split"].astype(str) == "test"].copy()
    if test.empty:
        raise ValueError("Locked test rows are required to verify the development boundary.")
    initial = int(profile["initial_train_dates"])
    block = int(profile["validation_block_dates"])
    fold_count = int(profile["fold_count"])
    required_dates = initial + fold_count * block
    dates = np.sort(development["feature_date"].dropna().unique())
    if len(dates) < required_dates:
        raise ValueError(
            f"Pilot needs {required_dates} development dates; only {len(dates)} exist."
        )
    selected_dates = dates[-required_dates:]
    selected_market = development.loc[
        development["feature_date"].isin(selected_dates)
    ].copy()
    if not pd.Timestamp(selected_dates[-1]) < test["feature_date"].min():
        raise AssertionError("Pilot development window overlaps the locked test.")
    missing_tickers = sorted(set(tickers).difference(selected_market["ticker"].unique()))
    if missing_tickers:
        raise AssertionError(f"Pilot market window is missing tickers: {missing_tickers}")
    fold_rows: list[dict[str, Any]] = []
    for fold_index in range(fold_count):
        train_end_index = initial + fold_index * block
        validation_end_index = train_end_index + block
        train_dates = selected_dates[:train_end_index]
        validation_dates = selected_dates[train_end_index:validation_end_index]
        if not pd.Timestamp(train_dates[-1]) < pd.Timestamp(validation_dates[0]):
            raise AssertionError(f"Pilot fold {fold_index + 1} violates chronology.")
        fold_rows.append(
            {
                "fold": fold_index + 1,
                "train_start": pd.Timestamp(train_dates[0]),
                "train_end": pd.Timestamp(train_dates[-1]),
                "validation_start": pd.Timestamp(validation_dates[0]),
                "validation_end": pd.Timestamp(validation_dates[-1]),
                "train_dates": len(train_dates),
                "validation_dates": len(validation_dates),
                "train_rows": int(
                    selected_market["feature_date"].isin(train_dates).sum()
                ),
                "validation_rows": int(
                    selected_market["feature_date"].isin(validation_dates).sum()
                ),
            }
        )
    canonical_path = resolve_shared_file(
        config,
        str(config["shared"]["canonical_events"]),
        kinds=("processed", "tables"),
    )
    assert canonical_path is not None
    events = read_table(canonical_path)
    validate_columns(
        events,
        (
            "event_id",
            "date",
            "news_level",
            "text",
            "text_hash",
            "target_ticker",
        ),
        "canonical events",
    )
    events["date"] = pd.to_datetime(events["date"], errors="raise").dt.normalize()
    events["target_ticker"] = events["target_ticker"].astype("string")
    warmup_days = int(max(profile["lookbacks"])) + 7
    event_start = pd.Timestamp(selected_dates[0]) - pd.Timedelta(days=warmup_days)
    event_end = pd.Timestamp(selected_dates[-1])
    selected_events = events.loc[
        events["news_level"].astype(str).isin(profile["news_levels"])
        & events["target_ticker"].isin(tickers)
        & events["date"].between(event_start, event_end)
    ].copy()
    selected_events = _map_news_to_feature_dates(selected_events, selected_market)
    selected_events = selected_events.loc[
        selected_events["feature_date"].isin(selected_dates)
    ].copy()
    selected_events["extraction_candidate"] = _candidate_mask(
        selected_events, profile
    ).astype(bool)
    if selected_events.duplicated("event_id").any():
        raise AssertionError("Pilot canonical event manifest contains duplicate event_id.")
    write_table(selected_market, paths["market"])
    write_table(selected_events, paths["event_manifest"])
    folds = pd.DataFrame(fold_rows)
    write_table(folds, paths["folds"])
    coverage_rows = []
    for ticker in tickers:
        subset = selected_events.loc[selected_events["target_ticker"].eq(ticker)]
        coverage_rows.append(
            {
                "ticker": ticker,
                "market_dates": int(
                    selected_market.loc[
                        selected_market["ticker"].eq(ticker), "feature_date"
                    ].nunique()
                ),
                "canonical_target_news": int(len(subset)),
                "extraction_candidate_news": int(
                    subset["extraction_candidate"].sum()
                ),
                "news_feature_dates": int(subset["feature_date"].nunique()),
                "news_day_rate": float(
                    subset["feature_date"].nunique() / max(len(selected_dates), 1)
                ),
            }
        )
    write_table(pd.DataFrame(coverage_rows), paths["coverage"])
    rate = float(profile.get("planning_events_per_minute", 8.0))
    candidate_count = int(selected_events["extraction_candidate"].sum())
    plan = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "market_source": str(market_source),
        "canonical_event_source": str(canonical_path),
        "development_only": True,
        "locked_test_read_for_forecasting": False,
        "pilot_feature_start": pd.Timestamp(selected_dates[0]).date().isoformat(),
        "pilot_feature_end": pd.Timestamp(selected_dates[-1]).date().isoformat(),
        "locked_test_start": test["feature_date"].min().date().isoformat(),
        "trading_dates": int(len(selected_dates)),
        "market_rows": int(len(selected_market)),
        "canonical_target_news": int(len(selected_events)),
        "extraction_candidate_news": candidate_count,
        "candidate_rate": candidate_count / max(len(selected_events), 1),
        "estimated_extraction_minutes": float(candidate_count / max(rate, 1.0e-9)),
        "folds": fold_rows,
        "tickers": list(tickers),
        "news_levels": list(profile["news_levels"]),
    }
    atomic_write_json(plan, paths["plan"])
    logger.info(
        "Pilot plan locked | dates=%s..%s | market_rows=%d | target_news=%d | "
        "LLM_candidates=%d (%.1f%%) | rough extraction ETA=%.1f min at "
        "%.1f news/min | locked test starts=%s",
        plan["pilot_feature_start"],
        plan["pilot_feature_end"],
        plan["market_rows"],
        plan["canonical_target_news"],
        plan["extraction_candidate_news"],
        100.0 * plan["candidate_rate"],
        plan["estimated_extraction_minutes"],
        rate,
        plan["locked_test_start"],
    )
    return paths


def run_extraction(
    config: Mapping[str, Any],
    *,
    logger: Any,
    resume: bool,
    force: bool,
    model_override: str | None = None,
    batch_size_override: int | None = None,
    disable_4bit: bool = False,
) -> dict[str, Path]:
    profile = _profile(config)
    root = output_root(config, profile)
    paths = run_plan(config, logger=logger, force=False)
    manifest = read_table(paths["event_manifest"])
    validate_columns(
        manifest,
        ("extraction_candidate",),
        "pilot event manifest",
    )
    manifest = manifest.loc[
        manifest["extraction_candidate"].fillna(False).astype(bool)
    ].copy()
    ontology = ontology_from_config(profile)
    return extract_events(
        manifest,
        ontology=ontology,
        tickers=tuple(map(str, config["universe"]["tickers"])),
        extractor_config=profile["extractor"],
        output_root=root,
        logger=logger,
        progress_state_path=paths["progress"],
        resume=resume,
        force=force,
        model_override=model_override,
        batch_size_override=batch_size_override,
        disable_4bit=disable_4bit,
    )


def _expected_structured_columns(
    ontology: Mapping[str, Sequence[str]],
) -> list[str]:
    expected = ["meta__structured_event_count"]
    expected.extend(f"etype__{value}" for value in ontology["event_types"])
    for field, key in (
        ("direction", "directions"),
        ("magnitude", "magnitudes"),
        ("certainty", "certainties"),
        ("time_horizon", "time_horizons"),
        ("entity_role", "entity_roles"),
    ):
        expected.extend(f"slot__{field}__{value}" for value in ontology[key])
    for event_type in ontology["event_types"]:
        for field, key in (
            ("direction", "directions"),
            ("magnitude", "magnitudes"),
            ("certainty", "certainties"),
            ("entity_role", "entity_roles"),
        ):
            expected.extend(
                f"interaction__{event_type}__{field}__{value}"
                for value in ontology[key]
            )
    return expected


def _structured_counts(
    structured_events: pd.DataFrame,
    ontology: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    keys = ["target_ticker", "feature_date"]
    expected = _expected_structured_columns(ontology)
    if structured_events.empty:
        return pd.DataFrame(columns=[*keys, *expected])
    records: list[dict[str, Any]] = []
    for row in structured_events.itertuples(index=False):
        values = {
            "event_type": str(row.event_type),
            "direction": str(row.direction),
            "magnitude": str(row.magnitude),
            "certainty": str(row.certainty),
            "time_horizon": str(row.time_horizon),
            "entity_role": str(row.entity_role),
        }
        record: dict[str, Any] = {
            "target_ticker": str(row.target_ticker),
            "feature_date": pd.Timestamp(row.feature_date),
            "meta__structured_event_count": 1.0,
        }
        record[f"etype__{values['event_type']}"] = 1.0
        for field in (
            "direction",
            "magnitude",
            "certainty",
            "time_horizon",
            "entity_role",
        ):
            record[f"slot__{field}__{values[field]}"] = 1.0
        for field in ("direction", "magnitude", "certainty", "entity_role"):
            record[
                f"interaction__{values['event_type']}__{field}__{values[field]}"
            ] = 1.0
        records.append(record)
    frame = pd.DataFrame(records).fillna(0.0)
    numeric = [column for column in frame.columns if column not in keys]
    aggregated = frame.groupby(keys, observed=True, as_index=False)[numeric].sum()
    # Create a stable ontology-defined column contract even if a class is absent.
    for column in expected:
        if column not in aggregated:
            aggregated[column] = 0.0
    return aggregated[keys + expected]


def _days_since_news(panel: pd.DataFrame) -> pd.Series:
    output = pd.Series(index=panel.index, dtype=float)
    for _, indices in panel.groupby("ticker", sort=False, observed=True).groups.items():
        count = 1_000.0
        for index in indices:
            if float(panel.at[index, "meta__news_count"]) > 0:
                count = 0.0
            else:
                count = min(count + 1.0, 1_000.0)
            output.at[index] = count
    return output


def run_features(
    config: Mapping[str, Any],
    *,
    logger: Any,
    force: bool = False,
) -> dict[str, Path]:
    profile = _profile(config)
    output_root(config, profile)
    paths = run_plan(config, logger=logger, force=False)
    if paths["feature_panel"].is_file() and not force:
        logger.info("Using existing structured feature panel %s", paths["feature_panel"])
        return paths
    if not paths["structured_events"].is_file():
        raise FileNotFoundError(
            f"{paths['structured_events']} is missing; run --stage extract first."
        )
    market = read_table(paths["market"])
    manifest = read_table(paths["event_manifest"])
    structured = read_table(paths["structured_events"])
    for frame in (market, manifest, structured):
        if "feature_date" in frame:
            frame["feature_date"] = pd.to_datetime(
                frame["feature_date"], errors="raise"
            ).dt.normalize()
    if not structured.empty:
        structured = structured.merge(
            manifest[["event_id", "feature_date"]].rename(
                columns={"event_id": "source_event_id"}
            ),
            on="source_event_id",
            how="left",
            validate="many_to_one",
        )
        if structured["feature_date"].isna().any():
            raise AssertionError("Some structured events lack mapped feature dates.")
    keys = market[["ticker", "feature_date"]].drop_duplicates().copy()
    news_daily = (
        manifest.groupby(["target_ticker", "feature_date"], observed=True)
        .size()
        .rename("meta__news_count")
        .reset_index()
        .rename(columns={"target_ticker": "ticker"})
    )
    candidate_daily = (
        manifest.loc[manifest["extraction_candidate"].fillna(False).astype(bool)]
        .groupby(["target_ticker", "feature_date"], observed=True)
        .size()
        .rename("meta__candidate_news_count")
        .reset_index()
        .rename(columns={"target_ticker": "ticker"})
    )
    panel = keys.merge(
        news_daily,
        on=["ticker", "feature_date"],
        how="left",
        validate="one_to_one",
    )
    panel["meta__news_count"] = panel["meta__news_count"].fillna(0.0).astype(float)
    panel = panel.merge(
        candidate_daily,
        on=["ticker", "feature_date"],
        how="left",
        validate="one_to_one",
    )
    panel["meta__candidate_news_count"] = (
        panel["meta__candidate_news_count"].fillna(0.0).astype(float)
    )
    panel["meta__has_candidate_news"] = panel[
        "meta__candidate_news_count"
    ].gt(0).astype(float)
    panel["meta__has_news"] = panel["meta__news_count"].gt(0).astype(float)
    panel["meta__no_news"] = 1.0 - panel["meta__has_news"]
    panel = panel.sort_values(["ticker", "feature_date"], kind="mergesort").reset_index(
        drop=True
    )
    panel["meta__days_since_news"] = _days_since_news(panel)
    ontology = ontology_from_config(profile)
    structured_counts = _structured_counts(structured, ontology).rename(
        columns={"target_ticker": "ticker"}
    )
    panel = panel.merge(
        structured_counts,
        on=["ticker", "feature_date"],
        how="left",
        validate="one_to_one",
    )
    numeric_columns = [
        column
        for column in panel.columns
        if column not in {"ticker", "feature_date"}
    ]
    panel[numeric_columns] = panel[numeric_columns].fillna(0.0).astype(float)
    panel["meta__has_structured_event"] = panel[
        "meta__structured_event_count"
    ].gt(0).astype(float)
    write_table(panel, paths["feature_panel"])
    audit_candidates = manifest.loc[
        manifest["extraction_candidate"].fillna(False).astype(bool)
    ].copy()
    audit_size = min(
        int(profile.get("audit_sample_size", 200)), len(audit_candidates)
    )
    audit = audit_candidates.sample(
        n=audit_size,
        random_state=int(config["project"]["seed"]),
    )[
        ["event_id", "date", "feature_date", "target_ticker", "text"]
    ].copy()
    if not structured.empty:
        event_fields = [
            "event_type",
            "direction",
            "magnitude",
            "certainty",
            "time_horizon",
            "entity_role",
            "evidence_text",
        ]
        serialized = pd.Series(
            {
                source_event_id: json.dumps(
                    group[event_fields].to_dict(orient="records"),
                    ensure_ascii=False,
                )
                for source_event_id, group in structured.groupby(
                    "source_event_id", observed=True
                )
            },
            name="predicted_events_json",
        )
        audit = audit.merge(
            serialized,
            left_on="event_id",
            right_index=True,
            how="left",
            validate="one_to_one",
        )
    audit["predicted_events_json"] = audit.get(
        "predicted_events_json", pd.Series(index=audit.index, dtype="string")
    ).fillna("[]")
    for column in (
        "audit_event_detection_correct",
        "audit_event_type_correct",
        "audit_direction_correct",
        "audit_entity_correct",
        "audit_evidence_correct",
        "audit_notes",
    ):
        audit[column] = ""
    write_table(audit, paths["audit_sample"])
    logger.info(
        "Feature panel completed | rows=%d columns=%d structured_events=%d "
        "audit_sample=%d",
        len(panel),
        len(panel.columns),
        len(structured),
        len(audit),
    )
    return paths


def _fold_frames(
    market: pd.DataFrame,
    fold_row: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = market.loc[
        market["feature_date"].between(
            pd.Timestamp(fold_row["train_start"]),
            pd.Timestamp(fold_row["train_end"]),
        )
    ].copy()
    validation = market.loc[
        market["feature_date"].between(
            pd.Timestamp(fold_row["validation_start"]),
            pd.Timestamp(fold_row["validation_end"]),
        )
    ].copy()
    if train.empty or validation.empty:
        raise ValueError(f"Fold {fold_row['fold']} has an empty segment.")
    if not train["feature_date"].max() < validation["feature_date"].min():
        raise AssertionError(f"Fold {fold_row['fold']} violates chronology.")
    return train, validation


def _apply_spike_labels(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    quantile: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    thresholds = (
        train.groupby("ticker", observed=True)["target_log_variance"]
        .quantile(float(quantile))
        .rename("spike_threshold")
        .reset_index()
    )
    train = train.merge(thresholds, on="ticker", how="left", validate="many_to_one")
    validation = validation.merge(
        thresholds, on="ticker", how="left", validate="many_to_one"
    )
    for frame in (train, validation):
        if frame["spike_threshold"].isna().any():
            raise AssertionError("Ticker-specific spike threshold is missing.")
        frame["spike_label"] = (
            frame["target_log_variance"] > frame["spike_threshold"]
        ).astype(int)
        if frame["spike_label"].nunique() < 2:
            raise ValueError("A pilot fold segment contains only one spike class.")
    return train, validation, thresholds


def _structured_columns(panel: pd.DataFrame) -> list[str]:
    return [
        column
        for column in panel.columns
        if column.startswith(("etype__", "slot__", "interaction__"))
    ]


def _lagged_panel(
    panel: pd.DataFrame,
    *,
    lookback: int,
    decay_rate: float,
) -> pd.DataFrame:
    output = panel.copy().sort_values(
        ["ticker", "feature_date"], kind="mergesort"
    )
    columns = _structured_columns(output)
    if int(lookback) == 1 or not columns:
        return output
    grouped = output.groupby("ticker", sort=False, observed=True)
    values = output[columns].copy()
    for lag in range(1, int(lookback)):
        weight = math.exp(-float(decay_rate) * lag)
        values = values.add(
            grouped[columns].shift(lag).fillna(0.0) * weight,
            fill_value=0.0,
        )
    output[columns] = values
    return output


def _rebuild_structured_panel(
    *,
    base_panel: pd.DataFrame,
    structured: pd.DataFrame,
    ontology: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    meta = base_panel[["ticker", "feature_date", *META_COLUMNS]].copy()
    counts = _structured_counts(structured, ontology).rename(
        columns={"target_ticker": "ticker"}
    )
    # Preserve true article-arrival metadata; only structured content is placeboed.
    counts = counts.drop(
        columns=["meta__structured_event_count"], errors="ignore"
    )
    output = meta.merge(
        counts,
        on=["ticker", "feature_date"],
        how="left",
        validate="one_to_one",
    )
    structured_columns = _structured_columns(output)
    output[structured_columns] = output[structured_columns].fillna(0.0)
    return output


def _segment_mask(frame: pd.DataFrame, start: Any, end: Any) -> pd.Series:
    return frame["feature_date"].between(pd.Timestamp(start), pd.Timestamp(end))


def _placebo_panel(
    *,
    base_panel: pd.DataFrame,
    structured: pd.DataFrame,
    ontology: Mapping[str, Sequence[str]],
    fold_row: pd.Series,
    kind: str,
    seed: int,
) -> pd.DataFrame:
    if kind not in PLACEBO_KINDS:
        raise ValueError(f"Unknown placebo kind {kind!r}.")
    rng = np.random.default_rng(int(seed))
    event_frame = structured.copy()
    segments = (
        (fold_row["train_start"], fold_row["train_end"]),
        (fold_row["validation_start"], fold_row["validation_end"]),
    )
    if kind == "shuffled_event_type":
        for start, end in segments:
            mask = _segment_mask(event_frame, start, end)
            values = event_frame.loc[mask, "event_type"].to_numpy(copy=True)
            rng.shuffle(values)
            event_frame.loc[mask, "event_type"] = values
        return _rebuild_structured_panel(
            base_panel=base_panel,
            structured=event_frame,
            ontology=ontology,
        )
    if kind == "shuffled_date":
        for start, end in segments:
            mask = _segment_mask(event_frame, start, end)
            dates = np.sort(event_frame.loc[mask, "feature_date"].unique())
            shuffled = dates.copy()
            rng.shuffle(shuffled)
            mapping = dict(zip(dates, shuffled))
            event_frame.loc[mask, "feature_date"] = event_frame.loc[
                mask, "feature_date"
            ].map(mapping)
        return _rebuild_structured_panel(
            base_panel=base_panel,
            structured=event_frame,
            ontology=ontology,
        )
    output = base_panel.copy()
    columns = _structured_columns(output)
    for start, end in segments:
        segment_mask = _segment_mask(output, start, end)
        segment = output.loc[segment_mask]
        for ticker, indices in segment.groupby(
            "ticker", sort=True, observed=True
        ).groups.items():
            del ticker
            for column in columns:
                values = output.loc[indices, column].to_numpy(copy=True)
                rng.shuffle(values)
                output.loc[indices, column] = values
    return output


def _representation_columns(
    representation: str,
    panel: pd.DataFrame,
    train_indices: pd.Index,
    minimum_count: float,
) -> list[str]:
    if representation == "F0_PRICE":
        return []
    if representation == "F1_META_BASIC":
        return list(BASIC_META_COLUMNS)
    if representation == "F1_EVENT_META":
        return list(META_COLUMNS)
    prefixes = {
        "F2_EVENT_TYPE": ("etype__",),
        "F3_SLOTS": ("etype__", "slot__"),
        "F4_INTERACTIONS": ("etype__", "slot__", "interaction__"),
    }
    if representation not in prefixes:
        raise ValueError(f"Unknown true representation {representation!r}.")
    candidates = [
        column
        for column in panel.columns
        if column.startswith(prefixes[representation])
    ]
    selected = [
        column
        for column in candidates
        if float(panel.loc[train_indices, column].sum()) >= float(minimum_count)
    ]
    return [*META_COLUMNS, *selected]


def _processor(numeric_columns: Sequence[str]) -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, list(numeric_columns)),
            (
                "ticker",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["ticker"],
            ),
        ],
        remainder="drop",
    )


def _fit_probability_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    numeric_columns: Sequence[str],
    c_values: Sequence[float],
    seed: int,
    inner_validation_fraction: float,
) -> tuple[np.ndarray, float]:
    dates = np.sort(train["feature_date"].unique())
    inner_count = max(int(round(len(dates) * inner_validation_fraction)), 30)
    if inner_count >= len(dates):
        raise ValueError("Inner chronological validation consumes all train dates.")
    inner_train_dates = dates[:-inner_count]
    inner_validation_dates = dates[-inner_count:]
    inner_train = train.loc[train["feature_date"].isin(inner_train_dates)]
    inner_validation = train.loc[
        train["feature_date"].isin(inner_validation_dates)
    ]
    if (
        inner_train["spike_label"].nunique() < 2
        or inner_validation["spike_label"].nunique() < 2
    ):
        raise ValueError("Inner split contains only one spike class.")
    model_columns = [*numeric_columns, "ticker"]
    best_c: float | None = None
    best_score = -np.inf
    for c_value in c_values:
        processor = _processor(numeric_columns)
        x_inner = processor.fit_transform(inner_train[model_columns])
        x_inner_validation = processor.transform(inner_validation[model_columns])
        model = LogisticRegression(
            C=float(c_value),
            penalty="l2",
            class_weight="balanced",
            max_iter=2000,
            random_state=int(seed),
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            model.fit(x_inner, inner_train["spike_label"].to_numpy(dtype=int))
        probability = model.predict_proba(x_inner_validation)[:, 1]
        score = float(
            binary_metrics(
                inner_validation["spike_label"].to_numpy(dtype=int),
                probability,
            )["pr_auc"]
        )
        if score > best_score:
            best_score = score
            best_c = float(c_value)
    if best_c is None:
        raise RuntimeError("No logistic C value produced a valid inner score.")
    processor = _processor(numeric_columns)
    x_train = processor.fit_transform(train[model_columns])
    x_validation = processor.transform(validation[model_columns])
    model = LogisticRegression(
        C=best_c,
        penalty="l2",
        class_weight="balanced",
        max_iter=2000,
        random_state=int(seed),
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(x_train, train["spike_label"].to_numpy(dtype=int))
    return model.predict_proba(x_validation)[:, 1], best_c


def _task_specs(profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    real_seed = int(profile.get("model_seed", 20260725))
    for representation in TRUE_REPRESENTATIONS:
        lookbacks = (
            (1,)
            if representation in {"F0_PRICE", "F1_META_BASIC", "F1_EVENT_META"}
            else profile["lookbacks"]
        )
        for lookback in lookbacks:
            tasks.append(
                {
                    "representation": representation,
                    "lookback": int(lookback),
                    "placebo_kind": "none",
                    "seed": real_seed,
                }
            )
    for representation in ("F2_EVENT_TYPE", "F3_SLOTS", "F4_INTERACTIONS"):
        for kind in PLACEBO_KINDS:
            for seed in profile["placebo_seeds"]:
                for lookback in profile["lookbacks"]:
                    tasks.append(
                        {
                            "representation": representation,
                            "lookback": int(lookback),
                            "placebo_kind": kind,
                            "seed": int(seed),
                        }
                    )
    return tasks


def _checkpoint_paths(root: Path, task_id: str) -> tuple[Path, Path]:
    return (
        root / "checkpoints" / f"{task_id}.json",
        root / "predictions" / f"{task_id}.parquet",
    )


def run_forecast(
    config: Mapping[str, Any],
    *,
    logger: Any,
    resume: bool,
    force: bool,
) -> dict[str, Path]:
    profile = _profile(config)
    root = output_root(config, profile)
    paths = run_features(config, logger=logger, force=False)
    market = read_table(paths["market"])
    panel = read_table(paths["feature_panel"])
    structured = read_table(paths["structured_events"])
    manifest = read_table(paths["event_manifest"])
    folds = read_table(paths["folds"])
    for frame in (market, panel, manifest, structured):
        if "feature_date" in frame:
            frame["feature_date"] = pd.to_datetime(
                frame["feature_date"], errors="raise"
            ).dt.normalize()
    if not structured.empty and "feature_date" not in structured:
        structured = structured.merge(
            manifest[["event_id", "feature_date"]].rename(
                columns={"event_id": "source_event_id"}
            ),
            on="source_event_id",
            how="left",
            validate="many_to_one",
        )
    market = market.merge(
        panel,
        on=["ticker", "feature_date"],
        how="left",
        validate="one_to_one",
    )
    price_columns = price_feature_columns(market, config)
    ontology = ontology_from_config(profile)
    task_templates = _task_specs(profile)
    all_tasks = [
        {"fold": int(fold), **task}
        for fold in folds["fold"].astype(int)
        for task in task_templates
    ]
    completed_rows: list[dict[str, Any]] = []
    ticker_rows: list[dict[str, Any]] = []
    progress = tqdm(
        total=len(all_tasks),
        desc="Structured-event forecasting",
        unit="task",
        dynamic_ncols=True,
    )
    started = time.monotonic()
    panel_cache: dict[tuple[int, str, int], pd.DataFrame] = {}
    for task_index, task in enumerate(all_tasks, start=1):
        task_id = stable_id(task, prefix="structured_pilot")
        checkpoint_path, prediction_path = _checkpoint_paths(root, task_id)
        if resume and not force and checkpoint_path.is_file() and prediction_path.is_file():
            row = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            completed_rows.append(row)
            predictions = read_table(prediction_path)
            for ticker, group in predictions.groupby(
                "ticker", sort=True, observed=True
            ):
                metrics = binary_metrics(
                    group["spike_label"].to_numpy(dtype=int),
                    group["probability"].to_numpy(dtype=float),
                )
                ticker_rows.append(
                    {**task, "task_id": task_id, "ticker": ticker, **metrics}
                )
            progress.update(1)
            continue
        fold_row = folds.loc[folds["fold"].astype(int).eq(task["fold"])].iloc[0]
        base_train, base_validation = _fold_frames(market, fold_row)
        train, validation, thresholds = _apply_spike_labels(
            base_train,
            base_validation,
            float(profile.get("spike_quantile", 0.90)),
        )
        cache_key = (
            int(task["fold"]),
            str(task["placebo_kind"]),
            int(task["seed"]),
        )
        if cache_key not in panel_cache:
            if task["placebo_kind"] == "none":
                content_panel = panel
            else:
                content_panel = _placebo_panel(
                    base_panel=panel,
                    structured=structured,
                    ontology=ontology,
                    fold_row=fold_row,
                    kind=str(task["placebo_kind"]),
                    seed=int(task["seed"]),
                )
            panel_cache[cache_key] = content_panel
        content_panel = _lagged_panel(
            panel_cache[cache_key],
            lookback=int(task["lookback"]),
            decay_rate=float(profile["decay_rate"]),
        )
        content_columns = [
            column
            for column in content_panel.columns
            if column not in {"ticker", "feature_date"}
        ]
        train = train.drop(columns=content_columns, errors="ignore").merge(
            content_panel,
            on=["ticker", "feature_date"],
            how="left",
            validate="one_to_one",
        )
        validation = validation.drop(columns=content_columns, errors="ignore").merge(
            content_panel,
            on=["ticker", "feature_date"],
            how="left",
            validate="one_to_one",
        )
        train_index = train.index
        representation_columns = _representation_columns(
            str(task["representation"]),
            train,
            train_index,
            float(profile.get("minimum_feature_count", 10)),
        )
        numeric_columns = list(dict.fromkeys([*price_columns, *representation_columns]))
        if train[numeric_columns].isna().all(axis=0).any():
            missing = train[numeric_columns].columns[
                train[numeric_columns].isna().all(axis=0)
            ].tolist()
            raise ValueError(f"Entirely missing training features: {missing}")
        probability, best_c = _fit_probability_model(
            train,
            validation,
            numeric_columns=numeric_columns,
            c_values=tuple(float(value) for value in profile["logistic_c_values"]),
            seed=int(task["seed"]),
            inner_validation_fraction=float(
                profile.get("inner_validation_fraction", 0.20)
            ),
        )
        metrics = binary_metrics(
            validation["spike_label"].to_numpy(dtype=int),
            probability,
        )
        prevalence = float(metrics["positive_rate"])
        row = {
            **task,
            "task_id": task_id,
            "target": "volatility_spike_q90_ticker",
            "model": "class_weighted_logistic",
            "best_c": best_c,
            "feature_count": len(numeric_columns),
            "structured_feature_count": len(representation_columns),
            "train_start": train["feature_date"].min(),
            "train_end": train["feature_date"].max(),
            "validation_start": validation["feature_date"].min(),
            "validation_end": validation["feature_date"].max(),
            "train_rows": len(train),
            "validation_rows": len(validation),
            "train_positive_rate": float(train["spike_label"].mean()),
            "pr_auc_over_prevalence": float(metrics["pr_auc"])
            / max(prevalence, 1.0e-12),
            "thresholds_json": json.dumps(
                dict(zip(thresholds["ticker"], thresholds["spike_threshold"])),
                sort_keys=True,
            ),
            **metrics,
        }
        prediction_frame = validation[
            [
                "ticker",
                "feature_date",
                "target_date",
                "target_log_variance",
                "spike_label",
            ]
        ].copy()
        prediction_frame["probability"] = probability
        prediction_frame["task_id"] = task_id
        prediction_frame["fold"] = int(task["fold"])
        prediction_frame["representation"] = str(task["representation"])
        prediction_frame["lookback"] = int(task["lookback"])
        prediction_frame["placebo_kind"] = str(task["placebo_kind"])
        prediction_frame["seed"] = int(task["seed"])
        write_table(prediction_frame, prediction_path)
        atomic_write_json(row, checkpoint_path)
        completed_rows.append(row)
        for ticker, group in prediction_frame.groupby(
            "ticker", sort=True, observed=True
        ):
            ticker_metrics = binary_metrics(
                group["spike_label"].to_numpy(dtype=int),
                group["probability"].to_numpy(dtype=float),
            )
            ticker_rows.append(
                {**task, "task_id": task_id, "ticker": ticker, **ticker_metrics}
            )
        progress.update(1)
        elapsed = max(time.monotonic() - started, 1.0e-9)
        rate = task_index / elapsed
        remaining = (len(all_tasks) - task_index) / max(rate, 1.0e-9)
        progress.set_postfix(
            fold=task["fold"],
            rep=task["representation"],
            lb=task["lookback"],
            placebo=task["placebo_kind"],
            pr=f"{float(metrics['pr_auc']):.3f}",
            eta_min=f"{remaining / 60.0:.1f}",
        )
        atomic_write_json(
            {
                "stage": "forecast",
                "status": "running",
                "completed": task_index,
                "total": len(all_tasks),
                "percent": 100.0 * task_index / max(len(all_tasks), 1),
                "elapsed_seconds": elapsed,
                "estimated_remaining_seconds": remaining,
                "current_task": task,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            paths["progress"],
        )
        logger.info(
            "Forecast %d/%d | fold=%d rep=%s lookback=%d placebo=%s seed=%d "
            "PR-AUC=%.6f Brier=%.6f ETA=%.1f min",
            task_index,
            len(all_tasks),
            task["fold"],
            task["representation"],
            task["lookback"],
            task["placebo_kind"],
            task["seed"],
            float(metrics["pr_auc"]),
            float(metrics["brier"]),
            remaining / 60.0,
        )
    progress.close()
    results = pd.DataFrame(completed_rows)
    ticker_results = pd.DataFrame(ticker_rows)
    write_table(results, paths["results"])
    write_table(ticker_results, paths["ticker_results"])
    atomic_write_json(
        {
            "stage": "forecast",
            "status": "completed",
            "completed": len(all_tasks),
            "total": len(all_tasks),
            "percent": 100.0,
            "elapsed_seconds": time.monotonic() - started,
            "estimated_remaining_seconds": 0.0,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
        paths["progress"],
    )
    return paths


def _gain(candidate: float, reference: float, *, lower_better: bool) -> float:
    direction = reference - candidate if lower_better else candidate - reference
    return 100.0 * direction / max(abs(reference), 1.0e-12)


def run_evaluation(
    config: Mapping[str, Any],
    *,
    logger: Any,
) -> dict[str, Path]:
    profile = _profile(config)
    root = output_root(config, profile)
    paths = _paths(root)
    if not paths["results"].is_file():
        raise FileNotFoundError(
            f"{paths['results']} is missing; run --stage forecast first."
        )
    results = read_table(paths["results"])
    true = results.loc[results["placebo_kind"].eq("none")].copy()
    aggregate = (
        true.groupby(["representation", "lookback"], observed=True)
        .agg(
            folds=("fold", "nunique"),
            mean_pr_auc=("pr_auc", "mean"),
            std_pr_auc=("pr_auc", "std"),
            mean_brier=("brier", "mean"),
            mean_ece=("ece", "mean"),
            mean_recall=("recall", "mean"),
            mean_pr_auc_over_prevalence=("pr_auc_over_prevalence", "mean"),
        )
        .reset_index()
    )
    candidates = aggregate.loc[
        aggregate["representation"].isin(
            ["F2_EVENT_TYPE", "F3_SLOTS", "F4_INTERACTIONS"]
        )
    ].sort_values(["mean_pr_auc", "mean_brier"], ascending=[False, True])
    if candidates.empty:
        raise ValueError("No structured-event candidate results exist.")
    best = candidates.iloc[0]
    reference = aggregate.loc[
        aggregate["representation"].eq("F1_EVENT_META")
        & aggregate["lookback"].astype(int).eq(1)
    ]
    if len(reference) != 1:
        raise ValueError("Expected exactly one aggregated F1_EVENT_META baseline.")
    reference_row = reference.iloc[0]
    basic_reference = aggregate.loc[
        aggregate["representation"].eq("F1_META_BASIC")
        & aggregate["lookback"].astype(int).eq(1)
    ]
    if len(basic_reference) != 1:
        raise ValueError("Expected exactly one aggregated F1_META_BASIC baseline.")
    basic_reference_row = basic_reference.iloc[0]
    fold_candidate = true.loc[
        true["representation"].eq(best["representation"])
        & true["lookback"].astype(int).eq(int(best["lookback"]))
    ]
    fold_reference = true.loc[
        true["representation"].eq("F1_EVENT_META")
        & true["lookback"].astype(int).eq(1)
    ]
    comparisons = fold_candidate.merge(
        fold_reference,
        on="fold",
        suffixes=("_candidate", "_reference"),
        validate="one_to_one",
    )
    comparisons["pr_auc_absolute_gain"] = (
        comparisons["pr_auc_candidate"] - comparisons["pr_auc_reference"]
    )
    comparisons["pr_auc_relative_gain_percent"] = [
        _gain(candidate, reference, lower_better=False)
        for candidate, reference in zip(
            comparisons["pr_auc_candidate"], comparisons["pr_auc_reference"]
        )
    ]
    comparisons["brier_gain_percent"] = [
        _gain(candidate, reference, lower_better=True)
        for candidate, reference in zip(
            comparisons["brier_candidate"], comparisons["brier_reference"]
        )
    ]
    write_table(comparisons, paths["comparisons"])
    placebo = results.loc[
        results["placebo_kind"].ne("none")
        & results["representation"].eq(best["representation"])
        & results["lookback"].astype(int).eq(int(best["lookback"]))
    ].copy()
    placebo_aggregate = (
        placebo.groupby(["placebo_kind", "seed"], observed=True)
        .agg(
            folds=("fold", "nunique"),
            mean_pr_auc=("pr_auc", "mean"),
            mean_brier=("brier", "mean"),
        )
        .reset_index()
    )
    true_score = float(best["mean_pr_auc"])
    placebo_aggregate["true_mean_pr_auc"] = true_score
    placebo_aggregate["true_beats_placebo"] = (
        true_score > placebo_aggregate["mean_pr_auc"]
    )
    write_table(placebo_aggregate, paths["placebos"])
    null_scores = placebo_aggregate["mean_pr_auc"].to_numpy(dtype=float)
    empirical_p = float(
        (1 + np.sum(null_scores >= true_score)) / (1 + len(null_scores))
    )
    mean_absolute_gain = float(comparisons["pr_auc_absolute_gain"].mean())
    mean_relative_gain = float(
        comparisons["pr_auc_relative_gain_percent"].mean()
    )
    positive_folds = int(comparisons["pr_auc_absolute_gain"].gt(0).sum())
    brier_degradation = max(
        0.0, -float(comparisons["brier_gain_percent"].mean())
    )
    extraction_summary_path = root / "tables" / "extraction_summary.csv"
    extraction_summary = read_table(extraction_summary_path).iloc[0]
    extraction_valid = float(extraction_summary["valid_rate"]) >= float(
        profile["extractor"].get("minimum_valid_rate", 0.95)
    )
    gain_pass = (
        mean_absolute_gain
        >= float(profile.get("minimum_absolute_pr_auc_gain", 0.01))
        or mean_relative_gain
        >= float(profile.get("minimum_relative_pr_auc_gain_percent", 5.0))
    )
    fold_pass = positive_folds == int(profile["fold_count"])
    brier_pass = brier_degradation <= float(
        profile.get("maximum_brier_degradation_percent", 1.0)
    )
    placebo_pass = bool(
        placebo_aggregate["true_beats_placebo"].all()
        if not placebo_aggregate.empty
        else False
    )
    event_meta_gain = float(reference_row["mean_pr_auc"]) - float(
        basic_reference_row["mean_pr_auc"]
    )
    if extraction_valid and gain_pass and fold_pass and brier_pass and placebo_pass:
        decision = "PILOT-PASS"
        next_step = (
            "Extract the full development corpus, freeze the extractor and test "
            "structured prototypes plus market-conditioned attention."
        )
    elif extraction_valid and event_meta_gain > 0 and mean_absolute_gain <= 0:
        decision = "PILOT-EVENT-DETECTION-ONLY"
        next_step = (
            "LLM event detection/count helps beyond the keyword gate, but event "
            "types/slots do not. Confirm event-count metadata before prototypes."
        )
    elif extraction_valid and mean_absolute_gain > 0 and positive_folds >= 1:
        decision = "PILOT-WEAK"
        next_step = (
            "Audit extraction and event coverage; add related news or one new "
            "development period before any full extraction."
        )
    else:
        decision = "PILOT-FAIL"
        next_step = (
            "Do not extract the full corpus. Verify extraction quality/coverage; "
            "stop if both are adequate."
        )
    decision_frame = pd.DataFrame(
        [
            {
                "decision": decision,
                "best_representation": best["representation"],
                "best_lookback": int(best["lookback"]),
                "best_mean_pr_auc": true_score,
                "meta_mean_pr_auc": float(reference_row["mean_pr_auc"]),
                "meta_basic_mean_pr_auc": float(
                    basic_reference_row["mean_pr_auc"]
                ),
                "event_meta_vs_basic_absolute_gain": event_meta_gain,
                "mean_absolute_pr_auc_gain": mean_absolute_gain,
                "mean_relative_pr_auc_gain_percent": mean_relative_gain,
                "positive_fold_count": positive_folds,
                "required_positive_folds": int(profile["fold_count"]),
                "mean_brier_degradation_percent": brier_degradation,
                "true_beats_all_placebos": placebo_pass,
                "empirical_placebo_p_value_descriptive": empirical_p,
                "extraction_valid_rate": float(extraction_summary["valid_rate"]),
                "gain_gate_passed": gain_pass,
                "fold_gate_passed": fold_pass,
                "brier_gate_passed": brier_pass,
                "placebo_gate_passed": placebo_pass,
                "next_step": next_step,
            }
        ]
    )
    write_table(decision_frame, paths["decision"])
    report = {
        "experiment": "structured_event_forecasting_feasibility",
        "development_only": True,
        "locked_test_evaluated": False,
        "decision": decision,
        "best_candidate": best.to_dict(),
        "event_meta_reference": reference_row.to_dict(),
        "basic_meta_reference": basic_reference_row.to_dict(),
        "gates": decision_frame.iloc[0].to_dict(),
        "warning": (
            "This is a screening pilot. PILOT-PASS authorizes a larger "
            "development-only experiment, not a GO forecasting claim."
        ),
    }
    atomic_write_json(report, paths["report"])
    figure = aggregate.copy()
    labels = [
        f"{row.representation}\nL={int(row.lookback)}"
        for row in figure.itertuples(index=False)
    ]
    plt.figure(figsize=(12, 5))
    plt.bar(labels, figure["mean_pr_auc"], color="#2A6F97")
    plt.axhline(
        float(reference_row["mean_pr_auc"]),
        color="#C1121F",
        linestyle="--",
        label="F1_EVENT_META",
    )
    plt.ylabel("Mean validation PR-AUC")
    plt.title("Structured-event feasibility: development folds only")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(root / "figures" / "structured_event_pilot_performance.png", dpi=180)
    plt.close()
    logger.info(
        "Pilot decision=%s | best=%s L=%d PR-AUC=%.6f | META=%.6f | "
        "absolute_gain=%.6f | positive_folds=%d/%d | placebo_p(descriptive)=%.4f",
        decision,
        best["representation"],
        int(best["lookback"]),
        true_score,
        float(reference_row["mean_pr_auc"]),
        mean_absolute_gain,
        positive_folds,
        int(profile["fold_count"]),
        empirical_p,
    )
    return paths


def print_report(config: Mapping[str, Any]) -> None:
    profile = _profile(config)
    paths = _paths(output_root(config, profile))
    if not paths["decision"].is_file():
        return
    row = read_table(paths["decision"]).iloc[0]
    print("\nSTRUCTURED EVENT FORECASTING FEASIBILITY PILOT")
    print("Development-only; locked test evaluated: False")
    print(f"Best representation: {row['best_representation']}")
    print(f"Best lookback: {int(row['best_lookback'])} trading day(s)")
    print(f"Best mean PR-AUC: {float(row['best_mean_pr_auc']):.6f}")
    print(
        "META_BASIC mean PR-AUC: "
        f"{float(row['meta_basic_mean_pr_auc']):.6f}"
    )
    print(
        "EVENT_META mean PR-AUC: "
        f"{float(row['meta_mean_pr_auc']):.6f}"
    )
    print(
        "Mean absolute PR-AUC gain: "
        f"{float(row['mean_absolute_pr_auc_gain']):+.6f}"
    )
    print(
        "Positive folds: "
        f"{int(row['positive_fold_count'])}/{int(row['required_positive_folds'])}"
    )
    print(f"True news beats all pilot placebos: {bool(row['true_beats_all_placebos'])}")
    print(f"Decision: {row['decision']}")
    print(f"Next step: {row['next_step']}")
