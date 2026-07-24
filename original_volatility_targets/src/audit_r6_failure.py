"""Post-hoc failure audit for the locked R6 confirmatory experiment.

This module is deliberately diagnostic. It reads only the three chronological
validation folds and never reads or evaluates the locked holdout test.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import adjusted_rand_score

from src.compare_representations import metric_gain
from src.evaluate_original_targets import load_predictions
from src.modeling import representation_row
from src.utils import (
    atomic_write_csv,
    get_logger,
    load_representation_frame,
    project_path,
    qlike_from_log,
    read_table,
    representation_feature_columns,
    resolve_shared_file,
    shared_root,
    validate_columns,
)


NEWS_LEVELS = ("macro", "sector", "related", "target")
KEY_COLUMNS = ("ticker", "feature_date", "target_date")
OUTPUT_NAMES = (
    "r6_ticker_diagnostics.csv",
    "r6_news_day_diagnostics.csv",
    "r6_news_level_diagnostics.csv",
    "r6_fold_distribution_shift.csv",
    "r6_prototype_drift.csv",
    "r6_failure_audit_summary.csv",
)


def audit_outputs() -> tuple[str, ...]:
    return tuple(f"outputs/tables/{name}" for name in OUTPUT_NAMES)


def _profile(config: Mapping[str, Any]) -> Mapping[str, Any]:
    profile = config.get("r6_confirmatory")
    if not isinstance(profile, Mapping):
        raise KeyError("Missing r6_confirmatory configuration section.")
    required = {
        "folds",
        "seeds",
        "representations",
        "comparison_representations",
        "representation_variant_family",
    }
    missing = sorted(required.difference(profile))
    if missing:
        raise KeyError(
            f"r6_confirmatory configuration is missing: {missing}"
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


def _parse_mixed_datetime(
    values: pd.Series,
    *,
    column_name: str,
) -> pd.Series:
    """Parse mixed date-only/ISO datetime values and reject invalid dates."""
    try:
        return pd.to_datetime(values, format="mixed", errors="raise")
    except ValueError as error:
        examples = values.dropna().astype(str).drop_duplicates().head(5).tolist()
        raise ValueError(
            f"Could not parse datetime column {column_name!r}; "
            f"example values={examples}"
        ) from error


def _locked_predictions(config: Mapping[str, Any]) -> pd.DataFrame:
    profile = _profile(config)
    predictions = load_predictions(config, mode="r6_confirmatory")
    alpha_values = tuple(
        float(value) for value in profile.get("ridge_alphas", [10.0])
    )
    if len(alpha_values) != 1:
        raise ValueError("R6 audit requires exactly one locked Ridge alpha.")
    alpha = alpha_values[0]
    selected = predictions.loc[
        predictions["target_family"].astype(str).eq("level")
        & predictions["target"].astype(str).eq("volatility_level")
        & predictions["evaluation_split"].astype(str).eq("validation")
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
    expected_inputs = np.where(
        selected["representation"].astype(str).eq("R0"),
        "price_only",
        "price_plus_text",
    )
    selected = selected.loc[
        selected["input_variant"].astype(str).to_numpy() == expected_inputs
    ].copy()
    for column in ("feature_date", "target_date"):
        selected[column] = _parse_mixed_datetime(
            selected[column],
            column_name=f"confirmatory_predictions.{column}",
        ).dt.normalize()
    if selected.duplicated(["task_id", *KEY_COLUMNS]).any():
        raise ValueError("Confirmatory predictions contain duplicate row keys.")

    task_rows = selected.drop_duplicates("task_id")
    expected = {
        (int(fold), int(seed), str(representation))
        for fold in profile["folds"]
        for seed in profile["seeds"]
        for representation in profile["representations"]
    }
    observed_counts = (
        task_rows.groupby(
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
            "Audit requires the exact locked checkpoint grid: "
            f"missing={missing}; duplicates={duplicates}; "
            f"unexpected={unexpected}"
        )
    if not _truthy_mask(
        task_rows["qualifies_for_robustness"]
    ).all():
        raise AssertionError(
            "A non-fold-safe checkpoint entered the R6 failure audit."
        )
    return selected.sort_values(
        ["fold", "seed", "representation", "feature_date", "ticker"],
        kind="mergesort",
    ).reset_index(drop=True)


def _feature_frame(
    config: Mapping[str, Any],
    fold: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    profile = _profile(config)
    family = str(profile["representation_variant_family"])
    row = representation_row(
        config,
        "R6",
        family,
        fold=fold,
        seed=seed,
        representation_variant_family=family,
    )
    frame = load_representation_frame(config, row, seed=seed)
    feature_columns = representation_feature_columns(frame)
    by_level = {
        level: [
            column
            for column in feature_columns
            if column.startswith(f"softproto__{level}__")
        ]
        for level in NEWS_LEVELS
    }
    missing = [
        level for level, columns in by_level.items() if not columns
    ]
    if missing:
        raise KeyError(
            f"Fold {fold}/seed {seed} R6 lacks prototype levels: {missing}"
        )
    if frame.duplicated(["ticker", "feature_date"]).any():
        raise ValueError(
            f"R6 fold feature frame has duplicate keys: fold={fold}, seed={seed}"
        )
    return frame, by_level


def _paired_cell(
    predictions: pd.DataFrame,
    fold: int,
    seed: int,
    references: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, str]]:
    cell = predictions.loc[
        predictions["fold"].eq(fold) & predictions["seed"].eq(seed)
    ].copy()
    r6 = cell.loc[cell["representation"].astype(str).eq("R6")]
    if r6["task_id"].nunique() != 1:
        raise ValueError(
            f"Expected one R6 task for fold={fold}, seed={seed}."
        )
    task_ids = {"R6": str(r6["task_id"].iloc[0])}
    paired = r6[
        [*KEY_COLUMNS, "y_true", "prediction"]
    ].rename(columns={"prediction": "prediction_R6"})
    for reference in references:
        source = cell.loc[
            cell["representation"].astype(str).eq(str(reference))
        ]
        if source["task_id"].nunique() != 1:
            raise ValueError(
                f"Expected one {reference} task for fold={fold}, seed={seed}."
            )
        task_ids[str(reference)] = str(source["task_id"].iloc[0])
        reference_frame = source[
            [*KEY_COLUMNS, "y_true", "prediction"]
        ].rename(
            columns={
                "y_true": f"y_true_{reference}",
                "prediction": f"prediction_{reference}",
            }
        )
        paired = paired.merge(
            reference_frame,
            on=list(KEY_COLUMNS),
            how="inner",
            validate="one_to_one",
        )
        if not np.allclose(
            paired["y_true"].to_numpy(dtype=float),
            paired[f"y_true_{reference}"].to_numpy(dtype=float),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise AssertionError(
                f"Target mismatch for R6/{reference}, fold={fold}, seed={seed}."
            )
        paired = paired.drop(columns=f"y_true_{reference}")
    expected_rows = len(r6)
    if len(paired) != expected_rows:
        raise ValueError(
            f"Paired prediction join lost rows for fold={fold}, seed={seed}: "
            f"{len(paired)}/{expected_rows}"
        )
    return paired, task_ids


def _paired_metrics(
    group: pd.DataFrame,
    reference: str,
    minimum_group_size: int,
) -> dict[str, Any]:
    y_true = group["y_true"].to_numpy(dtype=float)
    r6_prediction = group["prediction_R6"].to_numpy(dtype=float)
    reference_prediction = group[
        f"prediction_{reference}"
    ].to_numpy(dtype=float)
    r6_qlike = qlike_from_log(y_true, r6_prediction)
    reference_qlike = qlike_from_log(y_true, reference_prediction)
    return {
        "n": len(group),
        "minimum_group_size": int(minimum_group_size),
        "descriptive_group_eligible": bool(
            len(group) >= minimum_group_size
        ),
        "r6_qlike": r6_qlike,
        "reference_qlike": reference_qlike,
        "relative_gain": metric_gain(
            r6_qlike, reference_qlike, larger=False
        ),
        "r6_mae": float(np.mean(np.abs(y_true - r6_prediction))),
        "reference_mae": float(
            np.mean(np.abs(y_true - reference_prediction))
        ),
        "mean_target_log_variance": float(np.mean(y_true)),
    }


def _prediction_diagnostics(
    config: Mapping[str, Any],
    predictions: pd.DataFrame,
    logger: Any,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[tuple[int, int], tuple[pd.DataFrame, dict[str, list[str]]]],
]:
    profile = _profile(config)
    folds = tuple(int(value) for value in profile["folds"])
    seeds = tuple(int(value) for value in profile["seeds"])
    references = tuple(
        str(value) for value in profile["comparison_representations"]
    )
    minimum_group_size = int(profile.get("audit_min_group_size", 20))
    ticker_rows: list[dict[str, Any]] = []
    news_day_rows: list[dict[str, Any]] = []
    news_level_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    feature_cache: dict[
        tuple[int, int], tuple[pd.DataFrame, dict[str, list[str]]]
    ] = {}
    total = len(folds) * len(seeds)
    completed = 0
    for fold in folds:
        for seed in seeds:
            completed += 1
            logger.info(
                "R6 audit prediction strata | cell %d/%d | fold=%d | seed=%d",
                completed,
                total,
                fold,
                seed,
            )
            paired, task_ids = _paired_cell(
                predictions, fold, seed, references
            )
            features, by_level = _feature_frame(
                config, fold, seed
            )
            feature_cache[(fold, seed)] = (features, by_level)
            validation_features = features.loc[
                features["split"].astype(str).eq("validation")
            ].copy()
            join_columns = [
                column
                for columns in by_level.values()
                for column in columns
            ]
            enriched = paired.merge(
                validation_features[
                    ["ticker", "feature_date", *join_columns]
                ],
                on=["ticker", "feature_date"],
                how="left",
                validate="one_to_one",
            )
            if enriched[join_columns].isna().any().any():
                raise ValueError(
                    f"Missing R6 validation features for fold={fold}, seed={seed}."
                )
            for level, columns in by_level.items():
                enriched[f"has_{level}_news"] = (
                    enriched[columns]
                    .abs()
                    .sum(axis=1)
                    .gt(1.0e-12)
                )
            enriched["has_any_news"] = enriched[
                [f"has_{level}_news" for level in NEWS_LEVELS]
            ].any(axis=1)
            enriched["active_news_levels"] = enriched[
                [f"has_{level}_news" for level in NEWS_LEVELS]
            ].sum(axis=1)
            enriched["r6_feature_l2"] = np.linalg.norm(
                enriched[join_columns].to_numpy(dtype=float), axis=1
            )

            for reference in references:
                base = {
                    "fold": fold,
                    "prototype_seed": seed,
                    "model_seed": seed,
                    "reference": reference,
                    "r6_task_id": task_ids["R6"],
                    "reference_task_id": task_ids[reference],
                    "evidence_scope": "post_hoc_validation_audit",
                    "locked_test_used": False,
                }
                cell_rows.append(
                    {
                        **base,
                        **_paired_metrics(
                            enriched, reference, minimum_group_size
                        ),
                    }
                )
                for ticker, group in enriched.groupby(
                    "ticker", sort=True, observed=True
                ):
                    ticker_rows.append(
                        {
                            **base,
                            "ticker": str(ticker),
                            **_paired_metrics(
                                group, reference, minimum_group_size
                            ),
                            "news_day_rate": float(
                                group["has_any_news"].mean()
                            ),
                            "mean_active_news_levels": float(
                                group["active_news_levels"].mean()
                            ),
                        }
                    )
                for has_news, group in enriched.groupby(
                    "has_any_news", sort=True, observed=True
                ):
                    news_day_rows.append(
                        {
                            **base,
                            "condition": (
                                "has_any_news"
                                if bool(has_news)
                                else "no_news"
                            ),
                            **_paired_metrics(
                                group, reference, minimum_group_size
                            ),
                            "mean_active_news_levels": float(
                                group["active_news_levels"].mean()
                            ),
                            "mean_r6_feature_l2": float(
                                group["r6_feature_l2"].mean()
                            ),
                        }
                    )
                for level in NEWS_LEVELS:
                    flag = f"has_{level}_news"
                    for has_level, group in enriched.groupby(
                        flag, sort=True, observed=True
                    ):
                        news_level_rows.append(
                            {
                                **base,
                                "news_level": level,
                                "condition": (
                                    "has_level_news"
                                    if bool(has_level)
                                    else "no_level_news"
                                ),
                                **_paired_metrics(
                                    group,
                                    reference,
                                    minimum_group_size,
                                ),
                                "mean_r6_feature_l2": float(
                                    group["r6_feature_l2"].mean()
                                ),
                            }
                        )
    return (
        pd.DataFrame(ticker_rows),
        pd.DataFrame(news_day_rows),
        pd.DataFrame(news_level_rows),
        pd.DataFrame(cell_rows),
        feature_cache,
    )


def _safe_js_divergence(
    left: np.ndarray, right: np.ndarray
) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.sum() <= 0 or right.sum() <= 0:
        return np.nan
    left = left / left.sum()
    right = right / right.sum()
    return float(jensenshannon(left, right, base=2.0) ** 2)


def _coefficient_rows(
    config: Mapping[str, Any],
    r6_task_id: str,
    fold: int,
    seed: int,
) -> list[dict[str, Any]]:
    model_path = project_path(
        config, "outputs", "models", f"{r6_task_id}.joblib"
    )
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Missing R6 model artifact for coefficient audit: {model_path}"
        )
    payload = joblib.load(model_path)
    if not {"model", "processor", "features"}.issubset(payload):
        raise KeyError(
            f"R6 model payload is incomplete: {model_path.name}"
        )
    model = payload["model"]
    processor = payload["processor"]
    coefficients = np.asarray(model.coef_, dtype=float).reshape(-1)
    names = np.asarray(processor.get_feature_names_out(), dtype=str)
    if len(coefficients) != len(names):
        raise ValueError(
            f"Coefficient/name mismatch in {model_path.name}: "
            f"{len(coefficients)} != {len(names)}"
        )
    total_l2 = float(np.linalg.norm(coefficients))
    rows: list[dict[str, Any]] = []
    for level in NEWS_LEVELS:
        mask = np.char.find(
            names.astype(str), f"softproto__{level}__"
        ) >= 0
        values = coefficients[mask]
        if not len(values):
            raise KeyError(
                f"Model {r6_task_id} lacks coefficients for level={level}."
            )
        level_l2 = float(np.linalg.norm(values))
        rows.append(
            {
                "record_type": "ridge_text_coefficient_norm",
                "fold": fold,
                "prototype_seed": seed,
                "model_seed": seed,
                "news_level": level,
                "train_n": np.nan,
                "validation_n": np.nan,
                "train_news_rate": np.nan,
                "validation_news_rate": np.nan,
                "news_rate_change": np.nan,
                "train_mean_feature_l2": np.nan,
                "validation_mean_feature_l2": np.nan,
                "feature_l2_ratio_validation_to_train": np.nan,
                "jensen_shannon_divergence": np.nan,
                "mean_absolute_standardized_mean_difference": np.nan,
                "max_absolute_standardized_mean_difference": np.nan,
                "coefficient_count": len(values),
                "coefficient_l1": float(np.abs(values).sum()),
                "coefficient_l2": level_l2,
                "coefficient_max_abs": float(np.abs(values).max()),
                "coefficient_l2_share_of_model": (
                    level_l2 / total_l2 if total_l2 > 0 else np.nan
                ),
                "r6_task_id": r6_task_id,
                "evidence_scope": "post_hoc_validation_audit",
                "locked_test_used": False,
            }
        )
    return rows


def _distribution_shift(
    config: Mapping[str, Any],
    predictions: pd.DataFrame,
    feature_cache: Mapping[
        tuple[int, int], tuple[pd.DataFrame, dict[str, list[str]]]
    ],
) -> pd.DataFrame:
    epsilon = float(
        _profile(config).get("audit_distribution_epsilon", 1.0e-8)
    )
    market = read_table(
        project_path(
            config,
            "data",
            "processed",
            "original_market_targets.parquet",
        )
    )
    validate_columns(
        market,
        ("ticker", "feature_date", "target_date", "volatility_level"),
        "original market targets",
    )
    market["feature_date"] = _parse_mixed_datetime(
        market["feature_date"],
        column_name="market_targets.feature_date",
    ).dt.normalize()
    market["target_date"] = _parse_mixed_datetime(
        market["target_date"],
        column_name="market_targets.target_date",
    ).dt.normalize()
    rows: list[dict[str, Any]] = []
    r6_tasks = (
        predictions.loc[
            predictions["representation"].astype(str).eq("R6")
        ][["fold", "seed", "task_id"]]
        .drop_duplicates()
        .set_index(["fold", "seed"])["task_id"]
        .to_dict()
    )
    for (fold, seed), (features, by_level) in feature_cache.items():
        train = features.loc[
            features["split"].astype(str).eq("train")
        ]
        validation = features.loc[
            features["split"].astype(str).eq("validation")
        ]
        if train.empty or validation.empty:
            raise ValueError(
                f"Empty R6 feature split for fold={fold}, seed={seed}."
            )
        feature_targets = features[
            ["ticker", "feature_date", "split"]
        ].merge(
            market[
                [
                    "ticker",
                    "feature_date",
                    "target_date",
                    "volatility_level",
                ]
            ],
            on=["ticker", "feature_date"],
            how="left",
            validate="one_to_one",
        )
        if feature_targets[
            ["target_date", "volatility_level"]
        ].isna().any().any():
            raise ValueError(
                f"Missing target data for fold={fold}, seed={seed}."
            )
        train_target = feature_targets.loc[
            feature_targets["split"].astype(str).eq("train"),
            "volatility_level",
        ].to_numpy(dtype=float)
        validation_target = feature_targets.loc[
            feature_targets["split"].astype(str).eq("validation"),
            "volatility_level",
        ].to_numpy(dtype=float)
        train_target_dates = feature_targets.loc[
            feature_targets["split"].astype(str).eq("train"),
            "target_date",
        ]
        validation_target_dates = feature_targets.loc[
            feature_targets["split"].astype(str).eq("validation"),
            "target_date",
        ]
        if not train_target_dates.max() < validation_target_dates.min():
            raise AssertionError(
                f"Target chronology failed for fold={fold}, seed={seed}."
            )
        target_scale = np.sqrt(
            0.5
            * (
                np.var(train_target)
                + np.var(validation_target)
            )
        )
        rows.append(
            {
                "record_type": "target_distribution_shift",
                "fold": fold,
                "prototype_seed": seed,
                "model_seed": seed,
                "news_level": "all",
                "train_n": len(train_target),
                "validation_n": len(validation_target),
                "train_target_mean": float(np.mean(train_target)),
                "validation_target_mean": float(
                    np.mean(validation_target)
                ),
                "train_target_std": float(np.std(train_target)),
                "validation_target_std": float(
                    np.std(validation_target)
                ),
                "train_target_q50": float(
                    np.quantile(train_target, 0.50)
                ),
                "validation_target_q50": float(
                    np.quantile(validation_target, 0.50)
                ),
                "train_target_q90": float(
                    np.quantile(train_target, 0.90)
                ),
                "validation_target_q90": float(
                    np.quantile(validation_target, 0.90)
                ),
                "target_standardized_mean_difference": float(
                    abs(
                        np.mean(validation_target)
                        - np.mean(train_target)
                    )
                    / (target_scale + epsilon)
                ),
                "r6_task_id": r6_tasks[(fold, seed)],
                "evidence_scope": "post_hoc_train_validation_audit",
                "locked_test_used": False,
            }
        )
        for level, columns in by_level.items():
            train_matrix = train[columns].to_numpy(dtype=float)
            validation_matrix = validation[columns].to_numpy(dtype=float)
            train_norm = np.linalg.norm(train_matrix, axis=1)
            validation_norm = np.linalg.norm(validation_matrix, axis=1)
            train_mean = train_matrix.mean(axis=0)
            validation_mean = validation_matrix.mean(axis=0)
            pooled_scale = np.sqrt(
                0.5
                * (
                    train_matrix.var(axis=0)
                    + validation_matrix.var(axis=0)
                )
            )
            standardized_difference = np.abs(
                validation_mean - train_mean
            ) / (pooled_scale + epsilon)
            rows.append(
                {
                    "record_type": "r6_feature_distribution_shift",
                    "fold": fold,
                    "prototype_seed": seed,
                    "model_seed": seed,
                    "news_level": level,
                    "train_n": len(train),
                    "validation_n": len(validation),
                    "train_news_rate": float(
                        (np.abs(train_matrix).sum(axis=1) > 1.0e-12).mean()
                    ),
                    "validation_news_rate": float(
                        (
                            np.abs(validation_matrix).sum(axis=1)
                            > 1.0e-12
                        ).mean()
                    ),
                    "news_rate_change": float(
                        (
                            np.abs(validation_matrix).sum(axis=1)
                            > 1.0e-12
                        ).mean()
                        - (
                            np.abs(train_matrix).sum(axis=1)
                            > 1.0e-12
                        ).mean()
                    ),
                    "train_mean_feature_l2": float(train_norm.mean()),
                    "validation_mean_feature_l2": float(
                        validation_norm.mean()
                    ),
                    "feature_l2_ratio_validation_to_train": (
                        float(validation_norm.mean() / train_norm.mean())
                        if train_norm.mean() > epsilon
                        else np.nan
                    ),
                    "jensen_shannon_divergence": _safe_js_divergence(
                        train_matrix.sum(axis=0),
                        validation_matrix.sum(axis=0),
                    ),
                    "mean_absolute_standardized_mean_difference": float(
                        standardized_difference.mean()
                    ),
                    "max_absolute_standardized_mean_difference": float(
                        standardized_difference.max()
                    ),
                    "coefficient_count": np.nan,
                    "coefficient_l1": np.nan,
                    "coefficient_l2": np.nan,
                    "coefficient_max_abs": np.nan,
                    "coefficient_l2_share_of_model": np.nan,
                    "r6_task_id": r6_tasks[(fold, seed)],
                    "evidence_scope": "post_hoc_validation_audit",
                    "locked_test_used": False,
                }
            )
        rows.extend(
            _coefficient_rows(
                config,
                str(r6_tasks[(fold, seed)]),
                fold,
                seed,
            )
        )
    return pd.DataFrame(rows).sort_values(
        ["record_type", "fold", "prototype_seed", "news_level"],
        kind="mergesort",
    ).reset_index(drop=True)


def _resolve_fold_artifact(
    config: Mapping[str, Any],
    raw_path: Any,
    fold: int,
) -> Path:
    source = Path(str(raw_path))
    if source.is_file():
        return source.resolve()
    matches = [
        path
        for path in shared_root(config).rglob(source.name)
        if f"fold_{fold}" in path.parts
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Cannot resolve fold={fold} artifact {raw_path!r}; "
            f"found {len(matches)} matches."
        )
    return matches[0].resolve()


def _prototype_bundles(
    config: Mapping[str, Any],
    logger: Any,
) -> dict[tuple[int, int, str], dict[str, Any]]:
    profile = _profile(config)
    manifest_path = resolve_shared_file(
        config,
        "prototype_fold_manifest.csv",
        kinds=("processed", "tables"),
    )
    embedding_path = resolve_shared_file(
        config,
        "event_embeddings.npy",
        kinds=("embeddings",),
    )
    metadata_path = resolve_shared_file(
        config,
        "event_embedding_metadata.csv",
        kinds=("embeddings",),
    )
    assert manifest_path is not None
    assert embedding_path is not None
    assert metadata_path is not None
    manifest = read_table(manifest_path)
    metadata = read_table(metadata_path)
    validate_columns(
        manifest,
        (
            "fold_id",
            "prototype_seed",
            "news_level",
            "representation_variant_family",
            "eligible",
            "assignment_path",
            "k",
            "fit_scope",
            "train_end",
            "validation_start",
        ),
        "prototype fold manifest",
    )
    validate_columns(
        metadata,
        ("event_id", "embedding_index"),
        "event embedding metadata",
    )
    if metadata["event_id"].astype(str).duplicated().any():
        raise ValueError("Embedding metadata contains duplicate event IDs.")
    event_lookup = (
        metadata.assign(event_id=metadata["event_id"].astype(str))
        .set_index("event_id")["embedding_index"]
        .astype(int)
        .to_dict()
    )
    embeddings = np.load(embedding_path, mmap_mode="r")
    selected = manifest.loc[
        pd.to_numeric(manifest["fold_id"], errors="coerce").isin(
            profile["folds"]
        )
        & pd.to_numeric(
            manifest["prototype_seed"], errors="coerce"
        ).isin(profile["seeds"])
        & manifest["representation_variant_family"].astype(str).eq(
            str(profile["representation_variant_family"])
        )
        & manifest["news_level"].astype(str).isin(NEWS_LEVELS)
        & _truthy_mask(manifest["eligible"])
        & manifest["fit_scope"].astype(str).eq("fold_train_only")
    ].copy()
    expected = {
        (int(fold), int(seed), level)
        for fold in profile["folds"]
        for seed in profile["seeds"]
        for level in NEWS_LEVELS
    }
    observed = {
        (int(row.fold_id), int(row.prototype_seed), str(row.news_level))
        for row in selected.itertuples(index=False)
    }
    if observed != expected or len(selected) != len(expected):
        raise ValueError(
            "Prototype drift audit requires exactly one eligible assignment "
            f"per fold/seed/level; missing={sorted(expected - observed)}."
        )
    train_end = _parse_mixed_datetime(
        selected["train_end"],
        column_name="prototype_fold_manifest.train_end",
    )
    validation_start = _parse_mixed_datetime(
        selected["validation_start"],
        column_name="prototype_fold_manifest.validation_start",
    )
    if not (train_end < validation_start).all():
        raise AssertionError(
            "A prototype artifact violates train_end < validation_start."
        )

    bundles: dict[tuple[int, int, str], dict[str, Any]] = {}
    total = len(selected)
    for index, row in enumerate(selected.itertuples(index=False), start=1):
        fold = int(row.fold_id)
        seed = int(row.prototype_seed)
        level = str(row.news_level)
        if index == 1 or index % 5 == 0 or index == total:
            logger.info(
                "R6 audit prototype reconstruction | artifact %d/%d | "
                "fold=%d | seed=%d | level=%s",
                index,
                total,
                fold,
                seed,
                level,
            )
        assignment_path = _resolve_fold_artifact(
            config, row.assignment_path, fold
        )
        arrays = np.load(assignment_path, allow_pickle=False)
        required_arrays = {
            "event_ids",
            "fold_role",
            "hard_cluster_id",
        }
        missing_arrays = sorted(required_arrays.difference(arrays.files))
        if missing_arrays:
            raise KeyError(
                f"{assignment_path.name} lacks arrays: {missing_arrays}"
            )
        event_ids = arrays["event_ids"].astype(str)
        roles = arrays["fold_role"].astype(int)
        labels = arrays["hard_cluster_id"].astype(int)
        train_mask = roles == 0
        train_event_ids = event_ids[train_mask]
        train_labels = labels[train_mask]
        unknown_ids = sorted(
            set(train_event_ids).difference(event_lookup)
        )
        if unknown_ids:
            raise ValueError(
                f"Prototype assignments contain unknown events: "
                f"{unknown_ids[:5]}"
            )
        embedding_indices = np.fromiter(
            (event_lookup[event_id] for event_id in train_event_ids),
            dtype=np.int64,
            count=len(train_event_ids),
        )
        vectors = np.asarray(embeddings[embedding_indices], dtype=np.float32)
        k = int(row.k)
        centroids = np.zeros((k, vectors.shape[1]), dtype=np.float32)
        usage = np.zeros(k, dtype=float)
        for cluster in range(k):
            cluster_mask = train_labels == cluster
            usage[cluster] = int(cluster_mask.sum())
            if not cluster_mask.any():
                raise ValueError(
                    f"Dead cluster in eligible fold artifact: fold={fold}, "
                    f"seed={seed}, level={level}, cluster={cluster}."
                )
            centroid = vectors[cluster_mask].mean(axis=0)
            norm = float(np.linalg.norm(centroid))
            if norm <= 0:
                raise ValueError("Prototype centroid has zero L2 norm.")
            centroids[cluster] = centroid / norm
        usage /= usage.sum()
        bundles[(fold, seed, level)] = {
            "centroids": centroids,
            "usage": usage,
            "labels": dict(zip(train_event_ids, train_labels)),
            "n_train_events": len(train_event_ids),
            "assignment_path": str(assignment_path),
        }
    return bundles


def _prototype_comparison(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, Any]:
    similarity = np.asarray(left["centroids"]) @ np.asarray(
        right["centroids"]
    ).T
    left_index, right_index = linear_sum_assignment(-similarity)
    matched = similarity[left_index, right_index]
    left_usage = np.asarray(left["usage"])[left_index]
    right_usage = np.asarray(right["usage"])[right_index]
    shared_ids = sorted(
        set(left["labels"]).intersection(right["labels"])
    )
    ari = (
        adjusted_rand_score(
            [left["labels"][event_id] for event_id in shared_ids],
            [right["labels"][event_id] for event_id in shared_ids],
        )
        if len(shared_ids) >= 2
        else np.nan
    )
    return {
        "matched_centroid_cosine_mean": float(matched.mean()),
        "matched_centroid_cosine_min": float(matched.min()),
        "matched_centroid_cosine_std": float(matched.std(ddof=0)),
        "matched_usage_js_divergence": _safe_js_divergence(
            left_usage, right_usage
        ),
        "shared_event_count": len(shared_ids),
        "adjusted_rand_index_on_shared_events": ari,
        "left_train_event_count": int(left["n_train_events"]),
        "right_train_event_count": int(right["n_train_events"]),
    }


def _prototype_drift(
    config: Mapping[str, Any],
    logger: Any,
) -> pd.DataFrame:
    profile = _profile(config)
    folds = tuple(int(value) for value in profile["folds"])
    seeds = tuple(int(value) for value in profile["seeds"])
    logger.info("R6 audit loading fold prototype assignments and embeddings.")
    bundles = _prototype_bundles(config, logger)
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for level in NEWS_LEVELS:
            for left_fold, right_fold in zip(folds[:-1], folds[1:]):
                rows.append(
                    {
                        "comparison_type": "temporal_same_seed",
                        "news_level": level,
                        "left_fold": left_fold,
                        "right_fold": right_fold,
                        "left_seed": seed,
                        "right_seed": seed,
                        **_prototype_comparison(
                            bundles[(left_fold, seed, level)],
                            bundles[(right_fold, seed, level)],
                        ),
                        "evidence_scope": "post_hoc_train_artifact_audit",
                        "locked_test_used": False,
                    }
                )
    reference_seed = seeds[0]
    for fold in folds:
        for level in NEWS_LEVELS:
            for comparison_seed in seeds[1:]:
                rows.append(
                    {
                        "comparison_type": "seed_stability_same_fold",
                        "news_level": level,
                        "left_fold": fold,
                        "right_fold": fold,
                        "left_seed": reference_seed,
                        "right_seed": comparison_seed,
                        **_prototype_comparison(
                            bundles[(fold, reference_seed, level)],
                            bundles[(fold, comparison_seed, level)],
                        ),
                        "evidence_scope": "post_hoc_train_artifact_audit",
                        "locked_test_used": False,
                    }
                )
    return pd.DataFrame(rows).sort_values(
        [
            "comparison_type",
            "news_level",
            "left_fold",
            "left_seed",
            "right_seed",
        ],
        kind="mergesort",
    ).reset_index(drop=True)


def _audit_summary(
    config: Mapping[str, Any],
    cell_results: pd.DataFrame,
    ticker: pd.DataFrame,
    news_day: pd.DataFrame,
    news_level: pd.DataFrame,
    shift: pd.DataFrame,
    drift: pd.DataFrame,
) -> pd.DataFrame:
    profile = _profile(config)
    threshold = float(config["decision"]["minimum_relative_gain"])
    minimum_win_rate = float(config["decision"]["minimum_fold_win_rate"])
    rows: list[dict[str, Any]] = []
    comparison_passes: dict[str, bool] = {}
    for reference, group in cell_results.groupby(
        "reference", sort=True, observed=True
    ):
        values = pd.to_numeric(
            group["relative_gain"], errors="coerce"
        ).dropna()
        fold_means = (
            group.groupby("fold", sort=True, observed=True)[
                "relative_gain"
            ]
            .mean()
            .to_dict()
        )
        win_rate = float((values > 0).mean())
        passed = bool(
            len(values) == 15
            and float(values.mean()) >= threshold
            and win_rate >= minimum_win_rate
            and all(float(value) > 0 for value in fold_means.values())
        )
        comparison_passes[str(reference)] = passed
        rows.append(
            {
                "record_type": "overall_comparison",
                "key": str(reference),
                "fold": "all",
                "news_level": "all",
                "value": float(values.mean()),
                "secondary_value": win_rate,
                "threshold": threshold,
                "passed": passed,
                "details": json.dumps(
                    {
                        "fold_mean_gains": {
                            str(key): float(value)
                            for key, value in fold_means.items()
                        },
                        "n": len(values),
                    },
                    sort_keys=True,
                ),
                "interpretation": (
                    "R6 passed the locked comparator audit."
                    if passed
                    else "R6 failed mean-gain, win-rate, or all-fold stability."
                ),
                "evidence_scope": "post_hoc_validation_audit",
                "locked_test_used": False,
            }
        )
        for fold, fold_group in group.groupby(
            "fold", sort=True, observed=True
        ):
            rows.append(
                {
                    "record_type": "fold_comparison",
                    "key": str(reference),
                    "fold": int(fold),
                    "news_level": "all",
                    "value": float(fold_group["relative_gain"].mean()),
                    "secondary_value": float(
                        (fold_group["relative_gain"] > 0).mean()
                    ),
                    "threshold": 0.0,
                    "passed": bool(
                        fold_group["relative_gain"].mean() > 0
                    ),
                    "details": json.dumps(
                        {
                            "min_gain": float(
                                fold_group["relative_gain"].min()
                            ),
                            "max_gain": float(
                                fold_group["relative_gain"].max()
                            ),
                        },
                        sort_keys=True,
                    ),
                    "interpretation": "Fold-specific paired-seed result.",
                    "evidence_scope": "post_hoc_validation_audit",
                    "locked_test_used": False,
                }
            )

    r0_ticker = ticker.loc[
        ticker["reference"].astype(str).eq("R0")
        & _truthy_mask(ticker["descriptive_group_eligible"])
    ]
    earliest_fold = min(int(value) for value in profile["folds"])
    earliest = r0_ticker.loc[r0_ticker["fold"].eq(earliest_fold)]
    earliest_loss_fraction = float(
        (earliest["relative_gain"] < 0).mean()
    )
    for fold, group in r0_ticker.groupby(
        "fold", sort=True, observed=True
    ):
        rows.append(
            {
                "record_type": "ticker_breadth",
                "key": "R0",
                "fold": int(fold),
                "news_level": "all",
                "value": float((group["relative_gain"] > 0).mean()),
                "secondary_value": float(group["relative_gain"].mean()),
                "threshold": 0.5,
                "passed": bool(
                    (group["relative_gain"] > 0).mean() >= 0.5
                ),
                "details": json.dumps(
                    {
                        "ticker_seed_cells": len(group),
                        "loss_fraction": float(
                            (group["relative_gain"] < 0).mean()
                        ),
                    },
                    sort_keys=True,
                ),
                "interpretation": "Breadth of R6 wins across ticker-seed cells.",
                "evidence_scope": "post_hoc_validation_audit",
                "locked_test_used": False,
            }
        )

    r0_news = news_day.loc[
        news_day["reference"].astype(str).eq("R0")
        & _truthy_mask(news_day["descriptive_group_eligible"])
    ]
    news_means = (
        r0_news.groupby(
            ["fold", "condition"], sort=True, observed=True
        )["relative_gain"]
        .mean()
        .to_dict()
    )
    gate_pattern = bool(
        any(
            news_means.get((fold, "has_any_news"), np.nan) > 0
            and news_means.get((fold, "no_news"), np.nan) < 0
            for fold in profile["folds"]
        )
    )
    for (fold, condition), value in news_means.items():
        rows.append(
            {
                "record_type": "news_gate",
                "key": str(condition),
                "fold": int(fold),
                "news_level": "all",
                "value": float(value),
                "secondary_value": np.nan,
                "threshold": 0.0,
                "passed": bool(value > 0),
                "details": "",
                "interpretation": (
                    "Descriptive R6 gain vs R0 conditional on news presence."
                ),
                "evidence_scope": "post_hoc_validation_audit",
                "locked_test_used": False,
            }
        )

    r0_levels = news_level.loc[
        news_level["reference"].astype(str).eq("R0")
        & news_level["condition"].astype(str).eq("has_level_news")
        & _truthy_mask(news_level["descriptive_group_eligible"])
    ]
    level_candidates: list[str] = []
    for level, group in r0_levels.groupby(
        "news_level", sort=True, observed=True
    ):
        fold_means = group.groupby("fold")["relative_gain"].mean()
        mean_gain = float(group["relative_gain"].mean())
        candidate = bool(
            mean_gain >= threshold
            and len(fold_means) == len(profile["folds"])
            and (fold_means > 0).all()
        )
        if candidate:
            level_candidates.append(str(level))
        rows.append(
            {
                "record_type": "news_level_candidate",
                "key": "R0",
                "fold": "all",
                "news_level": str(level),
                "value": mean_gain,
                "secondary_value": float((group["relative_gain"] > 0).mean()),
                "threshold": threshold,
                "passed": candidate,
                "details": json.dumps(
                    {
                        str(key): float(value)
                        for key, value in fold_means.to_dict().items()
                    },
                    sort_keys=True,
                ),
                "interpretation": (
                    "Post-hoc level-specific candidate; requires new confirmation."
                ),
                "evidence_scope": "post_hoc_validation_audit",
                "locked_test_used": False,
            }
        )

    temporal_drift = drift.loc[
        drift["comparison_type"].astype(str).eq("temporal_same_seed")
    ]
    for level, group in temporal_drift.groupby(
        "news_level", sort=True, observed=True
    ):
        rows.append(
            {
                "record_type": "prototype_temporal_stability",
                "key": "matched_centroid_cosine",
                "fold": "all",
                "news_level": str(level),
                "value": float(
                    group["matched_centroid_cosine_mean"].mean()
                ),
                "secondary_value": float(
                    group[
                        "adjusted_rand_index_on_shared_events"
                    ].mean()
                ),
                "threshold": float(
                    profile.get(
                        "audit_min_matched_centroid_cosine", 0.80
                    )
                ),
                "passed": bool(
                    group["matched_centroid_cosine_mean"].mean()
                    >= float(
                        profile.get(
                            "audit_min_matched_centroid_cosine", 0.80
                        )
                    )
                ),
                "details": json.dumps(
                    {
                        "mean_usage_js_divergence": float(
                            group[
                                "matched_usage_js_divergence"
                            ].mean()
                        ),
                        "comparisons": len(group),
                    },
                    sort_keys=True,
                ),
                "interpretation": (
                    "Prototype drift measured in the original frozen "
                    "embedding space."
                ),
                "evidence_scope": "post_hoc_train_artifact_audit",
                "locked_test_used": False,
            }
        )

    shift_features = shift.loc[
        shift["record_type"].astype(str).eq(
            "r6_feature_distribution_shift"
        )
    ]
    for level, group in shift_features.groupby(
        "news_level", sort=True, observed=True
    ):
        rows.append(
            {
                "record_type": "feature_distribution_shift",
                "key": "mean_absolute_smd",
                "fold": "all",
                "news_level": str(level),
                "value": float(
                    group[
                        "mean_absolute_standardized_mean_difference"
                    ].mean()
                ),
                "secondary_value": float(
                    group["jensen_shannon_divergence"].mean()
                ),
                "threshold": float(
                    profile.get("audit_max_mean_absolute_smd", 0.25)
                ),
                "passed": bool(
                    group[
                        "mean_absolute_standardized_mean_difference"
                    ].mean()
                    <= float(
                        profile.get(
                            "audit_max_mean_absolute_smd", 0.25
                        )
                    )
                ),
                "details": "",
                "interpretation": "Train-to-validation R6 feature shift.",
                "evidence_scope": "post_hoc_train_validation_audit",
                "locked_test_used": False,
            }
        )

    broad_earliest_failure = earliest_loss_fraction >= float(
        profile.get("audit_broad_loss_fraction", 0.70)
    )
    if broad_earliest_failure:
        recommendation = "MOVE-TO-SPIKE-OR-MAGNITUDE"
        reason = (
            f"Fold {earliest_fold} loss is broad across "
            f"{earliest_loss_fraction:.1%} of ticker-seed cells."
        )
    elif gate_pattern:
        recommendation = "TRY-NEWS-GATING-EXPLORATORY"
        reason = (
            "R6 gain changes sign between news and no-news strata in at "
            "least one fold."
        )
    elif level_candidates:
        recommendation = "TRY-LEVEL-SPECIFIC-EXPLORATORY"
        reason = (
            "At least one post-hoc news level clears the descriptive gain "
            "and all-fold sign checks."
        )
    elif comparison_passes.get("R0", False):
        recommendation = "DESIGN-NEW-UNTOUCHED-CONFIRMATION"
        reason = "R6 passed price-only but not every locked comparator."
    else:
        recommendation = "STOP-DIRECT"
        reason = "R6 does not show a stable direct-volatility gain."
    rows.append(
        {
            "record_type": "final_recommendation",
            "key": recommendation,
            "fold": "all",
            "news_level": (
                ",".join(level_candidates) if level_candidates else "none"
            ),
            "value": earliest_loss_fraction,
            "secondary_value": np.nan,
            "threshold": float(
                profile.get("audit_broad_loss_fraction", 0.70)
            ),
            "passed": False,
            "details": json.dumps(
                {
                    "gate_pattern_detected": gate_pattern,
                    "post_hoc_level_candidates": level_candidates,
                    "confirmatory_decision_remains": "CONFIRMATORY-FAIL",
                },
                sort_keys=True,
            ),
            "interpretation": reason,
            "evidence_scope": "post_hoc_audit_not_confirmatory",
            "locked_test_used": False,
        }
    )
    return pd.DataFrame(rows)


def run(config: Mapping[str, Any]) -> dict[str, Path]:
    """Execute the R6 failure audit without reading the locked test."""

    logger = get_logger(
        "r6_failure_audit",
        config,
        project_path(config, "outputs", "logs", "r6_failure_audit.log"),
    )
    predictions = _locked_predictions(config)
    if predictions["evaluation_split"].astype(str).ne("validation").any():
        raise AssertionError("R6 audit received a non-validation prediction.")
    (
        ticker,
        news_day,
        news_level,
        cell_results,
        feature_cache,
    ) = _prediction_diagnostics(config, predictions, logger)
    shift = _distribution_shift(config, predictions, feature_cache)
    drift = _prototype_drift(config, logger)
    summary = _audit_summary(
        config,
        cell_results,
        ticker,
        news_day,
        news_level,
        shift,
        drift,
    )
    frames = {
        "ticker": ticker,
        "news_day": news_day,
        "news_level": news_level,
        "distribution_shift": shift,
        "prototype_drift": drift,
        "summary": summary,
    }
    empty = sorted(name for name, frame in frames.items() if frame.empty)
    if empty:
        raise ValueError(f"R6 audit produced empty outputs: {empty}")
    tables = project_path(config, "outputs", "tables")
    paths = {
        "ticker": tables / OUTPUT_NAMES[0],
        "news_day": tables / OUTPUT_NAMES[1],
        "news_level": tables / OUTPUT_NAMES[2],
        "distribution_shift": tables / OUTPUT_NAMES[3],
        "prototype_drift": tables / OUTPUT_NAMES[4],
        "summary": tables / OUTPUT_NAMES[5],
    }
    atomic_write_csv(ticker, paths["ticker"], index=False)
    atomic_write_csv(news_day, paths["news_day"], index=False)
    atomic_write_csv(news_level, paths["news_level"], index=False)
    atomic_write_csv(shift, paths["distribution_shift"], index=False)
    atomic_write_csv(drift, paths["prototype_drift"], index=False)
    atomic_write_csv(summary, paths["summary"], index=False)
    final = summary.loc[
        summary["record_type"].astype(str).eq("final_recommendation")
    ]
    if len(final) != 1:
        raise AssertionError("R6 audit must produce one final recommendation.")
    logger.info(
        "R6 failure audit complete | recommendation=%s | reason=%s",
        final.iloc[0]["key"],
        final.iloc[0]["interpretation"],
    )
    return paths
