"""Fold-safe prototype cross-attention without structured-event extraction.

The experiment consumes the existing daily R1/R6/R9/R10/R11 feature panels.
Price/HAR features form the market query.  Each (news level, prototype ID)
cell is retained as a token, so attention can condition prototype relevance on
the current market state instead of flattening the full R6 vector.
"""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.modeling import representation_row
from src.utils import (
    atomic_write_csv,
    atomic_write_json,
    binary_metrics,
    load_representation_frame,
    price_feature_columns,
    project_path,
    read_table,
    regression_metrics,
    set_global_seed,
    stable_id,
    task_split_frames,
    validate_columns,
    write_table,
)


NEWS_LEVELS = ("macro", "sector", "related", "target")
RIDGE_MODELS = {"PRICE_RIDGE", "PRICE_META_RIDGE", "R6_RIDGE"}
NEURAL_MODELS = {
    "PRICE_MLP",
    "META_BASIC_MLP",
    "R6_META_CONCAT_MLP",
    "R6_XATTN",
    "R6_META_XATTN",
    "R9_META_XATTN",
    "R10_META_XATTN",
    "R11_META_XATTN",
}
ATTENTION_MODELS = {
    "R6_XATTN",
    "R6_META_XATTN",
    "R9_META_XATTN",
    "R10_META_XATTN",
    "R11_META_XATTN",
}
MODEL_PROTOTYPE_SOURCE = {
    "PRICE_RIDGE": "R6",
    "PRICE_META_RIDGE": "R6",
    "R6_RIDGE": "R6",
    "PRICE_MLP": "R6",
    "META_BASIC_MLP": "R6",
    "R6_META_CONCAT_MLP": "R6",
    "R6_XATTN": "R6",
    "R6_META_XATTN": "R6",
    "R9_META_XATTN": "R9",
    "R10_META_XATTN": "R10",
    "R11_META_XATTN": "R11",
}


@dataclass
class NumericTransform:
    imputer: SimpleImputer
    scaler: StandardScaler

    def transform(self, values: pd.DataFrame) -> np.ndarray:
        array = self.imputer.transform(values)
        return np.asarray(self.scaler.transform(array), dtype=np.float32)


@dataclass
class PreparedArrays:
    train_frame: pd.DataFrame
    validation_frame: pd.DataFrame
    price_train: np.ndarray
    price_validation: np.ndarray
    meta_train: np.ndarray
    meta_validation: np.ndarray
    prototype_train: np.ndarray
    prototype_validation: np.ndarray
    prototype_mask_train: np.ndarray
    prototype_mask_validation: np.ndarray
    ticker_train: np.ndarray
    ticker_validation: np.ndarray
    y_train: np.ndarray
    y_validation: np.ndarray
    price_columns: list[str]
    metadata_columns: list[str]
    prototype_columns: list[str]
    k_per_level: int
    target_metadata: dict[str, Any]
    preprocessors: dict[str, Any]


def output_root(
    config: Mapping[str, Any], profile: Mapping[str, Any]
) -> Path:
    return project_path(config, str(profile["output_directory"]))


def ensure_output_directories(
    config: Mapping[str, Any], profile: Mapping[str, Any]
) -> Path:
    root = output_root(config, profile)
    for name in ("checkpoints", "models", "tables", "logs"):
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def _model_source(
    model_name: str,
    profile: Mapping[str, Any],
) -> str:
    if model_name not in MODEL_PROTOTYPE_SOURCE:
        raise ValueError(f"Unknown cross-attention model: {model_name}")
    return str(
        profile.get("prototype_source", {}).get(
            model_name, MODEL_PROTOTYPE_SOURCE[model_name]
        )
    )


def build_plan(
    config: Mapping[str, Any],
    profile: Mapping[str, Any],
    *,
    quick: bool = False,
    targets: Sequence[str] | None = None,
    models: Sequence[str] | None = None,
    folds: Sequence[int] | None = None,
    seeds: Sequence[int] | None = None,
) -> pd.DataFrame:
    selected_targets = list(targets or profile["targets"])
    selected_models = list(
        models
        or (profile["quick_models"] if quick else profile["models"])
    )
    selected_folds = [int(value) for value in (folds or profile["folds"])]
    selected_seeds = [int(value) for value in (seeds or profile["seeds"])]
    allowed_targets = {"volatility_level", "spike_q90"}
    unknown_targets = sorted(set(selected_targets).difference(allowed_targets))
    if unknown_targets:
        raise ValueError(f"Unsupported targets: {unknown_targets}")
    unknown_models = sorted(
        set(selected_models).difference(RIDGE_MODELS | NEURAL_MODELS)
    )
    if unknown_models:
        raise ValueError(f"Unsupported models: {unknown_models}")
    rows: list[dict[str, Any]] = []
    for target in selected_targets:
        for fold in selected_folds:
            for seed in selected_seeds:
                for model_name in selected_models:
                    payload = {
                        "target": str(target),
                        "fold": int(fold),
                        "seed": int(seed),
                        "model": str(model_name),
                        "prototype_source": _model_source(
                            model_name, profile
                        ),
                        "representation_variant_family": str(
                            profile["representation_variant_family"]
                        ),
                        "quick": bool(quick),
                    }
                    rows.append(
                        {
                            "task_id": stable_id(
                                {
                                    "experiment": "prototype_cross_attention",
                                    **payload,
                                },
                                prefix="xattn",
                            ),
                            **payload,
                        }
                    )
    plan = pd.DataFrame(rows)
    if plan.empty or plan["task_id"].duplicated().any():
        raise AssertionError("Cross-attention task plan is empty or duplicated.")
    return plan


def validate_plan_artifacts(
    config: Mapping[str, Any],
    profile: Mapping[str, Any],
    plan: pd.DataFrame,
) -> None:
    """Fail before training if a fold-safe prototype panel is unavailable."""

    family = str(profile["representation_variant_family"])
    cells = plan[
        ["prototype_source", "fold", "seed"]
    ].drop_duplicates()
    for row in cells.itertuples(index=False):
        representation_row(
            config,
            str(row.prototype_source),
            family,
            fold=int(row.fold),
            seed=int(row.seed),
            representation_variant_family=family,
        )
    # R1 is pure event-arrival metadata and is independent of prototype fitting.
    representation_row(
        config,
        "R1",
        "selected_default",
        fold=int(plan["fold"].iloc[0]),
        seed=int(plan["seed"].iloc[0]),
    )


def _load_feature_frame(
    config: Mapping[str, Any],
    profile: Mapping[str, Any],
    representation: str,
    fold: int,
    seed: int,
) -> tuple[pd.DataFrame, str]:
    if representation == "R1":
        row = representation_row(
            config,
            representation,
            "selected_default",
            fold=fold,
            seed=seed,
        )
    else:
        family = str(profile["representation_variant_family"])
        row = representation_row(
            config,
            representation,
            family,
            fold=fold,
            seed=seed,
            representation_variant_family=family,
        )
        if str(row.get("fit_scope", row.get("fit_split", ""))) != "fold_train_only":
            raise AssertionError(
                f"{representation} fold={fold}, seed={seed} is not fold-train-only."
            )
    frame = load_representation_frame(config, row, seed=seed)
    return frame, str(row.get("resolved_path", row.get("path", "")))


def _metadata_columns(
    frame: pd.DataFrame,
    profile: Mapping[str, Any],
) -> list[str]:
    levels = tuple(map(str, profile["news_levels"]))
    suffixes = set(map(str, profile["metadata_suffixes"]))
    columns = []
    for level in levels:
        prefix = f"meta__{level}__"
        columns.extend(
            column
            for column in frame.columns
            if column.startswith(prefix)
            and column.removeprefix(prefix) in suffixes
            and pd.api.types.is_numeric_dtype(frame[column])
        )
    if not columns:
        raise ValueError("No pure news-arrival metadata columns were found.")
    contaminated = [
        column
        for column in columns
        if any(
            token in column
            for token in ("entropy", "novelty", "distance", "prototype")
        )
    ]
    if contaminated:
        raise AssertionError(
            f"Metadata-only control contains prototype diagnostics: {contaminated}"
        )
    return list(dict.fromkeys(columns))


def prototype_columns_by_level(
    frame: pd.DataFrame,
    levels: Sequence[str],
) -> tuple[dict[str, list[str]], int]:
    blocks: dict[str, list[tuple[int, str]]] = {}
    for level in levels:
        token = f"__{level}__"
        candidates: list[tuple[int, str]] = []
        for column in frame.columns:
            if token not in column or column.startswith("meta__"):
                continue
            if not pd.api.types.is_numeric_dtype(frame[column]):
                continue
            try:
                prototype_id = int(column.rsplit("__", maxsplit=1)[-1])
            except ValueError:
                continue
            candidates.append((prototype_id, column))
        candidates.sort(key=lambda item: item[0])
        if not candidates:
            raise ValueError(f"No prototype columns found for news level {level!r}.")
        identifiers = [item[0] for item in candidates]
        if identifiers != list(range(len(identifiers))):
            raise ValueError(
                f"Non-contiguous prototype IDs for {level}: {identifiers}"
            )
        blocks[str(level)] = [item[1] for item in candidates]
    dimensions = {len(value) for value in blocks.values()}
    if len(dimensions) != 1:
        raise ValueError(
            f"News levels have inconsistent prototype dimensions: "
            f"{ {key: len(value) for key, value in blocks.items()} }"
        )
    return blocks, dimensions.pop()


def _fit_numeric(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    columns: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, NumericTransform]:
    if not columns:
        empty_train = np.empty((len(train), 0), dtype=np.float32)
        empty_validation = np.empty((len(validation), 0), dtype=np.float32)
        placeholder = NumericTransform(SimpleImputer(), StandardScaler())
        return empty_train, empty_validation, placeholder
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_values = imputer.fit_transform(train[list(columns)])
    validation_values = imputer.transform(validation[list(columns)])
    train_values = scaler.fit_transform(train_values)
    validation_values = scaler.transform(validation_values)
    return (
        np.asarray(train_values, dtype=np.float32),
        np.asarray(validation_values, dtype=np.float32),
        NumericTransform(imputer=imputer, scaler=scaler),
    )


def _ticker_matrix(
    values: pd.Series,
    configured_tickers: Sequence[str],
) -> np.ndarray:
    mapping = {
        str(ticker).upper(): index
        for index, ticker in enumerate(configured_tickers)
    }
    identifiers = values.astype(str).str.upper().map(mapping)
    if identifiers.isna().any():
        unknown = sorted(
            values.loc[identifiers.isna()].astype(str).unique().tolist()
        )
        raise ValueError(f"Unknown tickers in model frame: {unknown}")
    return np.eye(len(mapping), dtype=np.float32)[
        identifiers.to_numpy(dtype=int)
    ]


def _prototype_tensor(
    frame: pd.DataFrame,
    blocks: Mapping[str, Sequence[str]],
    levels: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    values = np.stack(
        [
            frame[list(blocks[level])]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
            for level in levels
        ],
        axis=1,
    )
    # A full level is masked only when no prototype mass is present. Individual
    # soft assignments remain visible as distinct tokens.
    active_level = np.abs(values).sum(axis=2) > 1.0e-10
    mask = np.repeat(active_level[:, :, None], values.shape[2], axis=2)
    return values, mask.reshape(len(frame), -1)


def _target_arrays(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    target: str,
    profile: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if target == "volatility_level":
        return (
            train["volatility_level"].to_numpy(dtype=np.float32),
            validation["volatility_level"].to_numpy(dtype=np.float32),
            {"target_type": "regression"},
        )
    if target != "spike_q90":
        raise ValueError(f"Unsupported target: {target}")
    quantile = float(profile.get("spike_quantile", 0.90))
    thresholds = (
        train.groupby("ticker", observed=True)["volatility_level"]
        .quantile(quantile)
        .to_dict()
    )

    def labels(frame: pd.DataFrame) -> np.ndarray:
        mapped = frame["ticker"].map(thresholds)
        if mapped.isna().any():
            raise ValueError("A validation ticker lacks a train spike threshold.")
        return (
            frame["volatility_level"].to_numpy(dtype=float)
            > mapped.to_numpy(dtype=float)
        ).astype(np.float32)

    return (
        labels(train),
        labels(validation),
        {
            "target_type": "classification",
            "spike_quantile": quantile,
            "ticker_thresholds": thresholds,
        },
    )


def prepare_arrays(
    config: Mapping[str, Any],
    profile: Mapping[str, Any],
    task: Mapping[str, Any],
) -> PreparedArrays:
    if bool(profile.get("locked_test", False)):
        raise AssertionError(
            "This validation experiment must not enable the locked test."
        )
    fold = int(task["fold"])
    seed = int(task["seed"])
    source = str(task["prototype_source"])
    market_path = project_path(
        config, "data", "processed", "original_market_targets.parquet"
    )
    market = read_table(market_path)
    validate_columns(
        market,
        ("ticker", "feature_date", "target_date", "volatility_level"),
        "original volatility targets",
    )
    market["ticker"] = market["ticker"].astype(str).str.upper()
    market["feature_date"] = pd.to_datetime(
        market["feature_date"], errors="raise"
    ).dt.normalize()
    market["target_date"] = pd.to_datetime(
        market["target_date"], errors="raise"
    ).dt.normalize()
    prototype_frame, prototype_path = _load_feature_frame(
        config, profile, source, fold, seed
    )
    metadata_frame, metadata_path = _load_feature_frame(
        config, profile, "R1", fold, seed
    )
    levels = tuple(map(str, profile["news_levels"]))
    blocks, k_per_level = prototype_columns_by_level(
        prototype_frame, levels
    )
    prototype_columns = [
        column for level in levels for column in blocks[level]
    ]
    metadata_columns = _metadata_columns(metadata_frame, profile)
    prototype_keep = prototype_frame[
        ["ticker", "feature_date", *prototype_columns]
    ].copy()
    metadata_keep = metadata_frame[
        ["ticker", "feature_date", *metadata_columns]
    ].copy()
    joined = market.merge(
        metadata_keep,
        on=["ticker", "feature_date"],
        how="left",
        validate="one_to_one",
        indicator="__meta_join",
    )
    if joined["__meta_join"].ne("both").any():
        raise AssertionError("Some market rows lack pure metadata controls.")
    joined = joined.drop(columns="__meta_join").merge(
        prototype_keep,
        on=["ticker", "feature_date"],
        how="left",
        validate="one_to_one",
        indicator="__prototype_join",
    )
    if joined["__prototype_join"].ne("both").any():
        raise AssertionError(
            f"Some market rows lack fold-safe {source} prototype features."
        )
    joined = joined.drop(columns="__prototype_join")
    train, validation, test = task_split_frames(
        joined, config, profile, fold
    )
    if not test.empty:
        raise AssertionError("A chronological validation fold exposed test rows.")
    if not train["target_date"].max() < validation["target_date"].min():
        raise AssertionError("Cross-attention train/validation chronology overlaps.")
    price_columns = price_feature_columns(joined, config)
    price_train, price_validation, price_transform = _fit_numeric(
        train, validation, price_columns
    )
    meta_train, meta_validation, meta_transform = _fit_numeric(
        train, validation, metadata_columns
    )
    prototype_train, mask_train = _prototype_tensor(
        train, blocks, levels
    )
    prototype_validation, mask_validation = _prototype_tensor(
        validation, blocks, levels
    )
    tickers = list(map(str, config["universe"]["tickers"]))
    ticker_train = _ticker_matrix(train["ticker"], tickers)
    ticker_validation = _ticker_matrix(validation["ticker"], tickers)
    y_train, y_validation, target_metadata = _target_arrays(
        train, validation, str(task["target"]), profile
    )
    target_metadata.update(
        {
            "prototype_artifact": prototype_path,
            "metadata_artifact": metadata_path,
            "fold": fold,
            "prototype_seed": seed,
        }
    )
    return PreparedArrays(
        train_frame=train.reset_index(drop=True),
        validation_frame=validation.reset_index(drop=True),
        price_train=price_train,
        price_validation=price_validation,
        meta_train=meta_train,
        meta_validation=meta_validation,
        prototype_train=prototype_train,
        prototype_validation=prototype_validation,
        prototype_mask_train=mask_train,
        prototype_mask_validation=mask_validation,
        ticker_train=ticker_train,
        ticker_validation=ticker_validation,
        y_train=y_train,
        y_validation=y_validation,
        price_columns=price_columns,
        metadata_columns=metadata_columns,
        prototype_columns=prototype_columns,
        k_per_level=k_per_level,
        target_metadata=target_metadata,
        preprocessors={
            "price": price_transform,
            "metadata": meta_transform,
            "ticker_order": tickers,
        },
    )


class FeedForwardFusion(nn.Module):
    def __init__(
        self,
        price_dim: int,
        ticker_dim: int,
        meta_dim: int,
        prototype_dim: int,
        *,
        use_meta: bool,
        use_prototype: bool,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.use_meta = bool(use_meta)
        self.use_prototype = bool(use_prototype)
        input_dim = price_dim + ticker_dim
        if self.use_meta:
            input_dim += meta_dim
        if self.use_prototype:
            input_dim += prototype_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        price: torch.Tensor,
        ticker: torch.Tensor,
        metadata: torch.Tensor,
        prototype: torch.Tensor,
        prototype_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del prototype_mask
        pieces = [price, ticker]
        if self.use_meta:
            pieces.append(metadata)
        if self.use_prototype:
            pieces.append(prototype.flatten(start_dim=1))
        output = self.network(torch.cat(pieces, dim=1)).squeeze(1)
        return output, {}


class PrototypeCrossAttention(nn.Module):
    """One market query attends to all level-specific prototype tokens."""

    def __init__(
        self,
        price_dim: int,
        ticker_dim: int,
        meta_dim: int,
        token_count: int,
        *,
        use_meta: bool,
        d_model: int,
        n_heads: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads.")
        self.use_meta = bool(use_meta)
        query_input_dim = price_dim + ticker_dim + (
            meta_dim if self.use_meta else 0
        )
        self.market_encoder = nn.Sequential(
            nn.Linear(query_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.prototype_identity = nn.Embedding(token_count, d_model)
        self.activation_encoder = nn.Sequential(
            nn.Linear(1, d_model),
            nn.Tanh(),
        )
        self.null_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.null_token, std=0.02)
        self.attention = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.context_norm = nn.LayerNorm(d_model)
        fusion_dim = 4 * d_model
        self.base_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.news_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.gate_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        price: torch.Tensor,
        ticker: torch.Tensor,
        metadata: torch.Tensor,
        prototype: torch.Tensor,
        prototype_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        query_parts = [price, ticker]
        if self.use_meta:
            query_parts.append(metadata)
        market_state = self.market_encoder(
            torch.cat(query_parts, dim=1)
        )
        batch_size = prototype.shape[0]
        activation = prototype.reshape(batch_size, -1, 1)
        token_ids = torch.arange(
            activation.shape[1], device=prototype.device
        )
        tokens = (
            self.prototype_identity(token_ids)[None, :, :]
            + self.activation_encoder(activation)
        )
        null_token = self.null_token.expand(batch_size, -1, -1)
        tokens = torch.cat([null_token, tokens], dim=1)
        active = torch.cat(
            [
                torch.ones(
                    (batch_size, 1),
                    dtype=torch.bool,
                    device=prototype.device,
                ),
                prototype_mask.bool(),
            ],
            dim=1,
        )
        context, weights = self.attention(
            market_state[:, None, :],
            tokens,
            tokens,
            key_padding_mask=~active,
            need_weights=True,
            average_attn_weights=False,
        )
        context = self.context_norm(context[:, 0, :])
        interaction = torch.cat(
            [
                market_state,
                context,
                market_state * context,
                torch.abs(market_state - context),
            ],
            dim=1,
        )
        gate = self.gate_head(interaction).squeeze(1)
        prediction = (
            self.base_head(market_state).squeeze(1)
            + gate * self.news_head(interaction).squeeze(1)
        )
        # [batch, heads, query=1, null+tokens]
        return prediction, {
            "attention": weights[:, :, 0, :],
            "gate": gate,
        }


def _torch_model(
    model_name: str,
    data: PreparedArrays,
    profile: Mapping[str, Any],
) -> nn.Module:
    common = {
        "price_dim": data.price_train.shape[1],
        "ticker_dim": data.ticker_train.shape[1],
        "meta_dim": data.meta_train.shape[1],
        "hidden_dim": int(profile["hidden_dim"]),
        "dropout": float(profile["dropout"]),
    }
    if model_name in ATTENTION_MODELS:
        return PrototypeCrossAttention(
            **common,
            token_count=len(profile["news_levels"]) * data.k_per_level,
            use_meta="META" in model_name,
            d_model=int(profile["d_model"]),
            n_heads=int(profile["n_heads"]),
        )
    if model_name == "PRICE_MLP":
        return FeedForwardFusion(
            **common,
            prototype_dim=data.prototype_train.shape[1]
            * data.prototype_train.shape[2],
            use_meta=False,
            use_prototype=False,
        )
    if model_name == "META_BASIC_MLP":
        return FeedForwardFusion(
            **common,
            prototype_dim=data.prototype_train.shape[1]
            * data.prototype_train.shape[2],
            use_meta=True,
            use_prototype=False,
        )
    if model_name == "R6_META_CONCAT_MLP":
        return FeedForwardFusion(
            **common,
            prototype_dim=data.prototype_train.shape[1]
            * data.prototype_train.shape[2],
            use_meta=True,
            use_prototype=True,
        )
    raise ValueError(f"No neural architecture for {model_name}.")


def _ridge_features(
    model_name: str,
    data: PreparedArrays,
    split: str,
) -> np.ndarray:
    if split == "train":
        price, ticker, metadata, prototype = (
            data.price_train,
            data.ticker_train,
            data.meta_train,
            data.prototype_train,
        )
    else:
        price, ticker, metadata, prototype = (
            data.price_validation,
            data.ticker_validation,
            data.meta_validation,
            data.prototype_validation,
        )
    pieces = [price, ticker]
    if model_name == "PRICE_META_RIDGE":
        pieces.append(metadata)
    elif model_name == "R6_RIDGE":
        pieces.append(prototype.reshape(len(prototype), -1))
    elif model_name != "PRICE_RIDGE":
        raise ValueError(f"Unknown linear comparator {model_name}.")
    return np.concatenate(pieces, axis=1).astype(np.float32)


def _fit_linear(
    model_name: str,
    target: str,
    data: PreparedArrays,
    profile: Mapping[str, Any],
) -> tuple[Any, np.ndarray]:
    x_train = _ridge_features(model_name, data, "train")
    x_validation = _ridge_features(model_name, data, "validation")
    if target == "volatility_level":
        estimator: Any = Ridge(alpha=float(profile["ridge_alpha"]))
        estimator.fit(x_train, data.y_train)
        prediction = estimator.predict(x_validation)
    else:
        positives = float(np.sum(data.y_train == 1))
        negatives = float(np.sum(data.y_train == 0))
        if positives == 0 or negatives == 0:
            raise ValueError("Spike training data contains only one class.")
        estimator = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=int(data.target_metadata["prototype_seed"]),
        )
        estimator.fit(x_train, data.y_train.astype(int))
        prediction = estimator.predict_proba(x_validation)[:, 1]
    return estimator, np.asarray(prediction, dtype=float)


def _tensor_dataset(
    data: PreparedArrays,
    split: str,
    indices: np.ndarray | None = None,
) -> TensorDataset:
    if split == "train":
        values = (
            data.price_train,
            data.ticker_train,
            data.meta_train,
            data.prototype_train,
            data.prototype_mask_train,
            data.y_train,
        )
    else:
        values = (
            data.price_validation,
            data.ticker_validation,
            data.meta_validation,
            data.prototype_validation,
            data.prototype_mask_validation,
            data.y_validation,
        )
    if indices is not None:
        values = tuple(value[indices] for value in values)
    return TensorDataset(
        torch.from_numpy(values[0]),
        torch.from_numpy(values[1]),
        torch.from_numpy(values[2]),
        torch.from_numpy(values[3]),
        torch.from_numpy(values[4]),
        torch.from_numpy(values[5]),
    )


def _inner_chronological_indices(
    frame: pd.DataFrame,
    fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.05 <= fraction <= 0.40:
        raise ValueError(
            "inner_validation_fraction must be between 0.05 and 0.40."
        )
    dates = pd.DatetimeIndex(frame["target_date"].unique()).sort_values()
    validation_dates = max(1, int(math.ceil(len(dates) * fraction)))
    if len(dates) - validation_dates < 22:
        raise ValueError(
            "Outer train has too few dates for chronological inner validation."
        )
    inner_start = dates[-validation_dates]
    train_mask = frame["target_date"].lt(inner_start).to_numpy()
    validation_mask = frame["target_date"].ge(inner_start).to_numpy()
    train_indices = np.flatnonzero(train_mask)
    validation_indices = np.flatnonzero(validation_mask)
    if not len(train_indices) or not len(validation_indices):
        raise AssertionError("Chronological inner split is empty.")
    if not (
        frame.iloc[train_indices]["target_date"].max()
        < frame.iloc[validation_indices]["target_date"].min()
    ):
        raise AssertionError("Chronological inner train/validation overlap.")
    return train_indices, validation_indices


def _predict_neural(
    model: nn.Module,
    data: PreparedArrays,
    target: str,
    device: torch.device,
    batch_size: int,
    target_mean: float,
    target_scale: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    loader = DataLoader(
        _tensor_dataset(data, "validation"),
        batch_size=batch_size,
        shuffle=False,
    )
    predictions: list[np.ndarray] = []
    gates: list[np.ndarray] = []
    attentions: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for price, ticker, metadata, prototype, mask, _ in loader:
            output, diagnostics = model(
                price.to(device),
                ticker.to(device),
                metadata.to(device),
                prototype.to(device),
                mask.to(device),
            )
            if target == "volatility_level":
                output = output * target_scale + target_mean
            else:
                output = torch.sigmoid(output)
            predictions.append(output.detach().cpu().numpy())
            if "gate" in diagnostics:
                gates.append(diagnostics["gate"].detach().cpu().numpy())
            if "attention" in diagnostics:
                attentions.append(
                    diagnostics["attention"].detach().cpu().numpy()
                )
    diagnostics_output: dict[str, np.ndarray] = {}
    if gates:
        diagnostics_output["gate"] = np.concatenate(gates)
    if attentions:
        diagnostics_output["attention"] = np.concatenate(attentions)
    return np.concatenate(predictions), diagnostics_output


def _fit_neural(
    model_name: str,
    target: str,
    data: PreparedArrays,
    profile: Mapping[str, Any],
    *,
    seed: int,
    quick: bool,
    logger: Any,
    epoch_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[nn.Module, np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    set_global_seed(seed, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _torch_model(model_name, data, profile).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(profile["learning_rate"]),
        weight_decay=float(profile["weight_decay"]),
    )
    if target == "volatility_level":
        target_mean = float(np.mean(data.y_train))
        target_scale = float(np.std(data.y_train))
        target_scale = max(target_scale, 1.0e-6)
        loss_function: nn.Module = nn.MSELoss()
    else:
        target_mean, target_scale = 0.0, 1.0
        positives = float(np.sum(data.y_train == 1))
        negatives = float(np.sum(data.y_train == 0))
        if positives == 0 or negatives == 0:
            raise ValueError("Spike training data contains only one class.")
        loss_function = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(negatives / positives, device=device)
        )
    batch_size = int(profile["batch_size"])
    inner_train_indices, inner_validation_indices = (
        _inner_chronological_indices(
            data.train_frame,
            float(profile["inner_validation_fraction"]),
        )
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        _tensor_dataset(data, "train", inner_train_indices),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=int(profile.get("num_workers", 0)),
    )
    validation_loader = DataLoader(
        _tensor_dataset(data, "train", inner_validation_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(profile.get("num_workers", 0)),
    )
    max_epochs = int(
        profile["quick_max_epochs"] if quick else profile["max_epochs"]
    )
    patience = int(profile["patience"])
    min_delta = float(profile["min_delta"])
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_gain = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_total = 0.0
        train_count = 0
        for price, ticker, metadata, prototype, mask, target_values in train_loader:
            optimizer.zero_grad(set_to_none=True)
            output, _ = model(
                price.to(device),
                ticker.to(device),
                metadata.to(device),
                prototype.to(device),
                mask.to(device),
            )
            target_batch = target_values.to(device)
            if target == "volatility_level":
                target_batch = (target_batch - target_mean) / target_scale
            loss = loss_function(output, target_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), float(profile["gradient_clip"])
            )
            optimizer.step()
            train_total += float(loss.detach().cpu()) * len(target_values)
            train_count += len(target_values)
        model.eval()
        validation_total = 0.0
        validation_count = 0
        with torch.inference_mode():
            for (
                price,
                ticker,
                metadata,
                prototype,
                mask,
                target_values,
            ) in validation_loader:
                output, _ = model(
                    price.to(device),
                    ticker.to(device),
                    metadata.to(device),
                    prototype.to(device),
                    mask.to(device),
                )
                target_batch = target_values.to(device)
                if target == "volatility_level":
                    target_batch = (target_batch - target_mean) / target_scale
                loss = loss_function(output, target_batch)
                validation_total += (
                    float(loss.detach().cpu()) * len(target_values)
                )
                validation_count += len(target_values)
        train_loss = train_total / max(train_count, 1)
        validation_loss = validation_total / max(validation_count, 1)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        improved = validation_loss < best_loss - min_delta
        if improved:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_gain = 0
        else:
            epochs_without_gain += 1
        epoch_state = {
            "epoch": epoch,
            "max_epochs": max_epochs,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "best_validation_loss": best_loss,
            "patience_remaining": max(patience - epochs_without_gain, 0),
        }
        if epoch_callback is not None:
            epoch_callback(epoch_state)
        log_every = int(profile.get("epoch_log_every", 5))
        if epoch == 1 or epoch % log_every == 0 or improved and epoch == max_epochs:
            logger.info(
                "epoch=%d/%d | train_loss=%.6f | val_loss=%.6f | "
                "best=%.6f | patience_remaining=%d",
                epoch,
                max_epochs,
                train_loss,
                validation_loss,
                best_loss,
                max(patience - epochs_without_gain, 0),
            )
        if epochs_without_gain >= patience:
            logger.info("Early stopping at epoch %d/%d", epoch, max_epochs)
            break
    if best_state is None:
        raise RuntimeError("Neural training did not produce a valid checkpoint.")
    model.load_state_dict(best_state)
    prediction, diagnostics = _predict_neural(
        model,
        data,
        target,
        device,
        batch_size,
        target_mean,
        target_scale,
    )
    metadata = {
        "device": str(device),
        "best_validation_loss": best_loss,
        "epochs_completed": len(history),
        "inner_train_samples": len(inner_train_indices),
        "inner_validation_samples": len(inner_validation_indices),
        "inner_train_target_end": data.train_frame.iloc[
            inner_train_indices
        ]["target_date"].max(),
        "inner_validation_target_start": data.train_frame.iloc[
            inner_validation_indices
        ]["target_date"].min(),
        "target_mean": target_mean,
        "target_scale": target_scale,
        "history": history,
    }
    return model, prediction, diagnostics, metadata


def _prediction_frame(
    task: Mapping[str, Any],
    data: PreparedArrays,
    prediction: np.ndarray,
    diagnostics: Mapping[str, np.ndarray],
    profile: Mapping[str, Any],
) -> pd.DataFrame:
    frame = data.validation_frame[
        ["ticker", "feature_date", "target_date"]
    ].copy()
    frame["task_id"] = str(task["task_id"])
    frame["evaluation_split"] = "validation"
    frame["target"] = str(task["target"])
    frame["model"] = str(task["model"])
    frame["prototype_source"] = str(task["prototype_source"])
    frame["fold"] = int(task["fold"])
    frame["seed"] = int(task["seed"])
    frame["y_true"] = data.y_validation
    frame["prediction"] = prediction
    frame["gate"] = np.nan
    for level in profile["news_levels"]:
        frame[f"attention__{level}"] = np.nan
    frame["attention_entropy"] = np.nan
    if "gate" in diagnostics:
        frame["gate"] = diagnostics["gate"]
    if "attention" in diagnostics:
        attention = np.asarray(diagnostics["attention"], dtype=float).mean(axis=1)
        # Drop the learned null token before level aggregation.
        semantic = attention[:, 1:]
        expected = len(profile["news_levels"]) * data.k_per_level
        if semantic.shape[1] != expected:
            raise AssertionError(
                f"Attention dimension {semantic.shape[1]} != expected {expected}."
            )
        semantic = semantic.reshape(
            len(frame), len(profile["news_levels"]), data.k_per_level
        )
        for level_index, level in enumerate(profile["news_levels"]):
            frame[f"attention__{level}"] = semantic[:, level_index, :].sum(
                axis=1
            )
        clipped = np.clip(attention, 1.0e-12, 1.0)
        frame["attention_entropy"] = -np.sum(
            clipped * np.log(clipped), axis=1
        )
    return frame


def run_task(
    config: Mapping[str, Any],
    profile: Mapping[str, Any],
    task: Mapping[str, Any],
    *,
    logger: Any,
    epoch_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Path]:
    root = ensure_output_directories(config, profile)
    task_id = str(task["task_id"])
    prediction_path = root / "checkpoints" / f"{task_id}.parquet"
    model_path = root / "models" / (
        f"{task_id}.joblib"
        if str(task["model"]) in RIDGE_MODELS
        else f"{task_id}.pt"
    )
    data = prepare_arrays(config, profile, task)
    model_name = str(task["model"])
    target = str(task["target"])
    started = time.monotonic()
    if model_name in RIDGE_MODELS:
        model, prediction = _fit_linear(
            model_name, target, data, profile
        )
        diagnostics: dict[str, np.ndarray] = {}
        training_metadata: dict[str, Any] = {
            "estimator": type(model).__name__,
            "elapsed_seconds": time.monotonic() - started,
        }
        joblib.dump(
            {
                "model": model,
                "preprocessors": data.preprocessors,
                "task": dict(task),
                "target_metadata": data.target_metadata,
                "price_columns": data.price_columns,
                "metadata_columns": data.metadata_columns,
                "prototype_columns": data.prototype_columns,
            },
            model_path,
        )
    else:
        model, prediction, diagnostics, training_metadata = _fit_neural(
            model_name,
            target,
            data,
            profile,
            seed=int(task["seed"]),
            quick=bool(task.get("quick", False)),
            logger=logger,
            epoch_callback=epoch_callback,
        )
        torch.save(
            {
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                "task": dict(task),
                "target_metadata": data.target_metadata,
                "training_metadata": training_metadata,
                "price_columns": data.price_columns,
                "metadata_columns": data.metadata_columns,
                "prototype_columns": data.prototype_columns,
                "preprocessors": data.preprocessors,
                "model_config": {
                    key: profile[key]
                    for key in (
                        "d_model",
                        "n_heads",
                        "hidden_dim",
                        "dropout",
                    )
                },
            },
            model_path,
        )
    predictions = _prediction_frame(
        task, data, prediction, diagnostics, profile
    )
    write_table(predictions, prediction_path)
    metadata_path = root / "checkpoints" / f"{task_id}.json"
    atomic_write_json(
        {
            "task": dict(task),
            "n_train": len(data.train_frame),
            "n_validation": len(data.validation_frame),
            "train_target_start": data.train_frame["target_date"].min(),
            "train_target_end": data.train_frame["target_date"].max(),
            "validation_target_start": data.validation_frame["target_date"].min(),
            "validation_target_end": data.validation_frame["target_date"].max(),
            "target_metadata": data.target_metadata,
            "training_metadata": training_metadata,
            "price_feature_count": len(data.price_columns),
            "metadata_feature_count": len(data.metadata_columns),
            "prototype_token_count": len(profile["news_levels"])
            * data.k_per_level,
        },
        metadata_path,
    )
    return {
        "predictions": prediction_path,
        "model": model_path,
        "metadata": metadata_path,
    }


def _metric_row(predictions: pd.DataFrame) -> dict[str, Any]:
    target = str(predictions["target"].iloc[0])
    if target == "volatility_level":
        metrics = regression_metrics(
            predictions["y_true"].to_numpy(dtype=float),
            predictions["prediction"].to_numpy(dtype=float),
        )
        primary_metric = "qlike"
        primary_value = metrics[primary_metric]
        larger_is_better = False
    else:
        metrics = binary_metrics(
            predictions["y_true"].to_numpy(dtype=int),
            predictions["prediction"].to_numpy(dtype=float),
        )
        primary_metric = "pr_auc"
        primary_value = metrics[primary_metric]
        larger_is_better = True
    first = predictions.iloc[0]
    return {
        "task_id": first["task_id"],
        "target": target,
        "model": first["model"],
        "prototype_source": first["prototype_source"],
        "fold": int(first["fold"]),
        "seed": int(first["seed"]),
        "primary_metric": primary_metric,
        "primary_value": primary_value,
        "larger_is_better": larger_is_better,
        **metrics,
    }


def _comparisons(
    results: pd.DataFrame,
    profile: Mapping[str, Any],
) -> pd.DataFrame:
    primary = str(profile.get("primary_model", "R6_META_XATTN"))
    references = tuple(map(str, profile["fixed_comparators"]))
    rows: list[dict[str, Any]] = []
    for target, target_results in results.groupby("target", sort=True):
        primary_rows = target_results.loc[
            target_results["model"].eq(primary)
        ][["fold", "seed", "primary_value", "larger_is_better"]].rename(
            columns={"primary_value": "primary_value"}
        )
        for reference in references:
            reference_rows = target_results.loc[
                target_results["model"].eq(reference)
            ][["fold", "seed", "primary_value"]].rename(
                columns={"primary_value": "reference_value"}
            )
            paired = primary_rows.merge(
                reference_rows,
                on=["fold", "seed"],
                how="inner",
                validate="one_to_one",
            )
            if paired.empty:
                rows.append(
                    {
                        "target": target,
                        "primary_model": primary,
                        "reference_model": reference,
                        "n_cells": 0,
                        "mean_relative_gain": np.nan,
                        "cell_win_rate": np.nan,
                        "all_fold_mean_wins": False,
                        "passed": False,
                        "reason": "missing paired results",
                    }
                )
                continue
            larger = bool(paired["larger_is_better"].iloc[0])
            if larger:
                difference = (
                    paired["primary_value"] - paired["reference_value"]
                )
            else:
                difference = (
                    paired["reference_value"] - paired["primary_value"]
                )
            denominator = np.maximum(
                np.abs(paired["reference_value"].to_numpy(dtype=float)),
                1.0e-12,
            )
            paired["relative_gain"] = (
                difference.to_numpy(dtype=float) / denominator
            )
            fold_gain = paired.groupby("fold", observed=True)[
                "relative_gain"
            ].mean()
            all_fold_wins = bool((fold_gain > 0).all())
            mean_gain = float(paired["relative_gain"].mean())
            win_rate = float((paired["relative_gain"] > 0).mean())
            passed = (
                mean_gain >= float(profile["minimum_relative_gain"])
                and win_rate >= float(profile["minimum_cell_win_rate"])
                and (
                    all_fold_wins
                    or not bool(profile.get("require_all_fold_mean_wins", True))
                )
            )
            rows.append(
                {
                    "target": target,
                    "primary_model": primary,
                    "reference_model": reference,
                    "n_cells": len(paired),
                    "mean_relative_gain": mean_gain,
                    "std_relative_gain": float(
                        paired["relative_gain"].std(ddof=1)
                    ),
                    "cell_win_rate": win_rate,
                    "all_fold_mean_wins": all_fold_wins,
                    "fold_mean_gains": json.dumps(
                        {
                            str(int(key)): float(value)
                            for key, value in fold_gain.items()
                        },
                        sort_keys=True,
                    ),
                    "passed": passed,
                    "reason": "passed" if passed else "gain/stability gate failed",
                }
            )
    return pd.DataFrame(rows)


def evaluate(
    config: Mapping[str, Any],
    profile: Mapping[str, Any],
    plan: pd.DataFrame,
) -> dict[str, Path]:
    root = ensure_output_directories(config, profile)
    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for task_id in plan["task_id"].astype(str):
        path = root / "checkpoints" / f"{task_id}.parquet"
        if not path.is_file():
            missing.append(task_id)
            continue
        predictions = read_table(path)
        metric_rows.append(_metric_row(predictions))
        prediction_frames.append(predictions)
    if missing:
        raise FileNotFoundError(
            f"Cannot evaluate: {len(missing)} planned tasks are incomplete. "
            f"Examples: {missing[:5]}"
        )
    results = pd.DataFrame(metric_rows).sort_values(
        ["target", "model", "fold", "seed"], kind="mergesort"
    )
    aggregate = (
        results.groupby(
            ["target", "model", "primary_metric"],
            observed=True,
            dropna=False,
        )["primary_value"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )
    comparisons = _comparisons(results, profile)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    attention_columns = [
        column
        for column in predictions.columns
        if column.startswith("attention__")
    ]
    attention = (
        predictions.loc[predictions["model"].isin(ATTENTION_MODELS)]
        .groupby(["target", "model", "fold"], observed=True)[
            ["gate", "attention_entropy", *attention_columns]
        ]
        .agg(["mean", "std"])
    )
    attention.columns = [
        "__".join(map(str, column)).strip("_")
        for column in attention.columns.to_flat_index()
    ]
    attention = attention.reset_index()
    decision_rows = []
    for target, group in comparisons.groupby("target", sort=True):
        complete = len(group) == len(profile["fixed_comparators"])
        passed = complete and bool(group["passed"].all())
        decision_rows.append(
            {
                "target": target,
                "decision": "XATTN-PASS" if passed else "XATTN-FAIL",
                "all_fixed_comparisons_passed": passed,
                "comparison_count": len(group),
                "interpretation": (
                    "Prototype cross-attention beat price, pure metadata, "
                    "concatenation and every fixed placebo."
                    if passed
                    else "Do not promote cross-attention to the locked holdout."
                ),
            }
        )
    decision = pd.DataFrame(decision_rows)
    paths = {
        "results": root / "tables" / "cross_attention_results.csv",
        "aggregate": root / "tables" / "cross_attention_aggregate.csv",
        "comparisons": root / "tables" / "cross_attention_comparisons.csv",
        "attention": root / "tables" / "cross_attention_attention_audit.csv",
        "decision": root / "tables" / "cross_attention_decision.csv",
    }
    atomic_write_csv(results, paths["results"], index=False)
    atomic_write_csv(aggregate, paths["aggregate"], index=False)
    atomic_write_csv(comparisons, paths["comparisons"], index=False)
    atomic_write_csv(attention, paths["attention"], index=False)
    atomic_write_csv(decision, paths["decision"], index=False)
    return paths


def print_report(
    config: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> None:
    path = output_root(config, profile) / "tables" / "cross_attention_decision.csv"
    if not path.is_file():
        print("PROTOTYPE CROSS-ATTENTION: evaluation has not completed.")
        return
    decision = read_table(path)
    print("\nPROTOTYPE CROSS-ATTENTION EXPERIMENT")
    print("Structured-event extraction required: False")
    print("Locked holdout evaluated: False")
    for row in decision.itertuples(index=False):
        print(
            f"{row.target}: {row.decision} | "
            f"all fixed comparisons passed={row.all_fixed_comparisons_passed}"
        )
        print(f"Next step: {row.interpretation}")
