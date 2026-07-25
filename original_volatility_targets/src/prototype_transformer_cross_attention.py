"""Transformer cross-attention with a frozen OOF R6-Ridge centre forecast.

This V2 experiment keeps the original daily FinTexTS prototype pipeline and
does not require structured-event extraction.  The centre forecast is always
the same fold-safe R6-Ridge model.  A Transformer decoder-style cross-attention
branch learns only an out-of-fold residual correction, making true R6 memory
directly comparable with R9/R10/R11 placebo memories.
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
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.prototype_cross_attention import (
    PreparedArrays,
    _fit_linear,
    _inner_chronological_indices,
    prepare_arrays,
    validate_plan_artifacts,
)
from src.utils import (
    atomic_write_csv,
    atomic_write_json,
    project_path,
    read_table,
    regression_metrics,
    set_global_seed,
    stable_id,
    write_table,
)


LINEAR_MODELS = {"PRICE_RIDGE", "PRICE_META_RIDGE", "R6_RIDGE"}
RESIDUAL_MLP_MODELS = {"META_RESIDUAL_MLP", "R6_RESIDUAL_CONCAT_MLP"}
TRANSFORMER_MODELS = {
    "R6_TRANSFORMER_XATTN",
    "R6_META_TRANSFORMER_XATTN",
    "R9_META_TRANSFORMER_XATTN",
    "R10_META_TRANSFORMER_XATTN",
    "R11_META_TRANSFORMER_XATTN",
}
NEURAL_MODELS = RESIDUAL_MLP_MODELS | TRANSFORMER_MODELS
MODEL_PROTOTYPE_SOURCE = {
    "PRICE_RIDGE": "R6",
    "PRICE_META_RIDGE": "R6",
    "R6_RIDGE": "R6",
    "META_RESIDUAL_MLP": "R6",
    "R6_RESIDUAL_CONCAT_MLP": "R6",
    "R6_TRANSFORMER_XATTN": "R6",
    "R6_META_TRANSFORMER_XATTN": "R6",
    "R9_META_TRANSFORMER_XATTN": "R9",
    "R10_META_TRANSFORMER_XATTN": "R10",
    "R11_META_TRANSFORMER_XATTN": "R11",
}


@dataclass
class TransformerPrepared:
    attention: PreparedArrays
    baseline: PreparedArrays


@dataclass
class FrozenBaseline:
    oof_prediction: np.ndarray
    oof_mask: np.ndarray
    validation_prediction: np.ndarray
    final_model: Ridge
    final_imputer: SimpleImputer
    final_scaler: StandardScaler
    oof_manifest: list[dict[str, Any]]


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
    model_name: str, profile: Mapping[str, Any]
) -> str:
    if model_name not in MODEL_PROTOTYPE_SOURCE:
        raise ValueError(f"Unknown Transformer experiment model: {model_name}")
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
    del config
    selected_targets = list(targets or profile["targets"])
    if set(selected_targets) != {"volatility_level"}:
        raise ValueError(
            "Transformer residual V2 currently supports volatility_level only."
        )
    selected_models = list(
        models
        or (profile["quick_models"] if quick else profile["models"])
    )
    unknown = sorted(
        set(selected_models).difference(LINEAR_MODELS | NEURAL_MODELS)
    )
    if unknown:
        raise ValueError(f"Unsupported Transformer V2 models: {unknown}")
    selected_folds = [
        int(value) for value in (folds or profile["folds"])
    ]
    selected_seeds = [
        int(value) for value in (seeds or profile["seeds"])
    ]
    version = str(profile["experiment_version"])
    rows: list[dict[str, Any]] = []
    for fold in selected_folds:
        for seed in selected_seeds:
            for model_name in selected_models:
                payload = {
                    "experiment_version": version,
                    "target": "volatility_level",
                    "fold": fold,
                    "seed": seed,
                    "model": str(model_name),
                    "prototype_source": _model_source(
                        str(model_name), profile
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
                                "experiment": (
                                    "prototype_transformer_cross_attention"
                                ),
                                **payload,
                            },
                            prefix="txattn",
                        ),
                        **payload,
                    }
                )
    plan = pd.DataFrame(rows)
    if plan.empty or plan["task_id"].duplicated().any():
        raise AssertionError("Transformer task plan is empty or duplicated.")
    return plan


def validate_artifacts(
    config: Mapping[str, Any],
    profile: Mapping[str, Any],
    plan: pd.DataFrame,
) -> None:
    validate_plan_artifacts(config, profile, plan)
    # Every correction, including placebos, uses the same semantic R6 centre.
    baseline_plan = plan.copy()
    baseline_plan["prototype_source"] = "R6"
    validate_plan_artifacts(config, profile, baseline_plan)


def _assert_aligned(
    attention: PreparedArrays, baseline: PreparedArrays
) -> None:
    for split, left, right in (
        (
            "train",
            attention.train_frame,
            baseline.train_frame,
        ),
        (
            "validation",
            attention.validation_frame,
            baseline.validation_frame,
        ),
    ):
        columns = ["ticker", "feature_date", "target_date"]
        if not left[columns].reset_index(drop=True).equals(
            right[columns].reset_index(drop=True)
        ):
            raise AssertionError(
                f"Attention and frozen-baseline {split} rows are misaligned."
            )
    if not np.allclose(attention.y_train, baseline.y_train):
        raise AssertionError("Attention and baseline train targets differ.")
    if not np.allclose(attention.y_validation, baseline.y_validation):
        raise AssertionError("Attention and baseline validation targets differ.")


def prepare_transformer_data(
    config: Mapping[str, Any],
    profile: Mapping[str, Any],
    task: Mapping[str, Any],
) -> TransformerPrepared:
    attention = prepare_arrays(config, profile, task)
    if str(task["prototype_source"]) == "R6":
        baseline = attention
    else:
        baseline_task = dict(task)
        baseline_task["prototype_source"] = "R6"
        baseline = prepare_arrays(config, profile, baseline_task)
    _assert_aligned(attention, baseline)
    return TransformerPrepared(attention=attention, baseline=baseline)


def _baseline_raw_numeric(
    data: PreparedArrays, split: str
) -> np.ndarray:
    frame = (
        data.train_frame if split == "train" else data.validation_frame
    )
    columns = [*data.price_columns, *data.prototype_columns]
    return (
        frame[columns]
        .replace([np.inf, -np.inf], np.nan)
        .to_numpy(dtype=np.float64)
    )


def _fit_ridge_window(
    train_numeric: np.ndarray,
    train_ticker: np.ndarray,
    train_target: np.ndarray,
    prediction_numeric: np.ndarray,
    prediction_ticker: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, Ridge, SimpleImputer, StandardScaler]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    transformed_train = scaler.fit_transform(
        imputer.fit_transform(train_numeric)
    )
    transformed_prediction = scaler.transform(
        imputer.transform(prediction_numeric)
    )
    x_train = np.concatenate(
        [transformed_train, train_ticker], axis=1
    )
    x_prediction = np.concatenate(
        [transformed_prediction, prediction_ticker], axis=1
    )
    model = Ridge(alpha=alpha)
    model.fit(x_train, train_target)
    return (
        np.asarray(model.predict(x_prediction), dtype=np.float32),
        model,
        imputer,
        scaler,
    )


def expanding_oof_blocks(
    target_dates: pd.Series,
    *,
    split_count: int,
    minimum_train_fraction: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    dates = pd.DatetimeIndex(target_dates.unique()).sort_values()
    if split_count < 2:
        raise ValueError("baseline_oof_splits must be at least two.")
    if not 0.20 <= minimum_train_fraction <= 0.80:
        raise ValueError(
            "baseline_oof_min_train_fraction must be in [0.20, 0.80]."
        )
    initial_dates = max(
        22, int(math.ceil(len(dates) * minimum_train_fraction))
    )
    if initial_dates >= len(dates):
        raise ValueError("Not enough train dates for expanding OOF residuals.")
    future_dates = dates[initial_dates:]
    date_blocks = [
        block for block in np.array_split(future_dates, split_count) if len(block)
    ]
    outputs: list[tuple[np.ndarray, np.ndarray]] = []
    normalized = pd.to_datetime(target_dates).dt.normalize()
    for block in date_blocks:
        validation_start = pd.Timestamp(block[0])
        validation_end = pd.Timestamp(block[-1])
        history = np.flatnonzero(
            normalized.lt(validation_start).to_numpy()
        )
        prediction = np.flatnonzero(
            normalized.between(
                validation_start, validation_end
            ).to_numpy()
        )
        if not len(history) or not len(prediction):
            raise AssertionError("An expanding OOF block is empty.")
        if not (
            normalized.iloc[history].max()
            < normalized.iloc[prediction].min()
        ):
            raise AssertionError("Expanding OOF chronology overlaps.")
        outputs.append((history, prediction))
    return outputs


def build_frozen_baseline(
    data: PreparedArrays,
    profile: Mapping[str, Any],
) -> FrozenBaseline:
    train_numeric = _baseline_raw_numeric(data, "train")
    validation_numeric = _baseline_raw_numeric(data, "validation")
    alpha = float(profile["ridge_alpha"])
    blocks = expanding_oof_blocks(
        data.train_frame["target_date"],
        split_count=int(profile["baseline_oof_splits"]),
        minimum_train_fraction=float(
            profile["baseline_oof_min_train_fraction"]
        ),
    )
    oof = np.full(len(data.train_frame), np.nan, dtype=np.float32)
    manifest: list[dict[str, Any]] = []
    for block_id, (history, prediction) in enumerate(blocks, start=1):
        values, _, _, _ = _fit_ridge_window(
            train_numeric[history],
            data.ticker_train[history],
            data.y_train[history],
            train_numeric[prediction],
            data.ticker_train[prediction],
            alpha,
        )
        oof[prediction] = values
        manifest.append(
            {
                "block": block_id,
                "train_start": data.train_frame.iloc[history][
                    "target_date"
                ].min(),
                "train_end": data.train_frame.iloc[history][
                    "target_date"
                ].max(),
                "prediction_start": data.train_frame.iloc[prediction][
                    "target_date"
                ].min(),
                "prediction_end": data.train_frame.iloc[prediction][
                    "target_date"
                ].max(),
                "n_train": len(history),
                "n_prediction": len(prediction),
            }
        )
    mask = np.isfinite(oof)
    if mask.mean() < 0.40:
        raise AssertionError(
            f"OOF baseline coverage is too low: {mask.mean():.1%}"
        )
    validation_prediction, model, imputer, scaler = _fit_ridge_window(
        train_numeric,
        data.ticker_train,
        data.y_train,
        validation_numeric,
        data.ticker_validation,
        alpha,
    )
    return FrozenBaseline(
        oof_prediction=oof,
        oof_mask=mask,
        validation_prediction=validation_prediction,
        final_model=model,
        final_imputer=imputer,
        final_scaler=scaler,
        oof_manifest=manifest,
    )


class TransformerCrossAttentionBlock(nn.Module):
    """Pre-norm decoder cross-attention followed by Add/Norm/FFN."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model)
        self.memory_norm = nn.LayerNorm(d_model)
        self.cross_attention = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        memory_active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_query = self.query_norm(query)
        normalized_memory = self.memory_norm(memory)
        attended, weights = self.cross_attention(
            normalized_query,
            normalized_memory,
            normalized_memory,
            key_padding_mask=~memory_active.bool(),
            need_weights=True,
            average_attn_weights=False,
        )
        query = query + self.attention_dropout(attended)
        query = query + self.ffn_dropout(self.ffn(self.ffn_norm(query)))
        return query, weights


class PrototypeTransformerResidual(nn.Module):
    """Market query sequence cross-attends to prototype key/value memory."""

    def __init__(
        self,
        *,
        price_dim: int,
        ticker_dim: int,
        meta_dim: int,
        level_count: int,
        k_per_level: int,
        use_meta: bool,
        d_model: int,
        n_heads: int,
        hidden_dim: int,
        ff_dim: int,
        market_encoder_layers: int,
        cross_attention_layers: int,
        dropout: float,
        gate_initial_value: float,
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads.")
        if not 0.0 < gate_initial_value < 0.5:
            raise ValueError("gate_initial_value must be in (0, 0.5).")
        self.use_meta = bool(use_meta)
        self.level_count = int(level_count)
        self.k_per_level = int(k_per_level)
        token_count = level_count * k_per_level
        self.market_cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.market_cls, std=0.02)
        self.price_value_projection = nn.Linear(1, d_model)
        self.price_feature_embedding = nn.Embedding(price_dim, d_model)
        self.ticker_projection = nn.Linear(ticker_dim, d_model)
        self.meta_projection = (
            nn.Linear(meta_dim, d_model) if self.use_meta else None
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.market_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=market_encoder_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        self.prototype_identity = nn.Embedding(token_count, d_model)
        self.news_level_embedding = nn.Embedding(level_count, d_model)
        self.prototype_activation_projection = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.null_memory = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.null_memory, std=0.02)
        self.cross_blocks = nn.ModuleList(
            [
                TransformerCrossAttentionBlock(
                    d_model, n_heads, ff_dim, dropout
                )
                for _ in range(cross_attention_layers)
            ]
        )
        fusion_dim = 4 * d_model
        self.correction_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.gate_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.correction_head[-1].weight)
        nn.init.zeros_(self.correction_head[-1].bias)
        gate_linear = self.gate_head[-2]
        nn.init.zeros_(gate_linear.weight)
        nn.init.constant_(
            gate_linear.bias,
            math.log(gate_initial_value / (1.0 - gate_initial_value)),
        )

    def _market_tokens(
        self,
        price: torch.Tensor,
        ticker: torch.Tensor,
        metadata: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, price_dim = price.shape
        feature_ids = torch.arange(price_dim, device=price.device)
        price_tokens = (
            self.price_value_projection(price[:, :, None])
            + self.price_feature_embedding(feature_ids)[None, :, :]
        )
        cls = self.market_cls.expand(batch_size, -1, -1)
        cls = cls + self.ticker_projection(ticker)[:, None, :]
        tokens = [cls, price_tokens]
        if self.use_meta:
            assert self.meta_projection is not None
            tokens.append(self.meta_projection(metadata)[:, None, :])
        return self.market_encoder(torch.cat(tokens, dim=1))

    def _prototype_memory(
        self,
        prototype: torch.Tensor,
        prototype_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = prototype.shape[0]
        activation = prototype.reshape(batch_size, -1, 1)
        token_ids = torch.arange(
            activation.shape[1], device=prototype.device
        )
        level_ids = (
            token_ids // self.k_per_level
        ).clamp_max(self.level_count - 1)
        memory = (
            self.prototype_identity(token_ids)[None, :, :]
            + self.news_level_embedding(level_ids)[None, :, :]
            + self.prototype_activation_projection(activation)
        )
        memory = torch.cat(
            [self.null_memory.expand(batch_size, -1, -1), memory],
            dim=1,
        )
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
        return memory, active

    def forward(
        self,
        price: torch.Tensor,
        ticker: torch.Tensor,
        metadata: torch.Tensor,
        prototype: torch.Tensor,
        prototype_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        market = self._market_tokens(price, ticker, metadata)
        initial_cls = market[:, 0, :]
        memory, active = self._prototype_memory(
            prototype, prototype_mask
        )
        last_weights: torch.Tensor | None = None
        for block in self.cross_blocks:
            market, last_weights = block(market, memory, active)
        if last_weights is None:
            raise AssertionError("At least one cross-attention layer is required.")
        final_cls = market[:, 0, :]
        fusion = torch.cat(
            [
                initial_cls,
                final_cls,
                final_cls - initial_cls,
                initial_cls * final_cls,
            ],
            dim=1,
        )
        gate = self.gate_head(fusion).squeeze(1)
        correction = gate * self.correction_head(fusion).squeeze(1)
        # Average query positions only for the audit; training uses the full
        # query sequence and all attention heads.
        audit_attention = last_weights.mean(dim=2)
        return correction, {
            "attention": audit_attention,
            "gate": gate,
        }


class ResidualMLP(nn.Module):
    def __init__(
        self,
        *,
        price_dim: int,
        ticker_dim: int,
        meta_dim: int,
        prototype_dim: int,
        include_prototype: bool,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.include_prototype = bool(include_prototype)
        input_dim = price_dim + ticker_dim + meta_dim
        if self.include_prototype:
            input_dim += prototype_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        price: torch.Tensor,
        ticker: torch.Tensor,
        metadata: torch.Tensor,
        prototype: torch.Tensor,
        prototype_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del prototype_mask
        pieces = [price, ticker, metadata]
        if self.include_prototype:
            pieces.append(prototype.flatten(start_dim=1))
        return self.network(torch.cat(pieces, dim=1)).squeeze(1), {}


def _build_model(
    model_name: str,
    data: PreparedArrays,
    profile: Mapping[str, Any],
) -> nn.Module:
    if model_name in TRANSFORMER_MODELS:
        return PrototypeTransformerResidual(
            price_dim=data.price_train.shape[1],
            ticker_dim=data.ticker_train.shape[1],
            meta_dim=data.meta_train.shape[1],
            level_count=len(profile["news_levels"]),
            k_per_level=data.k_per_level,
            use_meta="_META_" in model_name,
            d_model=int(profile["d_model"]),
            n_heads=int(profile["n_heads"]),
            hidden_dim=int(profile["hidden_dim"]),
            ff_dim=int(profile["transformer_ff_dim"]),
            market_encoder_layers=int(profile["market_encoder_layers"]),
            cross_attention_layers=int(profile["cross_attention_layers"]),
            dropout=float(profile["dropout"]),
            gate_initial_value=float(profile["gate_initial_value"]),
        )
    if model_name in RESIDUAL_MLP_MODELS:
        return ResidualMLP(
            price_dim=data.price_train.shape[1],
            ticker_dim=data.ticker_train.shape[1],
            meta_dim=data.meta_train.shape[1],
            prototype_dim=(
                data.prototype_train.shape[1]
                * data.prototype_train.shape[2]
            ),
            include_prototype=model_name == "R6_RESIDUAL_CONCAT_MLP",
            hidden_dim=int(profile["hidden_dim"]),
            dropout=float(profile["dropout"]),
        )
    raise ValueError(f"No residual architecture for {model_name}.")


def _residual_dataset(
    data: PreparedArrays,
    baseline_prediction: np.ndarray,
    residual_target: np.ndarray,
    indices: np.ndarray,
) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(data.price_train[indices]),
        torch.from_numpy(data.ticker_train[indices]),
        torch.from_numpy(data.meta_train[indices]),
        torch.from_numpy(data.prototype_train[indices]),
        torch.from_numpy(data.prototype_mask_train[indices]),
        torch.from_numpy(baseline_prediction[indices].astype(np.float32)),
        torch.from_numpy(residual_target[indices].astype(np.float32)),
        torch.from_numpy(data.y_train[indices].astype(np.float32)),
    )


def _hybrid_loss(
    normalized_correction: torch.Tensor,
    residual_target: torch.Tensor,
    baseline_prediction: torch.Tensor,
    y_true: torch.Tensor,
    residual_scale: float,
    qlike_weight: float,
) -> torch.Tensor:
    correction = normalized_correction * residual_scale
    residual_mse = torch.mean(
        torch.square(
            normalized_correction - residual_target / residual_scale
        )
    )
    final_prediction = baseline_prediction + correction
    log_ratio = torch.clamp(y_true - final_prediction, -10.0, 10.0)
    qlike = torch.mean(torch.exp(log_ratio) - log_ratio - 1.0)
    return residual_mse + qlike_weight * qlike


def _predict_correction(
    model: nn.Module,
    prepared: TransformerPrepared,
    baseline: FrozenBaseline,
    device: torch.device,
    batch_size: int,
    residual_scale: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    data = prepared.attention
    dataset = TensorDataset(
        torch.from_numpy(data.price_validation),
        torch.from_numpy(data.ticker_validation),
        torch.from_numpy(data.meta_validation),
        torch.from_numpy(data.prototype_validation),
        torch.from_numpy(data.prototype_mask_validation),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    corrections: list[np.ndarray] = []
    gates: list[np.ndarray] = []
    attentions: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for price, ticker, metadata, prototype, mask in loader:
            output, diagnostics = model(
                price.to(device),
                ticker.to(device),
                metadata.to(device),
                prototype.to(device),
                mask.to(device),
            )
            corrections.append(
                (output * residual_scale).detach().cpu().numpy()
            )
            if "gate" in diagnostics:
                gates.append(diagnostics["gate"].detach().cpu().numpy())
            if "attention" in diagnostics:
                attentions.append(
                    diagnostics["attention"].detach().cpu().numpy()
                )
    correction = np.concatenate(corrections)
    diagnostics_output: dict[str, np.ndarray] = {}
    if gates:
        diagnostics_output["gate"] = np.concatenate(gates)
    if attentions:
        diagnostics_output["attention"] = np.concatenate(attentions)
    diagnostics_output["baseline_prediction"] = (
        baseline.validation_prediction
    )
    diagnostics_output["correction"] = correction
    return baseline.validation_prediction + correction, diagnostics_output


def train_residual_model(
    model_name: str,
    prepared: TransformerPrepared,
    baseline: FrozenBaseline,
    profile: Mapping[str, Any],
    *,
    seed: int,
    quick: bool,
    logger: Any,
    epoch_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[nn.Module, np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    set_global_seed(seed, deterministic=True)
    data = prepared.attention
    eligible = np.flatnonzero(baseline.oof_mask)
    residual = (
        data.y_train.astype(np.float32)
        - baseline.oof_prediction.astype(np.float32)
    )
    residual_scale = max(
        float(np.std(residual[eligible], ddof=0)), 1.0e-6
    )
    eligible_frame = data.train_frame.iloc[eligible].reset_index(drop=True)
    inner_train_local, inner_validation_local = (
        _inner_chronological_indices(
            eligible_frame,
            float(profile["inner_validation_fraction"]),
        )
    )
    inner_train = eligible[inner_train_local]
    inner_validation = eligible[inner_validation_local]
    if not (
        data.train_frame.iloc[inner_train]["target_date"].max()
        < data.train_frame.iloc[inner_validation]["target_date"].min()
    ):
        raise AssertionError("Residual inner validation overlaps training.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(model_name, data, profile).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(profile["learning_rate"]),
        weight_decay=float(profile["weight_decay"]),
    )
    batch_size = int(profile["batch_size"])
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        _residual_dataset(
            data,
            baseline.oof_prediction,
            residual,
            inner_train,
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=int(profile.get("num_workers", 0)),
    )
    validation_loader = DataLoader(
        _residual_dataset(
            data,
            baseline.oof_prediction,
            residual,
            inner_validation,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(profile.get("num_workers", 0)),
    )
    qlike_weight = float(profile["qlike_loss_weight"])
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
        for (
            price,
            ticker,
            metadata,
            prototype,
            mask,
            base,
            residual_target,
            y_true,
        ) in train_loader:
            optimizer.zero_grad(set_to_none=True)
            correction, _ = model(
                price.to(device),
                ticker.to(device),
                metadata.to(device),
                prototype.to(device),
                mask.to(device),
            )
            loss = _hybrid_loss(
                correction,
                residual_target.to(device),
                base.to(device),
                y_true.to(device),
                residual_scale,
                qlike_weight,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), float(profile["gradient_clip"])
            )
            optimizer.step()
            train_total += float(loss.detach().cpu()) * len(y_true)
            train_count += len(y_true)
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
                base,
                residual_target,
                y_true,
            ) in validation_loader:
                correction, _ = model(
                    price.to(device),
                    ticker.to(device),
                    metadata.to(device),
                    prototype.to(device),
                    mask.to(device),
                )
                loss = _hybrid_loss(
                    correction,
                    residual_target.to(device),
                    base.to(device),
                    y_true.to(device),
                    residual_scale,
                    qlike_weight,
                )
                validation_total += (
                    float(loss.detach().cpu()) * len(y_true)
                )
                validation_count += len(y_true)
        train_loss = train_total / max(train_count, 1)
        validation_loss = validation_total / max(validation_count, 1)
        improved = validation_loss < best_loss - min_delta
        if improved:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_gain = 0
        else:
            epochs_without_gain += 1
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
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
        if epoch == 1 or epoch % log_every == 0:
            logger.info(
                "epoch=%d/%d | residual_train_loss=%.6f | "
                "residual_inner_val_loss=%.6f | best=%.6f | "
                "patience_remaining=%d",
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
        raise RuntimeError("Residual Transformer produced no valid checkpoint.")
    model.load_state_dict(best_state)
    prediction, diagnostics = _predict_correction(
        model,
        prepared,
        baseline,
        device,
        batch_size,
        residual_scale,
    )
    metadata = {
        "device": str(device),
        "best_inner_validation_loss": best_loss,
        "epochs_completed": len(history),
        "residual_scale": residual_scale,
        "oof_coverage": float(baseline.oof_mask.mean()),
        "oof_samples": int(baseline.oof_mask.sum()),
        "inner_train_samples": len(inner_train),
        "inner_validation_samples": len(inner_validation),
        "inner_train_target_end": data.train_frame.iloc[
            inner_train
        ]["target_date"].max(),
        "inner_validation_target_start": data.train_frame.iloc[
            inner_validation
        ]["target_date"].min(),
        "history": history,
    }
    return model, prediction, diagnostics, metadata


def _prediction_frame(
    task: Mapping[str, Any],
    prepared: TransformerPrepared,
    prediction: np.ndarray,
    diagnostics: Mapping[str, np.ndarray],
    profile: Mapping[str, Any],
) -> pd.DataFrame:
    data = prepared.attention
    frame = data.validation_frame[
        ["ticker", "feature_date", "target_date"]
    ].copy()
    frame["task_id"] = str(task["task_id"])
    frame["evaluation_split"] = "validation"
    frame["target"] = "volatility_level"
    frame["model"] = str(task["model"])
    frame["prototype_source"] = str(task["prototype_source"])
    frame["fold"] = int(task["fold"])
    frame["seed"] = int(task["seed"])
    frame["y_true"] = data.y_validation
    frame["prediction"] = prediction
    frame["baseline_prediction"] = diagnostics.get(
        "baseline_prediction", np.full(len(frame), np.nan)
    )
    frame["correction"] = diagnostics.get(
        "correction", np.full(len(frame), np.nan)
    )
    frame["gate"] = diagnostics.get(
        "gate", np.full(len(frame), np.nan)
    )
    for level in profile["news_levels"]:
        frame[f"attention__{level}"] = np.nan
    frame["attention_entropy"] = np.nan
    if "attention" in diagnostics:
        attention = np.asarray(diagnostics["attention"], dtype=float).mean(
            axis=1
        )
        semantic = attention[:, 1:]
        expected = len(profile["news_levels"]) * data.k_per_level
        if semantic.shape[1] != expected:
            raise AssertionError(
                f"Attention dimension {semantic.shape[1]} != {expected}."
            )
        semantic = semantic.reshape(
            len(frame), len(profile["news_levels"]), data.k_per_level
        )
        for level_index, level in enumerate(profile["news_levels"]):
            frame[f"attention__{level}"] = semantic[
                :, level_index, :
            ].sum(axis=1)
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
    model_name = str(task["model"])
    prediction_path = root / "checkpoints" / f"{task_id}.parquet"
    metadata_path = root / "checkpoints" / f"{task_id}.json"
    model_path = root / "models" / (
        f"{task_id}.joblib"
        if model_name in LINEAR_MODELS
        else f"{task_id}.pt"
    )
    prepared = prepare_transformer_data(config, profile, task)
    baseline = build_frozen_baseline(prepared.baseline, profile)
    started = time.monotonic()
    diagnostics: dict[str, np.ndarray] = {}
    if model_name in LINEAR_MODELS:
        if model_name == "R6_RIDGE":
            prediction = baseline.validation_prediction
            model: Any = baseline.final_model
            diagnostics = {
                "baseline_prediction": prediction,
                "correction": np.zeros_like(prediction),
            }
            training_metadata: dict[str, Any] = {
                "estimator": "FrozenR6Ridge",
                "oof_coverage": float(baseline.oof_mask.mean()),
            }
            payload = {
                "model": model,
                "imputer": baseline.final_imputer,
                "scaler": baseline.final_scaler,
            }
        else:
            model, prediction = _fit_linear(
                model_name,
                "volatility_level",
                prepared.baseline,
                profile,
            )
            training_metadata = {"estimator": type(model).__name__}
            payload = {"model": model}
        joblib.dump(
            {
                **payload,
                "task": dict(task),
                "baseline_oof_manifest": baseline.oof_manifest,
                "target_metadata": prepared.baseline.target_metadata,
            },
            model_path,
        )
    else:
        model, prediction, diagnostics, training_metadata = (
            train_residual_model(
                model_name,
                prepared,
                baseline,
                profile,
                seed=int(task["seed"]),
                quick=bool(task.get("quick", False)),
                logger=logger,
                epoch_callback=epoch_callback,
            )
        )
        torch.save(
            {
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                "task": dict(task),
                "training_metadata": training_metadata,
                "baseline_model": baseline.final_model,
                "baseline_imputer": baseline.final_imputer,
                "baseline_scaler": baseline.final_scaler,
                "baseline_oof_manifest": baseline.oof_manifest,
                "price_columns": prepared.attention.price_columns,
                "metadata_columns": prepared.attention.metadata_columns,
                "prototype_columns": prepared.attention.prototype_columns,
                "model_config": {
                    key: profile[key]
                    for key in (
                        "d_model",
                        "n_heads",
                        "hidden_dim",
                        "transformer_ff_dim",
                        "market_encoder_layers",
                        "cross_attention_layers",
                        "dropout",
                        "gate_initial_value",
                    )
                },
            },
            model_path,
        )
    predictions = _prediction_frame(
        task, prepared, prediction, diagnostics, profile
    )
    write_table(predictions, prediction_path)
    atomic_write_json(
        {
            "task": dict(task),
            "n_train": len(prepared.attention.train_frame),
            "n_validation": len(prepared.attention.validation_frame),
            "oof_coverage": float(baseline.oof_mask.mean()),
            "oof_manifest": baseline.oof_manifest,
            "training_metadata": training_metadata,
            "elapsed_seconds": time.monotonic() - started,
            "frozen_baseline": "R6_RIDGE",
            "baseline_prototype_source": "R6",
            "attention_prototype_source": str(task["prototype_source"]),
        },
        metadata_path,
    )
    return {
        "predictions": prediction_path,
        "model": model_path,
        "metadata": metadata_path,
    }


def _comparisons(
    results: pd.DataFrame, profile: Mapping[str, Any]
) -> pd.DataFrame:
    primary = str(profile["primary_model"])
    rows: list[dict[str, Any]] = []
    primary_rows = results.loc[
        results["model"].eq(primary)
    ][["fold", "seed", "primary_value"]].rename(
        columns={"primary_value": "primary_value"}
    )
    for reference in map(str, profile["fixed_comparators"]):
        reference_rows = results.loc[
            results["model"].eq(reference)
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
                    "target": "volatility_level",
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
        paired["relative_gain"] = (
            paired["reference_value"] - paired["primary_value"]
        ) / np.maximum(
            np.abs(paired["reference_value"].to_numpy(dtype=float)),
            1.0e-12,
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
                "target": "volatility_level",
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
    result_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for task in plan.itertuples(index=False):
        path = root / "checkpoints" / f"{task.task_id}.parquet"
        if not path.is_file():
            missing.append(str(task.task_id))
            continue
        frame = read_table(path)
        metrics = regression_metrics(
            frame["y_true"].to_numpy(dtype=float),
            frame["prediction"].to_numpy(dtype=float),
        )
        result_rows.append(
            {
                "task_id": task.task_id,
                "target": "volatility_level",
                "model": task.model,
                "prototype_source": task.prototype_source,
                "fold": int(task.fold),
                "seed": int(task.seed),
                "primary_metric": "qlike",
                "primary_value": metrics["qlike"],
                "larger_is_better": False,
                **metrics,
            }
        )
        prediction_frames.append(frame)
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} Transformer tasks are incomplete: {missing[:5]}"
        )
    results = pd.DataFrame(result_rows).sort_values(
        ["model", "fold", "seed"], kind="mergesort"
    )
    aggregate = (
        results.groupby(
            ["target", "model", "primary_metric"],
            observed=True,
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
    transformer_predictions = predictions.loc[
        predictions["model"].isin(TRANSFORMER_MODELS)
    ]
    attention = (
        transformer_predictions.groupby(
            ["target", "model", "fold"], observed=True
        )[
            [
                "gate",
                "correction",
                "attention_entropy",
                *attention_columns,
            ]
        ]
        .agg(["mean", "std"])
    )
    attention.columns = [
        "__".join(map(str, column)).strip("_")
        for column in attention.columns.to_flat_index()
    ]
    attention = attention.reset_index()
    passed = (
        len(comparisons) == len(profile["fixed_comparators"])
        and bool(comparisons["passed"].all())
    )
    decision = pd.DataFrame(
        [
            {
                "target": "volatility_level",
                "decision": (
                    "TRANSFORMER-XATTN-PASS"
                    if passed
                    else "TRANSFORMER-XATTN-FAIL"
                ),
                "all_fixed_comparisons_passed": passed,
                "comparison_count": len(comparisons),
                "locked_holdout_evaluated": False,
                "interpretation": (
                    "Promote the frozen-baseline Transformer correction."
                    if passed
                    else "Do not open the locked holdout."
                ),
            }
        ]
    )
    paths = {
        "results": root / "tables" / "transformer_results.csv",
        "aggregate": root / "tables" / "transformer_aggregate.csv",
        "comparisons": root / "tables" / "transformer_comparisons.csv",
        "attention": root / "tables" / "transformer_attention_audit.csv",
        "decision": root / "tables" / "transformer_decision.csv",
    }
    atomic_write_csv(results, paths["results"], index=False)
    atomic_write_csv(aggregate, paths["aggregate"], index=False)
    atomic_write_csv(comparisons, paths["comparisons"], index=False)
    atomic_write_csv(attention, paths["attention"], index=False)
    atomic_write_csv(decision, paths["decision"], index=False)
    return paths


def print_report(
    config: Mapping[str, Any], profile: Mapping[str, Any]
) -> None:
    path = output_root(config, profile) / "tables" / "transformer_decision.csv"
    if not path.is_file():
        print("PROTOTYPE TRANSFORMER: evaluation has not completed.")
        return
    row = read_table(path).iloc[0]
    print("\nPROTOTYPE TRANSFORMER CROSS-ATTENTION")
    print("Structured-event extraction required: False")
    print("Frozen centre forecast: R6_RIDGE")
    print("Residual target source: expanding-window OOF")
    print("Locked holdout evaluated: False")
    print(f"Decision: {row['decision']}")
    print(f"Next step: {row['interpretation']}")
