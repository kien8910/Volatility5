from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.prototype_transformer_cross_attention import (
    PrototypeTransformerResidual,
    TransformerCrossAttentionBlock,
    _hybrid_loss,
    expanding_oof_blocks,
)


def test_decoder_cross_attention_block_has_transformer_sublayers() -> None:
    block = TransformerCrossAttentionBlock(
        d_model=16,
        n_heads=4,
        ff_dim=32,
        dropout=0.0,
    )
    assert isinstance(block.cross_attention, nn.MultiheadAttention)
    assert isinstance(block.query_norm, nn.LayerNorm)
    assert isinstance(block.memory_norm, nn.LayerNorm)
    assert isinstance(block.ffn_norm, nn.LayerNorm)
    query = torch.randn(3, 5, 16, requires_grad=True)
    memory = torch.randn(3, 7, 16, requires_grad=True)
    active = torch.ones(3, 7, dtype=torch.bool)
    output, weights = block(query, memory, active)
    assert output.shape == query.shape
    assert weights.shape == (3, 4, 5, 7)
    output.sum().backward()
    assert query.grad is not None
    assert memory.grad is not None


def test_transformer_uses_only_null_memory_when_no_news() -> None:
    torch.manual_seed(19)
    model = PrototypeTransformerResidual(
        price_dim=6,
        ticker_dim=3,
        meta_dim=4,
        level_count=2,
        k_per_level=4,
        use_meta=True,
        d_model=16,
        n_heads=4,
        hidden_dim=24,
        ff_dim=32,
        market_encoder_layers=1,
        cross_attention_layers=2,
        dropout=0.0,
        gate_initial_value=0.05,
    )
    correction, diagnostics = model(
        torch.randn(5, 6),
        torch.eye(3)[torch.tensor([0, 1, 2, 0, 1])],
        torch.randn(5, 4),
        torch.zeros(5, 2, 4),
        torch.zeros(5, 8, dtype=torch.bool),
    )
    assert correction.shape == (5,)
    assert torch.isfinite(correction).all()
    assert torch.allclose(
        diagnostics["gate"],
        torch.full((5,), 0.05),
        atol=1.0e-6,
    )
    attention = diagnostics["attention"]
    assert attention.shape == (5, 4, 9)
    assert torch.allclose(
        attention[:, :, 0], torch.ones(5, 4), atol=1.0e-6
    )
    assert torch.allclose(
        attention[:, :, 1:], torch.zeros(5, 4, 8), atol=1.0e-6
    )


def test_expanding_oof_blocks_never_use_future_dates() -> None:
    dates = pd.date_range("2020-01-01", periods=60, freq="B")
    target_dates = pd.Series(np.repeat(dates, 2))
    blocks = expanding_oof_blocks(
        target_dates,
        split_count=5,
        minimum_train_fraction=0.40,
    )
    assert len(blocks) == 5
    predicted: list[int] = []
    for history, prediction in blocks:
        assert target_dates.iloc[history].max() < target_dates.iloc[
            prediction
        ].min()
        predicted.extend(prediction.tolist())
    assert len(predicted) == len(set(predicted))
    assert len(predicted) / len(target_dates) == 0.60


def test_hybrid_loss_penalizes_a_bad_residual_correction() -> None:
    residual = torch.tensor([0.2, -0.1])
    baseline = torch.tensor([-5.0, -4.0])
    y_true = baseline + residual
    scale = 0.2
    correct = residual / scale
    wrong = -correct
    correct_loss = _hybrid_loss(
        correct,
        residual,
        baseline,
        y_true,
        scale,
        0.25,
    )
    wrong_loss = _hybrid_loss(
        wrong,
        residual,
        baseline,
        y_true,
        scale,
        0.25,
    )
    assert correct_loss < wrong_loss
