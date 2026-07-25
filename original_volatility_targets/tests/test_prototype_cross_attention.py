from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.prototype_cross_attention import (
    PrototypeCrossAttention,
    _inner_chronological_indices,
    prototype_columns_by_level,
)


def test_prototype_columns_are_level_major_and_contiguous() -> None:
    frame = pd.DataFrame(
        {
            "softproto__macro__0001": np.zeros(2),
            "softproto__macro__0000": np.zeros(2),
            "softproto__target__0001": np.zeros(2),
            "softproto__target__0000": np.zeros(2),
        }
    )
    blocks, dimension = prototype_columns_by_level(
        frame, ("macro", "target")
    )
    assert dimension == 2
    assert blocks["macro"] == [
        "softproto__macro__0000",
        "softproto__macro__0001",
    ]
    assert blocks["target"] == [
        "softproto__target__0000",
        "softproto__target__0001",
    ]


def test_cross_attention_handles_an_all_no_news_batch() -> None:
    torch.manual_seed(7)
    model = PrototypeCrossAttention(
        price_dim=6,
        ticker_dim=3,
        meta_dim=4,
        token_count=8,
        use_meta=True,
        d_model=16,
        n_heads=4,
        hidden_dim=24,
        dropout=0.0,
    )
    prediction, diagnostics = model(
        torch.randn(5, 6),
        torch.eye(3)[torch.tensor([0, 1, 2, 0, 1])],
        torch.randn(5, 4),
        torch.zeros(5, 2, 4),
        torch.zeros(5, 8, dtype=torch.bool),
    )
    assert prediction.shape == (5,)
    assert torch.isfinite(prediction).all()
    assert diagnostics["gate"].shape == (5,)
    assert diagnostics["attention"].shape == (5, 4, 9)
    # Only the always-available null token can receive attention.
    assert torch.allclose(
        diagnostics["attention"][:, :, 0],
        torch.ones(5, 4),
        atol=1.0e-6,
    )
    assert torch.allclose(
        diagnostics["attention"][:, :, 1:],
        torch.zeros(5, 4, 8),
        atol=1.0e-6,
    )


def test_market_state_changes_prototype_attention() -> None:
    torch.manual_seed(17)
    model = PrototypeCrossAttention(
        price_dim=2,
        ticker_dim=1,
        meta_dim=0,
        token_count=2,
        use_meta=False,
        d_model=8,
        n_heads=2,
        hidden_dim=12,
        dropout=0.0,
    )
    model.eval()
    price = torch.tensor([[0.0, 0.0], [4.0, -3.0]])
    _, diagnostics = model(
        price,
        torch.ones(2, 1),
        torch.empty(2, 0),
        torch.tensor([[[0.8, 0.2]], [[0.8, 0.2]]]),
        torch.ones(2, 2, dtype=torch.bool),
    )
    attention = diagnostics["attention"].mean(dim=1)
    assert not torch.allclose(attention[0], attention[1])


def test_inner_early_stopping_split_is_chronological() -> None:
    dates = pd.date_range("2020-01-01", periods=40, freq="B")
    frame = pd.DataFrame(
        {
            "target_date": np.repeat(dates, 2),
            "ticker": ["AMD", "NVDA"] * len(dates),
        }
    )
    train_indices, validation_indices = _inner_chronological_indices(
        frame, 0.20
    )
    assert (
        frame.iloc[train_indices]["target_date"].max()
        < frame.iloc[validation_indices]["target_date"].min()
    )
    assert len(frame.iloc[validation_indices]["target_date"].unique()) == 8
