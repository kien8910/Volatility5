from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .utils import assert_unique, parse_dates_mixed, require_columns


@dataclass(frozen=True)
class FeatureGroups:
    price: list[str]
    metadata: list[str]
    semantic: list[str]

    def model_features(self) -> dict[str, list[str]]:
        return {
            "M0_PRICE": self.price,
            "M1_PRICE_METADATA": [*self.price, *self.metadata],
            "M2_PRICE_SEMANTIC": [*self.price, *self.semantic],
            "M3_PRICE_METADATA_SEMANTIC": [
                *self.price,
                *self.metadata,
                *self.semantic,
            ],
        }


def _discover_columns(
    market: pd.DataFrame,
    metadata: pd.DataFrame,
    semantic: pd.DataFrame,
    config: ExperimentConfig,
) -> FeatureGroups:
    price = [
        column
        for column in market.columns
        if any(column.startswith(prefix) for prefix in config.price_prefixes)
        and not any(token in column for token in config.price_exclude_tokens)
        and pd.api.types.is_numeric_dtype(market[column])
    ]
    metadata_columns = [
        column
        for column in metadata.columns
        if any(column.startswith(f"meta__{level}__") for level in config.metadata_levels)
        and pd.api.types.is_numeric_dtype(metadata[column])
    ]
    semantic_prefix = "raw" if config.semantic_representation == "R2" else "pca"
    semantic_columns = [
        column
        for column in semantic.columns
        if any(
            column.startswith(f"{semantic_prefix}__{level}__")
            for level in config.semantic_levels
        )
        and pd.api.types.is_numeric_dtype(semantic[column])
    ]
    groups = FeatureGroups(price, metadata_columns, semantic_columns)
    if not all([groups.price, groups.metadata, groups.semantic]):
        raise AssertionError(f"An input feature group is empty: {groups}")
    return groups


def prepare_dataset(
    target: pd.DataFrame,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, FeatureGroups, dict[str, Any]]:
    market = pd.read_parquet(config.market_path)
    metadata = pd.read_parquet(config.metadata_path)
    semantic = pd.read_parquet(config.semantic_path)
    for frame in [market, metadata, semantic]:
        frame["feature_date"] = parse_dates_mixed(frame["feature_date"])
        frame["ticker"] = frame["ticker"].astype(str)
        assert_unique(frame, ["ticker", "feature_date"], "source feature panel")

    groups = _discover_columns(market, metadata, semantic, config)
    keys = ["ticker", "feature_date"]
    market_keep = [*keys, *groups.price]
    metadata_keep = [*keys, *groups.metadata]
    semantic_keep = [*keys, *groups.semantic]

    prepared = target.rename(columns={"date": "feature_date"}).merge(
        market[market_keep],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    prepared = prepared.merge(
        metadata[metadata_keep],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    prepared = prepared.merge(
        semantic[semantic_keep],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    prepared = prepared.rename(columns={"feature_date": "date"})
    require_columns(prepared, ["date", "ticker", "t1_target"], "prepared dataset")
    if prepared[groups.price].isna().all(axis=1).any():
        raise AssertionError("Some rows contain no usable price features")
    if prepared[groups.metadata].isna().all(axis=1).any():
        raise AssertionError("Some rows lack the metadata panel join")
    if prepared[groups.semantic].isna().all(axis=1).any():
        raise AssertionError("Some rows lack the semantic panel join")
    if not np.isfinite(prepared["t1_target"]).all():
        raise AssertionError("Prepared target contains non-finite values")

    audit = {
        "rows": int(len(prepared)),
        "dates": int(prepared["date"].nunique()),
        "tickers": int(prepared["ticker"].nunique()),
        "price_feature_count": len(groups.price),
        "metadata_feature_count": len(groups.metadata),
        "semantic_feature_count": len(groups.semantic),
        "semantic_representation": config.semantic_representation,
        "metadata_levels": config.metadata_levels,
        "semantic_levels": config.semantic_levels,
        "feature_missing_counts": {
            "price": int(prepared[groups.price].isna().sum().sum()),
            "metadata": int(prepared[groups.metadata].isna().sum().sum()),
            "semantic": int(prepared[groups.semantic].isna().sum().sum()),
        },
    }
    return prepared, groups, audit
