from __future__ import annotations

import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml
from tqdm.auto import tqdm


LOGGER_NAME = "t1_sector_relative_volatility"


def configure_logging(output_directory: Path) -> logging.Logger:
    output_directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(
        output_directory / "pipeline.log", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        return


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def write_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def parse_dates_mixed(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="raise")
    return pd.to_datetime(series, format="mixed", errors="raise")


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise KeyError(f"{label} is missing required columns: {missing}")


def assert_unique(frame: pd.DataFrame, keys: list[str], label: str) -> None:
    duplicates = frame.duplicated(keys, keep=False)
    if duplicates.any():
        sample = frame.loc[duplicates, keys].head(10).to_dict("records")
        raise AssertionError(f"{label} has duplicate keys {keys}: {sample}")


def stage_message(index: int, total: int, label: str) -> None:
    print(f"\nStage {index}/{total}: {label}", flush=True)


def progress(
    iterable: Iterable[Any],
    *,
    total: int | None = None,
    description: str,
) -> tqdm:
    return tqdm(
        iterable,
        total=total,
        desc=description,
        unit="step",
        dynamic_ncols=True,
        mininterval=0.5,
    )


class StageTimer:
    def __init__(self, name: str) -> None:
        self.name = name
        self.started = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.started

