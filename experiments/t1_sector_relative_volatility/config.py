from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


TICKERS = [
    "ADI",
    "AMAT",
    "AMD",
    "AVGO",
    "INTC",
    "KLAC",
    "LRCX",
    "MU",
    "NVDA",
    "QCOM",
    "TXN",
]


@dataclass
class ExperimentConfig:
    """All parameters required to reproduce the T1 experiment."""

    workspace_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
    )
    source_profile: str = "light"
    tickers: list[str] = field(default_factory=lambda: TICKERS.copy())
    horizon: int = 5
    train_stride: int = 1
    eval_stride: int = 5
    daily_eval_stride: int = 1
    purge_anchor_days: int = 4
    embargo_days: int = 0
    min_cross_section_size: int = 8
    bootstrap_block_lengths: list[int] = field(
        default_factory=lambda: [10, 15, 20]
    )
    n_bootstrap: int = 2000
    hac_lags: list[int] = field(default_factory=lambda: [4, 5, 10, 20])
    seeds: list[int] = field(default_factory=lambda: [42, 123, 2026])
    ridge_alphas: list[float] = field(default_factory=lambda: [0.1, 1.0, 10.0])
    semantic_representation: str = "R3"
    metadata_levels: list[str] = field(default_factory=lambda: ["target"])
    semantic_levels: list[str] = field(default_factory=lambda: ["target"])
    huber_delta: float = 1.0
    bootstrap_confidence: float = 0.95
    debug: bool = False
    debug_train_dates: int = 126
    debug_validation_dates: int = 50
    debug_test_dates: int = 50
    debug_bootstrap: int = 100
    output_directory: Path | None = None

    price_prefixes: tuple[str, ...] = (
        "har_",
        "log_variance_lag_",
        "log_variance_roll_",
        "log_return",
        "absolute_log_return",
        "historical_mean_log_variance",
    )
    price_exclude_tokens: tuple[str, ...] = (
        "target_",
        "residual",
        "spike",
        "regime",
        "future",
    )

    def __post_init__(self) -> None:
        self.workspace_root = Path(self.workspace_root).resolve()
        if self.output_directory is None:
            suffix = "debug" if self.debug else "full"
            self.output_directory = self.experiment_root / "outputs" / suffix
        else:
            self.output_directory = Path(self.output_directory).resolve()
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        if self.purge_anchor_days < self.horizon - 1:
            raise ValueError("purge_anchor_days must be at least horizon - 1")
        if not 2 <= self.min_cross_section_size <= len(self.tickers):
            raise ValueError("min_cross_section_size is outside the ticker universe")
        if self.eval_stride != self.horizon:
            raise ValueError(
                "The confirmatory non-overlapping design requires eval_stride == horizon"
            )
        if self.semantic_representation not in {"R2", "R3"}:
            raise ValueError("semantic_representation must be R2 or R3")

    @property
    def experiment_root(self) -> Path:
        return Path(__file__).resolve().parent

    @property
    def source_processed_directory(self) -> Path:
        return (
            self.workspace_root
            / "fintexts_semiconductor_prototype"
            / "runs"
            / self.source_profile
            / "data"
            / "processed"
        )

    @property
    def market_path(self) -> Path:
        return self.source_processed_directory / "market_supervised.parquet"

    @property
    def metadata_path(self) -> Path:
        return self.source_processed_directory / "features_R1_mean.parquet"

    @property
    def semantic_path(self) -> Path:
        return (
            self.source_processed_directory
            / f"features_{self.semantic_representation}_mean.parquet"
        )

    @property
    def bootstrap_repetitions(self) -> int:
        return self.debug_bootstrap if self.debug else self.n_bootstrap

    def as_serializable_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value)
        return payload
