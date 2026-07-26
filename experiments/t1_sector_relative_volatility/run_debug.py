from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.t1_sector_relative_volatility.config import ExperimentConfig
    from experiments.t1_sector_relative_volatility.run_experiment import run
else:
    from .config import ExperimentConfig
    from .run_experiment import run


def main() -> None:
    run(ExperimentConfig(debug=True))


if __name__ == "__main__":
    main()
