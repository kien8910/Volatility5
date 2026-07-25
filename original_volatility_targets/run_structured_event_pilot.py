"""CLI for the fast structured-event forecasting feasibility pilot."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.structured_event_pilot import (  # noqa: E402
    output_root,
    print_report,
    run_evaluation,
    run_extraction,
    run_features,
    run_forecast,
    run_plan,
)
from src.utils import get_logger, load_config, set_global_seed  # noqa: E402


STAGES = ("plan", "extract", "features", "forecast", "evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a development-only pilot that tests whether frozen-LLM structured "
            "target-news events improve q90 semiconductor volatility-spike forecasts."
        )
    )
    parser.add_argument(
        "--stage",
        choices=("all", *STAGES),
        default="all",
        help="Pilot stage to run (default: all).",
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "config_original_volatility.yaml"),
        help="Configuration YAML containing structured_event_pilot.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse valid LLM cache and completed forecast checkpoints.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run the requested stage instead of reusing stage outputs.",
    )
    parser.add_argument(
        "--extractor-model",
        help="Override the configured Hugging Face extractor model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override LLM extraction batch size.",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable bitsandbytes 4-bit loading (requires more GPU memory).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    profile = config["structured_event_pilot"]
    root = output_root(config, profile)
    logger = get_logger(
        "structured_event_pilot",
        config,
        root / "logs" / "structured_event_pilot.log",
    )
    set_global_seed(
        int(config["project"]["seed"]),
        deterministic=bool(config["project"].get("deterministic", True)),
    )
    requested = STAGES if args.stage == "all" else (args.stage,)
    started = time.monotonic()
    logger.info(
        "Structured-event pilot START | stages=%s | resume=%s | force=%s",
        requested,
        args.resume,
        args.force,
    )
    for index, stage in enumerate(requested, start=1):
        stage_started = time.monotonic()
        logger.info("[Stage %d/%d] START %s", index, len(requested), stage)
        if stage == "plan":
            run_plan(config, logger=logger, force=args.force)
        elif stage == "extract":
            run_extraction(
                config,
                logger=logger,
                resume=bool(args.resume),
                force=bool(args.force),
                model_override=args.extractor_model,
                batch_size_override=args.batch_size,
                disable_4bit=bool(args.no_4bit),
            )
        elif stage == "features":
            run_features(config, logger=logger, force=args.force)
        elif stage == "forecast":
            run_forecast(
                config,
                logger=logger,
                resume=bool(args.resume),
                force=bool(args.force),
            )
        elif stage == "evaluate":
            run_evaluation(config, logger=logger)
        else:
            raise ValueError(f"Unsupported stage {stage!r}.")
        logger.info(
            "[Stage %d/%d] DONE %s | elapsed=%.1f seconds",
            index,
            len(requested),
            stage,
            time.monotonic() - stage_started,
        )
    logger.info(
        "Structured-event pilot DONE | total_elapsed=%.1f seconds",
        time.monotonic() - started,
    )
    print_report(config)


if __name__ == "__main__":
    main()
