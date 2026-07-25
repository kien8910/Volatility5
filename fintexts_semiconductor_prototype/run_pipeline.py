"""Command-line orchestrator for the FinTexTS prototype experiment."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils import ensure_directories, get_logger, load_config, project_path, write_json

STAGE_GROUPS: dict[str, list[str]] = {
    "download": ["download_data", "inspect_schema"],
    "market": ["preprocess_market"],
    "baseline": ["preprocess_market", "build_baseline", "generate_residuals"],
    "news": ["preprocess_news", "build_events"],
    "embeddings": ["build_embeddings"],
    "prototypes": [
        "preprocess_news",
        "build_events",
        "build_embeddings",
        "build_prototypes",
        "aggregate_features",
    ],
    "r6-confirmatory": ["build_fold_representations"],
    "target-mechanism-artifacts": ["build_target_mechanism_artifacts"],
    "targets": ["train_targets"],
    "evaluate": ["evaluate_targets", "analyze_prototypes", "placebo_tests"],
}
STAGE_GROUPS["all"] = [
    "download_data",
    "inspect_schema",
    "preprocess_market",
    "build_baseline",
    "generate_residuals",
    "preprocess_news",
    "build_events",
    "build_embeddings",
    "build_prototypes",
    "aggregate_features",
    "train_targets",
    "evaluate_targets",
    "analyze_prototypes",
    "placebo_tests",
]


def _run_module(module_name: str, config: dict[str, Any]) -> dict[str, Path]:
    module = importlib.import_module(f"src.{module_name}")
    runner = getattr(module, "run", None)
    if not callable(runner):
        raise AttributeError(f"src.{module_name} must expose run(config)")
    result = runner(config)
    if not isinstance(result, dict):
        raise TypeError(f"src.{module_name}.run must return dict[str, Path]")
    return {str(key): Path(value) for key, value in result.items()}


def _print_final_report(config: dict[str, Any]) -> None:
    decision_path = project_path(config, "outputs", "tables", "final_decision.csv")
    if not decision_path.exists():
        return
    decision = pd.read_csv(decision_path)
    if decision.empty:
        return
    final_rows = decision.loc[
        decision.get("record_type", pd.Series(index=decision.index, dtype=str))
        .astype(str)
        .eq("final")
    ]
    row = final_rows.iloc[-1] if not final_rows.empty else decision.iloc[-1]
    questions = [
        ("best_baseline", "1. Baseline tốt nhất"),
        ("prototype_vs_raw", "2. Prototype tốt hơn raw embedding"),
        ("prototype_vs_pca_random_projection", "3. Prototype tốt hơn PCA/random projection"),
        ("true_vs_shuffled", "4. Tin thật tốt hơn shuffled news"),
        ("stable", "5. Ổn định qua seed/fold"),
        ("best_news_level", "6. News level hữu ích nhất"),
        ("cross_stock_common_effect", "7. Có common semiconductor effect"),
        ("best_target", "8. Mục tiêu phù hợp nhất"),
        ("decision", "9. Kết luận"),
    ]
    print("\nFINAL EXPERIMENT REPORT")
    for column, label in questions:
        value = row.get(column, "not_available")
        print(f"{label}: {value}")


def _validate_required_artifacts(config: dict[str, Any]) -> None:
    output_config = config.get("outputs", {})
    tables = [
        project_path(config, "outputs", "tables", str(name))
        for name in output_config.get("required_tables", [])
    ]
    figures = [
        project_path(config, "outputs", "figures", str(name))
        for name in output_config.get("required_figures", [])
    ]
    missing = [str(path) for path in [*tables, *figures] if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Pipeline completed its modules but required artifacts are missing: "
            + "; ".join(missing)
        )


def run_stage_group(
    stage: str,
    config: dict[str, Any],
    dry_run: bool = False,
) -> dict[str, Any]:
    if stage not in STAGE_GROUPS:
        raise ValueError(f"Unknown stage {stage!r}; choose from {sorted(STAGE_GROUPS)}")
    ensure_directories(config)
    logger = get_logger(
        "run_pipeline",
        config,
        project_path(config, "outputs", "logs", "pipeline.log"),
    )
    modules = STAGE_GROUPS[stage]
    if dry_run:
        print(json.dumps({"stage": stage, "modules": modules}, indent=2))
        return {"stage": stage, "dry_run": True, "modules": modules}

    record: dict[str, Any] = {
        "stage": stage,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": config["_config_path"],
        "modules": [],
        "status": "running",
    }
    manifest_path = project_path(config, "outputs", "logs", "last_run_manifest.json")
    write_json(record, manifest_path)
    try:
        for module_name in modules:
            started = time.perf_counter()
            logger.info("Starting stage module: %s", module_name)
            outputs = _run_module(module_name, config)
            elapsed = time.perf_counter() - started
            missing_outputs = [
                str(path) for path in outputs.values() if not Path(path).exists()
            ]
            if missing_outputs:
                raise RuntimeError(
                    f"{module_name} returned outputs that do not exist: {missing_outputs}"
                )
            record["modules"].append(
                {
                    "module": module_name,
                    "elapsed_seconds": elapsed,
                    "outputs": {key: str(value) for key, value in outputs.items()},
                }
            )
            write_json(record, manifest_path)
            logger.info("Completed %s in %.1f seconds", module_name, elapsed)
    except (ValueError, TypeError, KeyError, FileNotFoundError, RuntimeError, AssertionError):
        record["status"] = "failed"
        record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(record, manifest_path)
        logger.exception("Pipeline stopped at a validated failure")
        raise
    if stage in {"evaluate", "all"}:
        try:
            _validate_required_artifacts(config)
        except FileNotFoundError:
            record["status"] = "failed"
            record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(record, manifest_path)
            logger.exception("Required artifact validation failed")
            raise
        _print_final_report(config)
    record["status"] = "completed"
    record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(record, manifest_path)
    return record


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=sorted(STAGE_GROUPS),
        help="Pipeline stage group to execute",
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "config" / "config.yaml"),
        help="Path to the YAML configuration",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the ordered modules without loading data or models",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    run_stage_group(args.stage, config, dry_run=args.dry_run)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
