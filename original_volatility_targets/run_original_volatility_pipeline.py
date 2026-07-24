"""Command-line orchestrator for the isolated original-volatility experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import (  # noqa: E402
    evaluate_r6_confirmatory,
    evaluate_original_targets,
    prepare_targets,
    train_sector_targets,
    train_volatility_level,
    train_volatility_regime,
    train_volatility_spike,
    train_volatility_uncertainty,
)
from src.progress_tracker import ProgressTracker, TaskSpec  # noqa: E402
from src.utils import (  # noqa: E402
    ensure_directories,
    get_logger,
    load_config,
    project_path,
    read_table,
    set_global_seed,
)


STAGES = (
    "prepare",
    "level",
    "spike",
    "regime",
    "uncertainty",
    "sector",
    "evaluate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Forecast direct original-volatility targets using cached FinTexTS "
            "news representations without modifying the residual pipeline."
        )
    )
    parser.add_argument(
        "--stage",
        choices=("all", *STAGES),
        default="all",
        help="Pipeline stage to execute (default: all).",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--quick",
        action="store_true",
        help="Small chronological smoke-test grid (default).",
    )
    mode.add_argument(
        "--full",
        action="store_true",
        help="Full seeds/folds/representations/model experiment grid.",
    )
    mode.add_argument(
        "--r6-confirmatory",
        action="store_true",
        help=(
            "Locked R6 follow-up: 3 chronological folds x 5 paired "
            "prototype/model seeds against fixed comparators."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed tasks whose declared outputs are still valid.",
    )
    parser.add_argument("--fold", action="append", help="Keep only this fold.")
    parser.add_argument("--seed", action="append", type=int, help="Keep only this seed.")
    parser.add_argument("--target", action="append", help="Keep matching target.")
    parser.add_argument(
        "--representation",
        action="append",
        help="Keep only this representation (for example R7).",
    )
    parser.add_argument("--model", action="append", help="Keep only this model.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run selected tasks even when valid checkpoints exist.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore this pipeline's task cache; shared residual artifacts stay read-only.",
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "config_original_volatility.yaml"),
        help="Path to YAML configuration.",
    )
    return parser.parse_args()


def _prepare_tasks() -> list[TaskSpec]:
    return [
        TaskSpec(
            stage="prepare",
            action="verify_shared",
            required=True,
            weight=1.0,
            outputs=(
                "outputs/tables/shared_artifact_compatibility.csv",
                "data/processed/shared_artifact_contract.json",
            ),
        ).with_id(),
        TaskSpec(
            stage="prepare",
            action="market_targets",
            required=True,
            weight=2.0,
            outputs=(
                "data/processed/original_market_targets.parquet",
                "outputs/tables/original_target_thresholds.csv",
                "data/processed/original_target_contract.json",
                "outputs/tables/original_target_summary.csv",
                "outputs/figures/volatility_distribution.png",
                "outputs/figures/volatility_by_ticker.png",
                "outputs/figures/volatility_spike_rate.png",
                "outputs/figures/volatility_regime_timeline.png",
            ),
        ).with_id(),
        TaskSpec(
            stage="prepare",
            action="sector_targets",
            required=True,
            weight=1.0,
            outputs=(
                "data/processed/sector_targets.parquet",
                "outputs/tables/sector_target_summary.csv",
                "outputs/figures/sector_volatility_breadth.png",
                "outputs/figures/sector_regime.png",
            ),
        ).with_id(),
    ]


def _evaluation_tasks(
    config: Mapping[str, Any], mode: str
) -> list[TaskSpec]:
    if mode == "r6_confirmatory":
        return [
            TaskSpec(
                stage="evaluate",
                action="r6_confirmatory",
                config={"experiment_profile": mode},
                required=True,
                weight=1.0,
                outputs=evaluate_r6_confirmatory.evaluation_outputs(),
            ).with_id()
        ]
    outputs = evaluate_original_targets.evaluation_outputs(config)
    return [
        TaskSpec(
            stage="evaluate",
            action="aggregate",
            required=True,
            weight=2.0,
            outputs=outputs["aggregate"],
        ).with_id(),
        TaskSpec(
            stage="evaluate",
            action="compare",
            required=True,
            weight=1.0,
            outputs=outputs["compare"],
        ).with_id(),
        TaskSpec(
            stage="evaluate",
            action="figures",
            required=True,
            weight=2.0,
            outputs=outputs["figures"],
        ).with_id(),
        TaskSpec(
            stage="evaluate",
            action="report",
            required=True,
            weight=1.0,
            outputs=("outputs/tables/final_report.json",),
        ).with_id(),
    ]


def plan_all_tasks(
    config: Mapping[str, Any],
    profile: Mapping[str, Any],
    *,
    quick: bool,
    stage: str = "all",
    mode: str = "quick",
) -> list[TaskSpec]:
    enabled = tuple(profile.get("enabled_stages", STAGES))
    unknown_enabled = sorted(set(enabled).difference(STAGES))
    if unknown_enabled:
        raise ValueError(
            f"Profile contains unknown enabled stages: {unknown_enabled}"
        )
    if stage != "all" and stage not in enabled:
        raise ValueError(
            f"Stage {stage!r} is disabled for experiment mode {mode!r}; "
            f"choose from {enabled}."
        )
    requested = set(enabled if stage == "all" else [stage])
    tasks: list[TaskSpec] = []
    if "prepare" in requested:
        tasks.extend(_prepare_tasks())
    if "level" in requested:
        tasks.extend(
            train_volatility_level.plan_tasks(config, profile, quick=quick)
        )
    if "spike" in requested:
        tasks.extend(
            train_volatility_spike.plan_tasks(config, profile, quick=quick)
        )
    if "regime" in requested:
        tasks.extend(
            train_volatility_regime.plan_tasks(config, profile, quick=quick)
        )
    if "uncertainty" in requested:
        tasks.extend(
            train_volatility_uncertainty.plan_tasks(
                config, profile, quick=quick
            )
        )
    if "sector" in requested:
        tasks.extend(
            train_sector_targets.plan_tasks(config, profile, quick=quick)
        )
    if "evaluate" in requested:
        tasks.extend(_evaluation_tasks(config, mode))
    return tasks


def _matches(value: Any, requested: list[Any] | None) -> bool:
    return requested is None or str(value) in {str(item) for item in requested}


def filter_tasks(
    tasks: list[TaskSpec],
    args: argparse.Namespace,
) -> list[TaskSpec]:
    selected_stages = set(STAGES if args.stage == "all" else [args.stage])
    filtered = []
    for task in tasks:
        if task.stage not in selected_stages:
            continue
        if task.stage in {"prepare", "evaluate"}:
            filtered.append(task)
            continue
        if not _matches(task.config.get("fold"), args.fold):
            continue
        if not _matches(task.config.get("seed"), args.seed):
            continue
        if not _matches(task.config.get("representation"), args.representation):
            continue
        if not _matches(task.config.get("model"), args.model):
            continue
        if args.target and not any(
            requested in str(task.config.get("target", ""))
            for requested in args.target
        ):
            continue
        filtered.append(task)
    if not filtered:
        raise ValueError("No task matches the requested stage/filter combination")
    return filtered


def _runner_for_task(
    task: TaskSpec,
) -> Callable[
    [TaskSpec, Mapping[str, Any], Mapping[str, Any], ProgressTracker],
    Mapping[str, Any],
]:
    mapping = {
        "level": train_volatility_level.run_task,
        "spike": train_volatility_spike.run_task,
        "regime": train_volatility_regime.run_task,
        "uncertainty": train_volatility_uncertainty.run_task,
        "sector": train_sector_targets.run_task,
    }
    if task.stage not in mapping:
        raise ValueError(f"No training runner for stage {task.stage}")
    return mapping[task.stage]


def _execute(
    task: TaskSpec,
    config: Mapping[str, Any],
    profile: Mapping[str, Any],
    tracker: ProgressTracker,
    mode: str,
) -> Mapping[str, Any]:
    if task.stage == "prepare":
        return prepare_targets.run_action(task.action, config)
    if task.stage == "evaluate":
        if task.action == "r6_confirmatory":
            return evaluate_r6_confirmatory.run(config)
        return evaluate_original_targets.run_action(
            task.action, config, mode=mode
        )
    return _runner_for_task(task)(task, config, profile, tracker)


def _best_representation(
    config: Mapping[str, Any], filename: str
) -> str:
    path = project_path(config, "outputs", "tables", filename)
    if not path.exists() or not path.stat().st_size:
        return "unavailable"
    frame = read_table(path)
    if frame.empty:
        return "unavailable"
    if "fold" in frame:
        holdout = frame.loc[frame["fold"].astype(str) == "holdout"]
        if not holdout.empty:
            frame = holdout
    larger = bool(frame["larger_is_better"].iloc[0])
    row = frame.sort_values(
        ["primary_value", "task_id"],
        ascending=[not larger, True],
        kind="mergesort",
    ).iloc[0]
    return (
        f"{row['representation']} ({row.get('model', '')}; "
        f"{row['primary_metric']}={float(row['primary_value']):.6g})"
    )


def _print_completion_report(
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    mode: str,
) -> None:
    if mode == "r6_confirmatory":
        decision_path = project_path(
            config,
            "outputs",
            "tables",
            "r6_confirmatory_decision.csv",
        )
        if not decision_path.exists():
            return
        row = read_table(decision_path).iloc[0]
        print("\nR6 CONFIRMATORY EXPERIMENT")
        print(f"Completed tasks: {summary['completed_tasks']}")
        print(f"Failed tasks: {summary['failed_tasks']}")
        print(f"Skipped tasks: {summary['skipped_tasks']}")
        print(
            f"Grid: {int(row['fold_count'])} folds x "
            f"{int(row['prototype_seed_count'])} paired seeds"
        )
        print(
            "All fixed comparisons passed: "
            f"{bool(row['all_comparisons_passed'])}"
        )
        print(f"Decision: {row['decision']}")
        print(f"Next step: {row['next_step']}")
        return
    decision_path = project_path(
        config,
        "outputs",
        "tables",
        "final_original_target_decision.csv",
    )
    if not decision_path.exists():
        return
    decision = read_table(decision_path)
    final = decision.loc[decision["record_type"].astype(str) == "final"]
    if len(final) != 1:
        return
    row = final.iloc[0]
    sector_path = project_path(
        config, "outputs", "tables", "sector_target_results.csv"
    )
    sector_best = "unavailable"
    if sector_path.exists() and sector_path.stat().st_size:
        sector = read_table(sector_path)
        validation = sector.loc[
            sector["evaluation_split"].astype(str) == "validation"
        ]
        if not validation.empty:
            pieces = []
            for target, group in validation.groupby(
                "target", sort=True, observed=True
            ):
                larger = bool(group["larger_is_better"].iloc[0])
                best = group.sort_values(
                    ["primary_value", "task_id"],
                    ascending=[not larger, True],
                    kind="mergesort",
                ).iloc[0]
                pieces.append(
                    f"{target}:{best['representation']}="
                    f"{float(best['primary_value']):.5g}"
                )
            sector_best = "; ".join(pieces)
    print("\nORIGINAL VOLATILITY TARGET EXPERIMENT")
    print(f"Completed tasks: {summary['completed_tasks']}")
    print(f"Failed tasks: {summary['failed_tasks']}")
    print(f"Skipped tasks: {summary['skipped_tasks']}")
    print(f"Overall completion: {summary['overall_completion']:.1f}%")
    print(
        "Best volatility-level representation: "
        + _best_representation(
            config, "volatility_level_results_validation.csv"
        )
    )
    print(
        "Best spike representation: "
        + _best_representation(
            config, "volatility_spike_results_validation.csv"
        )
    )
    print(
        "Best regime representation: "
        + _best_representation(
            config, "volatility_regime_results_validation.csv"
        )
    )
    print(
        "Best uncertainty representation: "
        + _best_representation(
            config, "volatility_uncertainty_results_validation.csv"
        )
    )
    print(f"Best sector-wide representation: {sector_best}")
    print(
        "Prototype better than price-only: "
        f"{row.get('prototype_better_than_price_only', False)}"
    )
    print(
        "Prototype better than raw embedding: "
        f"{row.get('prototype_better_than_raw_embedding', False)}"
    )
    print(
        "Prototype better than PCA/random projection: "
        f"{row.get('prototype_better_than_pca_random_projection', False)}"
    )
    print(
        "True news better than shuffled news: "
        f"{row.get('true_news_better_than_shuffled', False)}"
    )
    print(
        "Stable across folds and seeds: "
        f"{row.get('stable_across_folds_seeds', False)}"
    )
    comparison_path = project_path(
        config,
        "outputs",
        "tables",
        "comparison_with_residual_targets.csv",
    )
    if comparison_path.exists() and comparison_path.stat().st_size:
        comparison = read_table(comparison_path)
        print("Comparison with residual pipeline:")
        for interpretation in (
            "better_for_original_only",
            "better_for_residual_only",
            "better_for_both",
            "better_for_neither_or_evidence_missing",
        ):
            values = comparison.loc[
                comparison["interpretation"].astype(str) == interpretation,
                "original_target",
            ].astype(str)
            print(
                f"- {interpretation}: "
                + (", ".join(values) if len(values) else "none")
            )
    print(f"Final decision: {row.get('decision', 'NO-GO')}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_directories(config)
    mode = (
        "r6_confirmatory"
        if args.r6_confirmatory
        else "full"
        if args.full
        else "quick"
    )
    profile = config[mode]
    set_global_seed(
        int(config["project"]["seed"]),
        bool(config["project"]["deterministic"]),
    )
    logger = get_logger(
        "pipeline",
        config,
        project_path(config, "outputs", "logs", "pipeline.log"),
    )
    all_tasks = plan_all_tasks(
        config,
        profile,
        quick=(mode == "quick"),
        stage=args.stage,
        mode=mode,
    )
    tasks = filter_tasks(all_tasks, args)
    logger.info(
        "PLANNED %d selected tasks (%d in complete %s grid) | mode=%s | "
        "stage=%s | shared residual outputs are READ-ONLY",
        len(tasks),
        len(all_tasks),
        mode,
        mode,
        args.stage,
    )
    tracker = ProgressTracker(
        config,
        tasks,
        logger,
        resume=bool(args.resume),
        force=bool(args.force or args.no_cache),
    )
    integrity_failure: BaseException | None = None
    try:
        for task in tasks:
            if tracker.should_skip(task):
                continue
            tracker.start_task(task)
            try:
                result = _execute(task, config, profile, tracker, mode)
                tracker.complete_task(task, result.values())
            except Exception as error:
                tracker.fail_task(task, error)
                if task.required or not bool(
                    config["project"]["continue_on_experiment_error"]
                ):
                    integrity_failure = error
                    break
        summary = tracker.summary()
        logger.info(
            "PIPELINE SUMMARY | completed=%d | failed=%d | skipped=%d | "
            "selected_total=%d | overall=%.1f%%",
            summary["completed_tasks"],
            summary["failed_tasks"],
            summary["skipped_tasks"],
            summary["total_tasks"],
            summary["overall_completion"],
        )
        if summary["failures"]:
            logger.error(
                "FAILED TASKS:\n%s",
                json.dumps(summary["failures"], indent=2, default=str),
            )
        if integrity_failure is None and any(
            task.stage == "evaluate" for task in tasks
        ):
            _print_completion_report(config, summary, mode)
    finally:
        tracker.close()
    if integrity_failure is not None:
        raise integrity_failure


if __name__ == "__main__":
    main()
