"""CLI for the validation-only prototype cross-attention experiment."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prototype_cross_attention import (  # noqa: E402
    build_plan,
    ensure_output_directories,
    evaluate,
    print_report,
    run_task,
    validate_plan_artifacts,
)
from src.utils import (  # noqa: E402
    atomic_write_json,
    get_logger,
    load_config,
    read_table,
    set_global_seed,
    write_table,
)


STAGES = ("plan", "train", "evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare price-only, pure news metadata, concatenated R6 and "
            "market-query prototype cross-attention without LLM event extraction."
        )
    )
    parser.add_argument(
        "--stage",
        choices=("all", *STAGES),
        default="all",
    )
    parser.add_argument(
        "--config",
        default=str(
            PROJECT_ROOT / "config" / "config_original_volatility.yaml"
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use one fold, one seed, quick model subset and fewer epochs.",
    )
    parser.add_argument(
        "--target",
        action="append",
        choices=("volatility_level", "spike_q90"),
        help="Target to run; repeat to request both. Defaults to config targets.",
    )
    parser.add_argument(
        "--model",
        action="append",
        help="Model to run; repeat for multiple models.",
    )
    parser.add_argument(
        "--fold",
        action="append",
        type=int,
        help="Chronological fold; repeat for multiple folds.",
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        help="Paired model/prototype seed; repeat for multiple seeds.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip tasks with complete prediction/model/metadata outputs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run selected tasks even when outputs already exist.",
    )
    return parser.parse_args()


def _build_selected_plan(
    config: dict[str, Any],
    profile: dict[str, Any],
    args: argparse.Namespace,
) -> pd.DataFrame:
    folds = args.fold
    seeds = args.seed
    if args.quick:
        folds = folds or [int(profile["folds"][0])]
        seeds = seeds or [int(profile["seeds"][0])]
    return build_plan(
        config,
        profile,
        quick=bool(args.quick),
        targets=args.target,
        models=args.model,
        folds=folds,
        seeds=seeds,
    )


def _task_paths(root: Path, task_id: str, model_name: str) -> tuple[Path, ...]:
    model_suffix = ".joblib" if model_name.endswith("RIDGE") else ".pt"
    return (
        root / "checkpoints" / f"{task_id}.parquet",
        root / "checkpoints" / f"{task_id}.json",
        root / "models" / f"{task_id}{model_suffix}",
    )


def _outputs_valid(root: Path, row: Any) -> bool:
    return all(
        path.is_file() and path.stat().st_size > 0
        for path in _task_paths(root, str(row.task_id), str(row.model))
    )


def _load_state(path: Path, resume: bool, force: bool) -> dict[str, Any]:
    if resume and not force and path.is_file():
        payload = read_table(path) if path.suffix == ".csv" else None
        if payload is not None:
            raise AssertionError("JSON state path unexpectedly points to CSV.")
        import json

        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TypeError("Cross-attention progress state must be an object.")
        value.setdefault("tasks", {})
        return value
    return {
        "experiment": "prototype_cross_attention",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tasks": {},
    }


def _run_train(
    config: dict[str, Any],
    profile: dict[str, Any],
    plan: pd.DataFrame,
    *,
    logger: Any,
    resume: bool,
    force: bool,
) -> None:
    root = ensure_output_directories(config, profile)
    state_path = root / "logs" / "progress_state.json"
    state = _load_state(state_path, resume, force)
    total = len(plan)
    completed_before = sum(
        _outputs_valid(root, row)
        for row in plan.itertuples(index=False)
    ) if resume and not force else 0
    started = time.monotonic()
    durations: list[float] = []
    progress = tqdm(
        total=total,
        initial=completed_before,
        desc="Prototype cross-attention",
        unit="task",
        dynamic_ncols=True,
    )
    try:
        for task_number, row in enumerate(
            plan.itertuples(index=False), start=1
        ):
            task = row._asdict()
            task_id = str(task["task_id"])
            if resume and not force and _outputs_valid(root, row):
                progress.set_postfix(
                    task=f"{task_number}/{total}",
                    status="resume-skip",
                    model=task["model"],
                )
                continue
            task_started = time.monotonic()
            state["tasks"][task_id] = {
                "status": "RUNNING",
                "task": task,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "epoch": 0,
            }
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            atomic_write_json(state, state_path)
            progress.set_postfix(
                task=f"{task_number}/{total}",
                fold=task["fold"],
                seed=task["seed"],
                model=task["model"],
            )
            logger.info(
                "TASK START %d/%d | %s | target=%s | model=%s | "
                "fold=%s | seed=%s",
                task_number,
                total,
                task_id,
                task["target"],
                task["model"],
                task["fold"],
                task["seed"],
            )

            def epoch_callback(epoch_state: dict[str, Any]) -> None:
                state["tasks"][task_id].update(epoch_state)
                state["tasks"][task_id]["status"] = "RUNNING"
                state["updated_at"] = datetime.now(
                    timezone.utc
                ).isoformat()
                atomic_write_json(state, state_path)
                progress.set_postfix(
                    task=f"{task_number}/{total}",
                    model=task["model"],
                    epoch=(
                        f"{epoch_state['epoch']}/"
                        f"{epoch_state['max_epochs']}"
                    ),
                    val=f"{epoch_state['validation_loss']:.4f}",
                )

            try:
                outputs = run_task(
                    config,
                    profile,
                    task,
                    logger=logger,
                    epoch_callback=epoch_callback,
                )
            except Exception as error:
                state["tasks"][task_id].update(
                    {
                        "status": "FAILED",
                        "finished_at": datetime.now(
                            timezone.utc
                        ).isoformat(),
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                state["updated_at"] = datetime.now(
                    timezone.utc
                ).isoformat()
                atomic_write_json(state, state_path)
                logger.exception("TASK FAILED %s", task_id)
                raise
            elapsed = time.monotonic() - task_started
            durations.append(elapsed)
            state["tasks"][task_id].update(
                {
                    "status": "COMPLETED",
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "elapsed_seconds": elapsed,
                    "outputs": {
                        key: str(value) for key, value in outputs.items()
                    },
                }
            )
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            atomic_write_json(state, state_path)
            progress.update(1)
            remaining = total - task_number
            eta = (
                float(sum(durations) / len(durations) * remaining)
                if durations
                else float("nan")
            )
            logger.info(
                "TASK DONE %d/%d | %s | elapsed=%.1fs | estimated_remaining=%.1fs",
                task_number,
                total,
                task_id,
                elapsed,
                eta,
            )
    finally:
        progress.close()
    logger.info(
        "TRAIN DONE | tasks=%d | elapsed=%.1fs",
        total,
        time.monotonic() - started,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    profile = config["prototype_cross_attention"]
    root = ensure_output_directories(config, profile)
    logger = get_logger(
        "prototype_cross_attention",
        config,
        root / "logs" / "prototype_cross_attention.log",
    )
    set_global_seed(
        int(config["project"]["seed"]),
        deterministic=bool(config["project"].get("deterministic", True)),
    )
    plan = _build_selected_plan(config, profile, args)
    plan_path = root / "tables" / "cross_attention_task_plan.csv"
    requested = STAGES if args.stage == "all" else (args.stage,)
    for stage_index, stage in enumerate(requested, start=1):
        logger.info(
            "[Stage %d/%d] START %s | tasks=%d",
            stage_index,
            len(requested),
            stage,
            len(plan),
        )
        if stage == "plan":
            validate_plan_artifacts(config, profile, plan)
            write_table(plan, plan_path)
            print(
                f"Plan OK: {len(plan)} tasks; no structured-event extraction."
            )
        elif stage == "train":
            validate_plan_artifacts(config, profile, plan)
            write_table(plan, plan_path)
            _run_train(
                config,
                profile,
                plan,
                logger=logger,
                resume=bool(args.resume),
                force=bool(args.force),
            )
        elif stage == "evaluate":
            evaluate(config, profile, plan)
        else:
            raise ValueError(f"Unsupported stage: {stage}")
        logger.info(
            "[Stage %d/%d] DONE %s",
            stage_index,
            len(requested),
            stage,
        )
    print_report(config, profile)


if __name__ == "__main__":
    main()
