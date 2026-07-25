"""CLI for frozen-R6 Transformer prototype cross-attention V2."""

from __future__ import annotations

import argparse
import json
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

from src.prototype_transformer_cross_attention import (  # noqa: E402
    LINEAR_MODELS,
    build_plan,
    ensure_output_directories,
    evaluate,
    print_report,
    run_task,
    validate_artifacts,
)
from src.utils import (  # noqa: E402
    atomic_write_json,
    get_logger,
    load_config,
    set_global_seed,
    write_table,
)


STAGES = ("plan", "train", "evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a decoder-style Transformer cross-attention residual "
            "correction on a frozen fold-safe R6-Ridge forecast."
        )
    )
    parser.add_argument(
        "--stage", choices=("all", *STAGES), default="all"
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
        help="One fold, one seed, five models and at most 20 epochs.",
    )
    parser.add_argument(
        "--model",
        action="append",
        help="Model name; repeat to run multiple selected models.",
    )
    parser.add_argument("--fold", action="append", type=int)
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _selected_plan(
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
        models=args.model,
        folds=folds,
        seeds=seeds,
    )


def _task_paths(
    root: Path, task_id: str, model_name: str
) -> tuple[Path, ...]:
    suffix = ".joblib" if model_name in LINEAR_MODELS else ".pt"
    return (
        root / "checkpoints" / f"{task_id}.parquet",
        root / "checkpoints" / f"{task_id}.json",
        root / "models" / f"{task_id}{suffix}",
    )


def _outputs_valid(root: Path, row: Any) -> bool:
    return all(
        path.is_file() and path.stat().st_size > 0
        for path in _task_paths(root, str(row.task_id), str(row.model))
    )


def _load_state(
    path: Path, *, resume: bool, force: bool
) -> dict[str, Any]:
    if resume and not force and path.is_file():
        state = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(state, dict):
            raise TypeError("Transformer progress state must be an object.")
        state.setdefault("tasks", {})
        return state
    return {
        "experiment": "prototype_transformer_cross_attention",
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
    state = _load_state(state_path, resume=resume, force=force)
    completed_before = (
        sum(
            _outputs_valid(root, row)
            for row in plan.itertuples(index=False)
        )
        if resume and not force
        else 0
    )
    total = len(plan)
    durations: list[float] = []
    started = time.monotonic()
    progress = tqdm(
        total=total,
        initial=completed_before,
        desc="Prototype Transformer",
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
                "TASK START %d/%d | %s | model=%s | fold=%s | seed=%s | "
                "frozen_baseline=R6_RIDGE | memory=%s",
                task_number,
                total,
                task_id,
                task["model"],
                task["fold"],
                task["seed"],
                task["prototype_source"],
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
                    inner_val=(
                        f"{epoch_state['validation_loss']:.4f}"
                    ),
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
                "TASK DONE %d/%d | %s | elapsed=%.1fs | ETA=%.1fs",
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
    profile = config["prototype_transformer_cross_attention"]
    root = ensure_output_directories(config, profile)
    logger = get_logger(
        "prototype_transformer_cross_attention",
        config,
        root / "logs" / "prototype_transformer.log",
    )
    set_global_seed(
        int(config["project"]["seed"]),
        deterministic=bool(config["project"].get("deterministic", True)),
    )
    plan = _selected_plan(config, profile, args)
    plan_path = root / "tables" / "transformer_task_plan.csv"
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
            validate_artifacts(config, profile, plan)
            write_table(plan, plan_path)
            print(
                f"Plan OK: {len(plan)} tasks; frozen R6-Ridge + OOF residual."
            )
        elif stage == "train":
            validate_artifacts(config, profile, plan)
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
