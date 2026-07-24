"""Persistent task planning, terminal progress, resume, and history logging."""

from __future__ import annotations

import csv
import json
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tqdm.auto import tqdm

from src.utils import atomic_write_json, project_path, stable_id


@dataclass(frozen=True)
class TaskSpec:
    stage: str
    action: str
    config: dict[str, Any] = field(default_factory=dict)
    outputs: tuple[str, ...] = ()
    required: bool = False
    weight: float = 1.0
    task_id: str = ""

    def with_id(self) -> "TaskSpec":
        if self.task_id:
            return self
        payload = {
            "stage": self.stage,
            "action": self.action,
            "config": self.config,
            "outputs": self.outputs,
        }
        return TaskSpec(
            stage=self.stage,
            action=self.action,
            config=self.config,
            outputs=self.outputs,
            required=self.required,
            weight=self.weight,
            task_id=stable_id(payload, prefix=self.stage),
        )


HISTORY_COLUMNS = [
    "timestamp",
    "stage",
    "task_id",
    "fold",
    "seed",
    "target",
    "representation",
    "model",
    "completed_tasks",
    "total_tasks",
    "overall_percent",
    "remaining_percent",
    "elapsed_seconds",
    "estimated_remaining_seconds",
    "status",
]


class ProgressTracker:
    def __init__(
        self,
        config: Mapping[str, Any],
        tasks: Sequence[TaskSpec],
        logger: Any,
        *,
        resume: bool,
        force: bool,
    ) -> None:
        self.config = config
        self.logger = logger
        self.tasks = [task.with_id() for task in tasks]
        identifiers = [task.task_id for task in self.tasks]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Task manifest contains duplicate task IDs")
        progress_config = config["progress"]
        self.manifest_path = project_path(
            config, str(progress_config["task_manifest_file"])
        )
        self.state_path = project_path(config, str(progress_config["state_file"]))
        self.history_path = project_path(
            config, str(progress_config["history_file"])
        )
        self.resume = bool(resume)
        self.force = bool(force)
        self.started = time.monotonic()
        self.state = self._load_state()
        self._write_manifest()
        self.completed_weight = sum(
            task.weight
            for task in self.tasks
            if self._is_completed_and_valid(task)
        )
        self.total_weight = sum(task.weight for task in self.tasks)
        self.completed_count = sum(
            self._is_completed_and_valid(task) for task in self.tasks
        )
        self.processed_weight = self.completed_weight
        self.processed_count = self.completed_count
        self.failed_count = sum(
            self.state.get("tasks", {}).get(task.task_id, {}).get("status")
            == "FAILED"
            for task in self.tasks
        )
        self.skipped_count = 0
        self.stage_names = list(dict.fromkeys(task.stage for task in self.tasks))
        self.stage_totals = {
            stage: sum(task.stage == stage for task in self.tasks)
            for stage in self.stage_names
        }
        self.stage_processed = {
            stage: sum(
                task.stage == stage and self._is_completed_and_valid(task)
                for task in self.tasks
            )
            for stage in self.stage_names
        }
        self._overall = tqdm(
            total=self.total_weight,
            initial=self.completed_weight,
            desc="Overall",
            unit="task",
            dynamic_ncols=True,
            disable=not bool(progress_config.get("enabled", True)),
        )
        self._active_task: TaskSpec | None = None
        self._active_start = 0.0

    def close(self) -> None:
        self._overall.close()

    def _load_state(self) -> dict[str, Any]:
        if self.state_path.exists() and self.resume and not self.force:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise TypeError("progress_state.json must contain an object")
            payload.setdefault("tasks", {})
            return payload
        return {
            "pipeline": str(self.config["project"]["name"]),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "tasks": {},
        }

    def _write_manifest(self) -> None:
        payload = {
            "pipeline": self.config["project"]["name"],
            "config_path": self.config["_config_path"],
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_tasks": len(self.tasks),
            "total_weight": sum(task.weight for task in self.tasks),
            "tasks": [asdict(task) for task in self.tasks],
        }
        atomic_write_json(payload, self.manifest_path)

    def _write_state(self) -> None:
        self.state["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(self.state, self.state_path)

    def _outputs_valid(self, task: TaskSpec) -> bool:
        if not task.outputs:
            return True
        return all(
            project_path(self.config, output).is_file()
            and project_path(self.config, output).stat().st_size > 0
            for output in task.outputs
        )

    def _is_completed_and_valid(self, task: TaskSpec) -> bool:
        record = self.state.get("tasks", {}).get(task.task_id, {})
        return (
            not self.force
            and record.get("status") == "COMPLETED"
            and self._outputs_valid(task)
        )

    def should_skip(self, task: TaskSpec) -> bool:
        task = task.with_id()
        if not self.resume or not self._is_completed_and_valid(task):
            return False
        self.skipped_count += 1
        self.logger.info(
            "RESUME skip completed task %s | stage=%s | config=%s",
            task.task_id,
            task.stage,
            json.dumps(task.config, sort_keys=True, default=str),
        )
        self._history(task, "SKIPPED")
        return True

    def start_task(self, task: TaskSpec) -> None:
        task = task.with_id()
        self._active_task = task
        self._active_start = time.monotonic()
        record = {
            "task_id": task.task_id,
            "stage": task.stage,
            "action": task.action,
            "config": task.config,
            "status": "RUNNING",
            "start_time": datetime.now(timezone.utc).isoformat(),
            "end_time": None,
            "outputs": list(task.outputs),
            "error": None,
        }
        self.state.setdefault("tasks", {})[task.task_id] = record
        self._write_state()
        stage_number = self.stage_names.index(task.stage) + 1
        stage_current = self.stage_processed[task.stage] + 1
        self._overall.set_postfix(
            stage=f"{stage_number}/{len(self.stage_names)}:{task.stage}",
            task=f"{stage_current}/{self.stage_totals[task.stage]}",
            remaining=len(self.tasks) - self.processed_count,
        )
        self.logger.info(
            "[Stage %d/%d: %s] START %s | stage_task=%d/%d | "
            "overall=%.1f%% | remaining=%d/%d (%.1f%%) | elapsed=%s | "
            "ETA=%s | fold=%s | seed=%s | target=%s | representation=%s | "
            "model=%s | config=%s",
            stage_number,
            len(self.stage_names),
            task.stage,
            task.task_id,
            stage_current,
            self.stage_totals[task.stage],
            self.overall_percent,
            len(self.tasks) - self.processed_count,
            len(self.tasks),
            self.remaining_percent,
            _format_seconds(self.elapsed_seconds),
            _format_seconds(self.estimated_remaining_seconds),
            task.config.get("fold", ""),
            task.config.get("seed", ""),
            task.config.get("target", ""),
            task.config.get("representation", ""),
            task.config.get("model", ""),
            json.dumps(task.config, sort_keys=True, default=str),
        )
        self._history(task, "RUNNING")

    def complete_task(
        self,
        task: TaskSpec,
        outputs: Iterable[str | Path] = (),
    ) -> None:
        task = task.with_id()
        output_values = [str(value) for value in outputs] or list(task.outputs)
        record = self.state.setdefault("tasks", {}).setdefault(task.task_id, {})
        record.update(
            {
                "status": "COMPLETED",
                "end_time": datetime.now(timezone.utc).isoformat(),
                "outputs": output_values,
                "elapsed_seconds": time.monotonic() - self._active_start,
                "error": None,
            }
        )
        self.completed_count += 1
        self.completed_weight += task.weight
        self.processed_count += 1
        self.processed_weight += task.weight
        self.stage_processed[task.stage] += 1
        self._overall.update(task.weight)
        self._overall.set_postfix(
            stage=f"{self.stage_names.index(task.stage) + 1}/{len(self.stage_names)}:{task.stage}",
            task=f"{self.stage_processed[task.stage]}/{self.stage_totals[task.stage]}",
            remaining=len(self.tasks) - self.processed_count,
        )
        self._write_state()
        self._history(task, "COMPLETED")
        self.logger.info(
            "[Stage %s] DONE %s | stage task complete | overall=%.1f%% | "
            "remaining=%.1f%% | elapsed=%s | ETA=%s",
            task.stage,
            task.task_id,
            self.overall_percent,
            self.remaining_percent,
            _format_seconds(self.elapsed_seconds),
            _format_seconds(self.estimated_remaining_seconds),
        )
        self._active_task = None

    def fail_task(self, task: TaskSpec, error: BaseException) -> None:
        task = task.with_id()
        formatted = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )
        record = self.state.setdefault("tasks", {}).setdefault(task.task_id, {})
        record.update(
            {
                "status": "FAILED",
                "end_time": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.monotonic() - self._active_start,
                "error": formatted,
                "config": task.config,
            }
        )
        self.failed_count += 1
        self.processed_count += 1
        self.processed_weight += task.weight
        self.stage_processed[task.stage] += 1
        self._overall.update(task.weight)
        self._overall.set_postfix(
            stage=f"{self.stage_names.index(task.stage) + 1}/{len(self.stage_names)}:{task.stage}",
            task=f"{self.stage_processed[task.stage]}/{self.stage_totals[task.stage]}",
            remaining=len(self.tasks) - self.processed_count,
        )
        self._write_state()
        self._history(task, "FAILED")
        self.logger.error(
            "[Stage %s] FAILED %s | config=%s\n%s",
            task.stage,
            task.task_id,
            json.dumps(task.config, sort_keys=True, default=str),
            formatted,
        )
        self._active_task = None

    def log_epoch(
        self,
        task: TaskSpec,
        *,
        epoch: int,
        max_epochs: int,
        train_loss: float,
        validation_loss: float,
        best_validation_loss: float,
        patience_remaining: int,
    ) -> None:
        self.logger.info(
            "EPOCH %s | %d/%d (%.1f%%) | train_loss=%.6f | "
            "validation_loss=%.6f | best=%.6f | patience_remaining=%d",
            task.task_id,
            epoch,
            max_epochs,
            100.0 * epoch / max(max_epochs, 1),
            train_loss,
            validation_loss,
            best_validation_loss,
            patience_remaining,
        )
        self._history(task, f"EPOCH_{epoch}")

    @property
    def elapsed_seconds(self) -> float:
        return max(time.monotonic() - self.started, 0.0)

    @property
    def overall_percent(self) -> float:
        return (
            100.0 * self.processed_weight / self.total_weight
            if self.total_weight
            else 100.0
        )

    @property
    def remaining_percent(self) -> float:
        return max(0.0, 100.0 - self.overall_percent)

    @property
    def estimated_remaining_seconds(self) -> float | None:
        if self.processed_weight <= 0:
            return None
        rate = self.processed_weight / max(self.elapsed_seconds, 1.0e-9)
        return max(self.total_weight - self.processed_weight, 0.0) / rate

    def _history(self, task: TaskSpec, status: str) -> None:
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.history_path.exists()
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": task.stage,
            "task_id": task.task_id,
            "fold": task.config.get("fold", ""),
            "seed": task.config.get("seed", ""),
            "target": task.config.get("target", ""),
            "representation": task.config.get("representation", ""),
            "model": task.config.get("model", ""),
            "completed_tasks": self.completed_count,
            "total_tasks": len(self.tasks),
            "overall_percent": self.overall_percent,
            "remaining_percent": self.remaining_percent,
            "elapsed_seconds": self.elapsed_seconds,
            "estimated_remaining_seconds": self.estimated_remaining_seconds,
            "status": status,
        }
        with self.history_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=HISTORY_COLUMNS)
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def summary(self) -> dict[str, Any]:
        failures = [
            {
                "task_id": task_id,
                "stage": record.get("stage"),
                "config": record.get("config"),
                "error": record.get("error"),
            }
            for task_id, record in self.state.get("tasks", {}).items()
            if record.get("status") == "FAILED"
        ]
        return {
            "completed_tasks": self.completed_count,
            "failed_tasks": self.failed_count,
            "skipped_tasks": self.skipped_count,
            "total_tasks": len(self.tasks),
            "overall_completion": self.overall_percent,
            "failures": failures,
        }


def _format_seconds(value: float | None) -> str:
    if value is None:
        return "calculating..."
    seconds = max(int(round(value)), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
