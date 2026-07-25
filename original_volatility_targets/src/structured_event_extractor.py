"""Frozen local-LLM extraction with resumable cache and visible ETA."""

from __future__ import annotations

import json
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
from tqdm.auto import tqdm

from src.structured_event_schema import (
    build_messages,
    extraction_cache_key,
    ontology_digest,
    validate_response,
)
from src.utils import atomic_write_json, write_table


def _load_cache(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = line.strip()
            if not value:
                continue
            try:
                row = json.loads(value)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid extraction cache JSON at {path}:{line_number}"
                ) from error
            key = str(row.get("cache_key", ""))
            if not key:
                raise ValueError(f"Missing cache_key at {path}:{line_number}")
            records[key] = row
    return records


def _append_cache(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        handle.flush()


def _resolve_revision(model_id: str, configured_revision: Any) -> str:
    if configured_revision is not None and str(configured_revision).strip():
        return str(configured_revision).strip()
    local_path = Path(model_id)
    if local_path.is_dir():
        config_path = local_path / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Local extractor directory lacks config.json: {local_path}"
            )
        digest = hashlib.sha256(config_path.read_bytes()).hexdigest()[:16]
        return f"local-config-{digest}"
    from huggingface_hub import HfApi

    info = HfApi().model_info(model_id)
    if not info.sha:
        raise RuntimeError(f"Hugging Face did not return a revision SHA for {model_id}.")
    return str(info.sha)


def _torch_dtype(torch_module: Any) -> Any:
    if not torch_module.cuda.is_available():
        return torch_module.float32
    major, _ = torch_module.cuda.get_device_capability()
    return torch_module.bfloat16 if major >= 8 else torch_module.float16


def load_local_extractor(
    extractor_config: Mapping[str, Any],
    logger: Any,
    *,
    model_override: str | None = None,
    batch_size_override: int | None = None,
    disable_4bit: bool = False,
) -> tuple[Any, Any, dict[str, Any]]:
    """Load a frozen causal LM once, using 4-bit weights when requested."""

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_id = str(model_override or extractor_config["model_id"])
    revision = extractor_config.get("revision")
    load_revision = (
        None if str(revision).startswith("local-config-") else revision
    )
    use_4bit = bool(extractor_config.get("load_in_4bit", True)) and not disable_4bit
    if use_4bit and not torch.cuda.is_available():
        raise RuntimeError(
            "4-bit extraction requires CUDA. Run on the GPU server or pass --no-4bit."
        )
    dtype = _torch_dtype(torch)
    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    logger.info(
        "Loading frozen extractor model=%s revision=%s 4bit=%s dtype=%s device=%s",
        model_id,
        revision or "default",
        use_4bit,
        dtype,
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=load_revision,
        trust_remote_code=bool(extractor_config.get("trust_remote_code", False)),
        padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model_kwargs: dict[str, Any] = {
        "revision": load_revision,
        "trust_remote_code": bool(extractor_config.get("trust_remote_code", False)),
        "device_map": "auto",
        "torch_dtype": dtype,
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    runtime = {
        "model_id": model_id,
        "revision": revision,
        "use_4bit": use_4bit,
        "batch_size": int(
            batch_size_override or extractor_config.get("batch_size", 2)
        ),
        "device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        ),
        "dtype": str(dtype),
    }
    return model, tokenizer, runtime


def _render_prompts(
    rows: pd.DataFrame,
    tokenizer: Any,
    ontology: Mapping[str, Sequence[str]],
    tickers: Sequence[str],
    max_events: int,
) -> list[str]:
    prompts: list[str] = []
    for row in rows.itertuples(index=False):
        messages = build_messages(
            text=str(row.text),
            target_ticker=str(row.target_ticker),
            date=pd.Timestamp(row.date).date().isoformat(),
            ontology=ontology,
            allowed_tickers=tickers,
            max_events=max_events,
        )
        prompts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
    return prompts


def _generate_batch(
    rows: pd.DataFrame,
    *,
    model: Any,
    tokenizer: Any,
    extractor_config: Mapping[str, Any],
    ontology: Mapping[str, Sequence[str]],
    tickers: Sequence[str],
) -> tuple[list[str], list[int]]:
    import torch

    prompts = _render_prompts(
        rows,
        tokenizer,
        ontology,
        tickers,
        int(extractor_config["max_events_per_news"]),
    )
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(extractor_config.get("max_input_tokens", 3072)),
    )
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    input_width = int(inputs["input_ids"].shape[1])
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=int(extractor_config.get("max_new_tokens", 768)),
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    responses = tokenizer.batch_decode(
        generated[:, input_width:],
        skip_special_tokens=True,
    )
    token_counts = [
        int((row != tokenizer.pad_token_id).sum().item())
        for row in generated[:, input_width:]
    ]
    return responses, token_counts


def extract_events(
    source_events: pd.DataFrame,
    *,
    ontology: Mapping[str, Sequence[str]],
    tickers: Sequence[str],
    extractor_config: Mapping[str, Any],
    output_root: Path,
    logger: Any,
    progress_state_path: Path,
    resume: bool,
    force: bool,
    model_override: str | None = None,
    batch_size_override: int | None = None,
    disable_4bit: bool = False,
) -> dict[str, Path]:
    """Extract every selected canonical event and persist progress after each batch."""

    cache_path = output_root / "cache" / "structured_event_extraction.jsonl"
    cache = _load_cache(cache_path) if resume or cache_path.exists() else {}
    model_id = str(model_override or extractor_config["model_id"])
    configured_model = str(extractor_config["model_id"])
    configured_revision = (
        extractor_config.get("revision")
        if model_id == configured_model
        else None
    )
    revision = _resolve_revision(model_id, configured_revision)
    runtime_extractor_config = dict(extractor_config)
    runtime_extractor_config["revision"] = revision
    prompt_version = str(extractor_config["prompt_version"])
    ontology_hash = ontology_digest(ontology)
    selected = source_events.copy().sort_values(
        ["date", "target_ticker", "event_id"], kind="mergesort"
    )
    selected["cache_key"] = [
        extraction_cache_key(
            event_id=str(row.event_id),
            text=str(row.text),
            model_id=model_id,
            model_revision=None if revision is None else str(revision),
            prompt_version=prompt_version,
            ontology_hash=ontology_hash,
        )
        for row in selected.itertuples(index=False)
    ]
    cached_valid = {
        key
        for key, row in cache.items()
        if str(row.get("status", "")).startswith("valid") and not force
    }
    pending = selected.loc[~selected["cache_key"].isin(cached_valid)].copy()
    logger.info(
        "Structured extraction plan | selected=%d cached_valid=%d pending=%d",
        len(selected),
        len(selected) - len(pending),
        len(pending),
    )
    runtime: dict[str, Any] = {
        "model_id": model_id,
        "revision": revision,
        "batch_size": int(
            batch_size_override or extractor_config.get("batch_size", 2)
        ),
    }
    started = time.monotonic()
    if not pending.empty:
        model, tokenizer, runtime = load_local_extractor(
            runtime_extractor_config,
            logger,
            model_override=model_override,
            batch_size_override=batch_size_override,
            disable_4bit=disable_4bit,
        )
        batch_size = int(runtime["batch_size"])
        progress = tqdm(
            total=len(pending),
            desc="LLM structured-event extraction",
            unit="news",
            dynamic_ncols=True,
        )
        processed = 0
        for start in range(0, len(pending), batch_size):
            batch = pending.iloc[start : start + batch_size]
            batch_started = time.monotonic()
            responses, token_counts = _generate_batch(
                batch,
                model=model,
                tokenizer=tokenizer,
                extractor_config=runtime_extractor_config,
                ontology=ontology,
                tickers=tickers,
            )
            cache_rows: list[dict[str, Any]] = []
            for row, raw_response, token_count in zip(
                batch.itertuples(index=False), responses, token_counts
            ):
                validation = validate_response(
                    raw_response=raw_response,
                    source_text=str(row.text),
                    ontology=ontology,
                    allowed_tickers=tickers,
                    max_events=int(extractor_config["max_events_per_news"]),
                )
                cache_row = {
                    "cache_key": str(row.cache_key),
                    "event_id": str(row.event_id),
                    "date": pd.Timestamp(row.date).date().isoformat(),
                    "target_ticker": str(row.target_ticker),
                    "text_hash": str(row.text_hash),
                    "model_id": model_id,
                    "model_revision": revision,
                    "prompt_version": prompt_version,
                    "ontology_hash": ontology_hash,
                    "status": (
                        "valid_with_drops"
                        if validation.valid and validation.errors
                        else "valid"
                        if validation.valid
                        else "invalid"
                    ),
                    "events": validation.events,
                    "errors": validation.errors,
                    "dropped_events": validation.dropped_events,
                    "generated_tokens": int(token_count),
                    "raw_response": raw_response,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                cache_rows.append(cache_row)
                cache[str(row.cache_key)] = cache_row
            _append_cache(cache_path, cache_rows)
            processed += len(batch)
            progress.update(len(batch))
            elapsed = max(time.monotonic() - started, 1.0e-9)
            rate = processed / elapsed
            remaining = (len(pending) - processed) / max(rate, 1.0e-9)
            progress.set_postfix(
                valid=sum(
                    str(value["status"]).startswith("valid")
                    for value in cache_rows
                ),
                batch_s=f"{time.monotonic() - batch_started:.1f}",
                eta_min=f"{remaining / 60.0:.1f}",
            )
            atomic_write_json(
                {
                    "stage": "extract",
                    "status": "running",
                    "completed": processed,
                    "total": len(pending),
                    "percent": 100.0 * processed / max(len(pending), 1),
                    "elapsed_seconds": elapsed,
                    "estimated_remaining_seconds": remaining,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
                progress_state_path,
            )
            if processed % int(extractor_config.get("log_every", 20)) < len(batch):
                logger.info(
                    "Extraction progress %d/%d (%.1f%%) | %.2f news/s | ETA %.1f min",
                    processed,
                    len(pending),
                    100.0 * processed / max(len(pending), 1),
                    rate,
                    remaining / 60.0,
                )
        progress.close()
        del model
        if __import__("torch").cuda.is_available():
            __import__("torch").cuda.empty_cache()
    latest = _load_cache(cache_path)
    selected_records = [
        latest[str(key)]
        for key in selected["cache_key"]
        if str(key) in latest
    ]
    missing = len(selected) - len(selected_records)
    valid_records = [
        row
        for row in selected_records
        if str(row["status"]).startswith("valid")
    ]
    invalid_records = [
        row
        for row in selected_records
        if not str(row["status"]).startswith("valid")
    ]
    structured_rows: list[dict[str, Any]] = []
    for cache_row in valid_records:
        for event_index, event in enumerate(cache_row["events"]):
            structured_rows.append(
                {
                    "source_event_id": cache_row["event_id"],
                    "structured_event_id": (
                        f"{cache_row['event_id']}__{int(event_index):02d}"
                    ),
                    "event_index": int(event_index),
                    "date": pd.Timestamp(cache_row["date"]),
                    "target_ticker": cache_row["target_ticker"],
                    **event,
                    "extractor_model": cache_row["model_id"],
                    "prompt_version": cache_row["prompt_version"],
                }
            )
    structured_columns = [
        "source_event_id",
        "structured_event_id",
        "event_index",
        "date",
        "target_ticker",
        "event_type",
        "direction",
        "magnitude",
        "certainty",
        "time_horizon",
        "entity_role",
        "explicit_tickers",
        "evidence_text",
        "extractor_model",
        "prompt_version",
    ]
    structured = pd.DataFrame(structured_rows, columns=structured_columns)
    structured_path = output_root / "data" / "structured_events.parquet"
    write_table(structured, structured_path)
    issue_records = [
        row for row in selected_records if row.get("errors")
    ]
    failures = pd.DataFrame(
        [
            {
                "event_id": row["event_id"],
                "date": row["date"],
                "target_ticker": row["target_ticker"],
                "status": row["status"],
                "errors": json.dumps(row["errors"], ensure_ascii=False),
                "raw_response": row["raw_response"],
            }
            for row in issue_records
        ],
        columns=[
            "event_id",
            "date",
            "target_ticker",
            "status",
            "errors",
            "raw_response",
        ],
    )
    failures_path = output_root / "tables" / "extraction_failures.csv"
    write_table(failures, failures_path)
    valid_rate = len(valid_records) / max(len(selected), 1)
    articles_with_event = sum(bool(row["events"]) for row in valid_records)
    dropped_events = sum(
        int(row.get("dropped_events", 0)) for row in selected_records
    )
    accepted_events = sum(len(row.get("events", [])) for row in valid_records)
    dropped_event_rate = dropped_events / max(dropped_events + accepted_events, 1)
    summary = pd.DataFrame(
        [
            {
                "selected_news": len(selected),
                "cache_records": len(selected_records),
                "missing_records": missing,
                "valid_articles": len(valid_records),
                "invalid_articles": len(invalid_records),
                "valid_rate": valid_rate,
                "articles_with_event": articles_with_event,
                "articles_with_event_rate": articles_with_event
                / max(len(valid_records), 1),
                "structured_events": len(structured),
                "dropped_structured_events": dropped_events,
                "dropped_structured_event_rate": dropped_event_rate,
                "elapsed_seconds_this_run": time.monotonic() - started,
                **runtime,
            }
        ]
    )
    summary_path = output_root / "tables" / "extraction_summary.csv"
    write_table(summary, summary_path)
    minimum_valid_rate = float(extractor_config.get("minimum_valid_rate", 0.95))
    maximum_dropped_rate = float(
        extractor_config.get("maximum_dropped_event_rate", 0.10)
    )
    if missing:
        raise RuntimeError(f"Extraction cache is missing {missing} selected news records.")
    if valid_rate < minimum_valid_rate:
        raise RuntimeError(
            f"Extraction valid rate {valid_rate:.3f} is below "
            f"the configured gate {minimum_valid_rate:.3f}."
        )
    if dropped_event_rate > maximum_dropped_rate:
        raise RuntimeError(
            f"Dropped structured-event rate {dropped_event_rate:.3f} is above "
            f"the configured gate {maximum_dropped_rate:.3f}. Inspect "
            f"{failures_path} before forecasting."
        )
    atomic_write_json(
        {
            "stage": "extract",
            "status": "completed",
            "completed": len(selected),
            "total": len(selected),
            "percent": 100.0,
            "elapsed_seconds": time.monotonic() - started,
            "estimated_remaining_seconds": 0.0,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
        progress_state_path,
    )
    return {
        "cache": cache_path,
        "structured_events": structured_path,
        "summary": summary_path,
        "failures": failures_path,
    }
