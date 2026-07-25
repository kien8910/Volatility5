"""Schema, prompt and strict validation for structured financial events."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


REQUIRED_EVENT_FIELDS = (
    "event_type",
    "direction",
    "magnitude",
    "certainty",
    "time_horizon",
    "entity_role",
    "evidence_text",
)


@dataclass(frozen=True)
class ValidationResult:
    """Validated model response and diagnostics for one canonical news item."""

    events: list[dict[str, Any]]
    valid: bool
    errors: list[str]
    dropped_events: int


def normalized_text(value: Any) -> str:
    """Normalize whitespace without changing the evidence-bearing characters."""

    return re.sub(r"\s+", " ", str(value or "")).strip()


def ontology_from_config(profile: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    """Return an immutable, validated ontology from the pilot configuration."""

    keys = (
        "event_types",
        "directions",
        "magnitudes",
        "certainties",
        "time_horizons",
        "entity_roles",
    )
    ontology: dict[str, tuple[str, ...]] = {}
    for key in keys:
        values = tuple(dict.fromkeys(str(value).strip() for value in profile[key]))
        if not values or any(not value for value in values):
            raise ValueError(f"Structured-event ontology {key!r} is empty or invalid.")
        ontology[key] = values
    return ontology


def ontology_digest(ontology: Mapping[str, Sequence[str]]) -> str:
    payload = json.dumps(
        {key: list(values) for key, values in ontology.items()},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def extraction_cache_key(
    *,
    event_id: str,
    text: str,
    model_id: str,
    model_revision: str | None,
    prompt_version: str,
    ontology_hash: str,
) -> str:
    payload = {
        "event_id": str(event_id),
        "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "model_id": str(model_id),
        "model_revision": str(model_revision or "default"),
        "prompt_version": str(prompt_version),
        "ontology_hash": str(ontology_hash),
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_messages(
    *,
    text: str,
    target_ticker: str,
    date: str,
    ontology: Mapping[str, Sequence[str]],
    allowed_tickers: Sequence[str],
    max_events: int,
) -> list[dict[str, str]]:
    """Build a compact extraction prompt with an explicit closed ontology."""

    schema = {
        "events": [
            {
                "event_type": "one allowed event_types value",
                "direction": "one allowed directions value",
                "magnitude": "one allowed magnitudes value",
                "certainty": "one allowed certainties value",
                "time_horizon": "one allowed time_horizons value",
                "entity_role": "one allowed entity_roles value",
                "explicit_tickers": ["zero or more configured semiconductor tickers"],
                "evidence_text": "an exact non-empty substring copied from NEWS_TEXT",
            }
        ]
    }
    system = (
        "You are a financial event extraction system. Extract only factual corporate "
        "or economic events explicitly supported by NEWS_TEXT. Do not forecast prices, "
        "returns or volatility. Do not infer affected tickers merely from sector "
        "membership. A news item may contain zero, one or multiple events. "
        "evidence_text must be copied exactly from NEWS_TEXT. Use only the allowed enum "
        "values. Return one JSON object only, without markdown or explanation. "
        f"Return no more than {int(max_events)} events. If there is no identifiable "
        'financial event, return {"events":[]}.'
    )
    user = (
        "/no_think\n"
        f"RECORD_TARGET_TICKER: {target_ticker}\n"
        f"RECORD_DATE: {date}\n"
        f"ALLOWED_EXPLICIT_TICKERS: {json.dumps(list(allowed_tickers))}\n"
        "ALLOWED_ENUMS:\n"
        f"{json.dumps({key: list(values) for key, values in ontology.items()}, ensure_ascii=False)}\n"
        "OUTPUT_SCHEMA_EXAMPLE:\n"
        f"{json.dumps(schema, ensure_ascii=False)}\n"
        "NEWS_TEXT:\n---\n"
        f"{text}\n"
        "---\nReturn the JSON object now."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _first_json_object(raw: str) -> dict[str, Any]:
    """Decode the first complete JSON object without accepting prose as data."""

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", raw):
        try:
            value, _ = decoder.raw_decode(raw[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("No valid JSON object was found in the model response.")


def _enum_value(
    event: Mapping[str, Any],
    field: str,
    allowed: Sequence[str],
    errors: list[str],
    index: int,
) -> str | None:
    value = str(event.get(field, "")).strip().lower()
    if value not in set(allowed):
        errors.append(f"event[{index}].{field}={value!r} is outside the ontology")
        return None
    return value


def validate_response(
    *,
    raw_response: str,
    source_text: str,
    ontology: Mapping[str, Sequence[str]],
    allowed_tickers: Sequence[str],
    max_events: int,
) -> ValidationResult:
    """Strictly validate an extractor response and drop unsupported event rows."""

    errors: list[str] = []
    try:
        payload = _first_json_object(raw_response)
    except ValueError as error:
        return ValidationResult([], False, [str(error)], 0)
    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        return ValidationResult([], False, ["Top-level 'events' must be a list."], 0)
    if len(raw_events) > int(max_events):
        errors.append(
            f"Model returned {len(raw_events)} events; maximum is {int(max_events)}."
        )
        raw_events = raw_events[: int(max_events)]
    ticker_set = set(map(str, allowed_tickers))
    accepted: list[dict[str, Any]] = []
    dropped = 0
    for index, raw_event in enumerate(raw_events):
        event_errors: list[str] = []
        if not isinstance(raw_event, Mapping):
            errors.append(f"event[{index}] is not an object")
            dropped += 1
            continue
        values: dict[str, Any] = {}
        enum_mapping = {
            "event_type": "event_types",
            "direction": "directions",
            "magnitude": "magnitudes",
            "certainty": "certainties",
            "time_horizon": "time_horizons",
            "entity_role": "entity_roles",
        }
        for field, ontology_key in enum_mapping.items():
            value = _enum_value(
                raw_event,
                field,
                ontology[ontology_key],
                event_errors,
                index,
            )
            if value is not None:
                values[field] = value
        evidence = str(raw_event.get("evidence_text", "")).strip()
        if not evidence:
            event_errors.append(f"event[{index}].evidence_text is empty")
        elif evidence not in source_text:
            event_errors.append(
                f"event[{index}].evidence_text is not an exact source substring"
            )
        values["evidence_text"] = evidence
        tickers = raw_event.get("explicit_tickers", [])
        if not isinstance(tickers, list):
            event_errors.append(f"event[{index}].explicit_tickers must be a list")
            tickers = []
        normalized_tickers = list(
            dict.fromkeys(str(value).strip().upper() for value in tickers)
        )
        unknown_tickers = sorted(set(normalized_tickers).difference(ticker_set))
        if unknown_tickers:
            event_errors.append(
                f"event[{index}] contains unsupported tickers {unknown_tickers}"
            )
        values["explicit_tickers"] = [
            ticker for ticker in normalized_tickers if ticker in ticker_set
        ]
        if event_errors:
            errors.extend(event_errors)
            dropped += 1
            continue
        accepted.append(values)
    # Event-level violations are dropped and surfaced for audit. The response
    # remains structurally usable as long as the top-level JSON/schema parsed.
    return ValidationResult(accepted, True, errors, dropped)
