"""Unit tests for the structured-event pilot without loading an LLM."""

from __future__ import annotations

import json
import unittest

import pandas as pd

from src.structured_event_pilot import (
    BASIC_META_COLUMNS,
    META_COLUMNS,
    _candidate_mask,
    _placebo_panel,
    _profile,
    _structured_counts,
)
from src.structured_event_schema import ontology_from_config, validate_response
from src.utils import load_config


class StructuredEventPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config()
        cls.profile = _profile(cls.config)
        cls.ontology = ontology_from_config(cls.profile)

    def test_candidate_gate_is_deterministic(self) -> None:
        frame = pd.DataFrame(
            {
                "text": [
                    "Nvidia raised its revenue guidance.",
                    "The shares traded higher during the session.",
                ]
            }
        )
        self.assertEqual(
            _candidate_mask(frame, self.profile).tolist(),
            [True, False],
        )

    def test_validator_drops_unsupported_evidence(self) -> None:
        text = "Nvidia raised quarterly revenue guidance."
        response = json.dumps(
            {
                "events": [
                    {
                        "event_type": "earnings_guidance",
                        "direction": "positive",
                        "magnitude": "unknown",
                        "certainty": "confirmed",
                        "time_horizon": "next_quarter",
                        "entity_role": "target_company",
                        "explicit_tickers": ["NVDA"],
                        "evidence_text": "Nvidia cut quarterly guidance.",
                    }
                ]
            }
        )
        result = validate_response(
            raw_response=response,
            source_text=text,
            ontology=self.ontology,
            allowed_tickers=self.config["universe"]["tickers"],
            max_events=3,
        )
        self.assertTrue(result.valid)
        self.assertEqual(result.events, [])
        self.assertEqual(result.dropped_events, 1)
        self.assertTrue(result.errors)

    def test_empty_structured_counts_keep_column_contract(self) -> None:
        result = _structured_counts(pd.DataFrame(), self.ontology)
        self.assertIn("meta__structured_event_count", result.columns)
        self.assertIn("etype__earnings_guidance", result.columns)
        self.assertIn(
            "interaction__earnings_guidance__direction__negative",
            result.columns,
        )

    def test_placebo_preserves_metadata(self) -> None:
        dates = pd.to_datetime(["2022-01-03", "2022-01-04"])
        base = pd.DataFrame(
            {
                "ticker": ["NVDA", "NVDA"],
                "feature_date": dates,
                **{
                    column: [1.0, 0.0]
                    for column in META_COLUMNS
                },
                "etype__earnings_guidance": [1.0, 0.0],
            }
        )
        structured = pd.DataFrame(
            [
                {
                    "target_ticker": "NVDA",
                    "feature_date": dates[0],
                    "event_type": "earnings_guidance",
                    "direction": "positive",
                    "magnitude": "unknown",
                    "certainty": "confirmed",
                    "time_horizon": "next_quarter",
                    "entity_role": "target_company",
                }
            ]
        )
        fold = pd.Series(
            {
                "train_start": dates[0],
                "train_end": dates[0],
                "validation_start": dates[1],
                "validation_end": dates[1],
            }
        )
        placebo = _placebo_panel(
            base_panel=base,
            structured=structured,
            ontology=self.ontology,
            fold_row=fold,
            kind="random_structured_vector",
            seed=1001,
        )
        pd.testing.assert_frame_equal(
            base[["ticker", "feature_date", *META_COLUMNS]],
            placebo[["ticker", "feature_date", *META_COLUMNS]],
        )
        self.assertTrue(set(BASIC_META_COLUMNS).issubset(META_COLUMNS))


if __name__ == "__main__":
    unittest.main()
