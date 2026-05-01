from __future__ import annotations

import dataclasses
import json

import pytest

from ega.contract import EGA_SCHEMA_VERSION, ReasonCode
from ega.serialization import from_json, to_json
from ega.types import (
    AnswerCandidate,
    EnforcementResult,
    EvidenceItem,
    EvidenceSet,
    GateDecision,
    Unit,
    VerificationScore,
)


def _field_names(cls: type) -> set[str]:
    return {field.name for field in dataclasses.fields(cls)}


def test_public_surface_freeze_dataclass_fields() -> None:
    assert _field_names(Unit) == {"id", "text", "metadata", "source_ids"}
    assert _field_names(AnswerCandidate) == {"raw_answer_text", "units"}
    assert _field_names(EvidenceItem) == {"id", "text", "metadata"}
    assert _field_names(EvidenceSet) == {"items"}
    assert _field_names(VerificationScore) == {
        "unit_id",
        "entailment",
        "contradiction",
        "neutral",
        "label",
        "raw",
        "nli_score",
        "citation_overlap",
        "contradiction_flag",
        "conformal_decision",
        "conformal_raw_score",
    }
    assert _field_names(GateDecision) == {
        "allowed_units",
        "dropped_units",
        "refusal",
        "reason_code",
        "summary_stats",
    }
    assert _field_names(EnforcementResult) == {
        "final_text",
        "kept_units",
        "dropped_units",
        "refusal_message",
        "decision",
        "scores",
        "verified_units",
        "polished_units",
        "polish_status",
        "polish_fail_reasons",
        "ega_schema_version",
    }


def test_public_surface_freeze_reason_codes_and_schema_version() -> None:
    assert {member.name for member in ReasonCode} == {
        "OK_FULL",
        "OK_PARTIAL",
        "ALL_DROPPED",
        "PARTIAL_NOT_ALLOWED",
    }
    assert {member.value for member in ReasonCode} == {
        "OK_FULL",
        "OK_PARTIAL",
        "ALL_DROPPED",
        "PARTIAL_NOT_ALLOWED",
    }
    assert EGA_SCHEMA_VERSION == "1"


def test_schema_version_round_trip_preserves_version() -> None:
    result = EnforcementResult(
        final_text="answer",
        kept_units=["u1"],
        dropped_units=["u2"],
        refusal_message=None,
        decision=GateDecision(
            allowed_units=["u1"],
            dropped_units=["u2"],
            refusal=False,
            reason_code="OK_PARTIAL",
            summary_stats={"total_units": 2},
        ),
        scores=[
            VerificationScore(
                unit_id="u1",
                entailment=0.9,
                contradiction=0.1,
                neutral=0.0,
                label="entailment",
                raw={"model": "test"},
            )
        ],
    )

    payload = to_json(result)
    decoded = json.loads(payload)
    assert decoded["ega_schema_version"] == EGA_SCHEMA_VERSION
    assert decoded["data"]["ega_schema_version"] == EGA_SCHEMA_VERSION

    restored = from_json(payload, EnforcementResult)
    assert restored.ega_schema_version == EGA_SCHEMA_VERSION


def test_from_json_rejects_unsupported_schema_version() -> None:
    result = EnforcementResult(
        final_text=None,
        kept_units=[],
        dropped_units=["u1"],
        refusal_message="refuse",
        decision=GateDecision(
            allowed_units=[],
            dropped_units=["u1"],
            refusal=True,
            reason_code="ALL_DROPPED",
            summary_stats={},
        ),
        scores=[],
    )
    payload = json.loads(to_json(result))
    payload["ega_schema_version"] = "2"

    with pytest.raises(ValueError, match="Unsupported schema version"):
        from_json(json.dumps(payload), EnforcementResult)
