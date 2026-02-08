"""Serialization round-trip tests for stable EGA schemas."""

from ega.serialization import EGA_SCHEMA_VERSION, from_json, to_json
from ega.types import EnforcementResult, GateDecision, Unit, VerificationScore


def _sample_unit() -> Unit:
    return Unit(
        id="u-1",
        text="A supported claim.",
        metadata={"origin": "test"},
        source_ids=["doc-1", "doc-2"],
    )


def _sample_decision() -> GateDecision:
    unit = _sample_unit()
    return GateDecision(
        allowed_units=[unit.id],
        dropped_units=[],
        refusal=False,
        reason_code="allowed",
        summary_stats={"allowed_count": 1, "dropped_count": 0},
    )


def _sample_result() -> EnforcementResult:
    unit = _sample_unit()
    decision = _sample_decision()
    score = VerificationScore(
        unit_id=unit.id,
        entailment=0.91,
        contradiction=0.04,
        neutral=0.05,
        label="entailment",
        raw={"provider": "dummy", "logit": 2.2},
    )
    return EnforcementResult(
        final_text="A supported claim.",
        kept_units=[unit.id],
        dropped_units=[],
        refusal_message=None,
        decision=decision,
        scores=[score],
    )


def test_gate_decision_serialization_round_trip() -> None:
    decision = _sample_decision()

    payload = to_json(decision)
    restored = from_json(payload, GateDecision)

    assert restored == decision


def test_enforcement_result_serialization_round_trip() -> None:
    result = _sample_result()

    payload = to_json(result)
    restored = from_json(payload, EnforcementResult)

    assert restored == result


def test_serialized_payload_includes_schema_version() -> None:
    payload = to_json(_sample_decision())

    assert f'"ega_schema_version":"{EGA_SCHEMA_VERSION}"' in payload
