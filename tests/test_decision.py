"""Decision payload builder tests."""

from ega.decision import build_enforcement_result, build_gate_decision
from ega.types import GateDecision, Unit, VerificationScore


def test_build_gate_decision_shapes_payload() -> None:
    unit = Unit(id="u1", text="text", metadata={"a": 1}, source_ids=["s1"])

    decision = build_gate_decision(
        allowed_units=[unit.id],
        dropped_units=[],
        refusal=False,
        reason_code="ok",
        summary_stats={"kept": 1},
    )

    assert isinstance(decision, GateDecision)
    assert decision.allowed_units == [unit.id]
    assert decision.summary_stats["kept"] == 1


def test_build_enforcement_result_shapes_payload() -> None:
    unit = Unit(id="u1", text="text", metadata={})
    score = VerificationScore(
        unit_id="u1",
        entailment=0.7,
        contradiction=0.1,
        neutral=0.2,
        label="entailment",
        raw={"model": "demo"},
    )
    decision = build_gate_decision(
        allowed_units=[unit.id],
        dropped_units=[],
        refusal=False,
        reason_code="ok",
        summary_stats={},
    )

    result = build_enforcement_result(
        final_text="answer",
        kept_units=[unit.id],
        dropped_units=[],
        refusal_message=None,
        decision=decision,
        scores=[score],
    )

    assert result.final_text == "answer"
    assert result.scores == [score]
