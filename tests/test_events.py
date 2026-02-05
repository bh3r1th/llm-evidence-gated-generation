"""Structured event emission tests."""

from ega.events import DecisionEvent, event_from_result
from ega.policy import PolicyConfig
from ega.types import EnforcementResult, GateDecision, Unit


def test_event_from_result_builds_expected_schema() -> None:
    result = EnforcementResult(
        final_text="kept",
        kept_units=[Unit(id="u1", text="kept")],
        dropped_units=[Unit(id="u2", text="dropped")],
        refusal_message=None,
        decision=GateDecision(
            allowed_units=[Unit(id="u1", text="kept")],
            dropped_units=[Unit(id="u2", text="dropped")],
            refusal=False,
            reason_code="OK_PARTIAL",
            summary_stats={"total_units": 2, "kept_units": 1, "dropped_units": 1},
        ),
        scores=[],
    )

    event = event_from_result(
        result,
        {
            "run_id": "run-123",
            "timestamp": "2024-01-02T03:04:05+00:00",
            "model_name": "test-model",
            "policy_config": PolicyConfig(threshold=0.9, contradiction_max=0.1, partial_allowed=False),
        },
    )

    assert isinstance(event, DecisionEvent)
    assert event.run_id == "run-123"
    assert event.timestamp == "2024-01-02T03:04:05+00:00"
    assert event.model_name == "test-model"
    assert event.policy_config == {
        "threshold": 0.9,
        "contradiction_max": 0.1,
        "partial_allowed": False,
    }
    assert event.unit_count == 2
    assert event.kept_count == 1
    assert event.refusal is False
    assert event.summary_stats == {"total_units": 2, "kept_units": 1, "dropped_units": 1}
