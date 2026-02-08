"""Structured event emission tests."""

from ega.events import DecisionEvent, event_from_result
from ega.contract import PolicyConfig
from ega.types import EnforcementResult, GateDecision


def test_event_from_result_builds_expected_schema() -> None:
    result = EnforcementResult(
        final_text="kept",
        kept_units=["u1"],
        dropped_units=["u2"],
        refusal_message=None,
        decision=GateDecision(
            allowed_units=["u1"],
            dropped_units=["u2"],
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
            "policy_config": PolicyConfig(
                threshold_entailment=0.9,
                max_contradiction=0.1,
                partial_allowed=False,
            ),
        },
    )

    assert isinstance(event, DecisionEvent)
    assert event.run_id == "run-123"
    assert event.timestamp == "2024-01-02T03:04:05+00:00"
    assert event.model_name == "test-model"
    assert event.policy_config == {
        "threshold_entailment": 0.9,
        "max_contradiction": 0.1,
        "partial_allowed": False,
    }
    assert event.unit_count == 2
    assert event.kept_count == 1
    assert event.refusal is False
    assert event.summary_stats == {"total_units": 2, "kept_units": 1, "dropped_units": 1}
