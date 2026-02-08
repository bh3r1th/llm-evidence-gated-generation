from __future__ import annotations

import math

import pytest

from ega.contract import PolicyConfig, ReasonCode
from ega.policy import DefaultPolicy
from ega.types import Unit, VerificationScore


def _unit(unit_id: str) -> Unit:
    return Unit(id=unit_id, text=f"text-{unit_id}", metadata={})


def _score(unit_id: str, entailment: float, contradiction: float) -> VerificationScore:
    label = "entailment" if entailment >= 0.5 else "contradiction"
    return VerificationScore(
        unit_id=unit_id,
        entailment=entailment,
        contradiction=contradiction,
        neutral=0.0,
        label=label,
        raw={"source": "test"},
    )


@pytest.mark.parametrize(
    (
        "unit_ids",
        "scores",
        "config",
        "expected_refusal",
        "expected_allowed",
        "expected_dropped",
        "expected_reason",
        "expect_invalid_flag",
    ),
    [
        (
            ["u1"],
            [_score("u1", entailment=0.8, contradiction=0.1)],
            PolicyConfig(threshold_entailment=0.8, max_contradiction=0.2, partial_allowed=True),
            False,
            {"u1"},
            set(),
            ReasonCode.OK_FULL.value,
            False,
        ),
        (
            ["u1"],
            [_score("u1", entailment=0.9, contradiction=0.2)],
            PolicyConfig(threshold_entailment=0.8, max_contradiction=0.2, partial_allowed=True),
            False,
            {"u1"},
            set(),
            ReasonCode.OK_FULL.value,
            False,
        ),
        (
            ["u1", "u2"],
            [_score("u1", entailment=0.95, contradiction=0.01)],
            PolicyConfig(threshold_entailment=0.8, max_contradiction=0.2, partial_allowed=True),
            False,
            {"u1"},
            {"u2"},
            ReasonCode.OK_PARTIAL.value,
            False,
        ),
        (
            ["u1", "u2"],
            [_score("u1", entailment=0.95, contradiction=0.01)],
            PolicyConfig(threshold_entailment=0.8, max_contradiction=0.2, partial_allowed=False),
            True,
            {"u1"},
            {"u2"},
            ReasonCode.PARTIAL_NOT_ALLOWED.value,
            False,
        ),
        (
            ["u1", "u2"],
            [
                _score("u1", entailment=math.nan, contradiction=0.0),
                _score("u2", entailment=0.0, contradiction=0.9),
            ],
            PolicyConfig(threshold_entailment=0.8, max_contradiction=0.2, partial_allowed=True),
            True,
            set(),
            {"u1", "u2"},
            ReasonCode.ALL_DROPPED.value,
            True,
        ),
        (
            ["u1"],
            [_score("u1", entailment=1.1, contradiction=0.0)],
            PolicyConfig(threshold_entailment=0.8, max_contradiction=0.2, partial_allowed=True),
            True,
            set(),
            {"u1"},
            ReasonCode.ALL_DROPPED.value,
            True,
        ),
    ],
)
def test_default_policy_table_driven(
    unit_ids: list[str],
    scores: list[VerificationScore],
    config: PolicyConfig,
    expected_refusal: bool,
    expected_allowed: set[str],
    expected_dropped: set[str],
    expected_reason: str,
    expect_invalid_flag: bool,
) -> None:
    units = [_unit(unit_id) for unit_id in unit_ids]

    decision = DefaultPolicy().decide(scores=scores, units=units, config=config)

    assert decision.refusal is expected_refusal
    assert set(decision.allowed_units) == expected_allowed
    assert set(decision.dropped_units) == expected_dropped
    assert decision.reason_code == expected_reason
    if expect_invalid_flag:
        assert decision.summary_stats["invalid_entailment_count"] > 0
    else:
        assert decision.summary_stats["invalid_entailment_count"] == 0
