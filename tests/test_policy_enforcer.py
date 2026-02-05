"""Policy and enforcer behavior tests for deterministic gating."""

from ega.enforcer import Enforcer
from ega.policy import DefaultPolicy, PolicyConfig, ReasonCode
from ega.types import AnswerCandidate, EvidenceSet, Unit, VerificationScore


class FakeVerifier:
    """Deterministic verifier backed by static per-unit score mappings."""

    def __init__(self, scores_by_unit_id: dict[str, VerificationScore]) -> None:
        self._scores_by_unit_id = scores_by_unit_id

    def verify(self, *, unit_text: str, unit_id: str, evidence: EvidenceSet) -> VerificationScore:
        _ = (unit_text, evidence)
        return self._scores_by_unit_id[unit_id]


def _unit(unit_id: str, text: str) -> Unit:
    return Unit(id=unit_id, text=text)


def _score(
    unit_id: str,
    entailment: float,
    contradiction: float,
    neutral: float = 0.0,
    label: str = "entailment",
) -> VerificationScore:
    return VerificationScore(
        unit_id=unit_id,
        entailment=entailment,
        contradiction=contradiction,
        neutral=neutral,
        label=label,
        raw={},
    )


def test_default_policy_drops_unsupported_and_refuses_when_nothing_survives() -> None:
    units = [_unit("u1", "A"), _unit("u2", "B")]
    scores = [_score("u1", entailment=0.2, contradiction=0.7)]

    decision = DefaultPolicy().decide(
        scores=scores,
        units=units,
        config=PolicyConfig(threshold=0.8, contradiction_max=0.2, partial_allowed=True),
    )

    assert decision.refusal is True
    assert decision.reason_code == ReasonCode.ALL_DROPPED
    assert [unit.id for unit in decision.allowed_units] == []
    assert [unit.id for unit in decision.dropped_units] == ["u1", "u2"]


def test_default_policy_refuses_partial_when_not_allowed() -> None:
    units = [_unit("u1", "keep"), _unit("u2", "drop")]
    scores = [
        _score("u1", entailment=0.9, contradiction=0.05),
        _score("u2", entailment=0.3, contradiction=0.4),
    ]

    decision = DefaultPolicy().decide(
        scores=scores,
        units=units,
        config=PolicyConfig(partial_allowed=False),
    )

    assert decision.refusal is True
    assert decision.reason_code == ReasonCode.PARTIAL_NOT_ALLOWED
    assert [unit.id for unit in decision.allowed_units] == ["u1"]
    assert [unit.id for unit in decision.dropped_units] == ["u2"]


def test_enforcer_returns_partial_answer_without_rewriting() -> None:
    candidate = AnswerCandidate(
        raw_answer_text="one\ntwo",
        units=[_unit("u1", "one"), _unit("u2", "two")],
    )
    verifier = FakeVerifier(
        {
            "u1": _score("u1", entailment=0.95, contradiction=0.01),
            "u2": _score("u2", entailment=0.1, contradiction=0.5),
        }
    )

    result = Enforcer(
        verifier=verifier,
        policy=DefaultPolicy(),
        config=PolicyConfig(partial_allowed=True),
    ).enforce(candidate=candidate, evidence=EvidenceSet(items=[]))

    assert result.decision.refusal is False
    assert result.decision.reason_code == ReasonCode.OK_PARTIAL
    assert result.final_text == "one"
    assert [unit.id for unit in result.kept_units] == ["u1"]
    assert [unit.id for unit in result.dropped_units] == ["u2"]


def test_enforcer_refuses_with_message_when_all_units_drop() -> None:
    candidate = AnswerCandidate(
        raw_answer_text="x\ny",
        units=[_unit("u1", "x"), _unit("u2", "y")],
    )
    verifier = FakeVerifier(
        {
            "u1": _score("u1", entailment=0.0, contradiction=1.0, label="contradiction"),
            "u2": _score("u2", entailment=0.1, contradiction=0.9, label="contradiction"),
        }
    )

    result = Enforcer(
        verifier=verifier,
        refusal_message="refuse",
    ).enforce(candidate=candidate, evidence=EvidenceSet(items=[]))

    assert result.decision.refusal is True
    assert result.decision.reason_code == ReasonCode.ALL_DROPPED
    assert result.final_text is None
    assert result.refusal_message == "refuse"
