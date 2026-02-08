from __future__ import annotations

from ega.contract import PolicyConfig
from ega.enforcer import Enforcer
from ega.policy import DefaultPolicy
from ega.types import AnswerCandidate, EvidenceSet, Unit, VerificationScore


class FakeVerifier:
    def __init__(self, scores_by_unit_id: dict[str, VerificationScore]) -> None:
        self._scores_by_unit_id = scores_by_unit_id

    def verify(self, *, unit_text: str, unit_id: str, evidence: EvidenceSet) -> VerificationScore:
        _ = (unit_text, evidence)
        return self._scores_by_unit_id[unit_id]


def _score(unit_id: str, entailment: float, contradiction: float, label: str) -> VerificationScore:
    return VerificationScore(
        unit_id=unit_id,
        entailment=entailment,
        contradiction=contradiction,
        neutral=0.0,
        label=label,
        raw={"source": "fake"},
    )


def _candidate() -> AnswerCandidate:
    return AnswerCandidate(
        raw_answer_text="u1 text\nu2 text\nu3 text",
        units=[
            Unit(id="u1", text="unit1_text", metadata={}),
            Unit(id="u2", text="unit2_text", metadata={}),
            Unit(id="u3", text="unit3_text", metadata={}),
        ],
    )


def test_enforcer_final_text_joins_kept_units_in_original_order() -> None:
    candidate = _candidate()
    verifier = FakeVerifier(
        {
            "u1": _score("u1", entailment=0.9, contradiction=0.05, label="entailment"),
            "u2": _score("u2", entailment=0.2, contradiction=0.7, label="contradiction"),
            "u3": _score("u3", entailment=0.95, contradiction=0.02, label="entailment"),
        }
    )

    result = Enforcer(
        verifier=verifier,
        policy=DefaultPolicy(),
        config=PolicyConfig(threshold_entailment=0.8, max_contradiction=0.2, partial_allowed=True),
    ).enforce(candidate=candidate, evidence=EvidenceSet(items=[]))

    assert result.kept_units == ["u1", "u3"]
    assert result.dropped_units == ["u2"]
    assert result.final_text == "unit1_text\nunit3_text"


def test_enforcer_refusal_returns_none_final_text() -> None:
    candidate = _candidate()
    verifier = FakeVerifier(
        {
            "u1": _score("u1", entailment=0.1, contradiction=0.8, label="contradiction"),
            "u2": _score("u2", entailment=0.2, contradiction=0.7, label="contradiction"),
            "u3": _score("u3", entailment=0.1, contradiction=0.9, label="contradiction"),
        }
    )

    result = Enforcer(
        verifier=verifier,
        policy=DefaultPolicy(),
        config=PolicyConfig(threshold_entailment=0.8, max_contradiction=0.2, partial_allowed=True),
    ).enforce(candidate=candidate, evidence=EvidenceSet(items=[]))

    assert result.decision.refusal is True
    assert result.final_text is None
    assert isinstance(result.refusal_message, str)
    assert result.refusal_message.strip()
    assert result.kept_units == []
    assert result.dropped_units == ["u1", "u2", "u3"]
