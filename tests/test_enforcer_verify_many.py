from __future__ import annotations

from ega.contract import PolicyConfig
from ega.enforcer import Enforcer
from ega.types import AnswerCandidate, EvidenceSet, Unit, VerificationScore


class FakeBatchedVerifier:
    def __init__(self) -> None:
        self.verify_many_calls = 0
        self.verify_calls = 0

    def verify_many(
        self,
        candidate: AnswerCandidate,
        evidence: EvidenceSet,
    ) -> list[VerificationScore]:
        _ = evidence
        self.verify_many_calls += 1
        return [
            VerificationScore(
                unit_id=unit.id,
                entailment=0.95 if idx == 0 else 0.10,
                contradiction=0.01 if idx == 0 else 0.90,
                neutral=0.04 if idx == 0 else 0.0,
                label="entailment" if idx == 0 else "contradiction",
                raw={"source": "batched"},
            )
            for idx, unit in enumerate(candidate.units)
        ]

    def verify(self, *, unit_text: str, unit_id: str, evidence: EvidenceSet) -> VerificationScore:
        _ = (unit_text, unit_id, evidence)
        self.verify_calls += 1
        raise AssertionError("verify() should not be used when verify_many() is available")


def test_enforcer_prefers_verify_many_when_available() -> None:
    candidate = AnswerCandidate(
        raw_answer_text="A. B.",
        units=[
            Unit(id="u0001", text="A.", metadata={}),
            Unit(id="u0002", text="B.", metadata={}),
        ],
    )
    verifier = FakeBatchedVerifier()

    result = Enforcer(
        verifier=verifier,
        config=PolicyConfig(
            threshold_entailment=0.8,
            max_contradiction=0.2,
            partial_allowed=True,
        ),
    ).enforce(candidate=candidate, evidence=EvidenceSet(items=[]))

    assert verifier.verify_many_calls == 1
    assert verifier.verify_calls == 0
    assert result.kept_units == ["u0001"]
    assert result.dropped_units == ["u0002"]
