from __future__ import annotations

from ega.contract import PolicyConfig, ReasonCode
from ega.enforcer import Enforcer
from ega.serialization import from_json, to_json
from ega.types import EvidenceItem, EvidenceSet, EnforcementResult, VerificationScore
from ega.unitization import unitize_answer


class FakeVerifier:
    def __init__(self, scores_by_unit_id: dict[str, VerificationScore]) -> None:
        self._scores_by_unit_id = scores_by_unit_id

    def verify(self, *, unit_text: str, unit_id: str, evidence: EvidenceSet) -> VerificationScore:
        _ = (unit_text, evidence)
        return self._scores_by_unit_id[unit_id]

    def verify_unit(self, unit_text: str, evidence: EvidenceSet) -> VerificationScore:
        _ = unit_text
        return self.verify(unit_text="", unit_id="u0001", evidence=evidence)


def _evidence() -> EvidenceSet:
    return EvidenceSet(
        items=[
            EvidenceItem(id="e1", text="Alpha is true.", metadata={"source": "test"}),
            EvidenceItem(id="e2", text="Beta is true.", metadata={"source": "test"}),
            EvidenceItem(id="e3", text="Gamma is true.", metadata={"source": "test"}),
        ]
    )


def _candidate():
    return unitize_answer("Alpha is true. Beta is true. Gamma is true.", mode="sentence")


def _score(unit_id: str, entailment: float, contradiction: float, label: str) -> VerificationScore:
    return VerificationScore(
        unit_id=unit_id,
        entailment=entailment,
        contradiction=contradiction,
        neutral=max(0.0, 1.0 - entailment - contradiction),
        label=label,
        raw={"verifier": "fake", "unit_id": unit_id},
    )


def _assert_common(result: EnforcementResult, expected_ids: list[str]) -> None:
    assert expected_ids == ["u0001", "u0002", "u0003"]
    assert [score.unit_id for score in result.scores] == expected_ids
    restored = from_json(to_json(result), EnforcementResult)
    assert restored == result


def test_e2e_ok_full() -> None:
    candidate = _candidate()
    ids = [unit.id for unit in candidate.units]
    verifier = FakeVerifier(
        {
            "u0001": _score("u0001", entailment=0.95, contradiction=0.01, label="entailment"),
            "u0002": _score("u0002", entailment=0.91, contradiction=0.02, label="entailment"),
            "u0003": _score("u0003", entailment=0.89, contradiction=0.05, label="entailment"),
        }
    )

    result = Enforcer(
        verifier=verifier,
        config=PolicyConfig(threshold_entailment=0.8, max_contradiction=0.2, partial_allowed=True),
    ).enforce(candidate=candidate, evidence=_evidence())

    assert result.decision.refusal is False
    assert result.decision.reason_code == ReasonCode.OK_FULL.value
    assert result.kept_units == ids
    assert result.dropped_units == []
    _assert_common(result, ids)


def test_e2e_ok_partial() -> None:
    candidate = _candidate()
    ids = [unit.id for unit in candidate.units]
    verifier = FakeVerifier(
        {
            "u0001": _score("u0001", entailment=0.92, contradiction=0.03, label="entailment"),
            "u0002": _score("u0002", entailment=0.12, contradiction=0.7, label="contradiction"),
            "u0003": _score("u0003", entailment=0.9, contradiction=0.04, label="entailment"),
        }
    )

    result = Enforcer(
        verifier=verifier,
        config=PolicyConfig(threshold_entailment=0.8, max_contradiction=0.2, partial_allowed=True),
    ).enforce(candidate=candidate, evidence=_evidence())

    assert result.decision.refusal is False
    assert result.decision.reason_code == ReasonCode.OK_PARTIAL.value
    assert result.kept_units == ["u0001", "u0003"]
    assert result.dropped_units == ["u0002"]
    assert result.final_text == "Alpha is true.\nGamma is true."
    _assert_common(result, ids)


def test_e2e_all_dropped_refusal() -> None:
    candidate = _candidate()
    ids = [unit.id for unit in candidate.units]
    verifier = FakeVerifier(
        {
            "u0001": _score("u0001", entailment=0.1, contradiction=0.8, label="contradiction"),
            "u0002": _score("u0002", entailment=0.2, contradiction=0.6, label="contradiction"),
            "u0003": _score("u0003", entailment=0.3, contradiction=0.5, label="contradiction"),
        }
    )

    result = Enforcer(
        verifier=verifier,
        config=PolicyConfig(threshold_entailment=0.8, max_contradiction=0.2, partial_allowed=True),
    ).enforce(candidate=candidate, evidence=_evidence())

    assert result.decision.refusal is True
    assert result.decision.reason_code == ReasonCode.ALL_DROPPED.value
    assert result.kept_units == []
    assert result.dropped_units == ids
    assert result.final_text is None
    assert result.refusal_message
    _assert_common(result, ids)


def test_e2e_partial_not_allowed_refusal() -> None:
    candidate = _candidate()
    ids = [unit.id for unit in candidate.units]
    verifier = FakeVerifier(
        {
            "u0001": _score("u0001", entailment=0.92, contradiction=0.03, label="entailment"),
            "u0002": _score("u0002", entailment=0.12, contradiction=0.7, label="contradiction"),
            "u0003": _score("u0003", entailment=0.9, contradiction=0.04, label="entailment"),
        }
    )

    result = Enforcer(
        verifier=verifier,
        config=PolicyConfig(threshold_entailment=0.8, max_contradiction=0.2, partial_allowed=False),
    ).enforce(candidate=candidate, evidence=_evidence())

    assert result.decision.refusal is True
    assert result.decision.reason_code == ReasonCode.PARTIAL_NOT_ALLOWED.value
    assert result.kept_units == ["u0001", "u0003"]
    assert result.dropped_units == ["u0002"]
    assert result.refusal_message
    _assert_common(result, ids)
