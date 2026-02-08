from __future__ import annotations

import pytest

from ega.contract import EGA_SCHEMA_VERSION, PolicyConfig
from ega.enforcer import Enforcer
from ega.policy import DefaultPolicy
from ega.types import EvidenceItem, EvidenceSet, VerificationScore
from ega.unitization import unitize_answer
from ega.verifiers.nli_cross_encoder import NliCrossEncoderVerifier

pytestmark = pytest.mark.integration

pytest.importorskip("torch")
pytest.importorskip("transformers")


class _VerifierAdapter:
    def __init__(self, verifier: NliCrossEncoderVerifier) -> None:
        self._verifier = verifier

    def verify(self, *, unit_text: str, unit_id: str, evidence: EvidenceSet) -> VerificationScore:
        score = self._verifier.verify_unit(unit_text, evidence)
        return VerificationScore(
            unit_id=unit_id,
            entailment=score.entailment,
            contradiction=score.contradiction,
            neutral=score.neutral,
            label=score.label,
            raw=dict(score.raw),
        )


def test_nli_enforcer_manual_smoke() -> None:
    candidate = unitize_answer("Paris is in France. Berlin is in Germany.", mode="sentence")
    evidence = EvidenceSet(
        items=[
            EvidenceItem(id="e1", text="Paris is the capital of France.", metadata={}),
            EvidenceItem(id="e2", text="Berlin is the capital of Germany.", metadata={}),
        ]
    )

    verifier = _VerifierAdapter(NliCrossEncoderVerifier())
    result = Enforcer(
        verifier=verifier,
        policy=DefaultPolicy(),
        config=PolicyConfig(partial_allowed=True),
    ).enforce(candidate=candidate, evidence=evidence)

    assert result.ega_schema_version == EGA_SCHEMA_VERSION
    assert len(result.scores) == len(candidate.units)
    assert {
        score.label.lower() for score in result.scores
    } <= {"entailment", "contradiction", "neutral"}
