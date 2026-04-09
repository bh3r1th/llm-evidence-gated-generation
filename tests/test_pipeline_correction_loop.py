from __future__ import annotations

from ega.contract import PolicyConfig
from ega.pipeline import run_pipeline
from ega.types import EvidenceItem, EvidenceSet, Unit, VerificationScore


class _ConditionalVerifier:
    model_name = "fake-nli"

    def verify_many(self, candidate, evidence):  # type: ignore[no-untyped-def]
        del evidence
        scores: list[VerificationScore] = []
        for unit in candidate.units:
            is_supported = "good" in unit.text.lower() or "fixed" in unit.text.lower()
            entail = 0.95 if is_supported else 0.05
            scores.append(
                VerificationScore(
                    unit_id=unit.id,
                    entailment=entail,
                    contradiction=0.0 if is_supported else 0.9,
                    neutral=1.0 - entail,
                    label="entailment" if is_supported else "contradiction",
                    raw={"chosen_evidence_id": "e1", "per_item_probs": []},
                )
            )
        return scores

    @staticmethod
    def get_last_verify_trace() -> dict[str, float | int]:
        return {"n_pairs_scored": 1, "forward_seconds": 0.0}


def _evidence() -> EvidenceSet:
    return EvidenceSet(items=[EvidenceItem(id="e1", text="Good fact. Fixed fact.", metadata={})])


def test_correction_loop_enabled_regenerates_failed_units_only() -> None:
    calls: list[list[str]] = []

    def _generator(failed_units: list[Unit], _evidence: EvidenceSet, _retry: int) -> dict[str, str]:
        calls.append([unit.text for unit in failed_units])
        return {failed_units[0].id: "Fixed fact."}

    output = run_pipeline(
        llm_summary_text="Bad fact. Good fact.",
        evidence=_evidence(),
        policy_config=PolicyConfig(
            threshold_entailment=0.5,
            max_contradiction=0.2,
            partial_allowed=True,
        ),
        use_oss_nli=True,
        verifier=_ConditionalVerifier(),
        enable_correction=True,
        max_retries=1,
        correction_generator=_generator,
    )

    assert calls == [["Bad fact."]]
    assert output["stats"]["kept_units"] == 2
    assert output["correction"]["attempts"] == 1


def test_correction_loop_disabled_preserves_baseline_behavior() -> None:
    called = False

    def _generator(_failed_units: list[Unit], _evidence: EvidenceSet, _retry: int) -> dict[str, str]:
        nonlocal called
        called = True
        return {}

    output = run_pipeline(
        llm_summary_text="Bad fact. Good fact.",
        evidence=_evidence(),
        policy_config=PolicyConfig(
            threshold_entailment=0.5,
            max_contradiction=0.2,
            partial_allowed=True,
        ),
        use_oss_nli=True,
        verifier=_ConditionalVerifier(),
        enable_correction=False,
        max_retries=1,
        correction_generator=_generator,
    )

    assert called is False
    assert output["stats"]["kept_units"] == 1
    assert output["correction"]["attempts"] == 0
