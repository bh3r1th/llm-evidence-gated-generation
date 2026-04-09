from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

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

    trace_path = Path("data") / f"correction_trace_enabled_{uuid4().hex}.jsonl"
    try:
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
            trace_out=str(trace_path),
        )
        trace_row = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[0])
    finally:
        trace_path.unlink(missing_ok=True)

    assert calls == [["Bad fact."]]
    assert output["stats"]["kept_units"] == 2
    assert output["correction"]["attempts"] == 1
    assert trace_row["correction_enabled"] is True
    assert trace_row["correction_retries_attempted"] == 1
    assert trace_row["correction_corrected_unit_count"] == 1
    assert trace_row["correction_still_failed_count"] == 0
    assert trace_row["correction_reverify_occurred"] is True
    assert trace_row["correction_stopped_reason"] == "all_corrected"


def test_correction_loop_disabled_preserves_baseline_behavior() -> None:
    called = False

    def _generator(_failed_units: list[Unit], _evidence: EvidenceSet, _retry: int) -> dict[str, str]:
        nonlocal called
        called = True
        return {}

    trace_path = Path("data") / f"correction_trace_disabled_{uuid4().hex}.jsonl"
    try:
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
            trace_out=str(trace_path),
        )
        trace_row = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[0])
    finally:
        trace_path.unlink(missing_ok=True)

    assert called is False
    assert output["stats"]["kept_units"] == 1
    assert output["correction"]["attempts"] == 0
    assert trace_row["correction_enabled"] is False
    assert trace_row["correction_retries_attempted"] == 0
    assert trace_row["correction_corrected_unit_count"] == 0
    assert trace_row["correction_still_failed_count"] == 1
    assert trace_row["correction_reverify_occurred"] is False
    assert trace_row["correction_stopped_reason"] == "correction_disabled"
