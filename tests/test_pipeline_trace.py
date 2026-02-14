from __future__ import annotations

import json
import time
from pathlib import Path
from uuid import uuid4

import ega.verifiers.nli_cross_encoder as nli_mod
from ega.contract import PolicyConfig
from ega.pipeline import run_pipeline
from ega.types import EvidenceItem, EvidenceSet, VerificationScore


def test_pipeline_trace_includes_load_and_compute_breakdown() -> None:
    scores_path = Path("data") / f"trace_scores_{uuid4().hex}.jsonl"
    trace_path = Path("data") / f"trace_breakdown_{uuid4().hex}.jsonl"
    try:
        scores_path.write_text(
            json.dumps({"unit_id": "u0001", "score": 0.95}) + "\n",
            encoding="utf-8",
        )

        run_pipeline(
            llm_summary_text="Supported fact.",
            evidence=EvidenceSet(items=[EvidenceItem(id="e1", text="Supported fact.", metadata={})]),
            unitizer_mode="sentence",
            policy_config=PolicyConfig(
                threshold_entailment=0.8,
                max_contradiction=0.2,
                partial_allowed=True,
            ),
            scores_jsonl_path=str(scores_path),
            trace_out=str(trace_path),
        )

        lines = trace_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        payload = json.loads(lines[0])

        assert "load_seconds" in payload
        assert "verify_compute_seconds" in payload
        assert "verify_seconds" in payload
        assert "forward_seconds" in payload
        assert payload["verify_seconds"] >= payload["verify_compute_seconds"]
        assert payload["verify_compute_seconds"] >= payload["forward_seconds"]
    finally:
        scores_path.unlink(missing_ok=True)
        trace_path.unlink(missing_ok=True)


def test_pipeline_trace_load_seconds_positive_when_verifier_is_constructed(
    monkeypatch,
) -> None:
    class FakeVerifier:
        def __init__(self, model_name=None, **_kwargs):  # type: ignore[no-untyped-def]
            time.sleep(0.01)
            self.model_name = model_name or "fake-nli"

        def verify_many(self, candidate, evidence):  # type: ignore[no-untyped-def]
            chosen_id = evidence.items[0].id if evidence.items else None
            return [
                VerificationScore(
                    unit_id=unit.id,
                    entailment=1.0,
                    contradiction=0.0,
                    neutral=0.0,
                    label="entailment",
                    raw={"chosen_evidence_id": chosen_id, "per_item_probs": []},
                )
                for unit in candidate.units
            ]

        @staticmethod
        def get_last_verify_trace() -> dict[str, float | int]:
            return {"n_pairs_scored": 1, "forward_seconds": 0.0}

    trace_path = Path("data") / f"trace_load_positive_{uuid4().hex}.jsonl"
    try:
        monkeypatch.setattr(nli_mod, "NliCrossEncoderVerifier", FakeVerifier)
        run_pipeline(
            llm_summary_text="Supported fact.",
            evidence=EvidenceSet(items=[EvidenceItem(id="e1", text="Supported fact.", metadata={})]),
            unitizer_mode="sentence",
            policy_config=PolicyConfig(
                threshold_entailment=0.8,
                max_contradiction=0.2,
                partial_allowed=True,
            ),
            use_oss_nli=True,
            trace_out=str(trace_path),
        )
        payload = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[0])
        assert payload["load_seconds"] > 0.0
        assert payload["verify_seconds"] >= payload["verify_compute_seconds"]
    finally:
        trace_path.unlink(missing_ok=True)


def test_pipeline_trace_load_seconds_zero_when_reusing_verifier_instance() -> None:
    class FakeVerifier:
        def __init__(self) -> None:
            self.model_name = "fake-nli"

        def verify_many(self, candidate, evidence):  # type: ignore[no-untyped-def]
            chosen_id = evidence.items[0].id if evidence.items else None
            return [
                VerificationScore(
                    unit_id=unit.id,
                    entailment=1.0,
                    contradiction=0.0,
                    neutral=0.0,
                    label="entailment",
                    raw={"chosen_evidence_id": chosen_id, "per_item_probs": []},
                )
                for unit in candidate.units
            ]

        @staticmethod
        def get_last_verify_trace() -> dict[str, float | int]:
            return {"n_pairs_scored": 1, "forward_seconds": 0.0}

    trace_path = Path("data") / f"trace_load_reuse_{uuid4().hex}.jsonl"
    try:
        run_pipeline(
            llm_summary_text="Supported fact.",
            evidence=EvidenceSet(items=[EvidenceItem(id="e1", text="Supported fact.", metadata={})]),
            unitizer_mode="sentence",
            policy_config=PolicyConfig(
                threshold_entailment=0.8,
                max_contradiction=0.2,
                partial_allowed=True,
            ),
            use_oss_nli=True,
            verifier=FakeVerifier(),
            trace_out=str(trace_path),
        )
        payload = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[0])
        assert payload["load_seconds"] == 0.0
        assert payload["verify_seconds"] >= payload["verify_compute_seconds"]
    finally:
        trace_path.unlink(missing_ok=True)
