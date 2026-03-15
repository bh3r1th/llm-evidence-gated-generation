from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from ega.contract import PolicyConfig
from ega.pipeline import run_pipeline
from ega.types import EvidenceItem, EvidenceSet, VerificationScore
from ega.v2.coverage import CoverageConfig
from ega.v2.rewards import RewardConfig


def test_v2_end_to_end_smoke_coverage_rewards_and_training_emit() -> None:
    class FakeVerifier:
        def __init__(self) -> None:
            self.model_name = "fake-nli"

        def verify_many(self, candidate, evidence):  # type: ignore[no-untyped-def]
            chosen_id = evidence.items[0].id if evidence.items else None
            return [
                VerificationScore(
                    unit_id=unit.id,
                    entailment=0.95,
                    contradiction=0.03,
                    neutral=0.02,
                    label="entailment",
                    raw={"chosen_evidence_id": chosen_id, "per_item_probs": []},
                )
                for unit in candidate.units
            ]

        @staticmethod
        def get_last_verify_trace() -> dict[str, float | int]:
            return {
                "preselect_seconds": 0.01,
                "tokenize_seconds": 0.02,
                "forward_seconds": 0.03,
                "post_seconds": 0.01,
                "n_pairs_scored": 1,
            }

    trace_path = Path("data") / f"v2_smoke_trace_{uuid4().hex}.jsonl"
    train_path = Path("data") / f"v2_smoke_train_{uuid4().hex}.jsonl"
    try:
        result = run_pipeline(
            llm_summary_text="Supported fact.",
            evidence=EvidenceSet(
                items=[
                    EvidenceItem(id="e1", text="Supported fact.", metadata={}),
                    EvidenceItem(id="e2", text="Other fact.", metadata={}),
                ]
            ),
            unitizer_mode="sentence",
            policy_config=PolicyConfig(
                threshold_entailment=0.8,
                max_contradiction=0.2,
                partial_allowed=True,
            ),
            use_oss_nli=True,
            verifier=FakeVerifier(),
            coverage_config=CoverageConfig(pool_topk=2),
            reward_config=RewardConfig(),
            emit_training_example_path=str(train_path),
            training_example_id="smoke-1",
            trace_out=str(trace_path),
            render_safe_answer=True,
        )

        trace_row = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[0])
        assert "coverage_pool_topk" in trace_row
        assert "coverage_avg_score" in trace_row
        assert "coverage_unit_scores" in trace_row
        assert "coverage_missing_total" in trace_row
        assert "reward_total" in trace_row
        assert "reward_avg" in trace_row
        assert "reward_avg_support" in trace_row
        assert "reward_hallucination_rate" in trace_row
        assert "reward_abstention_rate" in trace_row
        assert "reward_avg_coverage" in trace_row
        assert "reward_unit_totals" in trace_row
        assert result["safe_answer_final_text"] == "Supported fact. [e1]"
        assert result["safe_answer_summary"] == {
            "accepted_count": 1,
            "abstained_count": 0,
            "rejected_count": 0,
        }
        assert trace_row["safe_answer_final_text"] == "Supported fact. [e1]"
        assert trace_row["safe_answer_summary"] == result["safe_answer_summary"]

        rows = [line for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(rows) == 1
        training_row = json.loads(rows[0])
        required = {
            "id",
            "input_prompt",
            "generated_text",
            "units",
            "pool_candidates",
            "used_evidence",
            "decisions",
            "verifier_scores",
            "coverage",
            "rewards",
            "reward_summary",
        }
        assert required.issubset(set(training_row.keys()))
        assert training_row["id"] == "smoke-1"
    finally:
        trace_path.unlink(missing_ok=True)
        train_path.unlink(missing_ok=True)
