from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from ega.contract import PolicyConfig
from ega.pipeline import run_pipeline
from ega.types import EvidenceItem, EvidenceSet


def test_pipeline_default_off_keeps_output_and_trace_shape() -> None:
    scores_path = Path("data") / f"default_off_scores_{uuid4().hex}.jsonl"
    trace_a = Path("data") / f"default_off_trace_a_{uuid4().hex}.jsonl"
    trace_b = Path("data") / f"default_off_trace_b_{uuid4().hex}.jsonl"
    try:
        scores_path.write_text(
            json.dumps({"unit_id": "u0001", "score": 0.95, "raw": {"chosen_evidence_id": "e1"}}) + "\n",
            encoding="utf-8",
        )
        kwargs = {
            "llm_summary_text": "Supported fact.",
            "evidence": EvidenceSet(items=[EvidenceItem(id="e1", text="Supported fact.", metadata={})]),
            "unitizer_mode": "sentence",
            "policy_config": PolicyConfig(
                threshold_entailment=0.8,
                max_contradiction=0.2,
                partial_allowed=True,
            ),
            "scores_jsonl_path": str(scores_path),
        }
        out_a = run_pipeline(**kwargs, trace_out=str(trace_a))
        out_b = run_pipeline(
            **kwargs,
            trace_out=str(trace_b),
            coverage_config=None,
            reward_config=None,
            emit_training_example_path=None,
            training_example_id=None,
        )

        assert out_a == out_b

        row_a = json.loads(trace_a.read_text(encoding="utf-8").splitlines()[0])
        row_b = json.loads(trace_b.read_text(encoding="utf-8").splitlines()[0])
        assert set(row_a.keys()) == set(row_b.keys())

        new_keys = {
            "coverage_pool_topk",
            "coverage_avg_score",
            "coverage_unit_scores",
            "coverage_missing_total",
            "reward_total",
            "reward_avg",
            "reward_avg_support",
            "reward_hallucination_rate",
            "reward_abstention_rate",
            "reward_avg_coverage",
            "reward_unit_totals",
        }
        assert new_keys.isdisjoint(set(row_a.keys()))
        assert new_keys.isdisjoint(set(out_a.get("stats", {}).keys()))
    finally:
        scores_path.unlink(missing_ok=True)
        trace_a.unlink(missing_ok=True)
        trace_b.unlink(missing_ok=True)
