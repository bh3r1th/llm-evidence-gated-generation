from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from ega.contract import PolicyConfig
from ega.pipeline import run_pipeline
from ega.types import EvidenceItem, EvidenceSet, VerificationScore
from ega.v2.budget import BudgetConfig
from ega.v2.budget_greedy import GreedyBudgetPolicy


class _TraceVerifier:
    def __init__(self) -> None:
        self.model_name = "test-nli"
        self._last_pairs = 0

    def verify_many(self, candidate, evidence):  # type: ignore[no-untyped-def]
        self._last_pairs = len(evidence.items)
        chosen_id = evidence.items[0].id if evidence.items else None
        return [
            VerificationScore(
                unit_id=unit.id,
                entailment=0.9 if chosen_id is not None else 0.0,
                contradiction=0.0 if chosen_id is not None else 1.0,
                neutral=0.1 if chosen_id is not None else 0.0,
                label="entailment" if chosen_id is not None else "contradiction",
                raw={"chosen_evidence_id": chosen_id, "per_item_probs": []},
            )
            for unit in candidate.units
        ]

    def get_last_verify_trace(self) -> dict[str, int | float]:
        return {"n_pairs_scored": self._last_pairs, "forward_seconds": 0.0}


def _policy() -> PolicyConfig:
    return PolicyConfig(threshold_entailment=0.5, max_contradiction=0.5, partial_allowed=True)


def _evidence() -> EvidenceSet:
    return EvidenceSet(
        items=[
            EvidenceItem(id="e1", text="short claim support", metadata={}),
            EvidenceItem(id="e2", text="medium evidence with overlap", metadata={}),
            EvidenceItem(id="e3", text="long ambiguous evidence with many extra words", metadata={}),
            EvidenceItem(id="e4", text="another unrelated sentence", metadata={}),
            EvidenceItem(id="e5", text="backup unrelated sentence", metadata={}),
        ]
    )


def test_budget_max_pairs_reduces_verifier_calls_proxy_below_baseline() -> None:
    verifier = _TraceVerifier()
    base_trace = Path("data") / f"budget_trace_base_{uuid4().hex}.jsonl"
    budget_trace = Path("data") / f"budget_trace_budget_{uuid4().hex}.jsonl"
    try:
        baseline = run_pipeline(
            llm_summary_text="Short claim. Medium overlap claim. Long ambiguous claim with extra tokens.",
            evidence=_evidence(),
            unitizer_mode="sentence",
            policy_config=_policy(),
            use_oss_nli=True,
            verifier=verifier,
            topk_per_unit=2,
            max_pairs_total=20,
            trace_out=str(base_trace),
        )
        budgeted = run_pipeline(
            llm_summary_text="Short claim. Medium overlap claim. Long ambiguous claim with extra tokens.",
            evidence=_evidence(),
            unitizer_mode="sentence",
            policy_config=_policy(),
            use_oss_nli=True,
            verifier=_TraceVerifier(),
            topk_per_unit=2,
            max_pairs_total=20,
            budget_policy=GreedyBudgetPolicy(),
            budget_config=BudgetConfig(max_pairs_total=3),
            trace_out=str(budget_trace),
        )
        baseline_trace = json.loads(base_trace.read_text(encoding="utf-8").splitlines()[0])
        budget_trace_payload = json.loads(budget_trace.read_text(encoding="utf-8").splitlines()[0])

        assert budget_trace_payload["n_pairs"] < baseline_trace["n_pairs"]
        assert budgeted["stats"]["effective_budget_max_pairs"] == 3
        assert budgeted["stats"]["planned_pairs_total"] > budgeted["stats"]["evaluated_pairs_total"]
        assert budgeted["stats"]["evaluated_pairs_total"] == 3
        assert budgeted["stats"]["pruned_pairs_total"] > 0
        assert budgeted["stats"]["pairs_allocated_to_high_risk_units"] >= (
            budgeted["stats"]["pairs_allocated_to_low_risk_units"]
        )
        assert sum(budgeted["per_unit_pair_budget"].values()) == 3
        assert budgeted["evaluated_pairs_count_per_unit"] == budgeted["per_unit_pair_budget"]
        assert budgeted["per_unit_pairs_before_budget"] != budgeted["per_unit_pairs_after_budget"]
    finally:
        base_trace.unlink(missing_ok=True)
        budget_trace.unlink(missing_ok=True)


def test_budget_disabled_leaves_pipeline_output_unchanged() -> None:
    kwargs = {
        "llm_summary_text": "One claim. Another claim.",
        "evidence": _evidence(),
        "unitizer_mode": "sentence",
        "policy_config": _policy(),
        "use_oss_nli": True,
        "verifier": _TraceVerifier(),
        "topk_per_unit": 2,
        "max_pairs_total": 20,
    }
    baseline = run_pipeline(**kwargs)
    with_budget_disabled = run_pipeline(
        **kwargs,
        budget_policy=None,
        budget_config=None,
    )

    assert with_budget_disabled == baseline
