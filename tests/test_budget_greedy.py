from __future__ import annotations

import pytest

from ega.types import EvidenceItem, EvidenceSet, Unit
from ega.v2.budget import BudgetConfig
from ega.v2.budget_greedy import GreedyBudgetPolicy
from ega.v2.risk import extract_unit_risks


def test_greedy_budget_allocates_more_to_high_risk_and_respects_cap() -> None:
    units = [
        Unit(id="u_high", text="high risk unit", metadata={}),
        Unit(id="u_mid", text="mid risk unit", metadata={}),
        Unit(id="u_low", text="low risk unit", metadata={}),
    ]
    evidence = EvidenceSet(
        items=[
            EvidenceItem(id="e1", text="doc1", metadata={}),
            EvidenceItem(id="e2", text="doc2", metadata={}),
            EvidenceItem(id="e3", text="doc3", metadata={}),
            EvidenceItem(id="e4", text="doc4", metadata={}),
            EvidenceItem(id="e5", text="doc5", metadata={}),
        ]
    )
    policy = GreedyBudgetPolicy()
    decision = policy.choose(
        units=units,
        evidence=evidence,
        base_params={
            "topk_per_unit": 1,
            "max_pairs_total": 20,
            "verifier_name": "nli_cross_encoder",
        },
        risk_features={"u_high": 0.95, "u_mid": 0.50, "u_low": 0.10},
        budget=BudgetConfig(
            max_pairs_total=8,
            latency_budget_ms=100,
            cost_per_pair=1.0,
        ),
    )
    trace = policy.get_last_trace()

    assert decision.max_pairs_total <= 8
    assert decision.topk_per_unit >= 1
    assert decision.per_unit_pair_budget is not None
    assert sum(decision.per_unit_pair_budget.values()) == decision.max_pairs_total
    assert trace.allocated_steps == decision.max_pairs_total
    assert decision.per_unit_pair_budget["u_high"] >= decision.per_unit_pair_budget["u_mid"]
    assert decision.per_unit_pair_budget["u_mid"] >= decision.per_unit_pair_budget["u_low"]
    assert trace.per_unit_pair_budget["u_high"] > trace.per_unit_pair_budget["u_low"]


def test_greedy_budget_respects_latency_derived_budget() -> None:
    units = [Unit(id="u1", text="A", metadata={}), Unit(id="u2", text="B", metadata={})]
    evidence = EvidenceSet(
        items=[
            EvidenceItem(id="e1", text="doc1", metadata={}),
            EvidenceItem(id="e2", text="doc2", metadata={}),
            EvidenceItem(id="e3", text="doc3", metadata={}),
            EvidenceItem(id="e4", text="doc4", metadata={}),
        ]
    )
    policy = GreedyBudgetPolicy()
    decision = policy.choose(
        units=units,
        evidence=evidence,
        base_params={
            "topk_per_unit": 1,
            "max_pairs_total": 20,
            "verifier_name": "nli_cross_encoder",
        },
        risk_features={"u1": 0.8, "u2": 0.8},
        budget=BudgetConfig(
            max_pairs_total=20,
            latency_budget_ms=4,
            cost_per_pair=2.0,
        ),
    )
    # latency budget maps to a total pair budget floor(4/2)=2
    assert decision.max_pairs_total == 2


def test_extract_unit_risks_from_similarity_and_fallback() -> None:
    units = [
        Unit(id="u1", text="A", metadata={"initial_retrieval_similarity": 0.9}),
        Unit(id="u2", text="B", metadata={}),
    ]
    evidence = EvidenceSet(items=[EvidenceItem(id="e1", text="doc1", metadata={})])

    risks = extract_unit_risks(units=units, evidence=evidence, similarity_by_unit=None)

    assert risks["u1"] == pytest.approx(0.1)
    assert 0.0 <= risks["u2"] <= 1.0
    assert risks["u2"] > risks["u1"]
