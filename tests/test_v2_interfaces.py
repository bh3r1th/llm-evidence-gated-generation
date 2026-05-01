from ega.types import EvidenceItem, EvidenceSet, Unit
from ega.v2 import (
    BudgetConfig,
    ConformalCalibrator,
    ConformalConfig,
    FixedBudgetPolicy,
    NoopReranker,
)


def test_v2_package_imports() -> None:
    _ = BudgetConfig()
    _ = ConformalCalibrator()
    _ = FixedBudgetPolicy()
    _ = NoopReranker()


def test_noop_reranker_returns_candidates_copy() -> None:
    reranker = NoopReranker()
    units = [Unit(id="u1", text="A", metadata={})]
    evidence = EvidenceSet(items=[EvidenceItem(id="e1", text="X", metadata={})])
    candidates = {"u1": ["e1", "e2"]}

    out = reranker.rerank(units=units, evidence=evidence, candidates=candidates, topk=1)

    assert out == candidates
    assert out is not candidates


def test_fixed_budget_policy_returns_base_params() -> None:
    policy = FixedBudgetPolicy()
    units = [Unit(id="u1", text="A", metadata={})]
    evidence = EvidenceSet(items=[EvidenceItem(id="e1", text="X", metadata={})])

    decision = policy.choose(
        units=units,
        evidence=evidence,
        base_params={
            "topk_per_unit": 12,
            "max_pairs_total": 200,
            "verifier_name": "nli_cross_encoder",
        },
        risk_features={},
        budget=BudgetConfig(),
    )

    assert decision.topk_per_unit == 12
    assert decision.max_pairs_total == 200
    assert decision.verifier_name == "nli_cross_encoder"


def test_conformal_calibrator_fit_and_gate() -> None:
    calibrator = ConformalCalibrator()
    config = ConformalConfig(epsilon=0.2, min_calib=5, abstain_k=0.0)
    rows = [
        {"score": 0.95, "supported": True},
        {"score": 0.90, "supported": True},
        {"score": 0.80, "supported": False},
        {"score": 0.30, "supported": False},
        {"score": 0.20, "supported": True},
    ]

    state = calibrator.fit(rows=rows, config=config)

    assert 0.0 <= state.threshold <= 1.0
    assert calibrator.gate(score=state.threshold, state=state) == "abstain"
    assert calibrator.gate(score=min(1.0, state.threshold + 0.01), state=state) == "accept"
    assert calibrator.gate(score=max(0.0, state.threshold - 0.01), state=state) == "reject"
