import pytest

from ega.types import Unit
from ega.v2.coverage import CoverageResult
from ega.v2.rewards import RewardComputer, RewardConfig


def _unit(unit_id: str) -> Unit:
    return Unit(id=unit_id, text=f"text-{unit_id}", metadata={})


def test_rewards_with_mixed_decisions_and_coverage() -> None:
    computer = RewardComputer()
    units = [_unit("u1"), _unit("u2"), _unit("u3")]
    verification = {
        "u1": {"entailment": 0.8},
        "u2": {"entailment": 0.2},
        "u3": {"entailment": 0.1},
    }
    decisions = {"u1": "accept", "u2": "reject", "u3": "abstain"}
    coverage = {
        "u1": CoverageResult(
            unit_id="u1",
            relevant_evidence_ids=["e1", "e2"],
            used_evidence_ids=["e1"],
            coverage_score=0.5,
            missing_evidence_ids=["e2"],
            meta={},
        ),
        "u2": CoverageResult(
            unit_id="u2",
            relevant_evidence_ids=["e3"],
            used_evidence_ids=[],
            coverage_score=0.0,
            missing_evidence_ids=["e3"],
            meta={},
        ),
        "u3": CoverageResult(
            unit_id="u3",
            relevant_evidence_ids=["e4", "e5"],
            used_evidence_ids=["e4"],
            coverage_score=0.5,
            missing_evidence_ids=["e5"],
            meta={},
        ),
    }

    unit_rewards, summary = computer.compute(
        units=units,
        verification=verification,
        decisions=decisions,
        coverage=coverage,
        config=RewardConfig(),
    )

    assert unit_rewards["u1"].total_reward == pytest.approx(1.3)
    assert unit_rewards["u2"].total_reward == pytest.approx(-1.8)
    assert unit_rewards["u3"].total_reward == pytest.approx(0.1)
    assert summary.total_reward == pytest.approx(-0.4)
    assert summary.avg_reward == pytest.approx(-0.4 / 3.0)
    assert summary.avg_support_score == pytest.approx((0.8 + 0.2 + 0.1) / 3.0)
    assert summary.hallucination_rate == pytest.approx(1.0 / 3.0)
    assert summary.abstention_rate == pytest.approx(1.0 / 3.0)
    assert summary.avg_coverage_score == pytest.approx((0.5 + 0.0 + 0.5) / 3.0)


def test_rewards_supports_keep_drop_mapping_without_coverage() -> None:
    computer = RewardComputer()
    units = [_unit("u1"), _unit("u2")]
    verification = {"u1": {"entailment": 1.0}, "u2": {"entailment": 0.9}}
    decisions = {"u1": "keep", "u2": "drop"}

    unit_rewards, summary = computer.compute(
        units=units,
        verification=verification,
        decisions=decisions,
        coverage=None,
        config=RewardConfig(),
    )

    assert unit_rewards["u1"].hallucination_penalty == 0.0
    assert unit_rewards["u2"].hallucination_penalty == 1.0
    assert unit_rewards["u1"].total_reward == pytest.approx(1.0)
    assert unit_rewards["u2"].total_reward == pytest.approx(-1.1)
    assert summary.total_reward == pytest.approx(-0.1)


def test_rewards_clamp_applies_to_extreme_values() -> None:
    computer = RewardComputer()
    units = [_unit("u1"), _unit("u2")]
    verification = {"u1": {"entailment": 1.0}, "u2": {"entailment": 1.0}}
    decisions = {"u1": "accept", "u2": "reject"}

    unit_rewards, _summary = computer.compute(
        units=units,
        verification=verification,
        decisions=decisions,
        coverage=None,
        config=RewardConfig(w_support=10.0, w_hallucination=20.0, clamp_min=-5.0, clamp_max=5.0),
    )

    assert unit_rewards["u1"].total_reward == 5.0
    assert unit_rewards["u2"].total_reward == -5.0
