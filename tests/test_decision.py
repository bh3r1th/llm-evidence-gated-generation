"""Decision layer skeleton tests."""

from ega.decision import decide
from ega.policy import GatingPolicy
from ega.types import DecisionOutcome, VerificationResult


def test_decide_allow_for_passing_result() -> None:
    policy = GatingPolicy(minimum_support_score=0.5)
    result = VerificationResult(verifier_name="dummy", score=0.9, passed=True)

    assert decide(policy, result) is DecisionOutcome.ALLOW
