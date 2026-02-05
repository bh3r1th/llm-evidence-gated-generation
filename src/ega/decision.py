"""Decision computation interfaces for EGA.

This module is responsible for translating policy + verification outputs into a
final decision outcome.
"""

from ega.policy import GatingPolicy
from ega.types import DecisionOutcome, VerificationResult


def decide(policy: GatingPolicy, result: VerificationResult) -> DecisionOutcome:
    """Produce a decision outcome from policy and verification data.

    TODO: Replace placeholder logic with full decision graph.
    """
    if result.passed and result.score >= policy.minimum_support_score:
        return DecisionOutcome.ALLOW
    if policy.allow_abstain:
        return DecisionOutcome.ABSTAIN
    return DecisionOutcome.BLOCK
