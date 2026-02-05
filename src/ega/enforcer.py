"""Enforcement entrypoints for EGA decision flow.

The enforcer orchestrates unitization, verification, decisioning, and optional
serialization for downstream consumers.
"""

from ega.decision import decide
from ega.policy import GatingPolicy
from ega.types import DecisionOutcome, VerificationResult


class Enforcer:
    """Placeholder enforcer for policy-driven answer gating.

    TODO: Add verifier registry and pipeline execution hooks.
    """

    def __init__(self, policy: GatingPolicy | None = None) -> None:
        self.policy = policy or GatingPolicy()

    def enforce(self, verification: VerificationResult) -> DecisionOutcome:
        """Apply policy to a verification result and return a decision."""
        return decide(self.policy, verification)
