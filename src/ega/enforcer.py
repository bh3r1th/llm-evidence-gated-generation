"""Enforcement orchestration for deterministic policy gating."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ega.policy import DefaultPolicy, Policy, PolicyConfig
from ega.types import (
    AnswerCandidate,
    EnforcementResult,
    EvidenceSet,
    VerificationScore,
)


class Verifier(Protocol):
    """Verifier interface used by the enforcer runtime."""

    def verify(self, *, unit_text: str, unit_id: str, evidence: EvidenceSet) -> VerificationScore:
        """Return a deterministic score for one answer unit."""


@dataclass(slots=True)
class Enforcer:
    """Deterministic enforcer that verifies units and applies a policy."""

    verifier: Verifier
    policy: Policy = DefaultPolicy()
    config: PolicyConfig = PolicyConfig()
    refusal_message: str = "I can’t provide a supported answer from the available evidence."

    def enforce(self, *, candidate: AnswerCandidate, evidence: EvidenceSet) -> EnforcementResult:
        """Verify units, apply policy, and emit final gated output."""

        scores = [
            self.verifier.verify(unit_text=unit.text, unit_id=unit.id, evidence=evidence)
            for unit in candidate.units
        ]
        decision = self.policy.decide(scores=scores, units=candidate.units, config=self.config)

        if decision.refusal:
            return EnforcementResult(
                final_text=None,
                kept_units=decision.allowed_units,
                dropped_units=decision.dropped_units,
                refusal_message=self.refusal_message,
                decision=decision,
                scores=scores,
            )

        final_text = "\n".join(unit.text for unit in decision.allowed_units)
        return EnforcementResult(
            final_text=final_text,
            kept_units=decision.allowed_units,
            dropped_units=decision.dropped_units,
            refusal_message=None,
            decision=decision,
            scores=scores,
        )
