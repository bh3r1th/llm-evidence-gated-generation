"""Enforcement entrypoints for EGA decision flow.

The enforcer module remains intentionally lightweight and free of policy logic.
"""

from __future__ import annotations

from ega.decision import build_enforcement_result
from ega.types import EnforcementResult, GateDecision, Unit, VerificationScore


class Enforcer:
    """Minimal enforcer façade that shapes explicit enforcement outputs."""

    def enforce(
        self,
        *,
        final_text: str | None,
        kept_units: list[Unit],
        dropped_units: list[Unit],
        refusal_message: str | None,
        decision: GateDecision,
        scores: list[VerificationScore],
    ) -> EnforcementResult:
        """Return a deterministic :class:`EnforcementResult` from explicit inputs."""

        return build_enforcement_result(
            final_text=final_text,
            kept_units=kept_units,
            dropped_units=dropped_units,
            refusal_message=refusal_message,
            decision=decision,
            scores=scores,
        )
