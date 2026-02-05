"""Deterministic policy definitions for answer gating behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ega.types import GateDecision, Unit, VerificationScore


class ReasonCode:
    """Stable reason codes emitted by gate decisions."""

    ALL_DROPPED = "ALL_DROPPED"
    PARTIAL_NOT_ALLOWED = "PARTIAL_NOT_ALLOWED"
    OK_PARTIAL = "OK_PARTIAL"
    OK_FULL = "OK_FULL"


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    """Runtime policy thresholds and output constraints."""

    threshold: float = 0.8
    contradiction_max: float = 0.2
    partial_allowed: bool = True


class Policy(Protocol):
    """Policy interface for deterministic, score-driven unit gating."""

    def decide(
        self,
        *,
        scores: list[VerificationScore],
        units: list[Unit],
        config: PolicyConfig,
    ) -> GateDecision:
        """Return a gate decision for the provided units and score outputs."""


class DefaultPolicy:
    """Default v1 policy with deterministic keep/drop behavior."""

    def decide(
        self,
        *,
        scores: list[VerificationScore],
        units: list[Unit],
        config: PolicyConfig,
    ) -> GateDecision:
        score_by_unit = {score.unit_id: score for score in scores}
        allowed_units: list[Unit] = []
        dropped_units: list[Unit] = []

        for unit in units:
            score = score_by_unit.get(unit.id)
            if score is None:
                dropped_units.append(unit)
                continue

            if (
                score.entailment >= config.threshold
                and score.contradiction <= config.contradiction_max
            ):
                allowed_units.append(unit)
            else:
                dropped_units.append(unit)

        refusal = False
        if not allowed_units:
            refusal = True
            reason_code = ReasonCode.ALL_DROPPED
        elif not config.partial_allowed and len(allowed_units) != len(units):
            refusal = True
            reason_code = ReasonCode.PARTIAL_NOT_ALLOWED
        elif len(allowed_units) == len(units):
            reason_code = ReasonCode.OK_FULL
        else:
            reason_code = ReasonCode.OK_PARTIAL

        return GateDecision(
            allowed_units=allowed_units,
            dropped_units=dropped_units,
            refusal=refusal,
            reason_code=reason_code,
            summary_stats={
                "total_units": len(units),
                "kept_units": len(allowed_units),
                "dropped_units": len(dropped_units),
                "threshold": config.threshold,
                "contradiction_max": config.contradiction_max,
                "partial_allowed": config.partial_allowed,
            },
        )
