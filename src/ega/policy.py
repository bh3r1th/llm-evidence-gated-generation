"""Deterministic policy definitions for answer gating behavior."""

from __future__ import annotations

import math
from typing import Protocol

from ega.contract import PolicyConfig, ReasonCode
from ega.types import GateDecision, Unit, VerificationScore


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
        allowed_units: list[str] = []
        dropped_units: list[str] = []
        invalid_entailment_count = 0

        for unit in units:
            score = score_by_unit.get(unit.id)
            if score is None:
                dropped_units.append(unit.id)
                continue

            # Conformal gate explicitly rejected or abstained this unit — honour it.
            if score.conformal_decision in {"reject", "abstain"}:
                dropped_units.append(unit.id)
                continue

            if (
                (not math.isfinite(score.entailment))
                or score.entailment < 0.0
                or score.entailment > 1.0
            ):
                invalid_entailment_count += 1
                dropped_units.append(unit.id)
                continue

            has_contradiction = bool(score.raw.get("has_contradiction", True))
            entailment_ok = score.entailment >= config.threshold_entailment
            contradiction_ok = (not has_contradiction) or (
                score.contradiction <= config.max_contradiction
            )

            if entailment_ok and contradiction_ok:
                allowed_units.append(unit.id)
            else:
                dropped_units.append(unit.id)

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
            reason_code=reason_code.value,
            summary_stats={
                "total_units": len(units),
                "kept_units": len(allowed_units),
                "dropped_units": len(dropped_units),
                "threshold_entailment": config.threshold_entailment,
                "max_contradiction": config.max_contradiction,
                "partial_allowed": config.partial_allowed,
                "invalid_entailment_count": invalid_entailment_count,
            },
        )
