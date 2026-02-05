"""Decision payload helpers for EGA.

This module intentionally contains only deterministic data shaping helpers and
no policy business logic.
"""

from __future__ import annotations

from typing import Any

from ega.types import EnforcementResult, GateDecision, Unit, VerificationScore


def build_gate_decision(
    *,
    allowed_units: list[Unit],
    dropped_units: list[Unit],
    refusal: bool,
    reason_code: str,
    summary_stats: dict[str, Any] | None = None,
) -> GateDecision:
    """Build a :class:`GateDecision` with explicit values."""

    return GateDecision(
        allowed_units=list(allowed_units),
        dropped_units=list(dropped_units),
        refusal=refusal,
        reason_code=reason_code,
        summary_stats=dict(summary_stats or {}),
    )


def build_enforcement_result(
    *,
    final_text: str | None,
    kept_units: list[Unit],
    dropped_units: list[Unit],
    refusal_message: str | None,
    decision: GateDecision,
    scores: list[VerificationScore],
) -> EnforcementResult:
    """Build an :class:`EnforcementResult` with explicit values."""

    return EnforcementResult(
        final_text=final_text,
        kept_units=list(kept_units),
        dropped_units=list(dropped_units),
        refusal_message=refusal_message,
        decision=decision,
        scores=list(scores),
    )
