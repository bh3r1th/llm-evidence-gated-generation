"""Structured event emission for enforcement decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from ega.policy import PolicyConfig
from ega.types import EnforcementResult


@dataclass(frozen=True, slots=True)
class DecisionEvent:
    """Structured event representing one enforcement decision."""

    run_id: str
    timestamp: str
    model_name: str
    policy_config: dict[str, Any]
    unit_count: int
    kept_count: int
    refusal: bool
    summary_stats: dict[str, Any]


def event_from_result(result: EnforcementResult, context: dict[str, Any]) -> DecisionEvent:
    """Build a structured decision event from an enforcement result and context."""

    policy_config = context.get("policy_config", PolicyConfig())
    if isinstance(policy_config, PolicyConfig):
        serialized_policy_config = asdict(policy_config)
    elif isinstance(policy_config, dict):
        serialized_policy_config = dict(policy_config)
    else:
        serialized_policy_config = {}

    timestamp = context.get("timestamp")
    if timestamp is None:
        timestamp = datetime.now(tz=timezone.utc).isoformat()

    return DecisionEvent(
        run_id=str(context.get("run_id", "")),
        timestamp=str(timestamp),
        model_name=str(context.get("model_name", "")),
        policy_config=serialized_policy_config,
        unit_count=len(result.kept_units) + len(result.dropped_units),
        kept_count=len(result.kept_units),
        refusal=result.decision.refusal,
        summary_stats=dict(result.decision.summary_stats),
    )
