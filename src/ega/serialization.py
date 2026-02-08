"""Serialization helpers for deterministic EGA payloads."""

from __future__ import annotations

import json
from typing import Any, TypeAlias

from ega.contract import EGA_SCHEMA_VERSION
from ega.types import EnforcementResult, GateDecision, Unit, VerificationScore

SerializablePayload: TypeAlias = GateDecision | EnforcementResult
SerializableType: TypeAlias = type[GateDecision] | type[EnforcementResult]


def _unit_to_dict(unit: Unit) -> dict[str, Any]:
    return {
        "id": unit.id,
        "text": unit.text,
        "metadata": dict(unit.metadata),
        "source_ids": list(unit.source_ids) if unit.source_ids is not None else None,
    }


def _unit_from_dict(data: dict[str, Any]) -> Unit:
    source_ids = data.get("source_ids")
    return Unit(
        id=str(data["id"]),
        text=str(data["text"]),
        metadata=dict(data.get("metadata", {})),
        source_ids=list(source_ids) if source_ids is not None else None,
    )


def _score_to_dict(score: VerificationScore) -> dict[str, Any]:
    return {
        "unit_id": score.unit_id,
        "entailment": score.entailment,
        "contradiction": score.contradiction,
        "neutral": score.neutral,
        "label": score.label,
        "raw": dict(score.raw),
    }


def _score_from_dict(data: dict[str, Any]) -> VerificationScore:
    return VerificationScore(
        unit_id=str(data["unit_id"]),
        entailment=float(data["entailment"]),
        contradiction=float(data["contradiction"]),
        neutral=float(data["neutral"]),
        label=str(data["label"]),
        raw=dict(data.get("raw", {})),
    )


def _decision_to_dict(decision: GateDecision) -> dict[str, Any]:
    return {
        "allowed_units": list(decision.allowed_units),
        "dropped_units": list(decision.dropped_units),
        "refusal": decision.refusal,
        "reason_code": decision.reason_code,
        "summary_stats": dict(decision.summary_stats),
    }


def _decision_from_dict(data: dict[str, Any]) -> GateDecision:
    return GateDecision(
        allowed_units=[str(item) for item in data.get("allowed_units", [])],
        dropped_units=[str(item) for item in data.get("dropped_units", [])],
        refusal=bool(data["refusal"]),
        reason_code=str(data["reason_code"]),
        summary_stats=dict(data.get("summary_stats", {})),
    )


def _enforcement_to_dict(result: EnforcementResult) -> dict[str, Any]:
    return {
        "final_text": result.final_text,
        "kept_units": list(result.kept_units),
        "dropped_units": list(result.dropped_units),
        "refusal_message": result.refusal_message,
        "decision": _decision_to_dict(result.decision),
        "scores": [_score_to_dict(score) for score in result.scores],
        "ega_schema_version": result.ega_schema_version,
    }


def _enforcement_from_dict(data: dict[str, Any]) -> EnforcementResult:
    return EnforcementResult(
        final_text=data.get("final_text"),
        kept_units=[str(item) for item in data.get("kept_units", [])],
        dropped_units=[str(item) for item in data.get("dropped_units", [])],
        refusal_message=data.get("refusal_message"),
        decision=_decision_from_dict(dict(data["decision"])),
        scores=[_score_from_dict(item) for item in data.get("scores", [])],
        ega_schema_version=str(data.get("ega_schema_version", EGA_SCHEMA_VERSION)),
    )


def to_json(obj: SerializablePayload) -> str:
    """Serialize a GateDecision or EnforcementResult to a stable JSON string."""

    payload: dict[str, Any]
    if isinstance(obj, GateDecision):
        payload = {"ega_schema_version": EGA_SCHEMA_VERSION}
        payload["kind"] = "gate_decision"
        payload["data"] = _decision_to_dict(obj)
    elif isinstance(obj, EnforcementResult):
        payload = {"ega_schema_version": obj.ega_schema_version}
        payload["kind"] = "enforcement_result"
        payload["data"] = _enforcement_to_dict(obj)
    else:
        raise TypeError(f"Unsupported object type for serialization: {type(obj)!r}")

    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def from_json(payload: str, target_type: SerializableType) -> SerializablePayload:
    """Deserialize a JSON payload into GateDecision or EnforcementResult."""

    decoded: dict[str, Any] = json.loads(payload)
    version = decoded.get("ega_schema_version", decoded.get("schema_version"))
    version = str(version) if version is not None else ""
    if version != EGA_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema version: {version!r}. Expected {EGA_SCHEMA_VERSION!r}."
        )

    kind = decoded.get("kind")
    data = dict(decoded["data"])

    if target_type is GateDecision:
        if kind != "gate_decision":
            raise ValueError(f"Payload kind mismatch: expected 'gate_decision', got {kind!r}")
        return _decision_from_dict(data)

    if target_type is EnforcementResult:
        if kind != "enforcement_result":
            raise ValueError(
                f"Payload kind mismatch: expected 'enforcement_result', got {kind!r}"
            )
        return _enforcement_from_dict(data)

    raise TypeError(f"Unsupported target type for deserialization: {target_type!r}")
