"""V4 output schema TypedDicts for EGA pipeline responses."""

from __future__ import annotations

from typing import Literal

try:
    from typing import TypedDict
except ImportError:  # pragma: no cover
    from typing_extensions import TypedDict  # type: ignore[assignment]


class FieldStatusEntry(TypedDict):
    field_path: str
    status: Literal["accepted", "rejected", "abstained"]
    score: float
    authority: str
    rejection_reason: str | None


class UnitAuditEntry(TypedDict):
    unit_id: str
    authority: str
    raw_score: float
    conformal_decision: str | None
    final_decision: str
    reason_code: str
    conformal_threshold: float | None
    conformal_band_width: float | None
    calibration_range: tuple[float, float] | None
    fallback_reason: str | None


class StrictAcceptedResponse(TypedDict):
    payload_status: Literal["ACCEPT"]
    verified_text: str
    verified_units: list[dict]
    tracking_id: str | None
    audit: list[UnitAuditEntry]


class StrictRejectedResponse(TypedDict):
    payload_status: Literal["REJECT"]
    rejection_reason: str
    failed_unit_ids: list[str]
    tracking_id: str | None
    route_status: str


class AdapterEnvelope(TypedDict):
    payload_status: Literal["ACCEPT", "REJECT"]
    accepted_fields: dict
    rejected_fields: dict
    field_status_map: dict[str, FieldStatusEntry]
    tracking_id: str | None


class PendingResponse(TypedDict):
    payload_status: Literal["PENDING"]
    tracking_id: str
    route_reason: str
    pending_expires_at: str | None
