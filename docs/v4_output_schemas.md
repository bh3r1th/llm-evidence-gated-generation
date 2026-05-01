# EGA V4 Output Schema Reference

**Audience:** Downstream engineer integrating against EGA V4 for the first time.
**Schema version:** `"2"` (V4). All response objects carry `ega_schema_version: "2"`.

---

## Conventions

**Downstream mode** — Every EGA response is produced under exactly one of two modes:

- **strict** (`STRICT_PASSTHROUGH`) — Emits the full verified payload on accept; emits no business content on reject. This is the default.
- **adapter** (`ADAPTER`) — Emits a validation envelope that carries per-field status for partial downstream processing.

**`tracking_id`** — An opaque, deterministic string of the form `ega4_<16-hex-chars>`, derived from a SHA-256 hash of the payload status, unit decisions, and unit texts. Null when no handoff is required (i.e., for terminal-accept responses). Stable for the same inputs; may be used for deduplication.

**`payload_status`** — The top-level routing signal. Must be the first field checked by any consumer. Valid values across all schemas: `ACCEPT`, `REJECT`, `PENDING`. These are the only three valid values; no other value will ever appear in a V4 response.

> **Bounded repair note:** Bounded repair and correction workflows are not represented as `payload_status = REPAIR`. Instead they use `payload_status = PENDING` with `route_reason = "BOUNDED_REPAIR"`. This keeps `payload_status` as a clean three-value enum and represents repair state through routing metadata. Adapter mode never emits `PENDING`; `payload_status = PENDING` is only valid in `PendingResponse`.

**Required fields** — All fields listed without `(optional)` are required and must be present in every response of that schema type. A missing required field is a schema violation and the consumer must reject the envelope rather than silently default.

---

## Helper Type — `ConformalStateArtifact`

The `ConformalStateArtifact` TypedDict defines the structure of the `ConformalState` artifact as surfaced in audit records. All fields are required when `authority` is `"conformal"` or `"fallback"`. When `authority` is `"threshold"`, all conformal artifact fields in the containing schema are null and `ConformalStateArtifact` is not populated.

```python
from typing import TypedDict


class ConformalStateArtifact(TypedDict):
    threshold: float
    # The calibrated decision threshold, derived from the quantile of calibration scores
    # at epsilon = (1 − target_coverage). Scores strictly above threshold + band_width
    # are accepted; scores strictly below threshold − band_width are rejected.

    band_width: float
    # Half-width of the abstain band, symmetric around threshold.
    # Computed as abstain_k × score_std from the calibration row set.

    calibration_score_min: float
    # The minimum entailment score observed in the calibration row set.
    # Used as the lower OOR boundary. A live score strictly less than this value
    # is out-of-calibration-range and is auto-rejected (reason_code CONFORMAL_OOR_LOW).

    calibration_score_max: float
    # The maximum entailment score observed in the calibration row set.
    # Used as the upper OOR boundary. A live score strictly greater than this value
    # is out-of-calibration-range and is auto-accepted (reason_code CONFORMAL_OOR_HIGH).

    n_samples: int
    # Number of rows in the calibration row set. Must be >= 50; artifacts with
    # n_samples < 50 are treated as corrupt and trigger fallback authority.

    score_mean: float
    # Mean entailment score across the calibration row set. Informational only;
    # not used for OOR boundary computation in V4.

    score_std: float
    # Standard deviation of entailment scores across the calibration row set.
    # Used to compute band_width. Informational for audit consumers.
```

---

## Schema 1 — `StrictAcceptedResponse`

**Applies to:** strict mode only.

**What it means:** All units in the candidate passed verification. The full verified business payload is safe to emit downstream.

**Consumer action:** Extract `verified_text` or iterate `verified_units`. Treat `tracking_id` as null (no handoff needed). Log `unit_audit` for compliance. Do not re-verify.

```python
from typing import Literal, TypedDict


class VerifiedUnitEntry(TypedDict):
    unit_id: str
    # Stable identifier for this unit within the request. Matches unit_id in unit_audit.

    text: str
    # The verified, evidence-supported text of the unit after cleaning.


class UnitAuditEntry(TypedDict):
    unit_id: str
    # Stable identifier matching the corresponding VerifiedUnitEntry.

    raw_score: float
    # Entailment score as produced by the verifier, before clipping or transformation.
    # Range: [0.0, 1.0] for well-behaved verifiers; recorded as-is otherwise.

    authority: Literal["conformal", "threshold", "fallback"]
    # Which decision authority made the final unit decision.
    # conformal  — ConformalState was loaded and valid; conformal gate governs.
    # threshold  — No ConformalState was loaded; accept_threshold governs.
    # fallback   — ConformalState was present but detected corrupt; accept_threshold governs.

    conformal_decision: Literal["accept", "reject", "abstain"] | None
    # Decision produced by the conformal gate, if authority is "conformal". Null otherwise.
    # When score is out of calibration range, this reflects the OOR decision (accept/reject).

    final_decision: Literal["accept"]
    # Always "accept" in this schema. Included for audit log uniformity.

    reason_code: Literal[
        "CONFORMAL_ACCEPT",
        "CONFORMAL_OOR_HIGH",
        "THRESHOLD_ACCEPT",
        "FALLBACK_ACCEPT",
    ]
    # The specific rule that produced the accept decision.
    # CONFORMAL_ACCEPT   — Score above threshold + band_width, inside calibration range.
    # CONFORMAL_OOR_HIGH — Score above calibration range maximum; auto-accepted.
    # THRESHOLD_ACCEPT   — accept_threshold authority; score >= threshold.
    # FALLBACK_ACCEPT    — Fallback authority (corrupt ConformalState); score >= threshold.

    conformal_threshold: float | None
    # Value of ConformalState.threshold at decision time. Null if authority is "threshold".

    conformal_band_width: float | None
    # Value of ConformalState.band_width at decision time. Null if authority is "threshold".

    calibration_range: tuple[float, float] | None
    # [calibration_score_min, calibration_score_max] from the ConformalStateArtifact.
    # Defines the OOR boundary: scores outside this interval trigger auto-accept or auto-reject.
    # Null if authority is "threshold".

    fallback_reason: None
    # Always null in StrictAcceptedResponse. Included for audit log uniformity.


class StrictAcceptedResponse(TypedDict):
    ega_schema_version: Literal["2"]
    # Schema version. Always "2" for V4. Consumers must reject envelopes with other values.

    payload_status: Literal["ACCEPT"]
    # Top-level routing signal. Always "ACCEPT" in this schema.

    route_status: Literal["READY"]
    # Downstream routing state. Always "READY" when payload_status is "ACCEPT".

    workflow_status: Literal["COMPLETED"]
    # Lifecycle state. Always "COMPLETED" for accepted payloads.

    business_payload_emitted: Literal[True]
    # Always True. Signals that verified_text and verified_units carry real content.

    tracking_id: None
    # Always null for accepted payloads. No handoff or downstream tracking required.

    verified_text: str
    # Concatenation of all accepted unit texts, separated by newline.
    # Safe to emit directly to downstream consumers.
    # Empty string only if the candidate had zero units (should not occur in practice).

    verified_units: list[VerifiedUnitEntry]
    # Ordered list of accepted units. Order matches the original candidate unit order.
    # Non-empty when payload_status is "ACCEPT".

    unit_audit: list[UnitAuditEntry]
    # Per-unit audit trail. One entry per unit in verified_units.
    # Order matches verified_units. Required for compliance logging.

    handoff_required: Literal[False]
    # Always False. No handoff is needed for accepted payloads.

    handoff_reason: None
    # Always null for accepted payloads.
```

---

## Schema 2 — `StrictRejectedResponse`

**Applies to:** strict mode only.

**What it means:** One or more units failed verification and the rejection policy prevents partial emission. No business content is present. The response is informational only.

**Consumer action:** Do not emit any business content from this response. Check `rejection_reason` to determine the failure category. Surface `failed_unit_ids` to operators if diagnostic logging is enabled. Use `tracking_id` for correlation if handoff is required.

```python
from typing import Literal, TypedDict


class FailedUnitEntry(TypedDict):
    unit_id: str
    # Stable identifier for the failed unit.

    failure_class: Literal["UNSUPPORTED_CLAIM", "MISSING_IN_SOURCE", "AMBIGUOUS_SOURCE"]
    # Per-unit diagnosis explaining why this unit was rejected.
    # UNSUPPORTED_CLAIM — The verifier scored the unit but the entailment score is
    #   low and contradiction score is high. The evidence actively contradicts the claim.
    # MISSING_IN_SOURCE — No evidence item matched the claim. The evidence pool contained
    #   no selected evidence ID for this unit.
    # AMBIGUOUS_SOURCE  — The verifier produced a score but neither UNSUPPORTED_CLAIM nor
    #   MISSING_IN_SOURCE applies. The claim cannot be confirmed or denied from the evidence.


class UnitAuditEntry(TypedDict):
    unit_id: str
    raw_score: float
    authority: Literal["conformal", "threshold", "fallback"]
    conformal_decision: Literal["accept", "reject", "abstain"] | None
    final_decision: Literal["reject", "abstain"]
    reason_code: Literal[
        "CONFORMAL_REJECT",
        "CONFORMAL_ABSTAIN",
        "CONFORMAL_OOR_LOW",
        "THRESHOLD_REJECT",
        "FALLBACK_REJECT",
    ]
    # The specific rule that produced the reject or abstain decision.
    # CONFORMAL_REJECT  — Score below threshold − band_width, inside calibration range.
    # CONFORMAL_ABSTAIN — Score within the abstain band (threshold ± band_width).
    # CONFORMAL_OOR_LOW — Score below calibration range minimum; auto-rejected.
    # THRESHOLD_REJECT  — accept_threshold authority; score < threshold.
    # FALLBACK_REJECT   — Fallback authority (corrupt ConformalState); score < threshold.

    conformal_threshold: float | None
    conformal_band_width: float | None
    calibration_range: tuple[float, float] | None
    fallback_reason: str | None
    # Human-readable explanation of why conformal authority fell back.
    # Non-null only when authority is "fallback".
    # Examples: "n_samples below minimum", "non-finite threshold", "missing field: score_std".


class StrictRejectedResponse(TypedDict):
    ega_schema_version: Literal["2"]
    # Schema version. Always "2" for V4.

    payload_status: Literal["REJECT"]
    # Top-level routing signal. Always "REJECT" in this schema.

    route_status: Literal["REJECTED"]
    # Downstream routing state. Always "REJECTED" when payload_status is "REJECT".

    workflow_status: Literal["COMPLETED", "PENDING"]
    # COMPLETED — Terminal rejection; no further processing expected.
    # PENDING   — Rejection requires human review or downstream action (handoff_required=True).

    business_payload_emitted: Literal[False]
    # Always False. No verified content is present in this response.

    tracking_id: str | None
    # Opaque handoff identifier. Non-null when handoff_required is True.
    # Consumers must not interpret the internal structure of this value.

    rejection_reason: Literal[
        "ALL_UNITS_FAILED",
        "PARTIAL_NOT_ALLOWED",
        "CONFORMAL_FALLBACK_CORRUPT_STATE",
    ]
    # Payload-level rejection category. One of:
    #
    # ALL_UNITS_FAILED
    #   Every unit in the candidate was rejected or abstained. No accepted units remain.
    #   Derived from gate reason_code ALL_DROPPED. The evidence does not support any
    #   part of the LLM output.
    #
    # PARTIAL_NOT_ALLOWED
    #   At least one unit passed but the active PolicyConfig disallows partial emission
    #   (partial_allowed=False). The response contains no business content even though
    #   some units were individually supported.
    #   Derived from gate reason_code PARTIAL_NOT_ALLOWED.
    #
    # CONFORMAL_FALLBACK_CORRUPT_STATE
    #   A ConformalState artifact was present but was detected as corrupt before evaluation.
    #   The entire request fell back to accept_threshold authority and all units were
    #   subsequently rejected. This reason code is only emitted when the corruption itself
    #   is the operationally notable event (i.e., conformal would otherwise have been used).
    #   The fallback_reason field on each UnitAuditEntry carries the per-state diagnosis.

    failed_unit_ids: list[str]
    # IDs of all units that were rejected or abstained. Ordered as they appeared
    # in the original candidate. Non-empty when payload_status is "REJECT".

    failed_units: list[FailedUnitEntry]
    # Per-unit rejection details. Parallel to failed_unit_ids.

    unit_audit: list[UnitAuditEntry]
    # Per-unit audit trail for all units, including any that passed before the
    # payload-level rejection was determined. One entry per unit in the original candidate.

    handoff_required: bool
    # True when the rejection requires downstream action (human review, repair).
    # False for terminal rejections.

    handoff_reason: str | None
    # Machine-readable reason for the handoff. Non-null when handoff_required is True.
    # Values: "REVIEW", "BOUNDED_REPAIR".
    # Null when handoff_required is False.
```

---

## Schema 3 — `AdapterEnvelope`

**Applies to:** adapter mode only.

**What it means:** Partial emission with a field-level validation envelope. The adapter mode allows a downstream consumer to receive accepted fields even when some fields were rejected, rather than treating the entire payload as all-or-nothing. The `field_status_map` is the authoritative record of what was verified and what was not.

**Consumer action:** Check `payload_status` first. Iterate `field_status_map` to determine which fields are safe to use. Never use a field whose status is `rejected` or `abstained` as business content. The `accepted_fields` list is a convenience view; `field_status_map` is authoritative in case of disagreement.

```python
from typing import Literal, TypedDict


class AdapterAcceptedUnit(TypedDict):
    unit_id: str
    # Stable identifier for this unit.

    text: str
    # Verified, evidence-supported text of the unit.


class AdapterRejectedUnit(TypedDict):
    unit_id: str
    # Stable identifier for this unit.

    text: str
    # Original (unverified) text of the unit. Must not be treated as verified content.

    decision: Literal["reject", "abstain"]
    # The gate decision for this unit.

    failure_class: Literal["UNSUPPORTED_CLAIM", "MISSING_IN_SOURCE", "AMBIGUOUS_SOURCE"] | None
    # Per-unit failure diagnosis. Null only if the unit was abstained and no diagnosis
    # could be assigned (rare). See StrictRejectedResponse.FailedUnitEntry for definitions.


class AdapterEnvelope(TypedDict):
    ega_schema_version: Literal["2"]
    # Schema version. Always "2" for V4.

    payload_status: Literal["ACCEPT", "REJECT"]
    # ACCEPT — All units passed; accepted_fields contains all input units.
    # REJECT — One or more units failed; some or all fields are rejected.
    # Note: PENDING is not a valid payload_status in AdapterEnvelope.
    # Pending responses are always delivered as PendingResponse regardless of mode.

    route_status: Literal["READY", "REJECTED"]
    # READY    — payload_status is ACCEPT.
    # REJECTED — payload_status is REJECT.

    business_payload_emitted: bool
    # True when at least one accepted unit is present and safe to emit.
    # True even when payload_status is REJECT, provided at least one unit was accepted.
    # False only when no units were accepted at all.

    tracking_id: str | None
    # Opaque handoff identifier. See conventions above.

    accepted_fields: list[AdapterAcceptedUnit]
    # Convenience list of all units that passed verification.
    # Empty list when no units were accepted.
    # Consumers must cross-check against field_status_map before using.

    rejected_fields: list[AdapterRejectedUnit]
    # Convenience list of all units that failed or abstained.
    # Empty list when all units were accepted.

    field_status_map: dict[str, "FieldStatusEntry"]
    # Authoritative per-field status record. Key is unit_id.
    # Every unit in the original candidate must appear as a key.
    # No unit may be absent. A missing key is a schema violation.
    # Consumers must not rely solely on accepted_fields or rejected_fields;
    # field_status_map is the ground truth.

    adapter_summary: "AdapterSummary"
    # Aggregate counts for quick inspection without iterating field_status_map.

    unit_audit: list["UnitAuditEntry"]
    # Per-unit audit trail. One entry per unit in the original candidate.
    # Order matches the original candidate unit order.


class AdapterSummary(TypedDict):
    total_units: int
    # Total number of units in the original candidate.

    accepted_units: int
    # Number of units with status "accepted".

    rejected_units: int
    # Number of units with status "rejected" or "abstained".

    supported_count: int
    # Alias for accepted_units. Retained for backward compatibility with V3 adapter output.

    unsupported_claim_count: int
    # Number of rejected units with failure_class UNSUPPORTED_CLAIM.

    missing_in_source_count: int
    # Number of rejected units with failure_class MISSING_IN_SOURCE.

    ambiguous_source_count: int
    # Number of rejected units with failure_class AMBIGUOUS_SOURCE.
```

---

## Schema 4 — `FieldStatusEntry`

**Applies to:** adapter mode only (as values in `AdapterEnvelope.field_status_map`).

**What it means:** The complete verification record for a single input field (unit). This is the atomic unit of the adapter contract. Every field in the original candidate has exactly one `FieldStatusEntry`.

**Consumer action:** Read `status` first. Use the field's text as business content only when `status` is `"accepted"`. When `status` is `"rejected"` or `"abstained"`, treat the corresponding text as unverified and do not emit it. Log `authority` and `score` for monitoring.

```python
from typing import Literal, TypedDict


class FieldStatusEntry(TypedDict):
    field_path: str
    # Dot-notation path identifying this field within the original structured input.
    # For unstructured (sentence-unitized) inputs, this is the unit_id.
    # For structured inputs (field-mode), this is the logical field path (e.g. "summary.findings[0]").
    # Consumers must treat this as an opaque string unless they control the unitizer mode.

    status: Literal["accepted", "rejected", "abstained"]
    # Verification outcome for this field.
    # accepted  — The field's content is evidentially supported. Safe to use downstream.
    # rejected  — The field's content failed verification. Must not be used as business content.
    # abstained — The conformal gate declined to make a confident decision (score within
    #             the abstain band). The field is treated as rejected for emission purposes.
    #             Abstained fields are never emitted as business content.

    score: float
    # Raw entailment score from the verifier, in the range [0.0, 1.0] for well-behaved
    # verifiers. Recorded as-is (not clipped) to preserve the true verifier output.

    authority: Literal["conformal", "threshold", "fallback"]
    # Which authority made the decision for this field.
    # conformal — ConformalState was loaded and valid.
    # threshold — No ConformalState was loaded; accept_threshold governs.
    # fallback  — ConformalState was present but corrupt; accept_threshold governs as fallback.

    reason_code: Literal[
        "CONFORMAL_ACCEPT",
        "CONFORMAL_REJECT",
        "CONFORMAL_ABSTAIN",
        "CONFORMAL_OOR_HIGH",
        "CONFORMAL_OOR_LOW",
        "THRESHOLD_ACCEPT",
        "THRESHOLD_REJECT",
        "FALLBACK_ACCEPT",
        "FALLBACK_REJECT",
    ]
    # Specific rule that produced this status. See the conformal authority spec for full
    # definitions. Included on all entries regardless of status.

    rejection_reason: Literal[
        "UNSUPPORTED_CLAIM",
        "MISSING_IN_SOURCE",
        "AMBIGUOUS_SOURCE",
    ] | None
    # Per-field failure diagnosis. Required when status is "rejected" or "abstained".
    # Null when status is "accepted".
    #
    # UNSUPPORTED_CLAIM — Evidence actively contradicts the claim. Low entailment + high
    #   contradiction score pattern from the verifier.
    # MISSING_IN_SOURCE — No evidence item matched the claim. The evidence pool contained
    #   no selected evidence ID for this field.
    # AMBIGUOUS_SOURCE  — The verifier scored the field but neither UNSUPPORTED_CLAIM nor
    #   MISSING_IN_SOURCE applies. The evidence does not clearly confirm or deny the claim.

    conformal_decision: Literal["accept", "reject", "abstain"] | None
    # Conformal authority decision actually applied to this field unit,
    # including any out-of-calibration-range override. Matches the
    # definition in Doc 1 Section 4. Never reflects a pre-override state.
    # Null when authority is "threshold".

    conformal_threshold: float | None
    # ConformalState.threshold value used at decision time. Null if authority is "threshold".

    conformal_band_width: float | None
    # ConformalState.band_width value used at decision time. Null if authority is "threshold".

    calibration_range: tuple[float, float] | None
    # [calibration_score_min, calibration_score_max] from the ConformalStateArtifact.
    # Defines the OOR boundary: scores outside this interval trigger auto-accept or auto-reject.
    # Null if authority is "threshold".

    fallback_reason: str | None
    # Human-readable description of why conformal fell back to threshold.
    # Non-null only when authority is "fallback".
```

---

## Schema 5 — `PendingResponse`

**Applies to:** both modes (strict and adapter).

**Status in V4:** Contract definition only. No polling or callback mechanism is implemented in V4. The `pending_expires_at` field is informational; the system does not enforce expiry or trigger any action when the timestamp passes.

**What it means:** The verification request was accepted but cannot be resolved synchronously. The consumer receives a stable `tracking_id` and an informational expiry hint. The consumer is responsible for any polling or retry logic until a final response is available.

**Consumer action:** Store `tracking_id` immediately. Do not emit any business content. Do not rely on `pending_expires_at` for enforcement; treat it as an operator hint only. When re-querying, present `tracking_id` as the correlation key. A subsequent response for the same `tracking_id` will arrive as one of `StrictAcceptedResponse`, `StrictRejectedResponse`, or `AdapterEnvelope`.

```python
from typing import Literal, TypedDict


class PendingResponse(TypedDict):
    ega_schema_version: Literal["2"]
    # Schema version. Always "2" for V4.

    payload_status: Literal["PENDING"]
    # Top-level routing signal. Always "PENDING" in this schema.
    # Consumers must not treat this response as a final accept or reject.

    tracking_id: str
    # Opaque, stable identifier for this pending request.
    # Format: ega4_<16-hex-chars>. Never null in PendingResponse.
    # Must be stored by the consumer and presented when re-querying.

    route_reason: str
    # Machine-readable code explaining why the response is pending.
    # Valid values in V4:
    #   "BOUNDED_REPAIR"  — One or more units were rejected with failure class
    #     UNSUPPORTED_CLAIM, and the correction loop has not yet exhausted its retry
    #     budget. A corrected candidate is being generated.
    #   "REVIEW_REQUIRED" — Operator or human review is needed before a final
    #     decision can be made. Used for requests flagged by policy as requiring
    #     manual inspection.
    # Additional values may be introduced in future schema versions.
    # Consumers must handle unknown route_reason values without failing.

    pending_expires_at: str | None
    # ISO 8601 UTC timestamp indicating when the pending state is expected to resolve,
    # e.g., "2026-04-29T14:00:00Z".
    # Null when no expiry estimate is available.
    #
    # V4 INFORMATIONAL ONLY: This timestamp is a hint from the producing system.
    # EGA V4 does not enforce expiry. No callback, notification, or automatic
    # transition occurs when this timestamp passes. Consumers must not build
    # automated retry logic that depends on this field being accurate or enforced.
    # Treat it as operator documentation, not as a system contract.

    workflow_status: Literal["PENDING"]
    # Lifecycle state. Always "PENDING" in this schema.

    business_payload_emitted: Literal[False]
    # Always False. No verified content is present in a pending response.

    handoff_required: Literal[True]
    # Always True. A pending response always requires downstream action.

    handoff_reason: str
    # Matches route_reason. Included for parity with StrictRejectedResponse structure.
    # Consumers may use either field; they carry the same value.
```

---

## Cross-Schema Reference: `UnitAuditEntry` (both modes)

`UnitAuditEntry` appears in both `StrictAcceptedResponse` and `StrictRejectedResponse` (strict mode) and in `AdapterEnvelope` (adapter mode). The field set is identical across all three schemas. The following table summarises which `reason_code` values are possible by `final_decision`:

| `final_decision` | Valid `reason_code` values |
|---|---|
| `accept` | `CONFORMAL_ACCEPT`, `CONFORMAL_OOR_HIGH`, `THRESHOLD_ACCEPT`, `FALLBACK_ACCEPT` |
| `reject` | `CONFORMAL_REJECT`, `CONFORMAL_OOR_LOW`, `THRESHOLD_REJECT`, `FALLBACK_REJECT` |
| `abstain` | `CONFORMAL_ABSTAIN` |

A `reason_code` of `CONFORMAL_ABSTAIN` can only appear with `final_decision = "abstain"`. Abstained units are always treated as rejected for emission purposes in both strict and adapter mode; `final_decision` records the gate decision, not the emission decision.

---

## Trace Metadata vs Response Schema

Some pipeline internals are recorded in the trace object (accessible via the `trace` key when `return_pipeline_output=True`) but are explicitly excluded from the public response schemas defined in this document.

**`field_query_fallback`** is a boolean trace field emitted per unit when the field-aware BM25 query produced no candidates and fell back to a value-only query. It is trace metadata only. It does not appear in `StrictAcceptedResponse`, `StrictRejectedResponse`, `AdapterEnvelope`, or `PendingResponse`. Consumers of the public response schema must not depend on it being present, and schema validators must not reject a response that omits it.

Trace fields in general are internal diagnostic data. No trace field is part of the public contract. Adding or removing trace fields is not a breaking change.

---

## Schema Discriminator Pattern

Every V4 response carries `payload_status` as the first field. Consumers must switch on this value before accessing any other field:

```python
response = ega_client.verify(...)

match response["payload_status"]:
    case "ACCEPT":
        # StrictAcceptedResponse or AdapterEnvelope with payload_status=ACCEPT
        emit(response["verified_text"])          # strict
        # or: emit fields where field_status_map[uid]["status"] == "accepted"  # adapter
    case "REJECT":
        # StrictRejectedResponse or AdapterEnvelope with payload_status=REJECT
        log_rejection(response["rejection_reason"])   # strict
        # or: inspect response["field_status_map"]     # adapter
    case "PENDING":
        # PendingResponse — store tracking_id, do not emit
        store(response["tracking_id"])
    case _:
        raise ValueError(f"Unknown payload_status: {response['payload_status']!r}")
```

No field other than `payload_status` is guaranteed to be present before the schema is identified. Consumers must not access `verified_text`, `rejection_reason`, or `tracking_id` before checking `payload_status`.
