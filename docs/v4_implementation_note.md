# EGA v4 implementation note (stabilization snapshot)

This note documents the current v4 behavior as implemented today. It does **not** introduce new product scope.

## What v4 adds

- Internal failure classification for rejected units (`UNSUPPORTED_CLAIM`, `MISSING_IN_SOURCE`, `AMBIGUOUS_SOURCE`).
- Payload-level aggregation (`payload_status`, `payload_action`, and `payload_failure_summary`).
- Structured field unitization/runtime wiring (`unitizer_mode="structured_field"` with `structured_candidate_payload`).
- Workflow contract fields for async/pending handoff (`workflow_status`, `handoff_required`, `handoff_reason`, `tracking_id`).
- Adapter compatibility mode for partial payload acceptance (`downstream_compatibility_mode="ADAPTER"`).

## Strict passthrough vs adapter behavior

- **STRICT_PASSTHROUGH** (default):
  - `ACCEPT` emits business payload.
  - `REJECT` emits no business payload.
  - `REPAIR` remains pending with no completed payload.
- **ADAPTER**:
  - Keeps the same payload status decisions, but may emit accepted subset payload on mixed reject outcomes.
  - Never includes rejected content in emitted adapter payload.
  - `REPAIR` still remains pending and does not emit completed business payload.

Backward-compatible aliases `STRICT` and `PASSTHROUGH` normalize to `STRICT_PASSTHROUGH`.

## Repair policy rule

- Repair retries are gated to units classified as `UNSUPPORTED_CLAIM`.
- `MISSING_IN_SOURCE` and `AMBIGUOUS_SOURCE` are terminal reject signals (no repair retry path).

## Pending/handoff semantics

- `payload_status="ACCEPT"` → workflow is `COMPLETED`, no handoff.
- `payload_status="REPAIR"` with bounded retry flow → workflow is `PENDING`, handoff reason `BOUNDED_REPAIR`.
- `payload_status="REJECT"` with terminal reject action → workflow is `COMPLETED` (no handoff).
- Non-terminal review-style reject actions are represented as `PENDING` with a deterministic `tracking_id`.

## Structured-mode limitations (current)

- Structured mode currently unitizes only scalar leaves and scalar array entries.
- Non-scalar leaves (objects/arrays as values) are not directly represented as verification units.
- Verification output/public response shape remains mostly legacy text-oriented.

## Intentionally out of scope (for this stabilization pass)

- Schema version bump without a deliberate public contract change.
- Broad verifier interface redesign.
- Enterprise workflow integrations beyond current handoff contract fields.
- Large logging/refactor churn unrelated to v4 stabilization.
