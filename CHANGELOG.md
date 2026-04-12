# Changelog

## v4.0.0-alpha.1 — 2026-04-12

> **Alpha warning:** the public contract is still largely legacy-shaped. Additive v4 output fields are available now, but may evolve before a stable v4 release.

### Release scope (implemented now)

- v4 failure classification categories for rejected content:
  - `UNSUPPORTED_CLAIM`
  - `MISSING_IN_SOURCE`
  - `AMBIGUOUS_SOURCE`
- Payload/workflow decision fields on pipeline output:
  - `payload_status`, `payload_action`, `payload_failure_summary`
  - `workflow_status`, `handoff_required`, `handoff_reason`, `tracking_id`
- Downstream compatibility modes:
  - `STRICT_PASSTHROUGH` (default; aliases `STRICT` and `PASSTHROUGH`)
  - `ADAPTER` for mixed reject outcomes where accepted subset payload can be emitted
- Bounded repair rule:
  - only `UNSUPPORTED_CLAIM` enters repair retry flow
  - `MISSING_IN_SOURCE` and `AMBIGUOUS_SOURCE` are terminal reject paths
- Pending/handoff semantics:
  - `REPAIR` + bounded retry -> `workflow_status=PENDING`, `handoff_reason=BOUNDED_REPAIR`
  - non-terminal reject actions -> `workflow_status=PENDING` with deterministic `tracking_id`
- Structured mode wiring:
  - enabled via `unitizer_mode="structured_field"` + `structured_candidate_payload`

### Known limitations (current alpha)

- Structured mode only unitizes scalar leaves and scalar array entries.
- Non-scalar nested values are not emitted as direct verification units.
- Public response shape remains mostly legacy text-oriented.
- Schema version remains `1` (intentionally unchanged for this alpha cut).
