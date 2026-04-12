# EGA v4 Alpha Validation (Post-release Focused Pass)

Date: 2026-04-12

This pass validates five realistic end-to-end scenarios without introducing new product features.

| Scenario | Expected behavior | Actual behavior | Pass/Fail | Notes |
|---|---|---|---|---|
| 1. Legacy text mode (normal accepted flow) | Text-mode accepted flow remains unchanged: payload accepted, business payload emitted, workflow completed. | `payload_status=ACCEPT`, `route_status=READY`, `workflow_status=COMPLETED`, `business_payload_emitted=true`, and no adapter-only payload fields in strict mode. | PASS | Regression guard confirms legacy text-mode path is stable. |
| 2. Structured strict passthrough (supported + failing required field) | Mixed structured content in strict mode must not emit business payload as success. | Structured strict run returns `payload_status=REJECT`, `route_status=REJECTED`, `passthrough_mode=STRICT`, and `business_payload_emitted=false`. | PASS | Confirms no partial-success semantics leak into strict passthrough. |
| 3. Structured adapter mode (supported + rejected fields) | Adapter payload may contain only supported content; rejected content must remain metadata-only. | Adapter run returns reject overall, but emits adapter-safe payload containing only supported unit(s); rejected unit appears only in `adapter_rejected_units` metadata with failure class. | PASS | Confirms field-level safe projection behavior in adapter compatibility mode. |
| 4. Repair path trigger policy | `UNSUPPORTED_CLAIM` should route to repair; `MISSING_IN_SOURCE` and `AMBIGUOUS_SOURCE` must not trigger repair. | Unsupported-only case returns `payload_status=REPAIR` and `route_status=REPAIR_PENDING`; missing+ambiguous case returns `payload_status=REJECT` and `route_status=REJECTED`. | PASS | Confirms repair trigger stays narrowly scoped to unsupported-claim failures. |
| 5. Pending/handoff route semantics | Repair/review style routes must be pending and require handoff, not synchronous completion. | Review-style contract derivation yields `workflow_status=PENDING`, `handoff_required=true`, and non-empty tracking id. | PASS | Confirms no fake synchronous completion for pending handoff routes. |

## Test run

- `pytest -q tests/test_alpha_validation_v4.py` → 5 passed.

## Findings summary

- Scenarios covered: 5/5.
- Failures found: none.
- Source fixes applied: none required.
- Remaining gaps: this pass is intentionally focused on the five alpha scenarios and does not expand coverage to unrelated modules or broader perf/load behaviors.
