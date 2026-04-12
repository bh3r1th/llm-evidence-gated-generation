# EGA v4 Alpha Validation (Post-release Focused Pass)

Date: 2026-04-12

This pass validates five realistic end-to-end scenarios without introducing new product features.
During hardening, each scenario was re-run after targeted source/test fixes.

| Scenario | Expected behavior | Final behavior (post-fix) | Pass/Fail | Notes |
|---|---|---|---|---|
| 1. Legacy text mode (normal accepted flow) | Text-mode accepted flow remains unchanged: payload accepted, business payload emitted, workflow completed. | `payload_status=ACCEPT`, `route_status=READY`, `workflow_status=COMPLETED`, `business_payload_emitted=true`, and no adapter-only payload fields in strict mode. | PASS | Regression guard confirms legacy text-mode path is stable. |
| 2. Structured strict passthrough (supported + failing required field) | Mixed structured content in strict mode must not emit business payload as success. | Structured strict run returns `payload_status=REJECT`, `route_status=REJECTED`, `passthrough_mode=STRICT`, `business_payload_emitted=false`, and no `adapter_payload` field. | PASS | Hardened strict-mode payload emission and metadata boundaries. |
| 3. Structured adapter mode (supported + rejected fields) | Adapter payload may contain only supported content; rejected content must remain metadata-only. | Adapter run returns reject overall, emits adapter-safe payload containing only supported unit(s), and rejected unit(s) remain only in `adapter_rejected_units` with reject decision + failure class. | PASS | Hardened adapter projection so rejected content does not leak into accepted payload. |
| 4. Repair path trigger policy | `UNSUPPORTED_CLAIM` should route to repair; `MISSING_IN_SOURCE` and `AMBIGUOUS_SOURCE` must not trigger repair. | Unsupported-only case returns `payload_status=REPAIR` and `route_status=REPAIR_PENDING`; missing+ambiguous case returns `payload_status=REJECT` and `route_status=REJECTED`; correction retry selection is scoped to `UNSUPPORTED_CLAIM` only. | PASS | Hardened retry filter to avoid non-unsupported repair attempts when class metadata is absent. |
| 5. Pending/handoff route semantics | Repair/review style routes must be pending and require handoff, not synchronous completion. | Review-style contract derivation yields `workflow_status=PENDING`, `handoff_required=true`, and non-empty deterministic tracking id; structured unit paths remain stable/distinct for mixed-type keys (no id collisions). | PASS | Hardened structured path/id stability to preserve deterministic pending-handoff tracking context. |

## Test run

- `pytest -q tests/test_alpha_validation_v4.py tests/test_pipeline_v4_structured_mode.py` → 15 passed.

## Findings summary

- Scenarios covered: 5/5.
- Failures found in hardening pass: strict-mode payload safety assertions, adapter rejected-content isolation assertions, correction retry filtering without failure-class metadata, and structured path/id stability under mixed-type keys.
- Source fixes applied: `src/ega/core/correction.py` and `src/ega/unitization.py`.
- Remaining gaps: this pass remains intentionally focused on the five alpha scenarios and does not expand coverage to unrelated modules or broader perf/load behaviors.
