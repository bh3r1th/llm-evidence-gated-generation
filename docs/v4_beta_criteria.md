# EGA v4 Beta Readiness Criteria

Date: 2026-04-12

This document defines concrete, testable beta-entry gates for EGA v4 based on the behavior that is already implemented. It does not add scope, features, or contract redesign.

## Scope

In-scope for this beta gate:
- Structured field verification path (`unitizer_mode="structured_field"`) and related unitization behavior.
- Failure classification outcomes (`UNSUPPORTED_CLAIM`, `MISSING_IN_SOURCE`, `AMBIGUOUS_SOURCE`).
- Bounded repair behavior limited to `UNSUPPORTED_CLAIM`.
- Payload aggregation/status/action semantics.
- Strict passthrough safety behavior.
- Async pending/handoff workflow signaling.
- Adapter-mode partial acceptance boundaries.
- Legacy text-mode stability.
- Documentation alignment with implemented behavior.

Out-of-scope for this beta gate:
- New runtime capabilities, schema/version changes, or public contract redesign.
- Broad architectural refactors not required for v4 behavior validation.

## Beta entry criteria

All criteria below must pass before beta entry.

1. **Structured mode stability**
   - Given representative structured payloads (scalars, scalar arrays, mixed accepted/rejected units), pipeline outputs are stable across repeated runs under identical inputs/configuration.
   - Pass condition: no nondeterministic status/action transitions for the same test vector.

2. **Deterministic path-based unit IDs**
   - Structured unit IDs derived from field paths remain deterministic and collision-free for mixed-type keys/scenarios covered by current implementation.
   - Pass condition: repeated unitization of identical input produces identical IDs; test fixtures detect no collisions.

3. **No crashes on malformed/empty structured input**
   - Malformed, empty, or minimally populated structured candidate payload inputs must not crash the runtime.
   - Pass condition: pipeline returns a bounded, valid route/result (reject/pending/complete semantics as applicable) without uncaught exceptions.

4. **Repair only for `UNSUPPORTED_CLAIM`**
   - Retry/repair targeting is strictly limited to units classified as `UNSUPPORTED_CLAIM`.
   - Pass condition: `MISSING_IN_SOURCE` and `AMBIGUOUS_SOURCE` units never enter repair retry selection.

5. **Strict passthrough never emits invalid business payload**
   - In strict passthrough mode, non-accept outcomes cannot emit completed business payload.
   - Pass condition:
     - `ACCEPT` may emit business payload.
     - `REJECT` emits no business payload.
     - `REPAIR`/pending routes do not emit completed business payload.

6. **Adapter mode never leaks rejected content into accepted payload**
   - Adapter compatibility projections may emit accepted subsets only.
   - Pass condition: any rejected unit content appears only in rejection metadata (for example rejected-unit collections), never in the emitted accepted adapter payload.

7. **Pending/handoff semantics never fake completion**
   - Pending workflows (repair/review-style routes) must remain pending and require handoff signaling.
   - Pass condition: pending routes set `workflow_status=PENDING` with `handoff_required=true`; completion flags/payloads are not emitted as final success.

8. **Legacy text mode remains stable**
   - Existing text-mode flows continue to behave consistently with pre-v4 expectations.
   - Pass condition: legacy text-mode regression scenarios keep expected acceptance/completion semantics with no v4-only field leakage into strict legacy responses.

9. **Docs match actual implemented behavior**
   - v4 docs and validation docs reflect actual current runtime behavior, limitations, and mode semantics.
   - Pass condition: no known contradiction between documented claims and passing implementation tests.

## Required validation areas

The beta decision must include evidence across all areas below:

1. **Mode behavior matrix**
   - Structured + strict passthrough.
   - Structured + adapter compatibility mode.
   - Legacy text mode baseline.

2. **Classification and repair policy**
   - Positive check: `UNSUPPORTED_CLAIM` can route to bounded repair path.
   - Negative checks: `MISSING_IN_SOURCE` and `AMBIGUOUS_SOURCE` are terminal (no repair retry).

3. **Payload safety and aggregation**
   - Verify payload-level status/action/failure-summary consistency.
   - Verify no invalid business payload emission under strict non-accept outcomes.

4. **Pending/handoff workflow contract**
   - Verify pending routes produce pending workflow semantics and deterministic handoff tracking identifiers.
   - Verify no synchronous-complete signaling on pending routes.

5. **Robustness and malformed input handling**
   - Verify structured input edge cases (empty, malformed, sparse) return controlled outcomes rather than runtime crashes.

6. **Determinism checks**
   - Re-run selected structured fixtures multiple times to confirm stable unit IDs and stable aggregate outcomes.

7. **Documentation conformance review**
   - Cross-check implementation note and alpha validation artifacts against current passing tests and observed runtime behavior.

## Release blockers

Any item below is a beta blocker:

- Any crash or uncaught exception on malformed/empty structured input.
- Any observed repair selection of non-`UNSUPPORTED_CLAIM` units.
- Any strict passthrough response that emits completed business payload on non-`ACCEPT` outcomes.
- Any adapter payload containing content from rejected units.
- Any pending/handoff route marked as completed success.
- Any nondeterministic/colliding structured path-based unit IDs for covered scenarios.
- Any validated regression in legacy text-mode behavior.
- Any material mismatch between v4 docs and implemented runtime behavior.

## Explicit non-goals for beta

The following are intentionally not required for v4 beta entry:

- Public contract redesign.
- Deep nested object semantic validation beyond current structured scalar-leaf/scalar-array scope.
- Enterprise workflow integrations beyond current handoff contract fields.
- Non-heuristic failure-classification redesign.

## Beta is NOT blocked by

Provided all beta entry criteria are met, beta is not blocked by the following deferred items:

- Public contract redesign work.
- Expanded deep semantic validation for arbitrarily nested object structures.
- Enterprise ecosystem/workflow integration expansion.
- Replacing current classification logic with a non-heuristic redesign.
