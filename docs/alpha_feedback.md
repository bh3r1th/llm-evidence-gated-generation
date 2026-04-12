# EGA v4 Alpha Feedback

## What was validated

- The post-release alpha validation pass executed five end-to-end scenarios and all five passed after hardening and re-runs on 2026-04-12.
- Validated paths covered:
  - legacy text-mode accept flow stability,
  - structured strict passthrough reject safety,
  - structured adapter-mode partial payload behavior,
  - repair routing only for `UNSUPPORTED_CLAIM`,
  - pending/handoff semantics and deterministic tracking IDs.
- Validation command: `pytest -q tests/test_alpha_validation_v4.py tests/test_pipeline_v4_structured_mode.py` (15 passed).

## What worked

- Legacy contract behavior stayed stable for text-mode acceptance, including business payload emission rules.
- Strict passthrough behavior now correctly blocks success payload emission on structured failures.
- Adapter mode now isolates rejected units to metadata and prevents rejected-unit leakage into adapter payload.
- Repair gating behavior is correctly scoped: `UNSUPPORTED_CLAIM` can route to repair; `MISSING_IN_SOURCE` and `AMBIGUOUS_SOURCE` remain terminal reject classes.
- Pending/handoff derivation is now stable for review/repair routes, including deterministic tracking metadata.

## What failed

- Before hardening, strict structured paths had payload safety assertion failures (success payload boundaries were too permissive).
- Before hardening, adapter mode had rejected-content isolation failures (rejected content could leak into adapter-shaped payload paths).
- Before hardening, retry selection was not strict enough when failure-class metadata was absent.
- Before hardening, structured unit path/id generation had mixed-type key stability issues (collision risk).
- These failures were addressed in the alpha hardening pass, but they confirm that v4 edge handling still needs deliberate stress coverage.

## Known limitations

- Public contract is still legacy-shaped (`verify_answer`, `PipelineConfig`, `PolicyConfig`, and legacy-oriented response shape), with v4 fields added additively rather than as a finalized v4-first public interface.
- Structured mode remains limited to scalar leaves and scalar array entries; non-scalar nested values are not first-class verification units yet.
- Failure classification is heuristic/rule-based (threshold and evidence-selection logic), not a calibrated or learned taxonomy.
- Async/handoff fields are additive contract semantics for pending work (`workflow_status`, `handoff_required`, `handoff_reason`, `tracking_id`), not a full workflow orchestration platform.
- Alpha validation scope was intentionally narrow (5 scenarios) and does not constitute broad load/perf or enterprise workflow validation.

## Recommended next priorities

1. **Structured mode edge-case hardening**
   - Expand tests for non-scalar nested payload shapes, mixed-type arrays, and deep-path stability guarantees.
   - Add explicit documentation for unsupported structures and expected fallback behavior.
2. **Failure classification quality improvements**
   - Improve class precision/recall with targeted eval sets for missing-vs-ambiguous-vs-unsupported splits.
   - Reduce threshold brittleness and tighten confidence reporting around classification outcomes.
3. **Adapter-mode leakage hardening**
   - Add contract tests ensuring adapter payload contains only accepted units across mixed reject scenarios.
   - Extend negative tests for path aliasing and metadata-to-payload boundary regressions.
4. **Public v4 contract decision**
   - Decide whether to keep additive legacy shape or publish a v4-first response contract and migration path.
   - Align README/CHANGELOG/API docs with that decision before beta.
5. **Beta readiness checklist**
   - Formalize exit criteria: scenario matrix, regression suite thresholds, docs completeness, and handoff semantics sign-off.
   - Include explicit “known non-goals” to prevent scope creep into full workflow platform expectations.
