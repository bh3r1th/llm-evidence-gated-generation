# EGA v4 Stable Release Criteria

Date: 2026-04-12

Stable release requires all gates below to pass.

1. **No critical payload-safety bugs**
   - Zero open critical issues that can emit unsafe/invalid payload outcomes.

2. **No adapter leakage**
   - Adapter output never includes rejected-unit content; rejected content appears only in rejection metadata.

3. **No incorrect pending/completed semantics**
   - Pending routes stay pending with required handoff signaling; completed routes are only truly terminal outcomes.

4. **Structured mode stable on beta matrix**
   - All scenarios in `docs/v4_beta_validation_matrix.md` are executed and marked PASS (or explicitly waived with approval).

5. **Acceptable classification quality on sampled real cases**
   - Agreed beta sample shows classification quality at or above team-defined acceptance threshold, with documented sampling method.

6. **No legacy text regressions**
   - Legacy text fixtures/regression checks match expected pre-v4 behavior.

7. **Docs match behavior**
   - Stable-facing docs reflect observed runtime behavior, known limitations, and gate outcomes with no contradictions.
