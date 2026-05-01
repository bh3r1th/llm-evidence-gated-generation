# Changelog

## [1.0.0] — First public release

### What EGA is

Runtime enforcement layer for LLM outputs. Verifies each claim against source evidence before allowing output downstream. Not an eval tool — enforces at runtime.

### What ships in v1.0.0

- Claim-level verification via NLI (DeBERTa-v3)
- Conformal gating with calibrated accept/abstain/reject decisions
- Explicit authority decision function: conformal (in-range) / conformal_oor / threshold fallback
- Two output modes: strict and adapter
- Bounded correction loop (disabled by default)
- Async pending contract with tracking_id
- `summarize_result()` utility for operational signal extraction
- CLI: `ega run`

### Known limitations

- Sentence-level segmentation, not semantic claim decomposition
- Calibration bootstrapped on pilot data, not production-grade
- Clean inputs frequently trigger conformal_oor (OOR-high) until recalibrated on production data
- Structured output BM25 routing untested on real pipelines
- Not benchmarked against RAGAS or TruLens

### Internal development history

Four internal development stages preceded this release.

| Stage | Focus |
|-------|-------|
| 1 | Whole-output verification |
| 2 | Claim-level verification + selective abstention |
| 3 | Bounded correction + conformal gating stabilization |
| 4 | Authority decision function + output schemas + async contract |

v1.0.0 is the first public release.
