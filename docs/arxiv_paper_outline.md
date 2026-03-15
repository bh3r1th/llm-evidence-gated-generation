# arXiv Paper Outline

## Title Candidates

- Evidence-Gated Generation with Reranking and Conformal Abstention
- A Reproducible POC for Evidence-Gated Answer Verification
- Evidence-Gated Answering: Verification, Abstention, and Safe Rendering

## Abstract Skeleton

- Problem: LLM summaries can mix supported and unsupported claims.
- Method: claim extraction, evidence retrieval, reranking, NLI verification, conformal abstention, safe-answer rendering.
- Setting: small pilot evaluation with fixed operating configuration.
- Main result: reranking gives modest value; conformal abstention reduces hallucination; combined mode is the recommended POC release configuration.
- Scope: reproducible POC artifact, not a broad benchmark.

## Section Outline

1. Introduction
2. Related Work
3. EGA Architecture
4. Verification and Abstention
5. Experimental Setup
6. Results
7. Limitations
8. Conclusion

## Repo-to-Paper Mapping

### 1. Introduction
- Motivation from unsupported-claim failure modes in the pilot examples.
- Safe-answer rendering as the end-user output constraint.

### 2. Related Work
- Retrieval-grounded verification.
- NLI-based factuality checking.
- Conformal risk control and abstention.

### 3. EGA Architecture
- `src/ega/pipeline.py`
- `docs/v2.md`

### 4. Verification and Abstention
- `src/ega/verifiers/nli_cross_encoder.py`
- `src/ega/v2/conformal.py`
- `src/ega/v2/calibrate.py`
- `src/ega/v2/export_calibration_rows.py`

### 5. Experimental Setup
- `examples/v2/eval_dataset_pilot.jsonl`
- `runs/v2_compare/calibration/pilot_conformal_state.json`
- recommended config from `docs/v2.md`

### 6. Results
- `runs/v2_compare/eval/final_poc_summary.json`
- `docs/poc_results.md`
- `v1_baseline`, `rerank_only`, `conformal_only`, `combined`

### 7. Limitations
- Pilot-scale evaluation only.
- Cost metrics are proxies.
- Budget controller remains experimental.

### 8. Conclusion
- Reproducible artifact claim.
- Honest scope for future scaling and stronger evaluation.
