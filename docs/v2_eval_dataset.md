# V2 Eval Dataset

`examples/v2/eval_dataset_pilot.jsonl` is the 10-example pilot dataset used for the current v2 POC comparison.

Schema per row:
- `id`
- `prompt`
- `llm_summary_text`
- `evidence_json`
- `gold_units`

Qualitative expectations:
- `v1_baseline` is the verifier-only reference.
- `rerank_only` should modestly reduce verifier work on distractor-heavy rows.
- `conformal_only` should increase abstention on uncertain claims.
- `combined` combines reranking and conformal abstention without budget claims.
- `budget_only` is experimental and should not be used in headline results.

Gold-aware metrics:
- `unsupported_claim_rate` comes from `gold_units.supported`.
- `gold_coverage_recall` uses `gold_units.relevant_evidence_ids` against the pipeline `used_evidence`.
