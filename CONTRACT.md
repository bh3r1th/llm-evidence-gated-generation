# EGA Public Contract

## Scope

EGA is an enforcement/gating layer. It consumes a generated summary and evidence, scores units
against evidence, and emits a deterministic gating decision.

## CLI Contracts

### `ega run`
- Input:
  - `--answer-file` plain text summary.
  - `--evidence-file` JSON array of `{id,text,metadata?}`.
  - Optional policy knobs (`--threshold`, `--partial-allowed`).
- Output:
  - Serialized `EnforcementResult`.

### `ega pipeline`
- Input:
  - `--llm-summary-file`, `--evidence-json`.
  - One scoring source: `--scores-jsonl` or `--use-oss-nli`.
  - Policy knobs (`--threshold-entailment`, `--max-contradiction`, `--partial-allowed`).
- Output:
  - JSON object with `verified_extract`, `verified_text`, `decision`, `stats`, and optional polish fields.

### `ega shell`
- Interactive mode:
  - prompts for summary and evidence file paths.
- JSONL mode (`--stdin-jsonl --stdout-jsonl`):
  - one JSON request per line, one JSON response per line.
  - supports `llm_summary` (or `summary`) and `evidence` (or `evidence_json`).

## Output Semantics

- `verified_extract`:
  - list of kept units (`unit_id`, `text`) in original unit order.
- `verified_text`:
  - newline join of `verified_extract[].text`.
- `decision`:
  - `allowed_units`: kept unit ids.
  - `dropped_units`: dropped unit ids.
  - `refusal`: whether output is refused.
  - `reason_code`: decision reason.
  - `summary_stats`: aggregate scores/counts used for policy decisions.

### Refusal and `partial_allowed`
- If all units are dropped, response is a refusal.
- If some units are dropped:
  - `partial_allowed=true`: keep supported units, non-refusal partial output.
  - `partial_allowed=false`: refusal.

## Non-Guarantees

- NLI verifier scores are heuristic model outputs, not proofs.
- Evidence quality and coverage materially affect outcomes.
- CPU latency, batching behavior, and throughput vary by hardware/runtime.
- Policy thresholds are operational controls, not guarantees of factual correctness.

## Trace Contract

When `--trace-out` is used, trace rows are JSON objects whose shape is documented by:
- `trace_schema.json`

Shell JSONL trace rows may be a reduced subset in invalid-input/error cases.
