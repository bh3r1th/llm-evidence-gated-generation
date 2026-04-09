# Evidence-Gated Answering (EGA)

EGA is a verification layer you run after an LLM produces an answer: it splits the answer into units, checks each unit against provided evidence with a verifier, applies policy/conformal gating, and returns only units that pass support checks so downstream systems can prefer partial truth over unsupported claims.

## Official package API (v3)

The package-level integration surface is intentionally small:

- `verify_answer`
- `PipelineConfig`

`PolicyConfig` is required to construct `PipelineConfig` and is part of the stable config contract.

## Minimal package usage

```python
import json
import tempfile
from pathlib import Path

from ega import PipelineConfig, PolicyConfig, verify_answer

# Minimal deterministic scoring source for local demos/tests.
with tempfile.TemporaryDirectory() as td:
    scores_path = Path(td) / "scores.jsonl"
    scores_path.write_text(
        json.dumps({"unit_id": "u0001", "score": 0.95, "label": "entailment"}) + "\n",
        encoding="utf-8",
    )

    result = verify_answer(
        llm_output="Paris is in France.",
        source_text="Paris is in France.",
        config=PipelineConfig(
            policy=PolicyConfig(threshold_entailment=0.5, max_contradiction=0.9),
            scores_jsonl_path=str(scores_path),
        ),
    )

print(result["verified_text"])
print(result["verified_units"])
print(result["dropped_units"])
print(result["trace"]["trace_schema_version"])
```

Returned top-level keys from `verify_answer(...)` are:

- `verified_text`
- `verified_units`
- `dropped_units`
- `trace`

## Minimal CLI usage (manual/local workflow)

Use the CLI when running local files or manual checks. It uses the same underlying execution model as the package pipeline.

```bash
ega pipeline \
  --llm-summary-file examples/pipeline_demo/llm_summary.txt \
  --evidence-json examples/pipeline_demo/evidence.json \
  --scores-jsonl examples/pipeline_demo/scores.jsonl \
  --threshold-entailment 0.8
```

## Bounded correction loop

Correction is optional and bounded (`enable_correction`, `max_retries` in `PipelineConfig`). Only failed units are retried, each retry is re-verified, and units still failing at the retry limit are dropped/abstained by normal decision rules.

## Trace output contract

`verify_answer(...)` always returns a `trace` object for observability/debugging. Stable fields include unit counts/ids, verifier metadata, keep/drop/abstain counts, correction-loop metadata, and stage timings (`total_seconds`, `unitize_seconds`, `verify_seconds`, `enforce_seconds`).

## SKILL-driven operating pattern

Operational workflow reference: [`docs/SKILL.md`](docs/SKILL.md).

## Public API vs internals

- **Public/stable**: `ega.verify_answer`, `ega.PipelineConfig`, and `ega.PolicyConfig`.
- **Internal/subject to change**: other modules, CLI subcommand arguments, and implementation details under `ega.*` not explicitly listed above.

## Repo boundaries

- Current package + CLI examples: `examples/minimal.py`, `examples/pipeline_demo/`
- Legacy v2 evaluation material (kept for reproducibility): `examples/v2/`, `docs/v2.md`
