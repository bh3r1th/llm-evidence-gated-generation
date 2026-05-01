#!/usr/bin/env bash
set -euo pipefail

ega pipeline \
  --llm-summary-file examples/pipeline_demo/llm_summary.txt \
  --evidence-json examples/pipeline_demo/evidence.json \
  --scores-jsonl examples/pipeline_demo/scores.jsonl \
  --unitizer spacy_sentence \
  --threshold-entailment 0.8 \
  --max-contradiction 0.2 \
  --partial-allowed \
  --polished-json examples/pipeline_demo/polished_ok.json

ega pipeline \
  --llm-summary-file examples/pipeline_demo/llm_summary.txt \
  --evidence-json examples/pipeline_demo/evidence.json \
  --scores-jsonl examples/pipeline_demo/scores.jsonl \
  --unitizer spacy_sentence \
  --threshold-entailment 0.8 \
  --max-contradiction 0.2 \
  --partial-allowed \
  --polished-json examples/pipeline_demo/polished_bad.json
