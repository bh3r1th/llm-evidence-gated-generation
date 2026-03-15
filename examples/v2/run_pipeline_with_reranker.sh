#!/usr/bin/env bash
set -euo pipefail

ega pipeline \
  --llm-summary-file examples/pipeline_demo/llm_summary.txt \
  --evidence-json examples/pipeline_demo/evidence.json \
  --use-oss-nli \
  --unitizer sentence \
  --use-reranker \
  --reranker-model cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --rerank-topk 6 \
  --partial-allowed
