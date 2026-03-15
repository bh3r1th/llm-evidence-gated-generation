#!/usr/bin/env bash
set -euo pipefail

mkdir -p artifacts

ega v2-eval \
  --dataset examples/v2/eval_dataset_tiny.jsonl \
  --out artifacts/v2_eval_summary.json \
  --conformal-state artifacts/conformal_state.tiny.json \
  --accept-threshold 0.05 \
  --rerank-topk 6
