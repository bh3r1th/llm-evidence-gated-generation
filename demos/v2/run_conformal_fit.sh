#!/usr/bin/env bash
set -euo pipefail

mkdir -p artifacts

ega conformal-calibrate \
  --in examples/v2/conformal_calibration_tiny.jsonl \
  --out artifacts/conformal_state.tiny.json \
  --epsilon 0.05 \
  --min-calib 5
