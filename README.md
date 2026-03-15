# EGA

EGA is an evidence-gated answer verification pipeline. The v2 POC extracts claims, retrieves evidence, optionally reranks evidence, verifies claims with NLI, applies conformal abstention, and can render a safe final answer from accepted claims.

## Install

```bash
python -m pip install -e ".[dev,nli]"
```

## Quickstart

```bash
ega v2-eval \
  --dataset examples/v2/eval_dataset_pilot.jsonl \
  --out runs/v2_compare/eval/pilot_threshold_005_recalibrated.json \
  --conformal-state runs/v2_compare/calibration/pilot_conformal_state.json \
  --accept-threshold 0.05 \
  --render-safe-answer
```

## Evaluation Workflow

Use the publish workflow in this order:

```bash
ega threshold-sweep \
  --dataset examples/v2/eval_dataset_pilot.jsonl \
  --out runs/v2_compare/eval/threshold_sweep.json \
  --accept-threshold 0.05

ega export-calibration-rows \
  --dataset examples/v2/eval_dataset_pilot.jsonl \
  --out runs/v2_compare/calibration/pilot_calibration_rows.jsonl \
  --accept-threshold 0.05

ega conformal-calibrate \
  --in runs/v2_compare/calibration/pilot_calibration_rows.jsonl \
  --out runs/v2_compare/calibration/pilot_conformal_state.json \
  --epsilon 0.05 \
  --min-calib 1

ega v2-eval \
  --dataset examples/v2/eval_dataset_pilot.jsonl \
  --out runs/v2_compare/eval/pilot_threshold_005_recalibrated.json \
  --conformal-state runs/v2_compare/calibration/pilot_conformal_state.json \
  --accept-threshold 0.05 \
  --reranker-model cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --render-safe-answer

ega generate-poc-report \
  --source-summary runs/v2_compare/eval/pilot_threshold_005_recalibrated.json \
  --dataset examples/v2/eval_dataset_pilot.jsonl \
  --conformal-state runs/v2_compare/calibration/pilot_conformal_state.json \
  --summary-out runs/v2_compare/eval/final_poc_summary.json \
  --report-out docs/poc_results.md
```

## Publishable Scope

Recommended headline variants:
- `v1_baseline`
- `rerank_only`
- `conformal_only`
- `combined`

Experimental:
- `budget_only`

Budget code remains in the repo, but it is not part of the recommended publish comparison unless a binding budget run is explicitly demonstrated and reported as experimental.

## Docs

- [v2 workflow](docs/v2.md)
- [POC results](docs/poc_results.md)
- [pilot dataset notes](docs/v2_eval_dataset.md)
- [paper outline scaffold](docs/arxiv_paper_outline.md)
