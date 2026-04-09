# Evidence-Gated Generation (EGA)

EGA is a verification layer for LLM systems that **emits only evidence-supported claims**.

Instead of treating an answer as fully correct or incorrect, EGA operates at the claim level:

- extract claims from an LLM answer  
- retrieve and rerank evidence  
- verify claims using NLI  
- apply conformal abstention  
- emit only supported claims with citations  

Key idea: Do not emit a claim unless it is supported by evidence

---

## Blog

Detailed explanation and results:
https://bh3r1th.medium.com/evidence-gated-generation-ega-v2-claim-level-verification-and-selective-abstention-for-safer-llm-638546b5632d

---

## Install

python -m pip install -e ".[dev,nli]"

---

## Quickstart

ega v2-eval \
  --dataset examples/v2/eval_dataset_pilot.jsonl \
  --out runs/v2_compare/eval/pilot_threshold_005_recalibrated.json \
  --conformal-state runs/v2_compare/calibration/pilot_conformal_state.json \
  --accept-threshold 0.05 \
  --render-safe-answer

---

## Package Usage (Public Contract)

```python
from ega import PipelineConfig, PolicyConfig, verify_answer

config = PipelineConfig(
    policy=PolicyConfig(threshold_entailment=0.5, max_contradiction=0.9),
    scores_jsonl_path="runs/scores.jsonl",
)

result = verify_answer(
    llm_output="Paris is in France.",
    source_text="Paris is in France.",
    config=config,
)

assert {"verified_text", "verified_units", "dropped_units", "trace"} <= set(result)
```

Official package-level API: `verify_answer`, `PipelineConfig`, and `PolicyConfig`.
Other modules and symbols should be treated as internal and may change without notice.

---

## Example Behavior

Input (LLM output):
The Eiffel Tower is located in Berlin. It was completed in 1889.

EGA Output:
The Eiffel Tower was completed in 1889. [source]

- incorrect claims → removed  
- uncertain claims → withheld  
- supported claims → emitted with citations  

---

## Evaluation Workflow

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

---

## Publishable Scope

Recommended variants:

- v1_baseline  
- rerank_only  
- conformal_only  
- combined  

Experimental:

- budget_only  

Budget variant is included but should not be used in headline comparisons unless explicitly reported as experimental.

---

## Docs

- docs/v2.md  
- docs/poc_results.md  
- docs/v2_eval_dataset.md  
- docs/arxiv_paper_outline.md  

---

## Status

This is an early proof-of-concept (v2).

- pilot dataset: 10 questions / 44 claims  
- larger evaluation in progress  

The goal is to validate claim-level verification + abstention as a practical mechanism for safer LLM outputs.
