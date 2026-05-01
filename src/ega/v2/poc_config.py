"""Central defaults for the publishable EGA v2 POC workflow."""

from __future__ import annotations

from pathlib import Path

from ega.verifiers.nli_cross_encoder import DEFAULT_MODEL_NAME

# Legacy fallback only when no conformal state is loaded.
DEFAULT_ACCEPT_THRESHOLD = 0.05
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_RERANK_TOPK = 6

DEFAULT_FINAL_DATASET = Path("examples") / "v2" / "eval_dataset_pilot.jsonl"
DEFAULT_FINAL_CALIBRATION_ROWS = (
    Path("runs") / "v2_compare" / "calibration" / "pilot_calibration_rows.jsonl"
)
DEFAULT_FINAL_CONFORMAL_STATE = (
    Path("runs") / "v2_compare" / "calibration" / "pilot_conformal_state.json"
)
DEFAULT_FINAL_SOURCE_SUMMARY = (
    Path("runs") / "v2_compare" / "eval" / "pilot_threshold_005_recalibrated.json"
)
DEFAULT_FINAL_SUMMARY = Path("runs") / "v2_compare" / "eval" / "final_poc_summary.json"
DEFAULT_FINAL_REPORT = Path("docs") / "poc_results.md"

RECOMMENDED_VARIANTS = ("v1_baseline", "rerank_only", "conformal_only", "combined")
EXPERIMENTAL_VARIANTS = ("budget_only",)

SAFE_ANSWER_EXAMPLE = """Supported claims are rendered as a compact answer:

```text
The Eiffel Tower is in Paris. [e1]
It opened in 1889. [e3]
```
"""

__all__ = [
    "DEFAULT_ACCEPT_THRESHOLD",
    "DEFAULT_FINAL_CALIBRATION_ROWS",
    "DEFAULT_FINAL_CONFORMAL_STATE",
    "DEFAULT_FINAL_DATASET",
    "DEFAULT_FINAL_REPORT",
    "DEFAULT_FINAL_SOURCE_SUMMARY",
    "DEFAULT_FINAL_SUMMARY",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_RERANKER_MODEL",
    "DEFAULT_RERANK_TOPK",
    "EXPERIMENTAL_VARIANTS",
    "RECOMMENDED_VARIANTS",
    "SAFE_ANSWER_EXAMPLE",
]
