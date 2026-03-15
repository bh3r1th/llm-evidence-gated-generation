"""Export conformal calibration rows from the current v2 verifier path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ega.v2.eval_harness import _extract_gold_units, _load_dataset_jsonl, _run_one

CALIBRATION_SCORE_DEFINITION = (
    "VerificationScore.entailment from the current verifier path for the chosen evidence, "
    "before conformal gating and before accept-threshold retention."
)


def export_calibration_rows(
    *,
    dataset_path: str | Path,
    out_path: str | Path,
    nli_model_name: str | None = None,
    topk_per_unit: int = 12,
    max_pairs_total: int = 200,
    nli_device: str = "auto",
    nli_dtype: str = "auto",
    accept_threshold: float | None = None,
) -> dict[str, Any]:
    """Export per-unit calibration rows using the same verifier path as v2 eval."""
    dataset_rows = _load_dataset_jsonl(dataset_path)
    out_rows: list[dict[str, Any]] = []

    for row in dataset_rows:
        example_id = str(row.get("id", "")).strip() or "<unknown>"
        gold_units = _extract_gold_units(row)
        if gold_units is None:
            raise ValueError(
                f"Calibration export requires gold_units or gold_unit_labels; missing for example {example_id}."
            )
        result, _trace = _run_one(
            row=row,
            variant={
                "kwargs": (
                    {"accept_threshold": float(accept_threshold)}
                    if accept_threshold is not None
                    else {}
                ),
                "config": (
                    {"accept_threshold": float(accept_threshold)}
                    if accept_threshold is not None
                    else {}
                ),
                "reranker_enabled": False,
            },
            topk_per_unit=topk_per_unit,
            max_pairs_total=max_pairs_total,
            nli_model_name=nli_model_name,
            nli_device=nli_device,
            nli_dtype=nli_dtype,
            accept_threshold=accept_threshold,
            render_safe_answer=False,
        )
        verifier_scores = result.get("verifier_scores", {})
        if not isinstance(verifier_scores, dict):
            verifier_scores = {}
        verifier_model_name = str(result.get("verifier_model_name", "") or "")
        recorded_threshold = result.get("accept_threshold")

        for unit_id, gold in gold_units.items():
            score_payload = verifier_scores.get(unit_id, {})
            if not isinstance(score_payload, dict):
                score_payload = {}
            out_rows.append(
                {
                    "example_id": example_id,
                    "unit_id": str(unit_id),
                    "score": _extract_calibration_score(score_payload),
                    "supported": bool(gold.get("supported", False)),
                    "chosen_evidence_id": score_payload.get("chosen_evidence_id"),
                    "verifier_model_name": verifier_model_name,
                    "accept_threshold": (
                        None if recorded_threshold is None else float(recorded_threshold)
                    ),
                }
            )

    resolved_out = Path(out_path)
    resolved_out.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in out_rows),
        encoding="utf-8",
    )
    return {
        "dataset_path": str(Path(dataset_path)),
        "out_path": str(resolved_out),
        "n_examples": len(dataset_rows),
        "n_rows": len(out_rows),
        "score_definition": CALIBRATION_SCORE_DEFINITION,
    }


def _extract_calibration_score(score_payload: dict[str, Any]) -> float:
    for key in ("conformal_score", "entailment"):
        value = score_payload.get(key)
        if value is not None:
            return float(value)
    return 0.0
