"""Threshold sweep utility for EGA v2 baseline verifier outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ega.contract import PolicyConfig
from ega.v2.eval_harness import (
    _extract_gold_units,
    _load_dataset_jsonl,
    _run_one,
    collect_unit_verifier_rows,
)

DEFAULT_THRESHOLDS = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)


def run_threshold_sweep(
    *,
    dataset_path: str | Path,
    out_path: str | Path,
    nli_model_name: str | None = None,
    topk_per_unit: int = 12,
    max_pairs_total: int = 200,
    nli_device: str = "auto",
    nli_dtype: str = "auto",
    accept_threshold: float | None = None,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    rows = _load_dataset_jsonl(dataset_path)
    sweep_totals: dict[float, dict[str, int]] = {
        float(threshold): {
            "kept_units": 0,
            "dropped_units": 0,
            "supported_kept": 0,
            "unsupported_kept": 0,
            "supported_gold_total": 0,
            "gold_unit_total": 0,
            "unit_total": 0,
        }
        for threshold in thresholds
    }

    for row in rows:
        gold_units = _extract_gold_units(row)
        if gold_units is None:
            example_id = str(row.get("id", "")).strip() or "<unknown>"
            raise ValueError(f"Threshold sweep requires gold_units; missing for example {example_id}.")
        policy = _policy_from_row_for_sweep(row)
        result, _trace = _run_one(
            row=_row_with_baseline_policy(row),
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
        unit_rows = collect_unit_verifier_rows(result=result, gold_units=gold_units)
        for threshold in thresholds:
            _accumulate_threshold_metrics(
                totals=sweep_totals[float(threshold)],
                unit_rows=unit_rows,
                threshold=float(threshold),
                policy=policy,
            )

    sweeps = [_finalize_threshold_metrics(threshold=threshold, totals=sweep_totals[threshold]) for threshold in thresholds]
    recommended = _select_best_threshold(sweeps)
    summary = {
        "dataset_path": str(Path(dataset_path)),
        "n_examples": len(rows),
        "accept_threshold": accept_threshold,
        "thresholds": [float(threshold) for threshold in thresholds],
        "sweeps": sweeps,
        "recommended_threshold": recommended["threshold"] if recommended is not None else None,
        "selection_metric": "f1",
    }
    Path(out_path).write_text(json.dumps(summary, sort_keys=True, indent=2), encoding="utf-8")
    return summary


def _row_with_baseline_policy(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    policy = _policy_from_row_for_sweep(row)
    out["policy"] = {
        "threshold_entailment": 0.0,
        "max_contradiction": float(policy.max_contradiction),
        "partial_allowed": True,
    }
    return out


def _policy_from_row_for_sweep(row: dict[str, Any]) -> PolicyConfig:
    payload = row.get("policy", {})
    if not isinstance(payload, dict):
        payload = {}
    return PolicyConfig(
        threshold_entailment=float(payload.get("threshold_entailment", 0.5)),
        max_contradiction=float(payload.get("max_contradiction", 0.2)),
        partial_allowed=bool(payload.get("partial_allowed", True)),
    )


def _accumulate_threshold_metrics(
    *,
    totals: dict[str, int],
    unit_rows: list[dict[str, Any]],
    threshold: float,
    policy: PolicyConfig,
) -> None:
    max_contradiction = float(policy.max_contradiction)
    for unit in unit_rows:
        totals["unit_total"] += 1
        supported = unit.get("supported")
        if isinstance(supported, bool):
            totals["gold_unit_total"] += 1
            if supported:
                totals["supported_gold_total"] += 1
        kept = float(unit.get("entailment", 0.0)) >= threshold and float(
            unit.get("contradiction", 0.0)
        ) <= max_contradiction
        if kept:
            totals["kept_units"] += 1
            if supported is True:
                totals["supported_kept"] += 1
            else:
                totals["unsupported_kept"] += 1
        else:
            totals["dropped_units"] += 1


def _finalize_threshold_metrics(*, threshold: float, totals: dict[str, int]) -> dict[str, Any]:
    kept_units = int(totals["kept_units"])
    supported_kept = int(totals["supported_kept"])
    unsupported_kept = int(totals["unsupported_kept"])
    supported_gold_total = int(totals["supported_gold_total"])
    unit_total = int(totals["unit_total"])
    precision = float(supported_kept) / float(kept_units) if kept_units > 0 else None
    recall = float(supported_kept) / float(supported_gold_total) if supported_gold_total > 0 else None
    f1 = _f1(precision=precision, recall=recall)
    unsupported_claim_rate = float(unsupported_kept) / float(kept_units) if kept_units > 0 else None
    hallucination_rate = float(unsupported_kept) / float(unit_total) if unit_total > 0 else None
    return {
        "threshold": float(threshold),
        "kept_units": kept_units,
        "dropped_units": int(totals["dropped_units"]),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "unsupported_claim_rate": unsupported_claim_rate,
        "hallucination_rate": hallucination_rate,
    }


def _f1(*, precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _select_best_threshold(sweeps: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [row for row in sweeps if isinstance(row.get("f1"), (int, float))]
    if not candidates:
        return None
    candidates.sort(
        key=lambda row: (
            float(row.get("f1", 0.0)),
            float(row.get("recall", 0.0) or 0.0),
            -float(row.get("unsupported_claim_rate", 1.0) or 1.0),
            float(row.get("threshold", 0.0)),
        ),
        reverse=True,
    )
    return candidates[0]
