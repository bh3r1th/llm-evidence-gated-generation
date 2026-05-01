"""Variant evaluation harness for EGA v2."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

from ega.contract import PolicyConfig
from ega.pipeline import run_pipeline
from ega.types import EvidenceItem, EvidenceSet
from ega.v2.budget import BudgetConfig
from ega.v2.budget_greedy import GreedyBudgetPolicy
from ega.v2.coverage import CoverageConfig
from ega.v2.cross_encoder_reranker import CrossEncoderReranker
from ega.v2.poc_config import DEFAULT_RERANKER_MODEL, DEFAULT_RERANK_TOPK
from ega.v2.rewards import RewardConfig
from ega.verifiers.nli_cross_encoder import DEFAULT_MODEL_NAME

DEFAULT_DEBUG_DUMP_PATH = Path("runs") / "v2_compare" / "eval" / "pilot_debug_examples.jsonl"


def run_v2_eval(
    *,
    dataset_path: str | Path,
    out_path: str | Path,
    conformal_state_path: str | None = None,
    reranker_model: str = DEFAULT_RERANKER_MODEL,
    rerank_topk: int = DEFAULT_RERANK_TOPK,
    latency_budget_ms: int | None = None,
    budget_max_pairs: int | None = None,
    topk_per_unit: int = 12,
    max_pairs_total: int = 200,
    nli_model_name: str | None = None,
    nli_device: str = "auto",
    nli_dtype: str = "auto",
    accept_threshold: float | None = None,
    cost_rerank_weight: float = 1.0,
    debug_dump_path: str | Path | None = None,
    render_safe_answer: bool = False,
) -> dict[str, Any]:
    _ = float(cost_rerank_weight)
    rows = _load_dataset_jsonl(dataset_path)
    debug_path = Path(debug_dump_path) if debug_dump_path is not None else None
    if debug_path is not None:
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.write_text("", encoding="utf-8")
    variants = _build_variants(
        conformal_state_path=conformal_state_path,
        reranker_model=reranker_model,
        rerank_topk=rerank_topk,
        latency_budget_ms=latency_budget_ms,
        budget_max_pairs=budget_max_pairs,
        accept_threshold=accept_threshold,
    )

    summary: dict[str, Any] = {
        "n_examples": len(rows),
        "variants": {},
    }

    for variant_name, variant in variants.items():
        metrics = _empty_metrics()
        errors: list[str] = []
        skipped = bool(variant.get("skip", False))
        if skipped:
            summary["variants"][variant_name] = {
                "status": "skipped",
                "reason": str(variant.get("skip_reason", "skipped")),
                "config": dict(variant.get("config", {})),
                "debug": _variant_debug_metadata(
                    variant=variant,
                    nli_model_name=nli_model_name,
                ),
                "metrics": _finalize_metrics(
                    metrics=metrics,
                ),
                "metrics_metadata": _finalize_metrics_metadata(metrics=metrics),
            }
            continue

        runtime_variant = _materialize_variant(
            variant=variant,
            nli_model_name=nli_model_name,
        )
        if bool(runtime_variant.get("skip", False)):
            summary["variants"][variant_name] = {
                "status": "skipped",
                "reason": str(runtime_variant.get("skip_reason", "skipped")),
                "config": dict(variant.get("config", {})),
                "debug": _variant_debug_metadata(
                    variant=runtime_variant,
                    nli_model_name=nli_model_name,
                ),
                "metrics": _finalize_metrics(
                    metrics=metrics,
                ),
                "metrics_metadata": _finalize_metrics_metadata(metrics=metrics),
            }
            continue

        for row in rows:
            try:
                result, trace = _run_one(
                    row=row,
                    variant=runtime_variant,
                    topk_per_unit=topk_per_unit,
                    max_pairs_total=max_pairs_total,
                    nli_model_name=nli_model_name,
                    nli_device=nli_device,
                    nli_dtype=nli_dtype,
                    accept_threshold=accept_threshold,
                    render_safe_answer=render_safe_answer,
                )
                _accumulate_metrics(
                    metrics=metrics,
                    result=result,
                    trace=trace,
                    gold_units=_extract_gold_units(row),
                )
                if debug_path is not None:
                    _append_debug_row(
                        path=debug_path,
                        example_id=str(row.get("id", "")),
                        variant=variant_name,
                        row=row,
                        result=result,
                    )
            except Exception as exc:
                errors.append(str(exc))

        status = "ok" if not errors else ("partial" if metrics["n_examples"] > 0 else "error")
        summary["variants"][variant_name] = {
            "status": status,
            "errors": errors,
            "config": dict(variant.get("config", {})),
            "debug": _variant_debug_metadata(
                variant=runtime_variant,
                nli_model_name=nli_model_name,
            ),
            "metrics": _finalize_metrics(
                metrics=metrics,
            ),
            "metrics_metadata": _finalize_metrics_metadata(metrics=metrics),
        }

    Path(out_path).write_text(json.dumps(summary, sort_keys=True, indent=2), encoding="utf-8")
    return summary


def _load_dataset_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Dataset line {line_no} must be an object.")
            if "llm_summary_text" not in payload:
                raise ValueError(f"Dataset line {line_no} missing 'llm_summary_text'.")
            if "evidence_json" not in payload:
                raise ValueError(f"Dataset line {line_no} missing 'evidence_json'.")
            rows.append(payload)
    return rows


def _build_variants(
    *,
    conformal_state_path: str | None,
    reranker_model: str,
    rerank_topk: int,
    latency_budget_ms: int | None,
    budget_max_pairs: int | None,
    accept_threshold: float | None,
) -> dict[str, dict[str, Any]]:
    threshold_config = (
        {"accept_threshold": float(accept_threshold)} if accept_threshold is not None else {}
    )
    variants: dict[str, dict[str, Any]] = {
        "v1_baseline": {
            "kwargs": dict(threshold_config),
            "config": dict(threshold_config),
            "reranker_enabled": False,
            "reranker_model_name": None,
        },
        "budget_only": {
            "kwargs": {
                **threshold_config,
                "budget_policy": GreedyBudgetPolicy(),
                "budget_config": BudgetConfig(
                    latency_budget_ms=latency_budget_ms,
                    max_pairs_total=budget_max_pairs,
                ),
            },
            "config": {
                **threshold_config,
                "use_budget": True,
                "latency_budget_ms": latency_budget_ms,
                "budget_max_pairs": budget_max_pairs,
            },
            "reranker_enabled": False,
            "reranker_model_name": None,
        },
    }

    if conformal_state_path:
        variants["conformal_only"] = {
            "kwargs": {
                **threshold_config,
                "conformal_state_path": conformal_state_path,
            },
            "config": {
                **threshold_config,
                "conformal_state_path": conformal_state_path,
            },
            "reranker_enabled": False,
            "reranker_model_name": None,
        }
    else:
        variants["conformal_only"] = {
            "skip": True,
            "skip_reason": "missing_conformal_state",
            "config": {},
            "reranker_enabled": False,
            "reranker_model_name": None,
        }

    variants["rerank_only"] = {
        "kwargs": {**threshold_config, "rerank_topk": rerank_topk},
        "config": {
            **threshold_config,
            "use_reranker": True,
            "reranker_model": reranker_model,
            "rerank_topk": rerank_topk,
        },
        "reranker_enabled": True,
        "reranker_model_name": reranker_model,
    }
    if conformal_state_path:
        variants["combined"] = {
            "kwargs": {
                **threshold_config,
                "rerank_topk": rerank_topk,
                "conformal_state_path": conformal_state_path,
            },
            "config": {
                **threshold_config,
                "use_reranker": True,
                "conformal_state_path": conformal_state_path,
                "reranker_model": reranker_model,
                "rerank_topk": rerank_topk,
            },
            "reranker_enabled": True,
            "reranker_model_name": reranker_model,
        }
    else:
        variants["combined"] = {
            "skip": True,
            "skip_reason": "missing_conformal_state",
            "config": {},
            "reranker_enabled": True,
            "reranker_model_name": reranker_model,
        }

    return variants


def _materialize_variant(
    *,
    variant: dict[str, Any],
    nli_model_name: str | None,
) -> dict[str, Any]:
    out = {
        key: (dict(value) if isinstance(value, dict) else value)
        for key, value in variant.items()
    }
    kwargs = dict(out.get("kwargs", {}))
    reranker_enabled = bool(out.get("reranker_enabled", False))
    reranker_model_name = out.get("reranker_model_name")
    if reranker_enabled:
        try:
            kwargs["reranker"] = CrossEncoderReranker(model_name=str(reranker_model_name))
        except ImportError:
            out["skip"] = True
            out["skip_reason"] = "missing_reranker_dependency"
            kwargs.pop("reranker", None)
    out["kwargs"] = kwargs
    out["verifier_model_name"] = nli_model_name or DEFAULT_MODEL_NAME
    return out


def _variant_debug_metadata(
    *,
    variant: dict[str, Any],
    nli_model_name: str | None,
) -> dict[str, Any]:
    budget_config = variant.get("kwargs", {}).get("budget_config")
    resolved_verifier_model_name = (
        str(nli_model_name) if isinstance(nli_model_name, str) and nli_model_name.strip() else DEFAULT_MODEL_NAME
    )
    return {
        "accept_threshold": variant.get("kwargs", {}).get("accept_threshold"),
        "verifier_model_name": resolved_verifier_model_name,
        "reranker_model_name": (
            str(variant.get("reranker_model_name"))
            if variant.get("reranker_enabled", False) and variant.get("reranker_model_name") is not None
            else None
        ),
        "reranker_enabled": bool(variant.get("reranker_enabled", False)),
        "budget_active": bool(variant.get("kwargs", {}).get("budget_policy") is not None),
        "requested_budget_max_pairs": (
            None
            if budget_config is None or getattr(budget_config, "max_pairs_total", None) is None
            else int(getattr(budget_config, "max_pairs_total"))
        ),
    }


def _run_one(
    *,
    row: dict[str, Any],
    variant: dict[str, Any],
    topk_per_unit: int,
    max_pairs_total: int,
    nli_model_name: str | None,
    nli_device: str,
    nli_dtype: str,
    accept_threshold: float | None,
    render_safe_answer: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    evidence = _evidence_from_payload(row["evidence_json"])
    policy = _policy_from_row(row)
    unitizer_mode = str(row.get("unitizer_mode", row.get("unitizer", "sentence")))

    trace_file = tempfile.NamedTemporaryFile(prefix="ega_v2_eval_", suffix=".jsonl", delete=False)
    trace_path = Path(trace_file.name)
    trace_file.close()
    t0 = time.perf_counter()
    try:
        kwargs: dict[str, Any] = {
            "llm_summary_text": str(row["llm_summary_text"]),
            "evidence": evidence,
            "unitizer_mode": unitizer_mode,
            "policy_config": policy,
            "accept_threshold": accept_threshold,
            "topk_per_unit": int(topk_per_unit),
            "max_pairs_total": int(max_pairs_total),
            "nli_model_name": nli_model_name,
            "nli_device": nli_device,
            "nli_dtype": nli_dtype,
            "coverage_config": CoverageConfig(),
            "reward_config": RewardConfig(),
            "trace_out": str(trace_path),
            "render_safe_answer": render_safe_answer,
            **dict(variant.get("kwargs", {})),
        }
        if "scores_jsonl_path" in row:
            kwargs["scores_jsonl_path"] = str(row["scores_jsonl_path"])
        else:
            kwargs["use_oss_nli"] = True
        result = run_pipeline(**kwargs)
    finally:
        elapsed = time.perf_counter() - t0

    trace = _read_last_trace_row(trace_path)
    if "total_seconds" not in trace:
        trace["total_seconds"] = elapsed
    trace_path.unlink(missing_ok=True)
    return result, trace


def _read_last_trace_row(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return {}
    payload = json.loads(lines[-1])
    if not isinstance(payload, dict):
        return {}
    return payload


def _evidence_from_payload(payload: Any) -> EvidenceSet:
    if isinstance(payload, str):
        resolved = Path(payload)
        obj = json.loads(resolved.read_text(encoding="utf-8-sig"))
    else:
        obj = payload
    if not isinstance(obj, list):
        raise ValueError("evidence_json must be a list or a path to a list JSON file.")
    items: list[EvidenceItem] = []
    for idx, row in enumerate(obj):
        if not isinstance(row, dict):
            raise ValueError(f"Evidence row {idx} must be an object.")
        if "id" not in row or "text" not in row:
            raise ValueError(f"Evidence row {idx} must include 'id' and 'text'.")
        metadata = row.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"Evidence row {idx} has non-object metadata.")
        items.append(
            EvidenceItem(
                id=str(row["id"]),
                text=str(row["text"]),
                metadata=dict(metadata),
            )
        )
    return EvidenceSet(items=items)


def _policy_from_row(row: dict[str, Any]) -> PolicyConfig:
    payload = row.get("policy", {})
    if not isinstance(payload, dict):
        payload = {}
    return PolicyConfig(
        threshold_entailment=float(payload.get("threshold_entailment", 0.5)),
        max_contradiction=float(payload.get("max_contradiction", 0.2)),
        partial_allowed=bool(payload.get("partial_allowed", True)),
    )


def _extract_gold_units(row: dict[str, Any]) -> dict[str, dict[str, Any]] | None:
    payload = row.get("gold_units")
    if isinstance(payload, list):
        out: dict[str, dict[str, Any]] = {}
        for idx, item in enumerate(payload, start=1):
            if not isinstance(item, dict):
                continue
            unit_id = str(item.get("unit_id", f"u{idx:04d}"))
            relevant_ids = _extract_string_list(item.get("relevant_evidence_ids"))
            required_ids = _extract_string_list(item.get("required_evidence_ids"))
            out[unit_id] = {
                "text": str(item.get("text", "")),
                "supported": bool(item.get("supported", False)),
                "required_evidence_ids": required_ids,
                "relevant_evidence_ids": relevant_ids,
            }
        if out:
            return out

    payload = row.get("gold_unit_labels")
    if payload is None:
        return None
    if isinstance(payload, dict):
        return {
            str(key): {
                "text": "",
                "supported": bool(value),
                "required_evidence_ids": [],
                "relevant_evidence_ids": [],
            }
            for key, value in payload.items()
        }
    if isinstance(payload, list):
        return {
            f"u{i + 1:04d}": {
                "text": "",
                "supported": bool(value),
                "required_evidence_ids": [],
                "relevant_evidence_ids": [],
            }
            for i, value in enumerate(payload)
        }
    return None


def _extract_string_list(payload: Any) -> list[str]:
    if not isinstance(payload, list):
        return []
    return [str(item) for item in payload]


def collect_unit_verifier_rows(
    *,
    result: dict[str, Any],
    gold_units: dict[str, dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    verifier_scores = result.get("verifier_scores", {})
    if not isinstance(verifier_scores, dict):
        verifier_scores = {}
    units_payload = result.get("units", [])
    out: list[dict[str, Any]] = []
    seen_unit_ids: set[str] = set()

    if isinstance(units_payload, list):
        for item in units_payload:
            if not isinstance(item, dict):
                continue
            unit_id = str(item.get("unit_id", "")).strip()
            if not unit_id:
                continue
            score_payload = verifier_scores.get(unit_id, {})
            if not isinstance(score_payload, dict):
                score_payload = {}
            gold = gold_units.get(unit_id) if gold_units is not None else None
            out.append(
                {
                    "unit_id": unit_id,
                    "text": str(item.get("text", "")),
                    "entailment": float(score_payload.get("entailment", 0.0)),
                    "contradiction": float(score_payload.get("contradiction", 0.0)),
                    "supported": (
                        bool(gold.get("supported", False)) if isinstance(gold, dict) else None
                    ),
                }
            )
            seen_unit_ids.add(unit_id)

    if gold_units is not None:
        for unit_id, gold in gold_units.items():
            if unit_id in seen_unit_ids:
                continue
            score_payload = verifier_scores.get(unit_id, {})
            if not isinstance(score_payload, dict):
                score_payload = {}
            out.append(
                {
                    "unit_id": unit_id,
                    "text": str(gold.get("text", "")),
                    "entailment": float(score_payload.get("entailment", 0.0)),
                    "contradiction": float(score_payload.get("contradiction", 0.0)),
                    "supported": bool(gold.get("supported", False)),
                }
            )
    return out


def _empty_metrics() -> dict[str, Any]:
    return {
        "n_examples": 0,
        "kept_units": 0,
        "dropped_units": 0,
        "abstain_units": 0,
        "total_units": 0,
        "n_pairs_sum": 0,
        "rerank_pairs_sum": 0,
        "verifier_cost_sum": 0,
        "reranker_cost_sum": 0,
        "total_seconds_samples": [],
        "unsupported_kept": 0,
        "accepted_gold_total": 0,
        "gold_available_rows": 0,
        "coverage_avg_score_sum": 0.0,
        "coverage_rows": 0,
        "gold_coverage_recall_sum": 0.0,
        "gold_coverage_rows": 0,
        "reward_total_sum": 0.0,
        "reward_avg_sum": 0.0,
        "reward_hallucination_rate_sum": 0.0,
        "reward_abstention_rate_sum": 0.0,
        "reward_rows": 0,
        "conformal_examples": 0,
        "conformal_threshold": None,
        "conformal_meta": None,
        "conformal_score_min": None,
        "conformal_score_max": None,
        "conformal_band_hit_count": 0,
        "conformal_decision_counts": {"accept": 0, "reject": 0, "abstain": 0},
        "budget_examples": 0,
        "budget_requested_max_pairs": None,
        "budget_effective_pairs_sum": 0,
        "budget_effective_topk_sum": 0,
        "budget_high_risk_pairs_sum": 0,
        "budget_low_risk_pairs_sum": 0,
        "planned_pairs_sum": 0,
        "evaluated_pairs_sum": 0,
        "pruned_pairs_sum": 0,
        "drift_samples": 0,
        "drift_flagged_count": 0,
        "drift_ks_statistic_sum": 0.0,
    }


def _accumulate_metrics(
    *,
    metrics: dict[str, Any],
    result: dict[str, Any],
    trace: dict[str, Any],
    gold_units: dict[str, dict[str, Any]] | None,
) -> None:
    metrics["n_examples"] += 1
    kept = int(result.get("stats", {}).get("kept_units", 0))
    dropped = int(result.get("stats", {}).get("dropped_units", 0))
    metrics["kept_units"] += kept
    metrics["dropped_units"] += dropped

    n_units = int(trace.get("n_units", kept + dropped))
    metrics["total_units"] += n_units
    metrics["abstain_units"] += int(trace.get("conformal_abstain_units", 0))
    metrics["n_pairs_sum"] += int(trace.get("n_pairs", 0))
    metrics["rerank_pairs_sum"] += int(trace.get("rerank_pairs_scored", 0))
    metrics["verifier_cost_sum"] += int(trace.get("n_pairs", 0))
    metrics["reranker_cost_sum"] += int(trace.get("rerank_pairs_scored", 0))
    metrics["total_seconds_samples"].append(float(trace.get("total_seconds", 0.0)))
    drift_payload = trace.get("distribution_drift")
    if isinstance(drift_payload, dict):
        ks_stat = drift_payload.get("ks_statistic")
        flagged = drift_payload.get("drift_flagged")
        if isinstance(ks_stat, (int, float)):
            metrics["drift_samples"] += 1
            metrics["drift_ks_statistic_sum"] += float(ks_stat)
            if bool(flagged):
                metrics["drift_flagged_count"] += 1
    stats = result.get("stats", {})
    if isinstance(stats, dict):
        if "coverage_avg_score" in stats:
            metrics["coverage_avg_score_sum"] += float(stats.get("coverage_avg_score", 0.0))
            metrics["coverage_rows"] += 1
        if "reward_total" in stats:
            metrics["reward_total_sum"] += float(stats.get("reward_total", 0.0))
            metrics["reward_rows"] += 1
        if "reward_avg" in stats:
            metrics["reward_avg_sum"] += float(stats.get("reward_avg", 0.0))
        if "reward_hallucination_rate" in stats:
            metrics["reward_hallucination_rate_sum"] += float(
                stats.get("reward_hallucination_rate", 0.0)
            )
        if "reward_abstention_rate" in stats:
            metrics["reward_abstention_rate_sum"] += float(stats.get("reward_abstention_rate", 0.0))
        if "budget_active" in stats:
            metrics["budget_examples"] += 1 if bool(stats.get("budget_active", False)) else 0
            requested = stats.get("requested_budget_max_pairs")
            if requested is not None:
                metrics["budget_requested_max_pairs"] = int(requested)
            metrics["budget_effective_pairs_sum"] += int(stats.get("effective_budget_max_pairs", 0))
            metrics["budget_effective_topk_sum"] += int(stats.get("effective_topk_per_unit", 0))
            metrics["budget_high_risk_pairs_sum"] += int(
                stats.get("pairs_allocated_to_high_risk_units", 0)
            )
            metrics["budget_low_risk_pairs_sum"] += int(
                stats.get("pairs_allocated_to_low_risk_units", 0)
            )
        metrics["planned_pairs_sum"] += int(stats.get("planned_pairs_total", trace.get("n_pairs", 0)))
        metrics["evaluated_pairs_sum"] += int(
            stats.get("evaluated_pairs_total", trace.get("n_pairs", 0))
        )
        metrics["pruned_pairs_sum"] += int(
            stats.get(
                "pruned_pairs_total",
                max(
                    0,
                    int(stats.get("planned_pairs_total", trace.get("n_pairs", 0)))
                    - int(stats.get("evaluated_pairs_total", trace.get("n_pairs", 0))),
                ),
            )
        )

    if gold_units is not None:
        metrics["gold_available_rows"] += 1
        used_evidence_payload = result.get("used_evidence", {})
        used_evidence = (
            {
                str(unit_id): [str(evidence_id) for evidence_id in evidence_ids]
                for unit_id, evidence_ids in used_evidence_payload.items()
                if isinstance(evidence_ids, list)
            }
            if isinstance(used_evidence_payload, dict)
            else {}
        )
        kept_ids = [
            str(row.get("unit_id"))
            for row in result.get("verified_extract", [])
            if isinstance(row, dict)
        ]
        for unit_id in kept_ids:
            gold = gold_units.get(unit_id)
            if gold is None:
                continue
            metrics["accepted_gold_total"] += 1
            if not bool(gold.get("supported", False)):
                metrics["unsupported_kept"] += 1
        for unit_id, gold in gold_units.items():
            if not bool(gold.get("supported", False)):
                continue
            relevant_ids = [str(item) for item in gold.get("relevant_evidence_ids", [])]
            if not relevant_ids:
                continue
            relevant_set = set(relevant_ids)
            used_relevant = relevant_set.intersection(used_evidence.get(unit_id, []))
            metrics["gold_coverage_recall_sum"] += float(len(used_relevant)) / float(len(relevant_set))
            metrics["gold_coverage_rows"] += 1

    conformal_payload = result.get("conformal")
    if isinstance(conformal_payload, dict):
        metrics["conformal_examples"] += 1
        metrics["conformal_threshold"] = float(conformal_payload.get("threshold", 0.0))
        raw_meta = conformal_payload.get("meta")
        if isinstance(raw_meta, dict):
            metrics["conformal_meta"] = dict(raw_meta)
        score_min = conformal_payload.get("score_min")
        score_max = conformal_payload.get("score_max")
        if isinstance(score_min, (int, float)):
            current = metrics["conformal_score_min"]
            value = float(score_min)
            metrics["conformal_score_min"] = value if current is None else min(float(current), value)
        if isinstance(score_max, (int, float)):
            current = metrics["conformal_score_max"]
            value = float(score_max)
            metrics["conformal_score_max"] = value if current is None else max(float(current), value)
        metrics["conformal_band_hit_count"] += int(conformal_payload.get("band_hit_count", 0))
        raw_counts = conformal_payload.get("decision_counts", {})
        if isinstance(raw_counts, dict):
            for key in ("accept", "reject", "abstain"):
                metrics["conformal_decision_counts"][key] += int(raw_counts.get(key, 0))


def _finalize_metrics(*, metrics: dict[str, Any]) -> dict[str, Any]:
    samples = [float(x) for x in metrics["total_seconds_samples"]]
    p50, p95 = _percentiles(samples)
    unsupported_claim_rate: float | None = None
    if metrics["gold_available_rows"] > 0:
        denom = int(metrics["accepted_gold_total"])
        unsupported_claim_rate = (
            float(metrics["unsupported_kept"]) / float(denom) if denom > 0 else None
        )

    verifier_cost = int(metrics["verifier_cost_sum"])
    reranker_cost = int(metrics["reranker_cost_sum"])
    cost_proxy = verifier_cost + reranker_cost
    abstention_rate = (
        float(metrics["abstain_units"]) / float(metrics["total_units"])
        if metrics["total_units"] > 0
        else 0.0
    )
    n_examples = int(metrics["n_examples"])
    denom_examples = float(n_examples) if n_examples > 0 else 1.0
    coverage_rows = int(metrics["coverage_rows"])
    gold_coverage_rows = int(metrics["gold_coverage_rows"])
    reward_rows = int(metrics["reward_rows"])
    coverage_denom = float(coverage_rows) if coverage_rows > 0 else 1.0
    gold_coverage_denom = float(gold_coverage_rows) if gold_coverage_rows > 0 else 1.0
    reward_denom = float(reward_rows) if reward_rows > 0 else 1.0
    reward_abstention_rate = (
        float(metrics["reward_abstention_rate_sum"]) / reward_denom if reward_rows > 0 else 0.0
    )
    budget_rows = int(metrics["budget_examples"])
    budget_denom = float(budget_rows) if budget_rows > 0 else 1.0
    return {
        "n_examples": n_examples,
        "kept_units": int(metrics["kept_units"]),
        "dropped_units": int(metrics["dropped_units"]),
        "abstention_rate": reward_abstention_rate if reward_rows > 0 else abstention_rate,
        "avg_coverage_score": (
            float(metrics["coverage_avg_score_sum"]) / coverage_denom if coverage_rows > 0 else None
        ),
        "gold_coverage_recall": (
            float(metrics["gold_coverage_recall_sum"]) / gold_coverage_denom
            if gold_coverage_rows > 0
            else None
        ),
        "reward_total": float(metrics["reward_total_sum"]) if reward_rows > 0 else None,
        "avg_reward": (
            float(metrics["reward_avg_sum"]) / reward_denom if reward_rows > 0 else None
        ),
        "hallucination_rate": (
            float(metrics["reward_hallucination_rate_sum"]) / reward_denom
            if reward_rows > 0
            else None
        ),
        "unsupported_claim_rate": unsupported_claim_rate,
        "p50_total_seconds": p50,
        "p95_total_seconds": p95,
        "verifier_calls_proxy": int(metrics["evaluated_pairs_sum"]),
        "verifier_cost": int(metrics["evaluated_pairs_sum"]),
        "reranker_cost": reranker_cost,
        "cost_proxy": int(metrics["evaluated_pairs_sum"]) + reranker_cost,
        "budget_active": bool(budget_rows > 0),
        "requested_budget_max_pairs": metrics["budget_requested_max_pairs"],
        "effective_budget_max_pairs": int(metrics["budget_effective_pairs_sum"]) if budget_rows > 0 else None,
        "effective_topk_per_unit": (
            int(round(float(metrics["budget_effective_topk_sum"]) / budget_denom))
            if budget_rows > 0
            else None
        ),
        "avg_pairs_per_unit": (
            float(metrics["evaluated_pairs_sum"]) / float(metrics["total_units"])
            if metrics["total_units"] > 0
            else 0.0
        ),
        "pairs_allocated_to_high_risk_units": int(metrics["budget_high_risk_pairs_sum"]),
        "pairs_allocated_to_low_risk_units": int(metrics["budget_low_risk_pairs_sum"]),
        "planned_pairs_total": int(metrics["planned_pairs_sum"]),
        "evaluated_pairs_total": int(metrics["evaluated_pairs_sum"]),
        "pruned_pairs_total": int(metrics["pruned_pairs_sum"]),
        "drift_flagged_count": int(metrics["drift_flagged_count"]),
        "drift_ks_statistic_mean": (
            float(metrics["drift_ks_statistic_sum"]) / float(metrics["drift_samples"])
            if int(metrics["drift_samples"]) > 0
            else None
        ),
    }


def _finalize_metrics_metadata(*, metrics: dict[str, Any]) -> dict[str, Any]:
    conformal_examples = int(metrics["conformal_examples"])
    if conformal_examples <= 0:
        return {}
    return {
        "conformal_threshold": metrics["conformal_threshold"],
        "conformal_meta": dict(metrics["conformal_meta"] or {}),
        "conformal_score_range": {
            "min": metrics["conformal_score_min"],
            "max": metrics["conformal_score_max"],
        },
        "conformal_band_hit_count": int(metrics["conformal_band_hit_count"]),
        "conformal_decision_counts": dict(metrics["conformal_decision_counts"]),
        "abstain_band_observed": bool(metrics["conformal_band_hit_count"] > 0),
    }


def _append_debug_row(
    *,
    path: Path,
    example_id: str,
    variant: str,
    row: dict[str, Any],
    result: dict[str, Any],
) -> None:
    payload = {
        "example_id": example_id,
        "variant": variant,
        "accept_threshold": result.get("accept_threshold"),
        "verifier_model_name": result.get("verifier_model_name"),
        "units": result.get("units", []),
        "gold_units": row.get("gold_units", row.get("gold_unit_labels")),
        "pool_candidates": result.get("pool_candidates", {}),
        "pre_rerank_candidates": result.get("pre_rerank_candidates", {}),
        "post_rerank_candidates": result.get("post_rerank_candidates"),
        "verification_pairs": result.get("verification_pairs", {}),
        "used_evidence": result.get("used_evidence", {}),
        "verifier_scores": result.get("verifier_scores", {}),
        "decisions": result.get("decisions", {}),
        "coverage": result.get("coverage", {}),
        "rewards": result.get("rewards", {}),
        "unit_risk_scores": result.get("unit_risk_scores", {}),
        "per_unit_pair_budget": result.get("per_unit_pair_budget", {}),
        "evaluated_pairs_count_per_unit": result.get("evaluated_pairs_count_per_unit", {}),
        "planned_pairs_total": result.get("stats", {}).get("planned_pairs_total"),
        "evaluated_pairs_total": result.get("stats", {}).get("evaluated_pairs_total"),
        "pruned_pairs_total": result.get("stats", {}).get("pruned_pairs_total"),
        "per_unit_pairs_before_budget": result.get("per_unit_pairs_before_budget", {}),
        "per_unit_pairs_after_budget": result.get("per_unit_pairs_after_budget", {}),
    }
    if "safe_answer_final_text" in result:
        payload["safe_answer_final_text"] = result["safe_answer_final_text"]
    if "safe_answer_summary" in result:
        payload["safe_answer_summary"] = result["safe_answer_summary"]
    if "safe_answer" in result:
        payload["safe_answer"] = result["safe_answer"]
    if "reranked_candidates" in result:
        payload["reranked_candidates"] = result["reranked_candidates"]
    if "conformal" in result:
        payload["conformal"] = result["conformal"]
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _percentiles(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    sorted_values = sorted(values)
    return _percentile(sorted_values, 0.50), _percentile(sorted_values, 0.95)


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    idx = int(round((len(sorted_values) - 1) * q))
    idx = max(0, min(idx, len(sorted_values) - 1))
    return float(sorted_values[idx])
