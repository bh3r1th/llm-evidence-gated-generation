"""Pure execution core for EGA pipeline stages."""

from __future__ import annotations

import time
from typing import Any

from ega.contract import PolicyConfig
from ega.interfaces import Verifier
from ega.enforcer import Enforcer
from ega.text_clean import clean_text
from ega.types import AnswerCandidate, EvidenceItem, EvidenceSet, Unit, VerificationScore
from ega.unitization import unitize_answer
from ega.v2.budget import BudgetConfig, BudgetPolicy
from ega.v2.conformal import ConformalCalibrator, ConformalState
from ega.v2.risk import extract_unit_risks
from ega.v2.reranker import EvidenceReranker


def run_core_pipeline(
    *,
    llm_summary_text: str,
    structured_candidate_payload: Any | None = None,
    evidence: EvidenceSet,
    unitizer_mode: str,
    policy_config: PolicyConfig,
    accept_threshold: float | None,
    scores_jsonl_path: str | None,
    verifier: Verifier | None,
    nli_model_name: str | None,
    nli_device: str,
    nli_dtype: str,
    topk_per_unit: int,
    max_pairs_total: int | None,
    max_evidence_per_request: int | None,
    max_batch_tokens: int | None,
    evidence_max_chars: int,
    evidence_max_sentences: int,
    reranker: EvidenceReranker | None,
    rerank_topk: int | None,
    conformal_state: ConformalState | None,
    budget_policy: BudgetPolicy | None,
    budget_config: BudgetConfig | None,
) -> dict[str, Any]:
    timings = {
        "unitize_seconds": 0.0,
        "verify_seconds": 0.0,
        "load_seconds": 0.0,
        "verify_compute_seconds": 0.0,
        "enforce_seconds": 0.0,
    }
    counts = {"n_units": 0, "n_evidence": 0, "n_pairs": 0}
    rerank_seconds = 0.0
    rerank_pairs_scored = 0
    conformal_threshold: float | None = None
    conformal_abstain_units = 0
    active_accept_threshold = (
        float(policy_config.threshold_entailment)
        if accept_threshold is None
        else float(accept_threshold)
    )
    budget_topk_per_unit: int | None = None
    budget_max_pairs_total: int | None = None
    budget_requested_max_pairs: int | None = None
    budget_unit_risk_scores: dict[str, float] | None = None
    budget_per_unit_pair_budget: dict[str, int] | None = None
    per_unit_pairs_before_budget: dict[str, int] | None = None
    per_unit_pairs_after_budget: dict[str, int] | None = None
    _ = (nli_model_name, nli_device, nli_dtype)
    verify_detail = {
        "preselect_seconds": 0.0,
        "tokenize_seconds": 0.0,
        "forward_seconds": 0.0,
        "post_seconds": 0.0,
        "num_batches": 0,
        "batch_size_mean": 0.0,
        "batch_size_max": 0,
        "seq_len_mean": 0.0,
        "seq_len_p50": 0.0,
        "seq_len_p95": 0.0,
        "tokens_total": 0,
        "device": None,
        "dtype": None,
        "amp_enabled": False,
        "compiled_enabled": False,
        "pairs_pruned_stage1": 0,
        "pairs_pruned_stage2": 0,
        "dtype_overridden": False,
        "evidence_truncated_frac": 0.0,
        "evidence_chars_mean_before": 0.0,
        "evidence_chars_mean_after": 0.0,
    }

    mode = "markdown_bullet" if unitizer_mode == "bullets" else unitizer_mode
    use_structured_candidate = mode == "structured_field"
    cleaned_summary = clean_text(llm_summary_text)
    cleaned_evidence = EvidenceSet(
        items=[
            EvidenceItem(id=item.id, text=clean_text(item.text), metadata=dict(item.metadata))
            for item in evidence.items
        ]
    )
    counts["n_evidence"] = len(cleaned_evidence.items)

    unitize_t0 = time.perf_counter()
    if use_structured_candidate:
        if not isinstance(structured_candidate_payload, (dict, list)):
            raise ValueError(
                "structured_field mode requires structured_candidate_payload as a dict or list."
            )
        unitize_source: Any = structured_candidate_payload
    else:
        unitize_source = cleaned_summary

    raw_candidate = unitize_answer(unitize_source, mode=mode)
    candidate = AnswerCandidate(
        raw_answer_text=cleaned_summary,
        units=[
            Unit(
                id=unit.id,
                text=clean_text(unit.text),
                metadata=dict(unit.metadata),
                source_ids=list(unit.source_ids) if unit.source_ids is not None else None,
            )
            for unit in raw_candidate.units
        ],
    )
    timings["unitize_seconds"] = time.perf_counter() - unitize_t0
    counts["n_units"] = len(candidate.units)
    model_name: str
    active_topk_per_unit = topk_per_unit
    active_max_pairs_total = max_pairs_total
    pool_candidates: dict[str, list[str]] = _build_pool_candidates(
        units=candidate.units,
        evidence_ids=[item.id for item in cleaned_evidence.items],
        topk=int(active_topk_per_unit),
    )
    initial_pool_candidates = {
        str(unit_id): [str(evidence_id) for evidence_id in ids]
        for unit_id, ids in pool_candidates.items()
    }
    reranked_candidates: dict[str, list[str]] | None = None
    verify_evidence = cleaned_evidence
    candidate_stage = "retrieval"

    if scores_jsonl_path:
        from ega.providers.jsonl_scores import JsonlScoresProvider

        verify_compute_t0 = time.perf_counter()
        model_name = "precomputed_scores_jsonl"
        scores = JsonlScoresProvider(path=scores_jsonl_path).load_scores(
            candidate=candidate,
            evidence=cleaned_evidence,
        )
        timings["load_seconds"] = 0.0
        timings["verify_compute_seconds"] = time.perf_counter() - verify_compute_t0
        timings["verify_seconds"] = timings["load_seconds"] + timings["verify_compute_seconds"]
        counts["n_pairs"] = 0
    else:
        if budget_policy is not None and budget_config is not None:
            risk_features = extract_unit_risks(units=candidate.units, evidence=cleaned_evidence)
            base_max_pairs = (
                int(max_pairs_total)
                if max_pairs_total is not None
                else int(max(0, len(candidate.units) * int(topk_per_unit)))
            )
            budget_decision = budget_policy.choose(
                units=candidate.units,
                evidence=cleaned_evidence,
                base_params={
                    "topk_per_unit": int(topk_per_unit),
                    "max_pairs_total": base_max_pairs,
                    "verifier_name": _verifier_runtime_name(verifier),
                },
                risk_features=risk_features,
                budget=budget_config,
            )
            active_topk_per_unit = int(budget_decision.topk_per_unit)
            active_max_pairs_total = int(budget_decision.max_pairs_total)
            budget_topk_per_unit = active_topk_per_unit
            budget_max_pairs_total = active_max_pairs_total
            budget_requested_max_pairs = (
                None if budget_config.max_pairs_total is None else int(budget_config.max_pairs_total)
            )
            budget_unit_risk_scores = {
                str(unit_id): float(value) for unit_id, value in risk_features.items()
            }
            if budget_decision.per_unit_pair_budget is not None:
                budget_per_unit_pair_budget = {
                    str(unit_id): int(value)
                    for unit_id, value in budget_decision.per_unit_pair_budget.items()
                }

        pool_candidates = _build_pool_candidates(
            units=candidate.units,
            evidence_ids=[item.id for item in cleaned_evidence.items],
            topk=int(active_topk_per_unit),
        )
        if reranker is not None:
            rerank_k = int(rerank_topk) if rerank_topk is not None else int(active_topk_per_unit)
            verify_evidence, rerank_pairs_scored, rerank_seconds, pool_candidates = (
                _apply_reranker_to_evidence(
                    reranker=reranker,
                    units=candidate.units,
                    evidence=cleaned_evidence,
                    topk=max(0, rerank_k),
                )
            )
            counts["n_evidence"] = len(verify_evidence.items)
            reranked_candidates = {
                str(unit_id): [str(evidence_id) for evidence_id in ids]
                for unit_id, ids in pool_candidates.items()
            }
            candidate_stage = "rerank"
        per_unit_pairs_before_budget = {
            str(unit.id): int(len(pool_candidates.get(unit.id, []))) for unit in candidate.units
        }
        if budget_per_unit_pair_budget is not None:
            pool_candidates = _apply_per_unit_pair_budget(
                units=candidate.units,
                pool_candidates=pool_candidates,
                per_unit_pair_budget=budget_per_unit_pair_budget,
            )
        per_unit_pairs_after_budget = {
            str(unit.id): int(len(pool_candidates.get(unit.id, []))) for unit in candidate.units
        }

        if verifier is None:
            raise ValueError("verifier is required when scores_jsonl_path is not provided")
        model_name = str(verifier.model_name or "")
        verify_compute_t0 = time.perf_counter()
        scores, trace_payload = _verify_with_candidate_mapping(
            verifier=verifier,
            candidate=candidate,
            evidence=verify_evidence,
            pool_candidates=pool_candidates,
        )
        timings["verify_compute_seconds"] = time.perf_counter() - verify_compute_t0
        timings["verify_seconds"] = timings["load_seconds"] + timings["verify_compute_seconds"]
        if trace_payload:
            verify_detail.update(trace_payload)
            counts["n_pairs"] = int(trace_payload.get("n_pairs_scored", 0))
        else:
            counts["n_pairs"] = sum(
                len(score.raw.get("per_item_probs", []))
                for score in scores
                if isinstance(score.raw, dict)
            )

    if scores_jsonl_path:
        pool_candidates = _build_pool_candidates(
            units=candidate.units,
            evidence_ids=[item.id for item in verify_evidence.items],
            topk=int(active_topk_per_unit),
        )

    if conformal_state is not None:
        conformal_threshold = float(conformal_state.threshold)
        scores, conformal_abstain_units, conformal_gate_meta = _apply_conformal_gate(
            scores=scores,
            state=conformal_state,
        )
    else:
        conformal_gate_meta = None
    timings["verify_compute_seconds"] = (
        float(verify_detail["preselect_seconds"])
        + float(verify_detail["tokenize_seconds"])
        + float(verify_detail["forward_seconds"])
        + float(verify_detail["post_seconds"])
    )

    enforce_t0 = time.perf_counter()
    result = Enforcer(
        config=PolicyConfig(
            threshold_entailment=active_accept_threshold,
            max_contradiction=float(policy_config.max_contradiction),
            partial_allowed=bool(policy_config.partial_allowed),
        )
    ).enforce(
        candidate=candidate,
        evidence=cleaned_evidence,
        scores=scores,
    )
    timings["enforce_seconds"] = time.perf_counter() - enforce_t0

    kept_ids = set(result.kept_units)
    verified_units = [unit for unit in candidate.units if unit.id in kept_ids]
    dropped_units = [unit for unit in candidate.units if unit.id not in kept_ids]
    decisions, failure_class_by_unit = _normalize_unit_decisions(
        units=candidate.units,
        result=result,
        scores=scores,
    )

    return {
        "units": candidate.units,
        "scores": scores,
        "decisions": decisions,
        "failure_class_by_unit": failure_class_by_unit,
        "verified_units": verified_units,
        "dropped_units": dropped_units,
        "intermediate_stats": {
            "timings": timings,
            "counts": counts,
            "cleaned_summary": cleaned_summary,
            "cleaned_evidence": cleaned_evidence,
            "candidate": candidate,
            "result": result,
            "active_accept_threshold": active_accept_threshold,
            "model_name": model_name,
            "active_topk_per_unit": active_topk_per_unit,
            "active_max_pairs_total": active_max_pairs_total,
            "pool_candidates": pool_candidates,
            "initial_pool_candidates": initial_pool_candidates,
            "reranked_candidates": reranked_candidates,
            "verify_evidence": verify_evidence,
            "candidate_stage": candidate_stage,
            "budget_topk_per_unit": budget_topk_per_unit,
            "budget_max_pairs_total": budget_max_pairs_total,
            "budget_requested_max_pairs": budget_requested_max_pairs,
            "budget_unit_risk_scores": budget_unit_risk_scores,
            "budget_per_unit_pair_budget": budget_per_unit_pair_budget,
            "per_unit_pairs_before_budget": per_unit_pairs_before_budget,
            "per_unit_pairs_after_budget": per_unit_pairs_after_budget,
            "verify_detail": verify_detail,
            "rerank_seconds": rerank_seconds,
            "rerank_pairs_scored": rerank_pairs_scored,
            "conformal_threshold": conformal_threshold,
            "conformal_abstain_units": conformal_abstain_units,
            "conformal_gate_meta": conformal_gate_meta,
            "conformal_state": conformal_state,
        },
    }


def _apply_conformal_gate(
    *,
    scores: list[VerificationScore],
    state: ConformalState,
) -> tuple[list[VerificationScore], int, dict[str, Any]]:
    calibrator = ConformalCalibrator()
    abstain_count = 0
    gated: list[VerificationScore] = []
    margin = float(state.meta.get("abstain_margin", 0.02))
    threshold = float(state.threshold)
    decision_counts = {"accept": 0, "reject": 0, "abstain": 0}
    score_values: list[float] = []
    band_hit_count = 0
    for score in scores:
        score_value = float(score.entailment)
        score_values.append(score_value)
        gate_decision = calibrator.gate(score.entailment, state)
        raw = dict(score.raw)
        raw["conformal_score"] = score_value
        raw["conformal_gate"] = gate_decision
        raw["conformal_threshold"] = threshold
        raw["conformal_abstain_margin"] = margin
        raw["conformal_abstain_band"] = [
            max(0.0, threshold - margin),
            min(1.0, threshold + margin),
        ]
        if abs(score_value - threshold) <= margin:
            band_hit_count += 1
        decision_counts[gate_decision] += 1
        if gate_decision == "accept":
            gated.append(
                VerificationScore(
                    unit_id=score.unit_id,
                    entailment=score.entailment,
                    contradiction=score.contradiction,
                    neutral=score.neutral,
                    label=score.label,
                    raw=raw,
                )
            )
            continue
        if gate_decision == "abstain":
            abstain_count += 1
        raw["has_contradiction"] = True
        gated.append(
            VerificationScore(
                unit_id=score.unit_id,
                entailment=0.0,
                contradiction=1.0,
                neutral=0.0,
                label=f"conformal_{gate_decision}",
                raw=raw,
            )
        )
    return gated, abstain_count, {
        "score_min": min(score_values) if score_values else None,
        "score_max": max(score_values) if score_values else None,
        "abstain_margin": margin,
        "abstain_band": [
            max(0.0, threshold - margin),
            min(1.0, threshold + margin),
        ],
        "band_hit_count": int(band_hit_count),
        "decision_counts": decision_counts,
    }


def _apply_reranker_to_evidence(
    *,
    reranker: EvidenceReranker,
    units: list[Unit],
    evidence: EvidenceSet,
    topk: int,
) -> tuple[EvidenceSet, int, float, dict[str, list[str]]]:
    evidence_ids = [item.id for item in evidence.items]
    candidates = {unit.id: evidence_ids[:topk] for unit in units}
    pairs_scored = sum(len(ids) for ids in candidates.values())
    rerank_t0 = time.perf_counter()
    rerank_with_stats = getattr(reranker, "rerank_with_stats", None)
    if callable(rerank_with_stats):
        reranked, stats = rerank_with_stats(
            units=units,
            evidence=evidence,
            candidates=candidates,
            topk=topk,
        )
        pairs_scored = int(getattr(stats, "n_pairs_scored", pairs_scored))
        rerank_seconds = float(getattr(stats, "seconds", 0.0))
    else:
        reranked = reranker.rerank(
            units=units,
            evidence=evidence,
            candidates=candidates,
            topk=topk,
        )
        rerank_seconds = time.perf_counter() - rerank_t0
        get_last_stats = getattr(reranker, "get_last_stats", None)
        if callable(get_last_stats):
            stats = get_last_stats()
            pairs_scored = int(getattr(stats, "n_pairs_scored", pairs_scored))
            rerank_seconds = float(getattr(stats, "seconds", rerank_seconds))

    id_to_item = {item.id: item for item in evidence.items}
    selected_ids: list[str] = []
    seen: set[str] = set()
    for unit in units:
        for evidence_id in reranked.get(unit.id, []):
            if evidence_id in id_to_item and evidence_id not in seen:
                seen.add(evidence_id)
                selected_ids.append(evidence_id)

    sanitized_reranked = _sanitize_pool_candidates(units=units, raw=reranked)
    if not selected_ids:
        return evidence, pairs_scored, rerank_seconds, sanitized_reranked

    reduced = EvidenceSet(items=[id_to_item[evidence_id] for evidence_id in selected_ids])
    return reduced, pairs_scored, rerank_seconds, sanitized_reranked


def _build_pool_candidates(
    *,
    units: list[Unit],
    evidence_ids: list[str],
    topk: int,
) -> dict[str, list[str]]:
    capped = evidence_ids[: max(0, int(topk))]
    return {unit.id: list(capped) for unit in units}


def _apply_per_unit_pair_budget(
    *,
    units: list[Unit],
    pool_candidates: dict[str, list[str]],
    per_unit_pair_budget: dict[str, int],
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for unit in units:
        cap = max(0, int(per_unit_pair_budget.get(unit.id, len(pool_candidates.get(unit.id, [])))))
        out[unit.id] = list(pool_candidates.get(unit.id, []))[:cap]
    return out


def _sanitize_pool_candidates(
    *,
    units: list[Unit],
    raw: dict[str, list[str]],
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for unit in units:
        rows = raw.get(unit.id, [])
        seen: set[str] = set()
        cleaned: list[str] = []
        for evidence_id in rows:
            eid = str(evidence_id)
            if eid in seen:
                continue
            seen.add(eid)
            cleaned.append(eid)
        out[unit.id] = cleaned
    return out


def _verifier_runtime_name(verifier: Verifier | None) -> str:
    if verifier is None:
        return "unknown"
    model_name = str(getattr(verifier, "model_name", "") or "").strip()
    return model_name or verifier.__class__.__name__


def _verify_with_candidate_mapping(
    *,
    verifier: Verifier,
    candidate: AnswerCandidate,
    evidence: EvidenceSet,
    pool_candidates: dict[str, list[str]],
) -> tuple[list[VerificationScore], dict[str, Any]]:
    id_to_item = {item.id: item for item in evidence.items}
    scores: list[VerificationScore] = []
    trace_totals: dict[str, Any] = {
        "preselect_seconds": 0.0,
        "tokenize_seconds": 0.0,
        "forward_seconds": 0.0,
        "post_seconds": 0.0,
        "num_batches": 0,
        "batch_size_mean": 0.0,
        "batch_size_max": 0,
        "seq_len_mean": 0.0,
        "seq_len_p50": 0.0,
        "seq_len_p95": 0.0,
        "tokens_total": 0,
        "device": None,
        "dtype": None,
        "amp_enabled": False,
        "compiled_enabled": False,
        "pairs_pruned_stage1": 0,
        "pairs_pruned_stage2": 0,
        "dtype_overridden": False,
        "n_pairs_scored": 0,
        "evaluated_pairs_count_per_unit": {},
        "evidence_truncated_frac": 0.0,
        "evidence_chars_mean_before": 0.0,
        "evidence_chars_mean_after": 0.0,
    }
    weighted_trace_keys = (
        "batch_size_mean",
        "seq_len_mean",
        "seq_len_p50",
        "seq_len_p95",
        "evidence_truncated_frac",
        "evidence_chars_mean_before",
        "evidence_chars_mean_after",
    )
    weighted_counts = {key: 0 for key in weighted_trace_keys}
    bool_trace_keys = ("amp_enabled", "compiled_enabled", "dtype_overridden")

    for unit in candidate.units:
        candidate_ids = [eid for eid in pool_candidates.get(unit.id, []) if eid in id_to_item]
        unit_evidence = EvidenceSet(items=[id_to_item[eid] for eid in candidate_ids])
        unit_scores = verifier.verify([unit], unit_evidence)
        if not unit_scores:
            raise ValueError("verifier returned no scores for unit verification")
        if len(unit_scores) != 1:
            raise ValueError("verifier must return exactly one score per unit")
        scores.append(unit_scores[0])
        unit_trace = verifier.get_last_verify_trace()
        if not unit_trace:
            continue
        pair_count = int(unit_trace.get("n_pairs_scored", 0))
        trace_totals["n_pairs_scored"] += pair_count
        trace_totals["evaluated_pairs_count_per_unit"][unit.id] = pair_count
        for key in (
            "preselect_seconds",
            "tokenize_seconds",
            "forward_seconds",
            "post_seconds",
            "num_batches",
            "pairs_pruned_stage1",
            "pairs_pruned_stage2",
            "tokens_total",
        ):
            trace_totals[key] += float(unit_trace.get(key, 0.0)) if "seconds" in key else int(
                unit_trace.get(key, 0)
            )
        trace_totals["batch_size_max"] = max(
            int(trace_totals["batch_size_max"]),
            int(unit_trace.get("batch_size_max", 0)),
        )
        for key in weighted_trace_keys:
            if key in unit_trace and unit_trace[key] is not None:
                trace_totals[key] += float(unit_trace[key]) * max(pair_count, 1)
                weighted_counts[key] += max(pair_count, 1)
        for key in ("device", "dtype"):
            if trace_totals[key] in (None, "") and unit_trace.get(key) not in (None, ""):
                trace_totals[key] = unit_trace.get(key)
        for key in bool_trace_keys:
            trace_totals[key] = bool(trace_totals[key] or unit_trace.get(key, False))

    for key in weighted_trace_keys:
        weight = weighted_counts[key]
        trace_totals[key] = (
            float(trace_totals[key]) / float(weight) if weight > 0 else float(trace_totals[key])
        )
    return scores, trace_totals


def _normalize_unit_decisions(
    *,
    units: list[Unit],
    result: Any,
    scores: list[VerificationScore],
) -> tuple[dict[str, str], dict[str, str]]:
    kept_ids = set(getattr(result, "kept_units", []))
    score_by_id = {score.unit_id: score for score in scores}
    decisions: dict[str, str] = {}
    failure_class_by_unit: dict[str, str] = {}

    def _missing_in_source(score: VerificationScore | None) -> bool:
        if score is None:
            return True
        raw = score.raw if isinstance(score.raw, dict) else {}
        chosen_evidence_id = raw.get("chosen_evidence_id")
        if chosen_evidence_id in (None, "", []):
            return True
        per_item_probs = raw.get("per_item_probs")
        if not isinstance(per_item_probs, list):
            return False
        if not per_item_probs:
            return False
        chosen_id = str(chosen_evidence_id)
        return not any(
            isinstance(row, dict) and str(row.get("evidence_id")) == chosen_id
            for row in per_item_probs
        )

    def _extract_chosen_probabilities(score: VerificationScore | None) -> tuple[float, float]:
        if score is None:
            return 0.0, 0.0
        raw = score.raw if isinstance(score.raw, dict) else {}
        entailment = float(getattr(score, "entailment", 0.0))
        contradiction = float(getattr(score, "contradiction", 0.0))
        chosen_evidence_id = raw.get("chosen_evidence_id")
        per_item_probs = raw.get("per_item_probs")
        if chosen_evidence_id in (None, "", []) or not isinstance(per_item_probs, list):
            return entailment, contradiction
        chosen_id = str(chosen_evidence_id)
        for row in per_item_probs:
            if not isinstance(row, dict) or str(row.get("evidence_id")) != chosen_id:
                continue
            row_entailment = row.get("entailment", row.get("p_entailment", entailment))
            row_contradiction = row.get("contradiction", row.get("p_contradiction", contradiction))
            return float(row_entailment), float(row_contradiction)
        return entailment, contradiction

    def _unsupported_claim(score: VerificationScore | None) -> bool:
        if score is None:
            return False
        entailment, contradiction = _extract_chosen_probabilities(score)
        return entailment <= 0.35 and contradiction >= 0.5

    for unit in units:
        if unit.id in kept_ids:
            decisions[unit.id] = "accept"
            failure_class_by_unit[unit.id] = "SUPPORTED"
            continue
        score = score_by_id.get(unit.id)
        raw = score.raw if (score is not None and isinstance(score.raw, dict)) else {}
        label = str(score.label).lower() if score is not None else ""
        gate = str(raw.get("conformal_gate", "")).lower()
        if gate == "abstain" or "abstain" in label:
            decisions[unit.id] = "abstain"
        else:
            decisions[unit.id] = "reject"
        if _missing_in_source(score):
            failure_class_by_unit[unit.id] = "MISSING_IN_SOURCE"
        elif _unsupported_claim(score):
            failure_class_by_unit[unit.id] = "UNSUPPORTED_CLAIM"
        else:
            failure_class_by_unit[unit.id] = "AMBIGUOUS_SOURCE"
    return decisions, failure_class_by_unit
