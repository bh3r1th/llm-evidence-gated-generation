"""File-driven end-to-end pipeline runner for provided summary inputs."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from ega.config import ADAPTER_MODE, STRICT_PASSTHROUGH_MODE, normalize_downstream_compatibility_mode
from ega.contract import PolicyConfig
from ega.interfaces import Verifier
from ega.core.correction import CorrectionConfig, run_correction_loop
from ega.core.pipeline_core import (
    run_core_pipeline,
    _apply_conformal_gate,
    _apply_reranker_to_evidence,
    _build_pool_candidates,
    _apply_per_unit_pair_budget,
    _sanitize_pool_candidates,
    _verify_with_candidate_mapping,
    _normalize_unit_decisions,
    UnitDecisionRecord,
    make_unit_authority_decision,
)
from ega.enforcer import Enforcer
from ega.polish.gate import PolishGateConfig, apply_polish_gate
from ega.polish.types import PolishedUnit
from ega.providers.jsonl_scores import JsonlScoresProvider
from ega.text_clean import clean_text
from ega.types import AnswerCandidate, EvidenceItem, EvidenceSet, Unit, VerificationScore
from ega.unitization import unitize_answer
from ega.v2.budget import BudgetConfig, BudgetPolicy
from ega.v2.conformal import ConformalCalibrator, ConformalState
from ega.v2.coverage import CoverageConfig, CoverageResult, EvidenceCoverageAnalyzer
from ega.v2.rewards import RewardComputer, RewardConfig
from ega.v2.render import SafeAnswerRenderer
from ega.v2.risk import extract_unit_risks
from ega.v2.reranker import EvidenceReranker
from ega.verifiers.adapter import LegacyVerifierAdapter


def _build_audit_entries(
    *,
    scores: list[VerificationScore],
    decisions: dict[str, str],
    conformal_state: ConformalState | None,
) -> list[dict[str, Any]]:
    """Build UnitAuditEntry dicts from pipeline scores for V4 schema output."""
    entries: list[dict[str, Any]] = []
    for score in scores:
        raw = score.raw if isinstance(score.raw, dict) else {}
        unit_id = score.unit_id
        raw_score = float(score.entailment)
        final_decision = str(decisions.get(unit_id, "reject"))
        if conformal_state is not None:
            authority = str(raw.get("conformal_authority", "conformal"))
            reason_code = str(raw.get("conformal_reason_code", "CONFORMAL_ACCEPT"))
            fallback_reason: str | None = raw.get("conformal_fallback_reason") or None
            conformal_decision: str | None = score.conformal_decision
            conformal_threshold: float | None = float(conformal_state.threshold)
            conformal_band_width: float | None = float(conformal_state.band_width)
            cal_min = conformal_state.calibration_score_min
            cal_max = conformal_state.calibration_score_max
            calibration_range: tuple[float, float] | None = (
                (float(cal_min), float(cal_max))
                if cal_min is not None and cal_max is not None
                else None
            )
        else:
            authority = "threshold"
            reason_code = "THRESHOLD_ACCEPT" if final_decision == "accept" else "THRESHOLD_REJECT"
            fallback_reason = None
            conformal_decision = None
            conformal_threshold = None
            conformal_band_width = None
            calibration_range = None
        entries.append({
            "unit_id": unit_id,
            "authority": authority,
            "raw_score": raw_score,
            "conformal_decision": conformal_decision,
            "final_decision": final_decision,
            "reason_code": reason_code,
            "conformal_threshold": conformal_threshold,
            "conformal_band_width": conformal_band_width,
            "calibration_range": calibration_range,
            "fallback_reason": fallback_reason,
        })
    return entries


def _assemble_v4_response(
    *,
    output_mode: str,
    internal_payload_status: str,
    payload_action: str,
    verified_text: str,
    verified_extract: list[dict[str, Any]],
    decisions: dict[str, str],
    failure_class_by_unit: dict[str, str] | None,
    audit_entries: list[dict[str, Any]],
    tracking_id: str | None,
    route_status: str,
    route_reason: str | None,
    pending_expires_at: str | None,
    unit_texts: dict[str, str],
) -> dict[str, Any]:
    """Assemble the V4 schema response object for strict, adapter, or pending mode."""
    v4_status = "PENDING" if internal_payload_status in {"REPAIR", "PENDING"} else internal_payload_status

    if v4_status == "PENDING":
        return {
            "payload_status": "PENDING",
            "tracking_id": tracking_id,
            "route_reason": str(route_reason or "BOUNDED_REPAIR"),
            "pending_expires_at": pending_expires_at,
        }

    if output_mode == "adapter":
        accepted_fields: dict[str, Any] = {}
        rejected_fields: dict[str, Any] = {}
        field_status_map: dict[str, Any] = {}
        for entry in audit_entries:
            uid = entry["unit_id"]
            dec = entry["final_decision"]
            status = (
                "accepted" if dec == "accept"
                else ("abstained" if dec == "abstain" else "rejected")
            )
            rejection_reason: str | None = None
            if status != "accepted":
                fc = str((failure_class_by_unit or {}).get(uid, "")).upper() or None
                rejection_reason = fc
            field_status_map[uid] = {
                "field_path": uid,
                "status": status,
                "score": entry["raw_score"],
                "authority": entry["authority"],
                "rejection_reason": rejection_reason,
            }
            unit_text = unit_texts.get(uid, "")
            if status == "accepted":
                accepted_fields[uid] = unit_text
            else:
                rejected_fields[uid] = unit_text
        return {
            "payload_status": v4_status,
            "accepted_fields": accepted_fields,
            "rejected_fields": rejected_fields,
            "field_status_map": field_status_map,
            "tracking_id": tracking_id,
        }

    # strict mode
    if v4_status == "ACCEPT":
        return {
            "payload_status": "ACCEPT",
            "verified_text": verified_text,
            "verified_units": list(verified_extract),
            "tracking_id": tracking_id,
            "audit": list(audit_entries),
        }

    # REJECT
    failed_unit_ids = [uid for uid, dec in decisions.items() if dec != "accept"]
    return {
        "payload_status": "REJECT",
        "rejection_reason": str(payload_action or "REJECT"),
        "failed_unit_ids": failed_unit_ids,
        "tracking_id": tracking_id,
        "route_status": route_status,
    }


def _read_summary_file(path: str | Path) -> str:
    p = Path(path)
    return clean_text(p.read_text(encoding="utf-8"))


def _read_evidence_json(path: str | Path) -> EvidenceSet:
    p = Path(path)
    payload = json.loads(p.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError("Evidence file must be a JSON list of {id,text,metadata} objects.")
    items: list[EvidenceItem] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Evidence item at index {idx} must be an object.")
        if "id" not in item or "text" not in item:
            raise ValueError(
                f"Evidence item at index {idx} must include 'id' and 'text' fields."
            )
        metadata = item.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"Evidence item at index {idx} has non-object metadata.")
        items.append(
            EvidenceItem(
                id=str(item["id"]),
                text=clean_text(str(item["text"])),
                metadata=metadata,
            )
        )
    return EvidenceSet(items=items)



def _derive_workflow_contract(
    *,
    payload_status: str,
    payload_action: str,
    candidate: AnswerCandidate,
    decisions_by_unit: dict[str, str],
) -> dict[str, Any]:
    def _build_tracking_id() -> str:
        tracking_context = {
            "payload_status": str(payload_status),
            "payload_action": str(payload_action),
            "units": [
                {"unit_id": str(unit.id), "text": str(unit.text)} for unit in candidate.units
            ],
            "decisions": {
                str(unit_id): str(decision)
                for unit_id, decision in sorted(decisions_by_unit.items(), key=lambda row: str(row[0]))
            },
        }
        digest = hashlib.sha256(
            json.dumps(tracking_context, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return f"ega4_{digest[:16]}"

    if payload_status == "ACCEPT":
        return {
            "workflow_status": "COMPLETED",
            "handoff_required": False,
            "handoff_reason": None,
            "tracking_id": None,
        }
    if payload_status in {"REPAIR", "PENDING"}:
        return {
            "workflow_status": "PENDING",
            "handoff_required": True,
            "handoff_reason": "BOUNDED_REPAIR",
            "tracking_id": _build_tracking_id(),
        }

    terminal_reject_actions = {"REJECT"}
    if payload_status == "REJECT" and payload_action in terminal_reject_actions:
        return {
            "workflow_status": "COMPLETED",
            "handoff_required": False,
            "handoff_reason": None,
            "tracking_id": None,
        }

    if payload_status == "REJECT":
        reason = str(payload_action or "REVIEW")
        return {
            "workflow_status": "PENDING",
            "handoff_required": True,
            "handoff_reason": reason,
            "tracking_id": _build_tracking_id(),
        }

    return {
        "workflow_status": "COMPLETED",
        "handoff_required": False,
        "handoff_reason": None,
        "tracking_id": None,
    }

def run_pipeline_request(
    *,
    llm_summary_text: str | None = None,
    llm_summary_file: str | Path | None = None,
    structured_candidate_payload: Any | None = None,
    evidence: EvidenceSet | None = None,
    evidence_json: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    if llm_summary_text is None:
        if llm_summary_file is None:
            raise ValueError("Provide llm_summary_text or llm_summary_file.")
        llm_summary_text = _read_summary_file(llm_summary_file)
    if evidence is None:
        if evidence_json is None:
            raise ValueError("Provide evidence or evidence_json.")
        evidence = _read_evidence_json(evidence_json)
    return run_pipeline(
        llm_summary_text=llm_summary_text,
        structured_candidate_payload=structured_candidate_payload,
        evidence=evidence,
        **kwargs,
    )


def run_pipeline(
    llm_summary_text: str,
    evidence: EvidenceSet,
    *,
    structured_candidate_payload: Any | None = None,
    unitizer_mode: str = "sentence",
    policy_config: PolicyConfig,
    accept_threshold: float | None = None,
    scores_jsonl_path: str | None = None,
    use_oss_nli: bool = False,
    verifier: Verifier | Any | None = None,
    nli_model_name: str | None = None,
    nli_device: str = "auto",
    nli_dtype: str = "auto",
    topk_per_unit: int = 12,
    max_pairs_total: int | None = 200,
    max_evidence_per_request: int | None = None,
    max_batch_tokens: int | None = None,
    evidence_max_chars: int = 800,
    evidence_max_sentences: int = 3,
    reranker: EvidenceReranker | None = None,
    rerank_topk: int | None = None,
    conformal_state_path: str | None = None,
    conformal_epsilon: float | None = None,
    budget_policy: BudgetPolicy | None = None,
    budget_config: BudgetConfig | None = None,
    coverage_config: CoverageConfig | None = None,
    reward_config: RewardConfig | None = None,
    emit_training_example_path: str | None = None,
    training_example_id: str | None = None,
    polished_json: dict[str, Any] | list[Any] | None = None,
    enable_polish_validation: bool = True,
    trace_out: str | None = None,
    render_safe_answer: bool = False,
    enable_correction: bool = False,
    max_retries: int = 1,
    correction_generator: Any | None = None,
    downstream_compatibility_mode: str = STRICT_PASSTHROUGH_MODE,
    output_mode: str = "strict",
    tracking_id: str | None = None,
    pending_expires_at: str | None = None,
) -> dict[str, Any]:
    """Run provided-summary -> unitize -> score -> gate -> optional polish validation."""
    total_t0 = time.perf_counter()
    timings = {
        "read_seconds": 0.0,
        "unitize_seconds": 0.0,
        "verify_seconds": 0.0,
        "load_seconds": 0.0,
        "verify_compute_seconds": 0.0,
        "enforce_seconds": 0.0,
        "polish_seconds": 0.0,
    }
    counts = {"n_units": 0, "n_evidence": 0, "n_pairs": 0}
    rerank_seconds = 0.0
    rerank_pairs_scored = 0
    conformal_threshold: float | None = None
    conformal_abstain_units = 0
    distribution_drift: dict[str, float | bool] | None = None
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
    verifier_type = "jsonl_scores" if scores_jsonl_path else ("custom_verifier" if verifier is not None else ("oss_nli" if use_oss_nli else "unknown"))
    resolved_tracking_id = str(tracking_id) if tracking_id is not None else str(uuid4())

    if not scores_jsonl_path and verifier is None and not use_oss_nli:
        raise ValueError(
            "A scoring source is required: pass `scores_jsonl_path`, provide `verifier`, or set `use_oss_nli=True`."
        )

    read_t0 = time.perf_counter()
    conformal_state = _load_conformal_state(conformal_state_path) if conformal_state_path else None
    if conformal_epsilon is not None:
        _ = float(conformal_epsilon)
    timings["read_seconds"] = time.perf_counter() - read_t0

    verifier_load_seconds = 0.0
    active_verifier: Verifier | None
    if scores_jsonl_path:
        active_verifier = None
    elif verifier is not None:
        active_verifier = LegacyVerifierAdapter(verifier)
    elif use_oss_nli:
        try:
            from ega.verifiers.nli_cross_encoder import NliCrossEncoderVerifier
        except ImportError as exc:
            raise ImportError(
                "OSS NLI verifier requires optional dependency: pip install 'ega[nli]'."
            ) from exc
        load_t0 = time.perf_counter()
        constructed_verifier = NliCrossEncoderVerifier(
            model_name=nli_model_name,
            device=nli_device,
            dtype=nli_dtype,
            topk_per_unit=topk_per_unit,
            max_pairs_total=max_pairs_total,
            max_evidence_per_request=max_evidence_per_request,
            max_batch_tokens=max_batch_tokens,
            evidence_max_chars=evidence_max_chars,
            evidence_max_sentences=evidence_max_sentences,
        )
        verifier_load_seconds = time.perf_counter() - load_t0
        active_verifier = LegacyVerifierAdapter(constructed_verifier)
    else:
        active_verifier = None

    core_output = run_core_pipeline(
        llm_summary_text=llm_summary_text,
        structured_candidate_payload=structured_candidate_payload,
        evidence=evidence,
        unitizer_mode=unitizer_mode,
        policy_config=policy_config,
        accept_threshold=accept_threshold,
        scores_jsonl_path=scores_jsonl_path,
        verifier=active_verifier,
        nli_model_name=nli_model_name,
        nli_device=nli_device,
        nli_dtype=nli_dtype,
        topk_per_unit=topk_per_unit,
        max_pairs_total=max_pairs_total,
        max_evidence_per_request=max_evidence_per_request,
        max_batch_tokens=max_batch_tokens,
        evidence_max_chars=evidence_max_chars,
        evidence_max_sentences=evidence_max_sentences,
        reranker=reranker,
        rerank_topk=rerank_topk,
        conformal_state=conformal_state,
        budget_policy=budget_policy,
        budget_config=budget_config,
    )
    core_intermediate = core_output["intermediate_stats"]
    cleaned_summary = core_intermediate["cleaned_summary"]
    cleaned_evidence = core_intermediate["cleaned_evidence"]
    candidate = core_intermediate["candidate"]
    scores = core_output["scores"]
    decisions = core_output["decisions"]
    failure_class_by_unit = core_output.get("failure_class_by_unit")
    verified_units = core_output["verified_units"]
    result = core_intermediate["result"]
    model_name = core_intermediate["model_name"]
    active_topk_per_unit = core_intermediate["active_topk_per_unit"]
    pool_candidates = core_intermediate["pool_candidates"]
    initial_pool_candidates = core_intermediate["initial_pool_candidates"]
    reranked_candidates = core_intermediate["reranked_candidates"]
    verify_evidence = core_intermediate["verify_evidence"]
    candidate_stage = core_intermediate["candidate_stage"]
    budget_topk_per_unit = core_intermediate["budget_topk_per_unit"]
    budget_max_pairs_total = core_intermediate["budget_max_pairs_total"]
    budget_requested_max_pairs = core_intermediate["budget_requested_max_pairs"]
    budget_unit_risk_scores = core_intermediate["budget_unit_risk_scores"]
    budget_per_unit_pair_budget = core_intermediate["budget_per_unit_pair_budget"]
    per_unit_pairs_before_budget = core_intermediate["per_unit_pairs_before_budget"]
    per_unit_pairs_after_budget = core_intermediate["per_unit_pairs_after_budget"]
    verify_detail.update(core_intermediate["verify_detail"])
    rerank_seconds = core_intermediate["rerank_seconds"]
    rerank_pairs_scored = core_intermediate["rerank_pairs_scored"]
    conformal_threshold = core_intermediate["conformal_threshold"]
    conformal_abstain_units = core_intermediate["conformal_abstain_units"]
    conformal_gate_meta = core_intermediate["conformal_gate_meta"]
    conformal_state = core_intermediate["conformal_state"]
    timings.update(core_intermediate["timings"])
    timings["load_seconds"] = float(timings.get("load_seconds", 0.0)) + float(verifier_load_seconds)
    timings["verify_seconds"] = float(timings.get("load_seconds", 0.0)) + float(timings.get("verify_compute_seconds", 0.0))
    counts.update(core_intermediate["counts"])
    active_accept_threshold = core_intermediate["active_accept_threshold"]

    correction_meta = {
        "enabled": bool(enable_correction),
        "attempts": 0,
        "max_retries": int(max(0, max_retries)),
        "retries_attempted": 0,
        "corrected_unit_count": 0,
        "still_failed_count": int(sum(1 for decision in decisions.values() if decision != "accept")),
        "reverify_occurred": False,
        "stopped_reason": "correction_disabled",
    }
    if enable_correction and correction_generator is not None:
        correction_cfg = CorrectionConfig(
            enable_correction=True,
            max_retries=max_retries,
            unitizer_mode=unitizer_mode,
        )

        def _rerun_core(updated_summary: str) -> dict[str, Any]:
            return run_core_pipeline(
                llm_summary_text=updated_summary,
                structured_candidate_payload=structured_candidate_payload,
                evidence=evidence,
                unitizer_mode=unitizer_mode,
                policy_config=policy_config,
                accept_threshold=accept_threshold,
                scores_jsonl_path=scores_jsonl_path,
                verifier=active_verifier,
                nli_model_name=nli_model_name,
                nli_device=nli_device,
                nli_dtype=nli_dtype,
                topk_per_unit=topk_per_unit,
                max_pairs_total=max_pairs_total,
                max_evidence_per_request=max_evidence_per_request,
                max_batch_tokens=max_batch_tokens,
                evidence_max_chars=evidence_max_chars,
                evidence_max_sentences=evidence_max_sentences,
                reranker=reranker,
                rerank_topk=rerank_topk,
                conformal_state=conformal_state,
                budget_policy=budget_policy,
                budget_config=budget_config,
            )

        core_output = run_correction_loop(
            core_output=core_output,
            generator=correction_generator,
            verifier=_rerun_core,
            config=correction_cfg,
        )
        correction_meta = dict(core_output.get("correction", correction_meta))
        correction_meta.setdefault("retries_attempted", int(correction_meta.get("attempts", 0)))
        correction_meta.setdefault("corrected_unit_count", 0)
        correction_meta.setdefault(
            "still_failed_count",
            int(sum(1 for decision in decisions.values() if decision != "accept")),
        )
        correction_meta.setdefault(
            "reverify_occurred",
            bool(int(correction_meta.get("retries_attempted", correction_meta.get("attempts", 0))) > 0),
        )
        correction_meta.setdefault("stopped_reason", "retry_limit_reached")

        core_intermediate = core_output["intermediate_stats"]
        cleaned_summary = core_intermediate["cleaned_summary"]
        cleaned_evidence = core_intermediate["cleaned_evidence"]
        candidate = core_intermediate["candidate"]
        scores = core_output["scores"]
        decisions = core_output["decisions"]
        failure_class_by_unit = core_output.get("failure_class_by_unit")
        verified_units = core_output["verified_units"]
        result = core_intermediate["result"]
        model_name = core_intermediate["model_name"]
        active_topk_per_unit = core_intermediate["active_topk_per_unit"]
        pool_candidates = core_intermediate["pool_candidates"]
        initial_pool_candidates = core_intermediate["initial_pool_candidates"]
        reranked_candidates = core_intermediate["reranked_candidates"]
        verify_evidence = core_intermediate["verify_evidence"]
        candidate_stage = core_intermediate["candidate_stage"]
        budget_topk_per_unit = core_intermediate["budget_topk_per_unit"]
        budget_max_pairs_total = core_intermediate["budget_max_pairs_total"]
        budget_requested_max_pairs = core_intermediate["budget_requested_max_pairs"]
        budget_unit_risk_scores = core_intermediate["budget_unit_risk_scores"]
        budget_per_unit_pair_budget = core_intermediate["budget_per_unit_pair_budget"]
        per_unit_pairs_before_budget = core_intermediate["per_unit_pairs_before_budget"]
        per_unit_pairs_after_budget = core_intermediate["per_unit_pairs_after_budget"]
        verify_detail.update(core_intermediate["verify_detail"])
        rerank_seconds = core_intermediate["rerank_seconds"]
        rerank_pairs_scored = core_intermediate["rerank_pairs_scored"]
        conformal_threshold = core_intermediate["conformal_threshold"]
        conformal_abstain_units = core_intermediate["conformal_abstain_units"]
        conformal_gate_meta = core_intermediate["conformal_gate_meta"]
        conformal_state = core_intermediate["conformal_state"]
        timings.update(core_intermediate["timings"])
        timings["load_seconds"] = float(timings.get("load_seconds", 0.0)) + float(verifier_load_seconds)
        timings["verify_seconds"] = float(timings.get("load_seconds", 0.0)) + float(timings.get("verify_compute_seconds", 0.0))
        counts.update(core_intermediate["counts"])
        active_accept_threshold = core_intermediate["active_accept_threshold"]

    verified_extract = [
        {"unit_id": unit.id, "text": clean_text(unit.text)} for unit in verified_units
    ]
    verified_text = clean_text("\n".join(unit["text"] for unit in verified_extract))
    used_evidence = _extract_used_evidence(scores=scores)
    if conformal_state is not None:
        live_scores = [float(score.entailment) for score in scores]
        if live_scores:
            calibrator = ConformalCalibrator()
            try:
                calibrator.load_reference_from_state(conformal_state)
                distribution_drift = calibrator.measure_drift(live_scores=live_scores)
            except Exception:
                distribution_drift = None
    coverage_results: dict[str, CoverageResult] | None = None
    verifier_scores = {
        score.unit_id: {
            "entailment": float(score.entailment),
            "contradiction": float(score.contradiction),
            "neutral": float(score.neutral),
            "label": str(score.label),
            "chosen_evidence_id": (
                score.raw.get("chosen_evidence_id") if isinstance(score.raw, dict) else None
            ),
            "chosen_evidence_id_source_stage": candidate_stage,
            "conformal_score": (
                score.raw.get("conformal_score")
                if isinstance(score.raw, dict) and "conformal_score" in score.raw
                else None
            ),
            "conformal_gate": (
                score.raw.get("conformal_gate")
                if isinstance(score.raw, dict) and "conformal_gate" in score.raw
                else None
            ),
        }
        for score in scores
    }

    planned_pairs_total = int(sum((per_unit_pairs_before_budget or {}).values()))
    evaluated_pairs_total = int(counts["n_pairs"])
    pruned_pairs_total = max(0, planned_pairs_total - evaluated_pairs_total)

    def _correction_final_outcome() -> str:
        if bool(result.decision.refusal):
            return "refusal"
        if int(result.decision.summary_stats.get("dropped_units", 0)) > 0:
            return "partial_accept"
        return "all_accepted"

    def _normalize_duration(seconds: float) -> float:
        value = max(0.0, float(seconds))
        return float(round(value, 2))

    def _build_trace(payload: dict[str, Any], *, total_seconds: float) -> dict[str, Any]:
        decision_payload = payload.get("decision", {})
        dropped_units_payload = decision_payload.get("dropped_units", [])
        trace_payload: dict[str, Any] = {
            "trace_schema_version": 1,
            "total_seconds": _normalize_duration(total_seconds),
            "read_seconds": _normalize_duration(timings["read_seconds"]),
            "unitize_seconds": _normalize_duration(timings["unitize_seconds"]),
            "verify_seconds": _normalize_duration(timings["verify_seconds"]),
            "load_seconds": _normalize_duration(timings["load_seconds"]),
            "verify_compute_seconds": _normalize_duration(timings["verify_compute_seconds"]),
            "enforce_seconds": _normalize_duration(timings["enforce_seconds"]),
            "polish_seconds": _normalize_duration(timings["polish_seconds"]),
            "preselect_seconds": _normalize_duration(verify_detail["preselect_seconds"]),
            "tokenize_seconds": _normalize_duration(verify_detail["tokenize_seconds"]),
            "forward_seconds": _normalize_duration(verify_detail["forward_seconds"]),
            "post_seconds": _normalize_duration(verify_detail["post_seconds"]),
            "num_batches": verify_detail["num_batches"],
            "batch_size_mean": verify_detail["batch_size_mean"],
            "batch_size_max": verify_detail["batch_size_max"],
            "seq_len_mean": verify_detail["seq_len_mean"],
            "seq_len_p50": verify_detail["seq_len_p50"],
            "seq_len_p95": verify_detail["seq_len_p95"],
            "tokens_total": verify_detail["tokens_total"],
            "device": verify_detail["device"],
            "dtype": verify_detail["dtype"],
            "amp_enabled": verify_detail["amp_enabled"],
            "compiled_enabled": verify_detail["compiled_enabled"],
            "pairs_pruned_stage1": verify_detail["pairs_pruned_stage1"],
            "pairs_pruned_stage2": verify_detail["pairs_pruned_stage2"],
            "dtype_overridden": verify_detail["dtype_overridden"],
            "evidence_truncated_frac": verify_detail["evidence_truncated_frac"],
            "evidence_chars_mean_before": verify_detail["evidence_chars_mean_before"],
            "evidence_chars_mean_after": verify_detail["evidence_chars_mean_after"],
            "rerank_seconds": _normalize_duration(rerank_seconds),
            "rerank_pairs_scored": rerank_pairs_scored,
            "conformal_threshold": conformal_threshold,
            "conformal_abstain_units": conformal_abstain_units,
            "accept_threshold": active_accept_threshold,
            "budget_topk_per_unit": budget_topk_per_unit,
            "budget_max_pairs_total": budget_max_pairs_total,
            "budget_requested_max_pairs": budget_requested_max_pairs,
            "planned_pairs_total": planned_pairs_total,
            "evaluated_pairs_total": evaluated_pairs_total,
            "pruned_pairs_total": pruned_pairs_total,
            "n_units": counts["n_units"],
            "unit_ids": [str(unit.id) for unit in candidate.units],
            "n_evidence": counts["n_evidence"],
            "n_pairs": counts["n_pairs"],
            "scored_units": int(len(scores)),
            "verifier_type": verifier_type,
            "kept_units": payload["stats"]["kept_units"],
            "dropped_units": payload["stats"]["dropped_units"],
            "abstained_units": int(conformal_abstain_units),
            "refusal": payload["decision"]["refusal"],
            "model_name": payload["stats"].get("model_name"),
            "tracking_id": payload.get("tracking_id"),
            "correction_enabled": bool(correction_meta.get("enabled", False)),
            "correction_max_retries": int(correction_meta.get("max_retries", 0)),
            "correction_retries_attempted": int(correction_meta.get("retries_attempted", correction_meta.get("attempts", 0))),
            "correction_corrected_unit_count": int(correction_meta.get("corrected_unit_count", 0)),
            "correction_still_failed_count": int(correction_meta.get("still_failed_count", len(dropped_units_payload))),
            "correction_reverify_occurred": bool(correction_meta.get("reverify_occurred", False)),
            "correction_stopped_reason": str(correction_meta.get("stopped_reason", "correction_disabled")),
            "correction_final_outcome": _correction_final_outcome(),
        }
        stats_payload = payload.get("stats", {})
        stats = stats_payload if isinstance(stats_payload, dict) else {}

        optional_trace_keys = (
            "coverage_pool_topk",
            "coverage_avg_score",
            "coverage_unit_scores",
            "coverage_missing_total",
            "reward_total",
            "reward_avg",
            "reward_avg_support",
            "reward_hallucination_rate",
            "reward_abstention_rate",
            "reward_avg_coverage",
            "reward_unit_totals",
        )
        for key in optional_trace_keys:
            if key in payload and payload[key] is not None:
                trace_payload[key] = payload[key]
                continue
            if key in stats and stats[key] is not None:
                trace_payload[key] = stats[key]
        if budget_unit_risk_scores is not None:
            trace_payload["unit_risk_scores"] = dict(budget_unit_risk_scores)
        if budget_per_unit_pair_budget is not None:
            trace_payload["per_unit_pair_budget"] = dict(budget_per_unit_pair_budget)
        if per_unit_pairs_before_budget is not None:
            trace_payload["per_unit_pairs_before_budget"] = dict(per_unit_pairs_before_budget)
        if per_unit_pairs_after_budget is not None:
            trace_payload["per_unit_pairs_after_budget"] = dict(per_unit_pairs_after_budget)
        if "evaluated_pairs_count_per_unit" in verify_detail:
            trace_payload["evaluated_pairs_count_per_unit"] = dict(
                verify_detail["evaluated_pairs_count_per_unit"]
            )
        if render_safe_answer:
            trace_payload["safe_answer_final_text"] = payload.get("safe_answer_final_text", "")
            trace_payload["safe_answer_summary"] = dict(payload.get("safe_answer_summary", {}))
        if "distribution_drift" in payload and isinstance(payload["distribution_drift"], dict):
            trace_payload["distribution_drift"] = dict(payload["distribution_drift"])
        return trace_payload

    def _aggregate_payload_decision(
        *,
        units: list[Unit],
        decisions_by_unit: dict[str, str],
        failure_classes_by_unit: dict[str, str] | None,
        correction: dict[str, Any],
    ) -> tuple[str, str, dict[str, int]]:
        summary = {
            "supported": 0,
            "unsupported_claim": 0,
            "missing_in_source": 0,
            "ambiguous_source": 0,
        }
        has_unsupported = False
        has_missing_or_ambiguous = False
        for unit in units:
            unit_id = unit.id
            decision = str(decisions_by_unit.get(unit_id, ""))
            if decision == "accept":
                summary["supported"] += 1
                continue

            failure_class = str((failure_classes_by_unit or {}).get(unit_id, "")).upper()
            if failure_class == "UNSUPPORTED_CLAIM":
                summary["unsupported_claim"] += 1
                has_unsupported = True
            elif failure_class == "MISSING_IN_SOURCE":
                summary["missing_in_source"] += 1
                has_missing_or_ambiguous = True
            elif failure_class == "AMBIGUOUS_SOURCE":
                summary["ambiguous_source"] += 1
                has_missing_or_ambiguous = True

        if units and all(str(decisions_by_unit.get(unit.id, "")) == "accept" for unit in units):
            return "ACCEPT", "EMIT", summary
        correction_enabled = bool(correction.get("enabled", False))
        retries_attempted = int(correction.get("retries_attempted", correction.get("attempts", 0)))
        still_failed_count = int(
            correction.get(
                "still_failed_count",
                sum(1 for unit in units if str(decisions_by_unit.get(unit.id, "")) != "accept"),
            )
        )
        if correction_enabled and retries_attempted > 0 and still_failed_count > 0:
            return "PENDING", "BOUNDED_REPAIR", summary
        if has_missing_or_ambiguous:
            return "REJECT", "REJECT", summary
        if has_unsupported:
            max_retries_allowed = int(correction.get("max_retries", 0))
            if correction_enabled and retries_attempted < max_retries_allowed:
                return "PENDING", "BOUNDED_REPAIR", summary
            return "REJECT", "REJECT", summary
        return "REJECT", "REJECT", summary
    normalized_downstream_mode = normalize_downstream_compatibility_mode(
        downstream_compatibility_mode
    )

    output: dict[str, Any] = {
        "accept_threshold": active_accept_threshold,
        "units": [{"unit_id": unit.id, "text": unit.text} for unit in candidate.units],
        "pool_candidates": {
            str(unit_id): [str(evidence_id) for evidence_id in ids]
            for unit_id, ids in pool_candidates.items()
        },
        "pre_rerank_candidates": _format_debug_candidates(initial_pool_candidates),
        "verified_extract": verified_extract,
        "verified_units": list(verified_extract),
        "verified_text": verified_text,
        "dropped_units": list(result.decision.dropped_units),
        "used_evidence": {
            str(unit_id): [str(evidence_id) for evidence_id in ids]
            for unit_id, ids in used_evidence.items()
        },
        "verifier_scores": verifier_scores,
        "verification_pairs": _extract_verification_pairs(scores=scores),
        "decisions": {str(unit_id): str(decision) for unit_id, decision in decisions.items()},
        "decision": asdict(result.decision),
        "verifier_model_name": model_name,
        "correction": correction_meta,
        "stats": {
            **dict(result.decision.summary_stats),
            "accept_threshold": active_accept_threshold,
            "model_name": model_name,
            "planned_pairs_total": planned_pairs_total,
            "evaluated_pairs_total": evaluated_pairs_total,
            "pruned_pairs_total": pruned_pairs_total,
        },
    }
    payload_status, payload_action, payload_failure_summary = _aggregate_payload_decision(
        units=candidate.units,
        decisions_by_unit=decisions,
        failure_classes_by_unit=failure_class_by_unit,
        correction=correction_meta,
    )
    output["payload_status"] = payload_status
    output["payload_action"] = payload_action
    output["payload_failure_summary"] = payload_failure_summary
    output.update(
        _derive_workflow_contract(
            payload_status=payload_status,
            payload_action=payload_action,
            candidate=candidate,
            decisions_by_unit=decisions,
        )
    )
    output["tracking_id"] = resolved_tracking_id
    route_status_by_payload_status = {
        "ACCEPT": "READY",
        "REJECT": "REJECTED",
        "PENDING": "REPAIR_PENDING",
        "REPAIR": "REPAIR_PENDING",
    }
    output["route_status"] = route_status_by_payload_status.get(payload_status, "REJECTED")
    if payload_status == "ACCEPT":
        output["business_payload_emitted"] = True
    elif payload_status == "REJECT":
        output["business_payload_emitted"] = False
        output["passthrough_mode"] = "STRICT"
    elif payload_status in {"REPAIR", "PENDING"}:
        output["business_payload_emitted"] = False
        output["passthrough_mode"] = "STRICT"
        output["repair_pending"] = True
        output["route_reason"] = "BOUNDED_REPAIR"
        output["pending_expires_at"] = pending_expires_at
    if normalized_downstream_mode == ADAPTER_MODE:
        accepted_units = [
            {"unit_id": unit.id, "text": clean_text(unit.text)}
            for unit in candidate.units
            if str(decisions.get(unit.id, "")) == "accept"
        ]
        rejected_units = [
            {
                "unit_id": unit.id,
                "text": clean_text(unit.text),
                "decision": str(decisions.get(unit.id, "")),
                "failure_class": str((failure_class_by_unit or {}).get(unit.id, "")).upper() or None,
            }
            for unit in candidate.units
            if str(decisions.get(unit.id, "")) != "accept"
        ]
        supported_count = int(payload_failure_summary.get("supported", 0))
        output["adapter_accepted_units"] = accepted_units
        output["adapter_rejected_units"] = rejected_units
        output["adapter_summary"] = {
            "total_units": int(len(candidate.units)),
            "accepted_units": int(len(accepted_units)),
            "rejected_units": int(len(rejected_units)),
            "supported_count": supported_count,
            "unsupported_claim_count": int(payload_failure_summary.get("unsupported_claim", 0)),
            "missing_in_source_count": int(payload_failure_summary.get("missing_in_source", 0)),
            "ambiguous_source_count": int(payload_failure_summary.get("ambiguous_source", 0)),
        }
        if payload_status == "ACCEPT":
            output["adapter_payload"] = list(verified_extract)
            output["business_payload_emitted"] = True
        elif payload_status == "REJECT":
            if supported_count > 0:
                output["adapter_payload"] = accepted_units
                output["business_payload_emitted"] = True
            else:
                output["adapter_payload"] = None
                output["business_payload_emitted"] = False
        else:
            output["adapter_payload"] = None
            output["business_payload_emitted"] = False

    unit_texts = {str(unit.id): clean_text(unit.text) for unit in candidate.units}
    audit_entries = _build_audit_entries(
        scores=scores,
        decisions=decisions,
        conformal_state=conformal_state,
    )
    output_mode_normalized = str(output_mode or "strict").strip().lower()
    if output_mode_normalized not in {"strict", "adapter"}:
        raise ValueError("output_mode must be 'strict' or 'adapter'.")
    schema_payload = _assemble_v4_response(
        output_mode=output_mode_normalized,
        internal_payload_status=payload_status,
        payload_action=payload_action,
        verified_text=verified_text,
        verified_extract=verified_extract,
        decisions=decisions,
        failure_class_by_unit=failure_class_by_unit,
        audit_entries=audit_entries,
        tracking_id=output.get("tracking_id"),
        route_status=str(output.get("route_status", "")),
        route_reason=(
            str(output.get("route_reason"))
            if output.get("route_reason") is not None
            else None
        ),
        pending_expires_at=pending_expires_at,
        unit_texts=unit_texts,
    )
    output["v4_response"] = dict(schema_payload)
    output.update(schema_payload)
    if render_safe_answer:
        safe_answer = SafeAnswerRenderer().render(
            units=candidate.units,
            decisions=decisions,
            used_evidence=used_evidence,
        )
        output["safe_answer"] = safe_answer.to_dict()
        output["safe_answer_final_text"] = safe_answer.final_text
        output["safe_answer_summary"] = dict(safe_answer.summary)
    if reranked_candidates is not None:
        output["reranked_candidates"] = reranked_candidates
        output["post_rerank_candidates"] = _format_debug_candidates(reranked_candidates)
    else:
        output["post_rerank_candidates"] = None
    if budget_unit_risk_scores is not None:
        output["unit_risk_scores"] = dict(budget_unit_risk_scores)
    if per_unit_pairs_before_budget is not None:
        output["per_unit_pairs_before_budget"] = dict(per_unit_pairs_before_budget)
    if per_unit_pairs_after_budget is not None:
        output["per_unit_pairs_after_budget"] = dict(per_unit_pairs_after_budget)
    if budget_per_unit_pair_budget is not None:
        output["per_unit_pair_budget"] = dict(budget_per_unit_pair_budget)
        budget_summary = _summarize_budget_allocation(
            unit_risk_scores=budget_unit_risk_scores or {},
            per_unit_pair_budget=(
                verify_detail.get("evaluated_pairs_count_per_unit", budget_per_unit_pair_budget)
                if isinstance(verify_detail.get("evaluated_pairs_count_per_unit"), dict)
                else budget_per_unit_pair_budget
            ),
        )
        output["stats"]["budget_active"] = True
        output["stats"]["requested_budget_max_pairs"] = budget_requested_max_pairs
        output["stats"]["effective_budget_max_pairs"] = evaluated_pairs_total
        output["stats"]["effective_topk_per_unit"] = int(
            max((per_unit_pairs_after_budget or budget_per_unit_pair_budget).values())
            if (per_unit_pairs_after_budget or budget_per_unit_pair_budget)
            else 0
        )
        output["stats"]["avg_pairs_per_unit"] = (
            float(evaluated_pairs_total) / float(len(candidate.units)) if candidate.units else 0.0
        )
        output["stats"]["pairs_allocated_to_high_risk_units"] = int(
            budget_summary["pairs_allocated_to_high_risk_units"]
        )
        output["stats"]["pairs_allocated_to_low_risk_units"] = int(
            budget_summary["pairs_allocated_to_low_risk_units"]
        )
    elif budget_policy is not None and budget_config is not None:
        output["stats"]["budget_active"] = False
        output["stats"]["requested_budget_max_pairs"] = budget_requested_max_pairs
        output["stats"]["effective_budget_max_pairs"] = 0
        output["stats"]["effective_topk_per_unit"] = 0
        output["stats"]["avg_pairs_per_unit"] = 0.0
        output["stats"]["pairs_allocated_to_high_risk_units"] = 0
        output["stats"]["pairs_allocated_to_low_risk_units"] = 0
        output["per_unit_pair_budget"] = {}
    if conformal_gate_meta is not None:
        output["conformal"] = {
            "threshold": float(conformal_state.threshold),
            "meta": dict(conformal_state.meta),
            **conformal_gate_meta,
        }
    if distribution_drift is not None:
        output["distribution_drift"] = dict(distribution_drift)
    if coverage_config is not None:
        coverage_results = EvidenceCoverageAnalyzer().analyze(
            units=candidate.units,
            evidence=verify_evidence,
            pool_candidates=pool_candidates,
            used_evidence=used_evidence,
            config=coverage_config,
        )
        coverage_scores = [row.coverage_score for row in coverage_results.values()]
        coverage_avg_score = (
            float(sum(coverage_scores)) / float(len(coverage_scores)) if coverage_scores else 0.0
        )
        coverage_missing_total = int(
            sum(len(row.missing_evidence_ids) for row in coverage_results.values())
        )
        output["stats"]["coverage_pool_topk"] = int(coverage_config.pool_topk)
        output["stats"]["coverage_avg_score"] = coverage_avg_score
        output["stats"]["coverage_unit_scores"] = {
            unit_id: float(row.coverage_score) for unit_id, row in coverage_results.items()
        }
        output["stats"]["coverage_missing_total"] = coverage_missing_total
        output["coverage"] = {
            unit_id: {
                "relevant_evidence_ids": list(row.relevant_evidence_ids),
                "used_evidence_ids": list(row.used_evidence_ids),
                "missing_evidence_ids": list(row.missing_evidence_ids),
                "coverage_score": float(row.coverage_score),
            }
            for unit_id, row in coverage_results.items()
        }

    if reward_config is not None:
        verification_payload = {
            score.unit_id: {
                "entailment": float(score.entailment),
                "contradiction": float(score.contradiction),
                "neutral": float(score.neutral),
                "label": str(score.label),
            }
            for score in scores
        }
        unit_rewards, reward_summary = RewardComputer().compute(
            units=candidate.units,
            verification=verification_payload,
            decisions=decisions,
            coverage=coverage_results,
            config=reward_config,
        )
        output["stats"]["reward_total"] = float(reward_summary.total_reward)
        output["stats"]["reward_avg"] = float(reward_summary.avg_reward)
        output["stats"]["reward_avg_support"] = float(reward_summary.avg_support_score)
        output["stats"]["reward_hallucination_rate"] = float(reward_summary.hallucination_rate)
        output["stats"]["reward_abstention_rate"] = float(reward_summary.abstention_rate)
        output["stats"]["reward_avg_coverage"] = float(reward_summary.avg_coverage_score)
        output["stats"]["reward_unit_totals"] = {
            unit_id: float(row.total_reward) for unit_id, row in unit_rewards.items()
        }
        output["rewards"] = {
            unit_id: {
                "total_reward": float(row.total_reward),
                "support_score": float(row.support_score),
                "hallucination_penalty": float(row.hallucination_penalty),
                "abstain_penalty": float(row.abstain_penalty),
                "coverage_score": float(row.coverage_score),
            }
            for unit_id, row in unit_rewards.items()
        }
    else:
        unit_rewards = None
        reward_summary = None

    def _append_trace(payload: dict[str, Any]) -> None:
        total_t1 = time.perf_counter()
        trace = _build_trace(payload, total_seconds=total_t1 - total_t0)
        payload["trace"] = dict(trace)
        if not trace_out:
            return
        with Path(trace_out).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trace, sort_keys=True) + "\n")

    def _append_training_example(payload: dict[str, Any]) -> None:
        if not emit_training_example_path:
            return
        out_path = Path(emit_training_example_path).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        verifier_scores = {
            score.unit_id: {
                "best_entail": float(score.entailment),
                "best_contradiction": float(score.contradiction),
                "best_neutral": float(score.neutral),
                "label": str(score.label),
                "chosen_evidence_id": (
                    score.raw.get("chosen_evidence_id")
                    if isinstance(score.raw, dict)
                    else None
                ),
            }
            for score in scores
        }
        training_row: dict[str, Any] = {
            "id": str(training_example_id or uuid4().hex),
            "input_prompt": None,
            "generated_text": cleaned_summary,
            "units": [{"unit_id": unit.id, "text": unit.text} for unit in candidate.units],
            "pool_candidates": {
                str(unit_id): [str(evidence_id) for evidence_id in ids]
                for unit_id, ids in pool_candidates.items()
            },
            "used_evidence": {
                str(unit_id): [str(evidence_id) for evidence_id in ids]
                for unit_id, ids in used_evidence.items()
            },
            "decisions": {str(unit_id): str(decision) for unit_id, decision in decisions.items()},
            "verifier_scores": verifier_scores,
        }
        if coverage_results is not None:
            training_row["coverage"] = {
                unit_id: {
                    "coverage_score": float(row.coverage_score),
                    "missing": list(row.missing_evidence_ids),
                }
                for unit_id, row in coverage_results.items()
            }
        if unit_rewards is not None:
            training_row["rewards"] = {
                unit_id: {
                    "total_reward": float(row.total_reward),
                    "support_score": float(row.support_score),
                    "hallucination_penalty": float(row.hallucination_penalty),
                    "abstain_penalty": float(row.abstain_penalty),
                    "coverage_score": float(row.coverage_score),
                }
                for unit_id, row in unit_rewards.items()
            }
            training_row["reward_summary"] = asdict(reward_summary) if reward_summary is not None else {}
        with out_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(training_row, sort_keys=True) + "\n")

    def _finalize(payload: dict[str, Any]) -> dict[str, Any]:
        if "evaluated_pairs_count_per_unit" in verify_detail:
            payload["evaluated_pairs_count_per_unit"] = dict(
                verify_detail["evaluated_pairs_count_per_unit"]
            )
        _append_trace(payload)
        _append_training_example(payload)
        return payload

    if polished_json is None:
        output["polish_status"] = "skipped"
        return _finalize(output)

    polish_t0 = time.perf_counter()
    polished_units = _parse_polished_units(polished_json)
    if enable_polish_validation:
        gated = apply_polish_gate(
            result=result,
            original_units=verified_units,
            polished_units=polished_units,
            config=PolishGateConfig(),
        )
        output["polish_status"] = gated.polish_status
        output["polish_fail_reasons"] = list(gated.polish_fail_reasons)
        if gated.polish_status == "passed" and gated.polished_units is not None:
            output["polished_extract"] = [dict(item) for item in gated.polished_units]
            output["polished_text"] = "\n".join(
                str(item["edited_text"]) for item in gated.polished_units
            )
        timings["polish_seconds"] = time.perf_counter() - polish_t0
        return _finalize(output)

    output["polish_status"] = "passed"
    output["polish_fail_reasons"] = []
    output["polished_extract"] = [
        {"unit_id": unit.unit_id, "edited_text": unit.edited_text}
        for unit in polished_units
    ]
    output["polished_text"] = "\n".join(unit.edited_text for unit in polished_units)
    timings["polish_seconds"] = time.perf_counter() - polish_t0
    return _finalize(output)


def _load_conformal_state(path: str | Path) -> ConformalState:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("Conformal state JSON must be an object.")
    if "threshold" not in payload:
        raise ValueError("Conformal state JSON must include 'threshold'.")
    threshold = float(payload["threshold"])
    raw_meta = payload.get("meta", {})
    meta = dict(raw_meta) if isinstance(raw_meta, dict) else {}
    band_width = float(
        payload.get(
            "band_width",
            meta.get("band_width", meta.get("abstain_margin", 0.02)),
        )
    )
    abstain_k = float(payload.get("abstain_k", meta.get("abstain_k", 1.0)))
    n_samples = int(payload.get("n_samples", meta.get("n_samples", meta.get("n_calib", 0))))
    score_mean = float(payload.get("score_mean", meta.get("score_mean", 0.0)))
    score_std = float(payload.get("score_std", meta.get("score_std", 0.0)))
    calibration_score_min = payload.get("calibration_score_min")
    calibration_score_max = payload.get("calibration_score_max")
    if calibration_score_min is not None:
        calibration_score_min = float(calibration_score_min)
    if calibration_score_max is not None:
        calibration_score_max = float(calibration_score_max)
    meta.setdefault("abstain_margin", band_width)
    meta.setdefault("band_width", band_width)
    meta.setdefault("abstain_k", abstain_k)
    meta.setdefault("n_samples", n_samples)
    meta.setdefault("score_mean", score_mean)
    meta.setdefault("score_std", score_std)
    meta.setdefault("calibration_score_min", calibration_score_min)
    meta.setdefault("calibration_score_max", calibration_score_max)
    return ConformalState(
        threshold=threshold,
        band_width=band_width,
        abstain_k=abstain_k,
        n_samples=n_samples,
        score_mean=score_mean,
        score_std=score_std,
        meta=meta,
        calibration_score_min=calibration_score_min,
        calibration_score_max=calibration_score_max,
    )


def _summarize_budget_allocation(
    *,
    unit_risk_scores: dict[str, float],
    per_unit_pair_budget: dict[str, int],
) -> dict[str, float | int]:
    if not per_unit_pair_budget:
        return {
            "avg_pairs_per_unit": 0.0,
            "pairs_allocated_to_high_risk_units": 0,
            "pairs_allocated_to_low_risk_units": 0,
        }
    ranked = sorted(
        per_unit_pair_budget.keys(),
        key=lambda unit_id: (-float(unit_risk_scores.get(unit_id, 0.0)), str(unit_id)),
    )
    split = max(1, len(ranked) // 2)
    high_risk = set(ranked[:split])
    high_pairs = sum(int(per_unit_pair_budget.get(unit_id, 0)) for unit_id in high_risk)
    low_pairs = sum(
        int(per_unit_pair_budget.get(unit_id, 0)) for unit_id in ranked if unit_id not in high_risk
    )
    total_pairs = sum(int(value) for value in per_unit_pair_budget.values())
    return {
        "avg_pairs_per_unit": float(total_pairs) / float(len(per_unit_pair_budget)),
        "pairs_allocated_to_high_risk_units": int(high_pairs),
        "pairs_allocated_to_low_risk_units": int(low_pairs),
    }


def _extract_used_evidence(*, scores: list[VerificationScore]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for score in scores:
        raw = score.raw if isinstance(score.raw, dict) else {}
        chosen = raw.get("chosen_evidence_id")
        if isinstance(chosen, str) and chosen:
            out[score.unit_id] = [chosen]
        else:
            out[score.unit_id] = []
    return out


def _format_debug_candidates(
    candidates: dict[str, list[str]],
) -> dict[str, list[dict[str, Any]]]:
    return {
        str(unit_id): [{"evidence_id": str(evidence_id), "score": None} for evidence_id in ids]
        for unit_id, ids in candidates.items()
    }


def _extract_verification_pairs(*, scores: list[VerificationScore]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for score in scores:
        raw = score.raw if isinstance(score.raw, dict) else {}
        per_item_probs = raw.get("per_item_probs", [])
        if isinstance(per_item_probs, list):
            out[score.unit_id] = [dict(row) for row in per_item_probs if isinstance(row, dict)]
        else:
            out[score.unit_id] = []
    return out


def _parse_polished_units(payload: dict[str, Any] | list[Any]) -> list[PolishedUnit]:
    rows: Any = payload
    if isinstance(payload, dict):
        if isinstance(payload.get("units"), list):
            rows = payload["units"]
        elif isinstance(payload.get("polished_units"), list):
            rows = payload["polished_units"]
        else:
            raise ValueError("Polished payload must include 'units' or 'polished_units' list.")
    if not isinstance(rows, list):
        raise ValueError("Polished payload must be a list or wrapper object containing a list.")

    units: list[PolishedUnit] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Polished row at index {idx} must be an object.")
        if "unit_id" not in row or "edited_text" not in row:
            raise ValueError(
                f"Polished row at index {idx} must include 'unit_id' and 'edited_text'."
            )
        units.append(
            PolishedUnit(
                unit_id=str(row["unit_id"]),
                edited_text=str(row["edited_text"]),
            )
        )
    return units
