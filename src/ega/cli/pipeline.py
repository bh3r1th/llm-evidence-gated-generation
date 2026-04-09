"""Handlers for pipeline and polish validation commands."""

from __future__ import annotations

import json
from pathlib import Path

from ega import api
from ega.benchmark import PolicyConfig
from ega.polish.gate import PolishGateConfig, apply_polish_gate
from ega.polish.types import PolishedUnit
from ega.serialization import from_json, to_json
from ega.text_clean import clean_text
from ega.types import EnforcementResult, EvidenceItem, EvidenceSet, Unit
from ega.v2.budget import BudgetConfig
from ega.v2.budget_greedy import GreedyBudgetPolicy
from ega.v2.coverage import CoverageConfig
from ega.v2.cross_encoder_reranker import CrossEncoderReranker
from ega.v2.rewards import RewardConfig


def _resolve_path(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    return p.resolve()


def _resolve_existing_path(path_str: str) -> Path:
    resolved_path = _resolve_path(path_str)
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"File not found: {resolved_path}. Current working directory: {Path.cwd()}"
        )
    return resolved_path


def _load_answer(path: str) -> str:
    return clean_text(_resolve_existing_path(path).read_text(encoding="utf-8"))


def _load_evidence(path: str) -> EvidenceSet:
    resolved_path = _resolve_existing_path(path)
    try:
        payload = json.loads(resolved_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in file {resolved_path}: {exc.msg}.") from exc

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


def _load_polished_units(path: str) -> list[PolishedUnit]:
    resolved_path = _resolve_existing_path(path)
    try:
        payload = json.loads(resolved_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in file {resolved_path}: {exc.msg}.") from exc

    rows = payload
    if isinstance(payload, dict):
        if isinstance(payload.get("units"), list):
            rows = payload["units"]
        elif isinstance(payload.get("polished_units"), list):
            rows = payload["polished_units"]
        else:
            raise ValueError(
                "Polished JSON object must include a 'units' or 'polished_units' list."
            )
    if not isinstance(rows, list):
        raise ValueError("Polished JSON must be a list or wrapper object containing a list.")

    units: list[PolishedUnit] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Polished unit at index {idx} must be an object.")
        if "unit_id" not in row or "edited_text" not in row:
            raise ValueError(
                f"Polished unit at index {idx} must include 'unit_id' and 'edited_text'."
            )
        units.append(PolishedUnit(unit_id=str(row["unit_id"]), edited_text=str(row["edited_text"])))
    return units


def _load_enforcement_result(path: str) -> EnforcementResult:
    payload = _resolve_existing_path(path).read_text(encoding="utf-8-sig")
    return from_json(payload, EnforcementResult)


def handle_polish_validate(args: object) -> int:
    result = _load_enforcement_result(args.verified_json)
    original_units = [
        Unit(
            id=str(item["unit_id"]),
            text=str(item["text"]),
            metadata={},
        )
        for item in result.verified_units
    ]
    polished_units = _load_polished_units(args.polished_json)
    merged = apply_polish_gate(
        result=result,
        original_units=original_units,
        polished_units=polished_units,
        config=PolishGateConfig(),
    )
    print(to_json(merged))
    return 0


def handle_pipeline(args: object) -> int:
    if not getattr(args, "llm_summary_file", None):
        raise ValueError("Missing required argument: --llm-summary-file")
    if not getattr(args, "evidence_json", None):
        raise ValueError("Missing required argument: --evidence-json")
    if not (bool(args.scores_jsonl) or bool(args.use_oss_nli)):
        raise ValueError("Must provide either --scores-jsonl or --use-oss-nli")

    polished_payload = None
    if isinstance(args.polished_json, str) and args.polished_json.strip():
        polished_path = _resolve_existing_path(args.polished_json)
        try:
            polished_payload = json.loads(polished_path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in file {polished_path}: {exc.msg}.") from exc

    scores_jsonl_path = None
    if isinstance(args.scores_jsonl, str) and args.scores_jsonl.strip():
        scores_jsonl_path = str(_resolve_existing_path(args.scores_jsonl))

    conformal_state_path = None
    if isinstance(args.conformal_state, str) and args.conformal_state.strip():
        conformal_state_path = str(_resolve_existing_path(args.conformal_state))

    reranker = None
    if args.use_reranker:
        try:
            reranker = CrossEncoderReranker(model_name=args.reranker_model)
        except ImportError as exc:
            raise ImportError(
                "v2 reranker requires optional dependency: pip install 'ega[rerank]'."
            ) from exc

    budget_policy = None
    budget_config = None
    if args.use_budget:
        budget_policy = GreedyBudgetPolicy()
        budget_config = BudgetConfig(
            latency_budget_ms=args.latency_budget_ms,
            max_pairs_total=args.budget_max_pairs,
        )

    coverage_config = CoverageConfig(pool_topk=int(args.coverage_pool_topk)) if args.coverage else None
    reward_config = None
    if args.rewards:
        reward_config = RewardConfig(
            w_support=float(args.reward_w_support),
            w_hallucination=float(args.reward_w_hallucination),
            w_abstain=float(args.reward_w_abstain),
            w_coverage=float(args.reward_w_coverage),
            clamp_min=float(args.reward_clamp_min),
            clamp_max=float(args.reward_clamp_max),
        )

    config = {
        "unitizer_mode": args.unitizer,
        "policy_config": PolicyConfig(
            threshold_entailment=args.threshold_entailment,
            max_contradiction=args.max_contradiction,
            partial_allowed=args.partial_allowed,
        ),
        "accept_threshold": args.accept_threshold,
        "scores_jsonl_path": scores_jsonl_path,
        "use_oss_nli": args.use_oss_nli,
        "nli_model_name": args.nli_model_name,
        "nli_device": args.device,
        "nli_dtype": args.dtype,
        "topk_per_unit": args.topk_per_unit,
        "max_pairs_total": args.max_pairs_total,
        "reranker": reranker,
        "rerank_topk": args.rerank_topk,
        "conformal_state_path": conformal_state_path,
        "budget_policy": budget_policy,
        "budget_config": budget_config,
        "max_evidence_per_request": args.max_evidence_per_request,
        "max_batch_tokens": args.max_batch_tokens,
        "evidence_max_chars": args.evidence_max_chars,
        "evidence_max_sentences": args.evidence_max_sentences,
        "polished_json": polished_payload,
        "enable_polish_validation": not args.no_polish_validation,
        "trace_out": args.trace_out,
        "coverage_config": coverage_config,
        "reward_config": reward_config,
        "emit_training_example_path": args.emit_training_jsonl,
        "training_example_id": args.training_example_id,
        "render_safe_answer": args.render_safe_answer,
    }
    payload = api.verify_answer(
        llm_output=_load_answer(args.llm_summary_file),
        source_text="",
        evidence=_load_evidence(args.evidence_json),
        config=config,
        return_pipeline_output=True,
    )
    print(json.dumps(payload, sort_keys=True))
    return 0
