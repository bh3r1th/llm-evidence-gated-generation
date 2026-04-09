"""Interactive and JSONL shell command handlers."""

from __future__ import annotations

import json
import time
from pathlib import Path

from ega import api
from ega.benchmark import PolicyConfig
from ega.text_clean import clean_text
from ega.types import EvidenceItem, EvidenceSet
from ega.v2.budget import BudgetConfig
from ega.v2.budget_greedy import GreedyBudgetPolicy
from ega.v2.cross_encoder_reranker import CrossEncoderReranker
from ega.verifiers.nli_cross_encoder import NliCrossEncoderVerifier


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
    payload = json.loads(_resolve_existing_path(path).read_text(encoding="utf-8-sig"))
    return _evidence_from_rows(payload)


def _evidence_from_rows(rows: list[object]) -> EvidenceSet:
    items: list[EvidenceItem] = []
    for idx, item in enumerate(rows):
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


def _load_evidence_payload(payload: object) -> EvidenceSet:
    if isinstance(payload, str):
        return _load_evidence(payload)
    if isinstance(payload, list):
        return _evidence_from_rows(payload)
    raise ValueError("evidence_json must be a file path string or a list of evidence objects.")


def _ensure_trace_parent(trace_out: str | None) -> None:
    if trace_out:
        Path(trace_out).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def _append_shell_trace_row(trace_out: str | None, row: dict[str, object]) -> None:
    if not trace_out:
        return
    _ensure_trace_parent(trace_out)
    with Path(trace_out).expanduser().resolve().open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _shell_trace_row(
    *,
    total_seconds: float = 0.0,
    verify_trace: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "trace_schema_version": 1,
        "total_seconds": total_seconds,
        "read_seconds": 0.0,
        "unitize_seconds": 0.0,
        "verify_seconds": 0.0,
        "enforce_seconds": 0.0,
        "polish_seconds": 0.0,
    }
    if isinstance(verify_trace, dict):
        payload.update(verify_trace)
        if "n_pairs" not in payload and "n_pairs_scored" in payload:
            payload["n_pairs"] = payload["n_pairs_scored"]
    return payload


def _strip_bom(s: str) -> str:
    return s.lstrip("\ufeff")


def _build_pipeline_config(args: object, *, policy: PolicyConfig, verifier: object, reranker: object, budget_policy: object, budget_config: object, conformal_state_path: str | None, trace_out: str | None, overrides: dict[str, object] | None = None) -> dict[str, object]:
    overrides = overrides or {}
    return {
        "policy_config": policy,
        "unitizer_mode": str(overrides.get("unitizer_mode", getattr(args, "unitizer", "sentence"))),
        "use_oss_nli": True,
        "verifier": verifier,
        "nli_model_name": getattr(args, "model_name", None),
        "nli_device": str(overrides.get("device", getattr(args, "device", "auto"))),
        "nli_dtype": str(overrides.get("dtype", getattr(args, "dtype", "auto"))),
        "topk_per_unit": int(overrides.get("topk_per_unit", getattr(args, "topk_per_unit", 12))),
        "max_pairs_total": overrides.get("max_pairs_total", getattr(args, "max_pairs_total", 200)),
        "reranker": reranker,
        "rerank_topk": getattr(args, "rerank_topk", 6),
        "conformal_state_path": conformal_state_path,
        "budget_policy": budget_policy,
        "budget_config": budget_config,
        "max_evidence_per_request": overrides.get("max_evidence_per_request", getattr(args, "max_evidence_per_request", None)),
        "max_batch_tokens": overrides.get("max_batch_tokens", getattr(args, "max_batch_tokens", None)),
        "evidence_max_chars": int(overrides.get("evidence_max_chars", getattr(args, "evidence_max_chars", 800))),
        "evidence_max_sentences": int(overrides.get("evidence_max_sentences", getattr(args, "evidence_max_sentences", 3))),
        "trace_out": trace_out,
    }


def handle_shell(args: object, sys_module: object) -> int:
    if bool(args.stdin_jsonl) != bool(args.stdout_jsonl):
        raise ValueError("Use --stdin-jsonl and --stdout-jsonl together.")

    conformal_state_path = None
    if isinstance(args.conformal_state, str) and args.conformal_state.strip():
        conformal_state_path = str(_resolve_existing_path(args.conformal_state))

    reranker = None
    if args.use_reranker:
        try:
            reranker = CrossEncoderReranker(model_name=args.reranker_model)
        except ImportError as exc:
            raise ImportError("v2 reranker requires optional dependency: pip install 'ega[rerank]'.") from exc

    budget_policy = None
    budget_config = None
    if args.use_budget:
        budget_policy = GreedyBudgetPolicy()
        budget_config = BudgetConfig(latency_budget_ms=args.latency_budget_ms, max_pairs_total=args.budget_max_pairs)

    base_policy = PolicyConfig(
        threshold_entailment=args.threshold_entailment,
        max_contradiction=args.max_contradiction,
        partial_allowed=args.partial_allowed,
    )
    verifier = NliCrossEncoderVerifier(model_name=args.model_name)

    if args.stdin_jsonl and args.stdout_jsonl:
        for raw_line in sys_module.stdin:
            line = raw_line.strip()
            if not line:
                continue
            line_trace_out: str | None = args.trace_out
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Each stdin-jsonl line must be a JSON object.")
                llm_summary = row.get("llm_summary", row.get("summary"))
                if not isinstance(llm_summary, str) or not llm_summary.strip():
                    minimal = {
                        "verified_extract": [],
                        "verified_text": "",
                        "decision": {
                            "allowed_units": [],
                            "dropped_units": [],
                            "refusal": True,
                            "reason_code": "invalid_input",
                            "summary_stats": {"kept_units": 0, "dropped_units": 0},
                        },
                        "stats": {
                            "kept_units": 0,
                            "dropped_units": 0,
                            "model_name": str(getattr(verifier, "model_name", "") or ""),
                        },
                        "polish_status": "skipped",
                    }
                    trace_out = str(row.get("trace_out", args.trace_out or "")).strip() or None
                    line_trace_out = trace_out
                    verify_trace: dict[str, object] = {}
                    get_trace = getattr(verifier, "get_last_verify_trace", None)
                    if callable(get_trace):
                        payload = get_trace()
                        if isinstance(payload, dict):
                            verify_trace = dict(payload)
                    _append_shell_trace_row(trace_out, _shell_trace_row(verify_trace=verify_trace))
                    print(json.dumps(minimal, sort_keys=True))
                    continue

                evidence_payload = row.get("evidence", row.get("evidence_json"))
                if evidence_payload is None:
                    raise ValueError("evidence is required.")
                evidence = _load_evidence_payload(evidence_payload)
                overrides = row.get("overrides", {})
                if not isinstance(overrides, dict):
                    overrides = {}
                policy = PolicyConfig(
                    threshold_entailment=float(overrides.get("threshold_entailment", row.get("threshold_entailment", base_policy.threshold_entailment))),
                    max_contradiction=float(overrides.get("max_contradiction", row.get("max_contradiction", base_policy.max_contradiction))),
                    partial_allowed=bool(overrides.get("partial_allowed", row.get("partial_allowed", base_policy.partial_allowed))),
                )
                trace_out = str(overrides.get("trace_out", row.get("trace_out", args.trace_out or ""))).strip() or None
                line_trace_out = trace_out
                _ensure_trace_parent(trace_out)
                t0 = time.perf_counter()
                result = api.verify_answer(
                    llm_output=llm_summary,
                    source_text="",
                    evidence=evidence,
                    config=_build_pipeline_config(
                        args,
                        policy=policy,
                        verifier=verifier,
                        reranker=reranker,
                        budget_policy=budget_policy,
                        budget_config=budget_config,
                        conformal_state_path=conformal_state_path,
                        trace_out=trace_out,
                        overrides=overrides,
                    ),
                    return_pipeline_output=True,
                )
                _ = time.perf_counter() - t0
                print(json.dumps(result, sort_keys=True))
            except Exception as exc:
                verify_trace: dict[str, object] = {}
                get_trace = getattr(verifier, "get_last_verify_trace", None)
                if callable(get_trace):
                    payload = get_trace()
                    if isinstance(payload, dict):
                        verify_trace = dict(payload)
                _append_shell_trace_row(line_trace_out, _shell_trace_row(verify_trace=verify_trace))
                print(json.dumps({"error": str(exc)}))
        return 0

    print("[EGA] Interactive shell starting. Loading verifier...")
    print("[EGA] Ready. Enter paths or 'exit' to quit.")
    while True:
        summary_path = input("summary path> ").strip()
        if summary_path.lower() in ("exit", "quit"):
            break
        if not summary_path:
            print("[EGA] summary path required.")
            continue
        summary_file = Path(summary_path).expanduser()
        if summary_file.is_dir():
            print(f"[EGA ERROR] Expected file path, got directory: {summary_path}")
            continue
        if not summary_file.exists():
            print(f"[EGA ERROR] File not found: {summary_path}")
            continue

        evidence_path = input("evidence path> ").strip()
        if evidence_path.lower() in ("exit", "quit"):
            break
        if not evidence_path:
            print("[EGA] evidence path required.")
            continue
        evidence_file = Path(evidence_path).expanduser()
        if evidence_file.is_dir():
            print(f"[EGA ERROR] Expected file path, got directory: {evidence_path}")
            continue
        if not evidence_file.exists():
            print(f"[EGA ERROR] File not found: {evidence_path}")
            continue

        try:
            t0 = time.perf_counter()
            _ensure_trace_parent(args.trace_out)
            result = api.verify_answer(
                llm_output=_load_answer(summary_path),
                source_text="",
                evidence=_load_evidence(evidence_path),
                config=_build_pipeline_config(
                    args,
                    policy=base_policy,
                    verifier=verifier,
                    reranker=reranker,
                    budget_policy=budget_policy,
                    budget_config=budget_config,
                    conformal_state_path=conformal_state_path,
                    trace_out=args.trace_out,
                ),
                return_pipeline_output=True,
            )
            t1 = time.perf_counter()
            if isinstance(result.get("verified_text"), str):
                result["verified_text"] = _strip_bom(result["verified_text"])
            verified_extract = result.get("verified_extract")
            if isinstance(verified_extract, list):
                for row in verified_extract:
                    if isinstance(row, dict) and isinstance(row.get("text"), str):
                        row["text"] = _strip_bom(row["text"])
            stats = result.get("stats", {})
            decision = result.get("decision", {})
            print(
                f"[EGA] request_seconds={t1 - t0:.6f} "
                f"kept={stats.get('kept_units')} dropped={stats.get('dropped_units')} refusal={decision.get('refusal')}"
            )
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"[EGA ERROR] {e}")

    return 0
