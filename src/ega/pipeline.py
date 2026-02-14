"""File-driven end-to-end pipeline runner for provided summary inputs."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ega.contract import PolicyConfig
from ega.enforcer import Enforcer
from ega.polish.gate import PolishGateConfig, apply_polish_gate
from ega.polish.types import PolishedUnit
from ega.providers.jsonl_scores import JsonlScoresProvider
from ega.text_clean import clean_text
from ega.types import AnswerCandidate, EvidenceItem, EvidenceSet, Unit, VerificationScore
from ega.unitization import unitize_answer


class _NliVerifierAdapter:
    def __init__(self, verifier: Any) -> None:
        self._verifier = verifier

    def verify(self, *, unit_text: str, unit_id: str, evidence: EvidenceSet) -> VerificationScore:
        score = self._verifier.verify_unit(unit_text, evidence)
        return VerificationScore(
            unit_id=unit_id,
            entailment=score.entailment,
            contradiction=score.contradiction,
            neutral=score.neutral,
            label=score.label,
            raw=dict(score.raw),
        )

    def verify_many(
        self,
        candidate: AnswerCandidate,
        evidence: EvidenceSet,
    ) -> list[VerificationScore]:
        verify_many = getattr(self._verifier, "verify_many", None)
        if callable(verify_many):
            scores = verify_many(candidate, evidence)
            return [
                VerificationScore(
                    unit_id=score.unit_id,
                    entailment=score.entailment,
                    contradiction=score.contradiction,
                    neutral=score.neutral,
                    label=score.label,
                    raw=dict(score.raw),
                )
                for score in scores
            ]

        mapped: list[VerificationScore] = []
        for unit in candidate.units:
            score = self._verifier.verify_unit(unit.text, evidence)
            mapped.append(
                VerificationScore(
                    unit_id=unit.id,
                    entailment=score.entailment,
                    contradiction=score.contradiction,
                    neutral=score.neutral,
                    label=score.label,
                    raw=dict(score.raw),
                )
            )
        return mapped

    def get_last_verify_trace(self) -> dict[str, Any]:
        getter = getattr(self._verifier, "get_last_verify_trace", None)
        if callable(getter):
            payload = getter()
            if isinstance(payload, dict):
                return dict(payload)
        return {}


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


def run_pipeline_request(
    *,
    llm_summary_text: str | None = None,
    llm_summary_file: str | Path | None = None,
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
    return run_pipeline(llm_summary_text=llm_summary_text, evidence=evidence, **kwargs)


def run_pipeline(
    llm_summary_text: str,
    evidence: EvidenceSet,
    *,
    unitizer_mode: str = "sentence",
    policy_config: PolicyConfig,
    scores_jsonl_path: str | None = None,
    use_oss_nli: bool = False,
    verifier: Any | None = None,
    nli_model_name: str | None = None,
    nli_device: str = "auto",
    nli_dtype: str = "auto",
    topk_per_unit: int = 12,
    max_pairs_total: int | None = 200,
    max_evidence_per_request: int | None = None,
    max_batch_tokens: int | None = None,
    evidence_max_chars: int = 800,
    evidence_max_sentences: int = 3,
    polished_json: dict[str, Any] | list[Any] | None = None,
    enable_polish_validation: bool = True,
    trace_out: str | None = None,
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

    if not scores_jsonl_path and not use_oss_nli:
        raise ValueError(
            "A scoring source is required: pass `scores_jsonl_path` or `use_oss_nli=True`."
        )

    read_t0 = time.perf_counter()
    mode = "markdown_bullet" if unitizer_mode == "bullets" else unitizer_mode
    cleaned_summary = clean_text(llm_summary_text)
    cleaned_evidence = EvidenceSet(
        items=[
            EvidenceItem(id=item.id, text=clean_text(item.text), metadata=dict(item.metadata))
            for item in evidence.items
        ]
    )
    timings["read_seconds"] = time.perf_counter() - read_t0
    counts["n_evidence"] = len(cleaned_evidence.items)

    unitize_t0 = time.perf_counter()
    raw_candidate = unitize_answer(cleaned_summary, mode=mode)
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

    if scores_jsonl_path:
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
        try:
            from ega.verifiers.nli_cross_encoder import NliCrossEncoderVerifier
        except ImportError as exc:
            raise ImportError(
                "OSS NLI verifier requires optional dependency: pip install 'ega[nli]'."
            ) from exc
        if verifier is None:
            load_t0 = time.perf_counter()
            nli_verifier = NliCrossEncoderVerifier(
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
            timings["load_seconds"] += time.perf_counter() - load_t0
        else:
            nli_verifier = verifier
        model_name = str(nli_verifier.model_name or "")
        verifier = _NliVerifierAdapter(nli_verifier)
        verify_compute_t0 = time.perf_counter()
        verify_many = getattr(verifier, "verify_many", None)
        if callable(verify_many):
            scores = verify_many(candidate, cleaned_evidence)
        else:
            scores = [
                verifier.verify(unit_text=unit.text, unit_id=unit.id, evidence=cleaned_evidence)
                for unit in candidate.units
            ]
        timings["verify_compute_seconds"] = time.perf_counter() - verify_compute_t0
        timings["verify_seconds"] = timings["load_seconds"] + timings["verify_compute_seconds"]
        trace_payload = verifier.get_last_verify_trace()
        if trace_payload:
            verify_detail.update(trace_payload)
            counts["n_pairs"] = int(trace_payload.get("n_pairs_scored", 0))
        else:
            counts["n_pairs"] = sum(
                len(score.raw.get("per_item_probs", []))
                for score in scores
                if isinstance(score.raw, dict)
            )
    timings["verify_compute_seconds"] = (
        float(verify_detail["preselect_seconds"])
        + float(verify_detail["tokenize_seconds"])
        + float(verify_detail["forward_seconds"])
        + float(verify_detail["post_seconds"])
    )

    enforce_t0 = time.perf_counter()
    result = Enforcer(config=policy_config).enforce(
        candidate=candidate,
        evidence=cleaned_evidence,
        scores=scores,
    )
    timings["enforce_seconds"] = time.perf_counter() - enforce_t0

    kept_ids = set(result.kept_units)
    verified_units = [unit for unit in candidate.units if unit.id in kept_ids]
    verified_extract = [
        {"unit_id": unit.id, "text": clean_text(unit.text)} for unit in verified_units
    ]
    verified_text = clean_text("\n".join(unit["text"] for unit in verified_extract))

    output: dict[str, Any] = {
        "verified_extract": verified_extract,
        "verified_text": verified_text,
        "decision": asdict(result.decision),
        "stats": {
            **dict(result.decision.summary_stats),
            "model_name": model_name,
        },
    }

    def _append_trace(payload: dict[str, Any]) -> None:
        if not trace_out:
            return
        total_t1 = time.perf_counter()
        trace = {
            "total_seconds": total_t1 - total_t0,
            "read_seconds": timings["read_seconds"],
            "unitize_seconds": timings["unitize_seconds"],
            "verify_seconds": timings["verify_seconds"],
            "load_seconds": timings["load_seconds"],
            "verify_compute_seconds": timings["verify_compute_seconds"],
            "enforce_seconds": timings["enforce_seconds"],
            "polish_seconds": timings["polish_seconds"],
            "preselect_seconds": verify_detail["preselect_seconds"],
            "tokenize_seconds": verify_detail["tokenize_seconds"],
            "forward_seconds": verify_detail["forward_seconds"],
            "post_seconds": verify_detail["post_seconds"],
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
            "n_units": counts["n_units"],
            "n_evidence": counts["n_evidence"],
            "n_pairs": counts["n_pairs"],
            "kept_units": payload["stats"]["kept_units"],
            "dropped_units": payload["stats"]["dropped_units"],
            "refusal": payload["decision"]["refusal"],
            "model_name": payload["stats"].get("model_name"),
        }
        with Path(trace_out).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trace) + "\n")

    if polished_json is None:
        output["polish_status"] = "skipped"
        _append_trace(output)
        return output

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
        _append_trace(output)
        return output

    output["polish_status"] = "passed"
    output["polish_fail_reasons"] = []
    output["polished_extract"] = [
        {"unit_id": unit.unit_id, "edited_text": unit.edited_text}
        for unit in polished_units
    ]
    output["polished_text"] = "\n".join(unit.edited_text for unit in polished_units)
    timings["polish_seconds"] = time.perf_counter() - polish_t0
    _append_trace(output)
    return output


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
