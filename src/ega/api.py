"""Public package API for answer verification."""

from __future__ import annotations

from typing import Any

from ega.config import PipelineConfig
from ega.pipeline import run_pipeline
from ega.types import EvidenceItem, EvidenceSet


def _pipeline_kwargs_from_config(config: PipelineConfig | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config, dict):
        return dict(config)

    pipeline_kwargs: dict[str, Any] = {
        "policy_config": config.policy,
        "nli_model_name": config.verifier.model,
        "nli_device": config.verifier.device,
        "nli_dtype": config.verifier.dtype,
        "topk_per_unit": config.verifier.top_k,
        "max_pairs_total": config.verifier.max_pairs,
        "use_oss_nli": config.verifier.use_oss_nli,
        "verifier": config.verifier.verifier,
        "conformal_epsilon": None if config.conformal is None else config.conformal.epsilon,
        "conformal_state_path": config.conformal_state_path,
        "budget_config": config.budget,
        "budget_policy": config.budget_policy,
        "reranker": config.reranker.reranker if config.reranker.enabled else None,
        "rerank_topk": config.reranker.top_k,
        "render_safe_answer": config.output.render_safe_answer,
        "trace_out": config.output.trace_out,
        "enable_polish_validation": config.output.enable_polish_validation,
        "scores_jsonl_path": config.scores_jsonl_path,
        "unitizer_mode": config.unitizer_mode,
        "accept_threshold": config.accept_threshold,
        "enable_correction": config.enable_correction,
        "max_retries": config.max_retries,
    }
    if config.extras:
        pipeline_kwargs.update(config.extras)
    return pipeline_kwargs


def verify_answer(
    *,
    llm_output: str,
    source_text: str,
    config: PipelineConfig | dict[str, Any],
    prompt: str | None = None,
    evidence: EvidenceSet | None = None,
    return_pipeline_output: bool = False,
) -> dict[str, Any]:
    """Verify an answer against source text via the existing pipeline."""
    pipeline_kwargs = _pipeline_kwargs_from_config(config)
    evidence_set = evidence
    if evidence_set is None:
        evidence_set = EvidenceSet(
            items=[
                EvidenceItem(
                    id="source",
                    text=source_text,
                    metadata={},
                )
            ]
        )
    pipeline_output = run_pipeline(
        llm_summary_text=llm_output,
        evidence=evidence_set,
        **pipeline_kwargs,
    )

    if return_pipeline_output:
        return pipeline_output

    response: dict[str, Any] = {
        "verified_text": pipeline_output.get("verified_text", ""),
        "verified_units": pipeline_output.get("verified_extract", []),
        "dropped_units": pipeline_output.get("decision", {}).get("dropped_units", []),
    }
    if "trace" in pipeline_output:
        response["trace"] = pipeline_output["trace"]
    return response
