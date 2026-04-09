"""Public package API for answer verification."""

from __future__ import annotations

from typing import Any

from ega.pipeline import run_pipeline
from ega.types import EvidenceItem, EvidenceSet


def verify_answer(
    *,
    llm_output: str,
    source_text: str,
    config: dict[str, Any],
    prompt: str | None = None,
) -> dict[str, Any]:
    """Verify an answer against source text via the existing pipeline."""
    pipeline_kwargs = dict(config)
    pipeline_output = run_pipeline(
        llm_summary_text=llm_output,
        evidence=EvidenceSet(
            items=[
                EvidenceItem(
                    id="source",
                    text=source_text,
                    metadata={},
                )
            ]
        ),
        **pipeline_kwargs,
    )

    response: dict[str, Any] = {
        "verified_text": pipeline_output.get("verified_text", ""),
        "verified_units": pipeline_output.get("verified_extract", []),
        "dropped_units": pipeline_output.get("decision", {}).get("dropped_units", []),
    }
    if "trace" in pipeline_output:
        response["trace"] = pipeline_output["trace"]
    return response
