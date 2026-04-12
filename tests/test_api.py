from __future__ import annotations

import json
from pathlib import Path

import pytest

from ega.api import verify_answer
from ega.config import OutputConfig, PipelineConfig
from ega.contract import PolicyConfig


def test_verify_answer_maps_inputs_and_outputs(tmp_path: Path) -> None:
    scores_path = tmp_path / "scores.jsonl"
    scores_path.write_text(
        json.dumps({"unit_id": "u0001", "score": 0.9, "label": "entailment"}) + "\n",
        encoding="utf-8",
    )

    output = verify_answer(
        prompt="optional prompt",
        source_text="Paris is in France.",
        llm_output="Paris is in France.",
        config={
            "policy_config": PolicyConfig(threshold_entailment=0.5, max_contradiction=0.9),
            "scores_jsonl_path": str(scores_path),
        },
    )

    assert set(output) == {"verified_text", "verified_units", "dropped_units", "trace"}
    assert output["verified_text"] == "Paris is in France."
    assert output["verified_units"] == [{"unit_id": "u0001", "text": "Paris is in France."}]
    assert output["dropped_units"] == []
    assert isinstance(output["trace"], dict)
    required_trace_fields = {
        "trace_schema_version",
        "n_units",
        "unit_ids",
        "scored_units",
        "verifier_type",
        "kept_units",
        "dropped_units",
        "abstained_units",
        "correction_enabled",
        "correction_max_retries",
        "correction_retries_attempted",
        "correction_corrected_unit_count",
        "correction_still_failed_count",
        "correction_reverify_occurred",
        "correction_stopped_reason",
        "total_seconds",
        "unitize_seconds",
        "verify_seconds",
        "enforce_seconds",
    }
    assert required_trace_fields.issubset(set(output["trace"]))
    assert output["trace"]["unit_ids"] == ["u0001"]
    assert output["trace"]["verifier_type"] == "jsonl_scores"
    assert "coverage_pool_topk" not in output["trace"]


def test_verify_answer_accepts_pipeline_config(tmp_path: Path) -> None:
    scores_path = tmp_path / "scores.jsonl"
    scores_path.write_text(
        json.dumps({"unit_id": "u0001", "score": 0.9, "label": "entailment"}) + "\n",
        encoding="utf-8",
    )

    output = verify_answer(
        source_text="Paris is in France.",
        llm_output="Paris is in France.",
        config=PipelineConfig(
            policy=PolicyConfig(threshold_entailment=0.5, max_contradiction=0.9),
            scores_jsonl_path=str(scores_path),
        ),
    )

    assert set(output) == {"verified_text", "verified_units", "dropped_units", "trace"}
    assert output["verified_text"] == "Paris is in France."
    assert output["verified_units"] == [{"unit_id": "u0001", "text": "Paris is in France."}]
    assert output["dropped_units"] == []
    assert isinstance(output["trace"], dict)


def test_verify_answer_rejects_invalid_inputs() -> None:
    with pytest.raises(TypeError, match="llm_output must be a string"):
        verify_answer(  # type: ignore[arg-type]
            llm_output=123,
            source_text="source",
            config={"policy_config": PolicyConfig(threshold_entailment=0.5, max_contradiction=0.9)},
        )

    with pytest.raises(ValueError, match="config dict must include policy_config"):
        verify_answer(
            llm_output="answer",
            source_text="source",
            config={},
        )


def test_verify_answer_accepts_downstream_mode_aliases_in_pipeline_config(tmp_path: Path) -> None:
    scores_path = tmp_path / "scores.jsonl"
    scores_path.write_text(
        json.dumps({"unit_id": "u0001", "score": 0.9, "label": "entailment"}) + "\n",
        encoding="utf-8",
    )

    output = verify_answer(
        source_text="Paris is in France.",
        llm_output="Paris is in France.",
        config=PipelineConfig(
            policy=PolicyConfig(threshold_entailment=0.5, max_contradiction=0.9),
            scores_jsonl_path=str(scores_path),
            output=OutputConfig(downstream_compatibility_mode="strict"),
        ),
        return_pipeline_output=True,
    )

    assert output["route_status"] == "READY"
    assert output["payload_status"] == "ACCEPT"


def test_verify_answer_rejects_invalid_downstream_mode_in_pipeline_config(tmp_path: Path) -> None:
    scores_path = tmp_path / "scores.jsonl"
    scores_path.write_text(
        json.dumps({"unit_id": "u0001", "score": 0.9, "label": "entailment"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="downstream_compatibility_mode"):
        verify_answer(
            source_text="Paris is in France.",
            llm_output="Paris is in France.",
            config=PipelineConfig(
                policy=PolicyConfig(threshold_entailment=0.5, max_contradiction=0.9),
                scores_jsonl_path=str(scores_path),
                output=OutputConfig(downstream_compatibility_mode="legacy_v0"),
            ),
            return_pipeline_output=True,
        )
