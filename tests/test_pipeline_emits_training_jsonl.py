from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from ega.contract import PolicyConfig
from ega.pipeline import run_pipeline
from ega.types import EvidenceItem, EvidenceSet
from ega.v2.coverage import CoverageConfig
from ega.v2.rewards import RewardConfig


def test_pipeline_emits_training_jsonl_row_with_expected_keys(tmp_path: Path) -> None:
    scores_path = tmp_path / f"emit_training_scores_{uuid4().hex}.jsonl"
    emit_path = tmp_path / f"training_examples_{uuid4().hex}.jsonl"
    scores_path.write_text(
        json.dumps({"unit_id": "u0001", "score": 0.95, "raw": {"chosen_evidence_id": "e1"}}) + "\n",
        encoding="utf-8",
    )

    run_pipeline(
        llm_summary_text="Supported fact.",
        evidence=EvidenceSet(items=[EvidenceItem(id="e1", text="Supported fact.", metadata={})]),
        unitizer_mode="sentence",
        policy_config=PolicyConfig(
            threshold_entailment=0.8,
            max_contradiction=0.2,
            partial_allowed=True,
        ),
        scores_jsonl_path=str(scores_path),
        coverage_config=CoverageConfig(pool_topk=5),
        reward_config=RewardConfig(),
        emit_training_example_path=str(emit_path),
        training_example_id="train-row-1",
    )

    lines = [line for line in emit_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    row = json.loads(lines[0])

    expected_core = {
        "id",
        "input_prompt",
        "generated_text",
        "units",
        "pool_candidates",
        "used_evidence",
        "decisions",
        "verifier_scores",
        "coverage",
        "rewards",
        "reward_summary",
    }
    assert expected_core.issubset(set(row.keys()))
    assert row["id"] == "train-row-1"
    assert isinstance(row["units"], list)
    assert isinstance(row["pool_candidates"], dict)
    assert isinstance(row["used_evidence"], dict)
    assert isinstance(row["decisions"], dict)
    assert isinstance(row["verifier_scores"], dict)
    assert isinstance(row["coverage"], dict)
    assert isinstance(row["rewards"], dict)
    assert isinstance(row["reward_summary"], dict)
