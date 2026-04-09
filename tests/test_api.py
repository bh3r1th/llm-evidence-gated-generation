from __future__ import annotations

import json
from pathlib import Path

from ega.api import verify_answer
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

    assert set(output) == {"verified_text", "verified_units", "dropped_units"}
    assert output["verified_text"] == "Paris is in France."
    assert output["verified_units"] == [{"unit_id": "u0001", "text": "Paris is in France."}]
    assert output["dropped_units"] == []
