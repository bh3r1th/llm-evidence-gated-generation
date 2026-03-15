from __future__ import annotations

import json
from pathlib import Path

from ega.contract import PolicyConfig
from ega.pipeline import run_pipeline
from ega.types import EvidenceItem, EvidenceSet


def test_pipeline_conformal_gating_drops_reject_and_abstain(tmp_path: Path) -> None:
    scores_path = tmp_path / "scores.jsonl"
    state_path = tmp_path / "conformal_state.json"

    scores_rows = [
        {"unit_id": "u0001", "score": 0.90},
        {"unit_id": "u0002", "score": 0.20},
        {"unit_id": "u0003", "score": 0.60},
    ]
    scores_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in scores_rows) + "\n",
        encoding="utf-8",
    )
    state_path.write_text(
        json.dumps(
            {
                "threshold": 0.6,
                "meta": {
                    "abstain_margin": 0.0,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    output = run_pipeline(
        llm_summary_text="A fact. B fact. C fact.",
        evidence=EvidenceSet(items=[EvidenceItem(id="e1", text="A fact.", metadata={})]),
        unitizer_mode="sentence",
        policy_config=PolicyConfig(
            threshold_entailment=0.0,
            max_contradiction=0.2,
            partial_allowed=True,
        ),
        scores_jsonl_path=str(scores_path),
        conformal_state_path=str(state_path),
    )

    assert output["decision"]["refusal"] is False
    assert output["decision"]["reason_code"] == "OK_PARTIAL"
    assert output["stats"]["kept_units"] == 1
    assert output["stats"]["dropped_units"] == 2
    assert [row["unit_id"] for row in output["verified_extract"]] == ["u0001"]
    assert output["decisions"]["u0003"] == "abstain"
    assert output["conformal"]["decision_counts"]["abstain"] == 1
    assert output["conformal"]["band_hit_count"] == 1
