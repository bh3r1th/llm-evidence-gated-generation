"""Minimal public-package usage example for EGA v3."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from ega import PipelineConfig, PolicyConfig, verify_answer


def run() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        scores_path = Path(tmp_dir) / "scores.jsonl"
        scores_path.write_text(
            json.dumps({"unit_id": "u0001", "score": 0.95, "label": "entailment"}) + "\n",
            encoding="utf-8",
        )

        result = verify_answer(
            llm_output="Paris is in France.",
            source_text="Paris is in France.",
            config=PipelineConfig(
                policy=PolicyConfig(threshold_entailment=0.5, max_contradiction=0.9),
                scores_jsonl_path=str(scores_path),
            ),
        )

    print("verified_text:", result["verified_text"])
    print("verified_units:", result["verified_units"])
    print("dropped_units:", result["dropped_units"])
    print("trace_schema_version:", result["trace"]["trace_schema_version"])


if __name__ == "__main__":
    run()
