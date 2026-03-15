from __future__ import annotations

import json
from pathlib import Path


def test_eval_dataset_pilot_rows_have_required_fields_and_consistent_ids() -> None:
    dataset_path = Path("examples/v2/eval_dataset_pilot.jsonl")
    rows = [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(rows) == 10
    seen_example_ids: set[str] = set()

    for row in rows:
        assert set(("id", "prompt", "llm_summary_text", "evidence_json", "gold_units")).issubset(row)
        example_id = str(row["id"])
        assert example_id not in seen_example_ids
        seen_example_ids.add(example_id)

        evidence = row["evidence_json"]
        assert isinstance(evidence, list)
        evidence_ids = [str(item["id"]) for item in evidence]
        assert len(evidence_ids) == len(set(evidence_ids))

        gold_units = row["gold_units"]
        assert isinstance(gold_units, list)
        unit_ids = [str(item["unit_id"]) for item in gold_units]
        assert len(unit_ids) == len(set(unit_ids))

        for item in evidence:
            assert set(("id", "text")).issubset(item)
            assert isinstance(item["text"], str)

        for unit in gold_units:
            assert set(
                (
                    "unit_id",
                    "text",
                    "supported",
                    "required_evidence_ids",
                    "relevant_evidence_ids",
                )
            ).issubset(unit)
            required_ids = [str(item) for item in unit["required_evidence_ids"]]
            relevant_ids = [str(item) for item in unit["relevant_evidence_ids"]]
            assert set(required_ids).issubset(relevant_ids) or not required_ids
            assert set(required_ids).issubset(evidence_ids)
            assert set(relevant_ids).issubset(evidence_ids)
            if not bool(unit["supported"]):
                assert required_ids == []
