from __future__ import annotations

import json
from pathlib import Path

from ega.benchmark import calibrate_policies
from ega.types import EvidenceSet, VerificationScore


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def test_calibration_initializes_verifier_once_and_reuses_cached_scores(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_path = tmp_path / "calibration.jsonl"
    _write_rows(
        data_path,
        [
            {
                "id": "ex1",
                "answer": "Only unit.",
                "evidence": [{"id": "e1", "text": "Only unit."}],
            }
        ],
    )

    class FakeCrossEncoder:
        init_count = 0
        verify_count = 0

        def __init__(self, model_name: str | None = None) -> None:
            type(self).init_count += 1
            self.model_name = model_name or "fake-nli"

        def verify_unit(self, unit_text: str, evidence: EvidenceSet) -> VerificationScore:
            _ = evidence
            type(self).verify_count += 1
            return VerificationScore(
                unit_id="unit",
                entailment=0.61,
                contradiction=0.34,
                neutral=0.05,
                label="entailment",
                raw={"source": "fake"},
            )

    monkeypatch.setattr("ega.benchmark.NliCrossEncoderVerifier", FakeCrossEncoder)

    calibration = calibrate_policies(data_path=data_path)

    assert FakeCrossEncoder.init_count == 1
    assert FakeCrossEncoder.verify_count == 1
    assert calibration["best_policy_config"] == {
        "threshold_entailment": 0.6,
        "max_contradiction": 0.35,
        "partial_allowed": True,
    }
