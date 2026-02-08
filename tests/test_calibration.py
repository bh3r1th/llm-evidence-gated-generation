from __future__ import annotations

import json
from pathlib import Path

from ega.benchmark import calibrate_policies
from ega.types import EvidenceSet, VerificationScore


class FakeVerifier:
    model_name = "fake-nli"

    def __init__(self, by_text: dict[str, tuple[float, float, str]]) -> None:
        self._by_text = by_text

    def verify(self, *, unit_text: str, unit_id: str, evidence: EvidenceSet) -> VerificationScore:
        _ = evidence
        entailment, contradiction, label = self._by_text[unit_text]
        return VerificationScore(
            unit_id=unit_id,
            entailment=entailment,
            contradiction=contradiction,
            neutral=max(0.0, 1.0 - entailment - contradiction),
            label=label,
            raw={"source": "fake"},
        )


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def test_calibration_prefers_configs_meeting_refusal_target(tmp_path: Path) -> None:
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
    verifier = FakeVerifier({"Only unit.": (0.61, 0.34, "entailment")})

    calibration = calibrate_policies(data_path=data_path, verifier=verifier)

    assert calibration["best_policy_config"] == {
        "threshold_entailment": 0.6,
        "max_contradiction": 0.35,
        "partial_allowed": True,
    }


def test_calibration_tiebreak_prefers_higher_threshold_then_lower_contradiction(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "tie.jsonl"
    _write_rows(
        data_path,
        [
            {
                "id": "ex1",
                "answer": "A. B.",
                "evidence": [{"id": "e1", "text": "A"}, {"id": "e2", "text": "B"}],
            }
        ],
    )
    verifier = FakeVerifier(
        {
            "A.": (0.95, 0.01, "entailment"),
            "B.": (0.95, 0.01, "entailment"),
        }
    )

    calibration = calibrate_policies(data_path=data_path, verifier=verifier)

    assert calibration["best_policy_config"] == {
        "threshold_entailment": 0.8,
        "max_contradiction": 0.2,
        "partial_allowed": True,
    }
    assert len(calibration["top_configs"]) == 5
    assert calibration["top_configs"][0]["policy_config"] == calibration["best_policy_config"]
