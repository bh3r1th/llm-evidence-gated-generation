from __future__ import annotations

import json
from pathlib import Path

from ega.benchmark import run_benchmark
from ega.contract import PolicyConfig
from ega.types import EvidenceSet, VerificationScore


class FakeVerifier:
    model_name = "fake-nli"

    def __init__(self, scores_by_text: dict[str, tuple[float, float, str]]) -> None:
        self._scores_by_text = scores_by_text

    def verify(self, *, unit_text: str, unit_id: str, evidence: EvidenceSet) -> VerificationScore:
        _ = evidence
        entailment, contradiction, label = self._scores_by_text[unit_text]
        return VerificationScore(
            unit_id=unit_id,
            entailment=entailment,
            contradiction=contradiction,
            neutral=max(0.0, 1.0 - entailment - contradiction),
            label=label,
            raw={"verifier": "fake"},
        )


def _write_jsonl(path: Path) -> None:
    rows = [
        {
            "id": "ex1",
            "answer": "A one. B two. C three.",
            "evidence": [
                {"id": "e1", "text": "A one.", "metadata": {"src": "t"}},
                {"id": "e2", "text": "C three.", "metadata": {"src": "t"}},
            ],
        },
        {
            "id": "ex2",
            "answer": "- D one\n- E two",
            "unitizer": "bullets",
            "policy": {"threshold_entailment": 0.85},
            "evidence": [
                {"id": "e3", "text": "D one", "metadata": {}},
                {"id": "e4", "text": "E two", "metadata": {}},
            ],
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def test_run_benchmark_aggregates_expected_metrics(tmp_path: Path) -> None:
    data_path = tmp_path / "bench.jsonl"
    out_path = tmp_path / "summary.json"
    _write_jsonl(data_path)

    verifier = FakeVerifier(
        {
            "A one.": (0.9, 0.05, "entailment"),
            "B two.": (0.2, 0.7, "contradiction"),
            "C three.": (0.8, 0.2, "entailment"),
            "- D one": (0.84, 0.1, "entailment"),
            "- E two": (0.1, 0.9, "contradiction"),
        }
    )

    summary = run_benchmark(
        data_path=data_path,
        out_path=out_path,
        policy_config=PolicyConfig(),
        verifier=verifier,
    )

    assert summary["n_examples"] == 2
    assert summary["total_units"] == 5
    assert summary["kept_units"] == 2
    assert summary["dropped_units"] == 3
    assert summary["keep_rate"] == 0.4
    assert summary["refusal_rate"] == 0.5
    assert summary["avg_entailment_kept"] == 0.85
    assert summary["policy_config"] == {
        "threshold_entailment": 0.8,
        "max_contradiction": 0.2,
        "partial_allowed": True,
    }
    assert summary["model_name"] == "fake-nli"

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == summary
