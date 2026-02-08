from __future__ import annotations

import json
from pathlib import Path

from ega.benchmark import run_benchmark as run_benchmark_core
from ega.cli import main
from ega.types import EvidenceSet, VerificationScore


class FakeVerifier:
    model_name = "fake-nli-smoke"

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
            raw={"source": "fake"},
        )


def test_benchmark_cli_smoke_uses_example_data(monkeypatch, capsys) -> None:
    verifier = FakeVerifier(
        {
            "Paris is the capital of France.": (0.95, 0.02, "entailment"),
            "France has one capital.": (0.91, 0.03, "entailment"),
            "The Moon is made of cheese.": (0.05, 0.9, "contradiction"),
            "The Moon orbits Earth.": (0.9, 0.04, "entailment"),
            "Water boils at 100 C at sea level.": (0.92, 0.03, "entailment"),
            "Water freezes at 0 C.": (0.9, 0.04, "entailment"),
            "Tokyo is in Japan.": (0.94, 0.03, "entailment"),
            "Tokyo is in Brazil.": (0.08, 0.88, "contradiction"),
            "Python is a programming language.": (0.93, 0.03, "entailment"),
            "Python was first released in 1991.": (0.88, 0.05, "entailment"),
            "Mount Everest is the tallest mountain above sea level.": (0.91, 0.04, "entailment"),
            "Mount Everest is in Nepal.": (0.85, 0.08, "entailment"),
        }
    )

    def fake_run_benchmark(*, data_path, out_path=None, model_name=None, policy_config=None):
        return run_benchmark_core(
            data_path=data_path,
            out_path=out_path,
            model_name=model_name,
            policy_config=policy_config,
            verifier=verifier,
        )

    monkeypatch.setattr("ega.cli.run_benchmark", fake_run_benchmark)
    monkeypatch.setattr(
        "sys.argv",
        ["ega", "benchmark", "--data", str(Path("data") / "benchmark_example.jsonl")],
    )

    exit_code = main()

    captured = capsys.readouterr()
    payload = json.loads(captured.out.strip())
    assert exit_code == 0
    assert payload["n_examples"] == 6
    assert payload["total_units"] == 12
    assert "keep_rate" in payload
    assert "refusal_rate" in payload
    assert "avg_entailment_kept" in payload
