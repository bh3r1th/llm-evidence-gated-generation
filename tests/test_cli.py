"""CLI behavior tests for the `ega run` command."""

from __future__ import annotations

import json
from pathlib import Path

from ega.cli import main
from ega.serialization import from_json
from ega.types import EnforcementResult


def _write_files(tmp_path: Path, *, answer: str) -> tuple[Path, Path]:
    answer_file = tmp_path / "answer.txt"
    evidence_file = tmp_path / "evidence.json"

    answer_file.write_text(answer, encoding="utf-8")
    evidence_file.write_text(
        json.dumps(
            [
                {
                    "id": "doc-1",
                    "text": "Paris is the capital of France.",
                    "metadata": {"source": "test"},
                }
            ]
        ),
        encoding="utf-8",
    )
    return answer_file, evidence_file


def test_run_outputs_enforcement_json_for_partial_answer(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    answer_file, evidence_file = _write_files(
        tmp_path,
        answer="Paris is the capital of France.\nFrance has 1000 capitals.",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "ega",
            "run",
            "--answer-file",
            str(answer_file),
            "--evidence-file",
            str(evidence_file),
            "--partial-allowed",
        ],
    )

    exit_code = main()

    captured = capsys.readouterr()
    result = from_json(captured.out.strip(), EnforcementResult)

    assert exit_code == 0
    assert result.decision.refusal is False
    assert result.final_text == "Paris is the capital of France."
    assert result.kept_units == ["u0001"]
    assert result.dropped_units == ["u0002"]


def test_run_refuses_when_partial_not_allowed(tmp_path: Path, monkeypatch, capsys) -> None:
    answer_file, evidence_file = _write_files(
        tmp_path,
        answer="Paris is the capital of France.\nFrance has 1000 capitals.",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "ega",
            "run",
            "--answer-file",
            str(answer_file),
            "--evidence-file",
            str(evidence_file),
        ],
    )

    exit_code = main()

    captured = capsys.readouterr()
    result = from_json(captured.out.strip(), EnforcementResult)

    assert exit_code == 0
    assert result.decision.refusal is True
    assert result.final_text is None
    assert result.refusal_message is not None
