from __future__ import annotations

import json
from pathlib import Path

from ega.cli import main


def _write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_pipeline_cli_paths_work_with_relative_files(tmp_path: Path, monkeypatch, capsys) -> None:
    summary = _write(tmp_path / "llm_summary.txt", "One.\nTwo.\n")
    evidence = _write(
        tmp_path / "evidence.json",
        json.dumps([{"id": "e1", "text": "One.", "metadata": {}}]),
    )
    scores = _write(
        tmp_path / "scores.jsonl",
        "\n".join(
            [
                json.dumps({"unit_id": "u0001", "score": 0.95}),
                json.dumps({"unit_id": "u0002", "score": 0.10}),
            ]
        )
        + "\n",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "ega",
            "pipeline",
            "--llm-summary-file",
            summary.name,
            "--evidence-json",
            evidence.name,
            "--scores-jsonl",
            scores.name,
            "--unitizer",
            "sentence",
            "--partial-allowed",
        ],
    )

    exit_code = main()
    payload = json.loads(capsys.readouterr().out.strip())

    assert exit_code == 0
    assert payload["verified_extract"]
    assert "decision" in payload


def test_pipeline_cli_missing_file_returns_clean_error(tmp_path: Path, monkeypatch, capsys) -> None:
    summary = _write(tmp_path / "llm_summary.txt", "One.\n")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "ega",
            "pipeline",
            "--llm-summary-file",
            summary.name,
            "--evidence-json",
            "missing_evidence.json",
            "--scores-jsonl",
            "missing_scores.jsonl",
            "--unitizer",
            "sentence",
        ],
    )

    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "[EGA ERROR]" in captured.err
    assert "File not found:" in captured.err
    assert "Traceback" not in captured.err
