from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from ega.cli import main
from ega.serialization import from_json, to_json
from ega.types import EnforcementResult, GateDecision


def _path(name: str) -> Path:
    return Path("data") / f"{name}_{uuid4().hex}.json"


def _result_payload() -> EnforcementResult:
    return EnforcementResult(
        final_text="Paris is in France.",
        kept_units=["u1"],
        dropped_units=[],
        refusal_message=None,
        decision=GateDecision(
            allowed_units=["u1"],
            dropped_units=[],
            refusal=False,
            reason_code="OK_FULL",
            summary_stats={},
        ),
        scores=[],
        verified_units=[{"unit_id": "u1", "text": "Paris is in France."}],
    )


def test_cli_polish_validate_suppresses_polished_output_on_failure(monkeypatch, capsys) -> None:
    verified_path = _path("verified")
    polished_path = _path("polished")
    try:
        verified_path.write_text(to_json(_result_payload()), encoding="utf-8")
        polished_path.write_text(
            json.dumps([{"unit_id": "u1", "edited_text": "Paris is in Germany."}]),
            encoding="utf-8",
        )

        monkeypatch.setattr(
            "sys.argv",
            [
                "ega",
                "polish-validate",
                "--verified-json",
                str(verified_path),
                "--polished-json",
                str(polished_path),
            ],
        )

        exit_code = main()
        payload = from_json(capsys.readouterr().out.strip(), EnforcementResult)

        assert exit_code == 0
        assert payload.polish_status == "failed"
        assert payload.polished_units is None
        assert payload.polish_fail_reasons
    finally:
        verified_path.unlink(missing_ok=True)
        polished_path.unlink(missing_ok=True)


def test_cli_run_uses_scores_jsonl_provider(monkeypatch, capsys) -> None:
    answer_path = _path("answer")
    evidence_path = _path("evidence")
    scores_path = _path("scores").with_suffix(".jsonl")
    try:
        answer_path.write_text("A claim.\nAnother claim.", encoding="utf-8")
        evidence_path.write_text(
            json.dumps([{"id": "doc-1", "text": "unrelated evidence", "metadata": {}}]),
            encoding="utf-8",
        )
        scores_path.write_text(
            "\n".join(
                [
                    json.dumps({"unit_id": "u0001", "score": 0.95}),
                    json.dumps({"unit_id": "u0002", "score": 0.92}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(
            "sys.argv",
            [
                "ega",
                "run",
                "--answer-file",
                str(answer_path),
                "--evidence-file",
                str(evidence_path),
                "--scores-jsonl",
                str(scores_path),
            ],
        )

        exit_code = main()
        payload = from_json(capsys.readouterr().out.strip(), EnforcementResult)

        assert exit_code == 0
        assert payload.decision.refusal is False
        assert payload.final_text == "A claim.\nAnother claim."
        assert payload.kept_units == ["u0001", "u0002"]
    finally:
        answer_path.unlink(missing_ok=True)
        evidence_path.unlink(missing_ok=True)
        scores_path.unlink(missing_ok=True)
