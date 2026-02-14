from __future__ import annotations

import json
from pathlib import Path

import pytest

from ega.cli import main

pytestmark = pytest.mark.integration

pytest.importorskip("torch")
pytest.importorskip("transformers")

_FIXTURES_DIR = Path("examples") / "pipeline_demo"


def test_pipeline_cli_oss_nli_integration_smoke(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "ega",
            "pipeline",
            "--llm-summary-file",
            str(_FIXTURES_DIR / "llm_summary.txt"),
            "--evidence-json",
            str(_FIXTURES_DIR / "evidence.json"),
            "--use-oss-nli",
            "--unitizer",
            "sentence",
            "--partial-allowed",
        ],
    )

    exit_code = main()
    payload = json.loads(capsys.readouterr().out.strip())

    assert exit_code == 0
    assert isinstance(payload.get("verified_extract"), list)
    assert "decision" in payload
    assert "stats" in payload
    assert "model_name" in payload["stats"]

