from __future__ import annotations

import ega.cli as cli


def test_pipeline_cli_accepts_and_passes_device_dtype(monkeypatch, capsys) -> None:
    seen: dict[str, object] = {}

    def fake_run_pipeline(**kwargs):  # type: ignore[no-untyped-def]
        seen.update(kwargs)
        return {
            "verified_extract": [],
            "verified_text": "",
            "decision": {"refusal": False, "reason_code": "ok", "summary_stats": {}},
            "stats": {"kept_units": 0, "dropped_units": 0, "model_name": "fake"},
        }

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "ega",
            "pipeline",
            "--llm-summary-file",
            "examples/pipeline_demo/llm_summary.txt",
            "--evidence-json",
            "examples/pipeline_demo/evidence.json",
            "--scores-jsonl",
            "examples/pipeline_demo/scores.jsonl",
            "--device",
            "cpu",
            "--dtype",
            "float16",
        ],
    )

    exit_code = cli.main()
    _ = capsys.readouterr()

    assert exit_code == 0
    assert seen["nli_device"] == "cpu"
    assert seen["nli_dtype"] == "float16"
