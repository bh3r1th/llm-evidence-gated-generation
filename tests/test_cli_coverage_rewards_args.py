from __future__ import annotations

import json
from pathlib import Path

import ega.cli as cli


def _write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_pipeline_cli_coverage_rewards_and_emit_args_flow_into_run_pipeline(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    seen: dict[str, object] = {}

    def fake_verify_answer(**kwargs):  # type: ignore[no-untyped-def]
        seen.update(kwargs)
        return {
            "verified_extract": [],
            "verified_text": "",
            "decision": {"refusal": False, "reason_code": "ok", "summary_stats": {}},
            "stats": {"kept_units": 0, "dropped_units": 0, "model_name": "fake"},
        }

    summary = _write(tmp_path / "llm_summary.txt", "One.\n")
    evidence = _write(
        tmp_path / "evidence.json",
        json.dumps([{"id": "e1", "text": "One.", "metadata": {}}]),
    )
    scores = _write(tmp_path / "scores.jsonl", json.dumps({"unit_id": "u0001", "score": 0.9}) + "\n")
    emit_path = tmp_path / "train.jsonl"

    monkeypatch.setattr(cli, "verify_answer", fake_verify_answer)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "ega",
            "pipeline",
            "--llm-summary-file",
            str(summary),
            "--evidence-json",
            str(evidence),
            "--scores-jsonl",
            str(scores),
            "--coverage",
            "--coverage-pool-topk",
            "33",
            "--rewards",
            "--reward-w-support",
            "1.2",
            "--reward-w-hallucination",
            "2.3",
            "--reward-w-abstain",
            "0.7",
            "--reward-w-coverage",
            "1.4",
            "--reward-clamp-min",
            "-3.0",
            "--reward-clamp-max",
            "4.0",
            "--emit-training-jsonl",
            str(emit_path),
            "--training-example-id",
            "example-123",
        ],
    )

    exit_code = cli.main()
    _ = capsys.readouterr()

    assert exit_code == 0
    config = seen["config"]
    extras = getattr(config, "extras")
    coverage_config = extras.get("coverage_config")
    assert coverage_config is not None
    assert getattr(coverage_config, "pool_topk") == 33
    reward_config = extras.get("reward_config")
    assert reward_config is not None
    assert getattr(reward_config, "w_support") == 1.2
    assert getattr(reward_config, "w_hallucination") == 2.3
    assert getattr(reward_config, "w_abstain") == 0.7
    assert getattr(reward_config, "w_coverage") == 1.4
    assert getattr(reward_config, "clamp_min") == -3.0
    assert getattr(reward_config, "clamp_max") == 4.0
    assert extras.get("emit_training_example_path") == str(emit_path)
    assert extras.get("training_example_id") == "example-123"


def test_pipeline_cli_coverage_rewards_default_off_passes_none(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    seen: dict[str, object] = {}

    def fake_verify_answer(**kwargs):  # type: ignore[no-untyped-def]
        seen.update(kwargs)
        return {
            "verified_extract": [],
            "verified_text": "",
            "decision": {"refusal": False, "reason_code": "ok", "summary_stats": {}},
            "stats": {"kept_units": 0, "dropped_units": 0, "model_name": "fake"},
        }

    summary = _write(tmp_path / "llm_summary.txt", "One.\n")
    evidence = _write(
        tmp_path / "evidence.json",
        json.dumps([{"id": "e1", "text": "One.", "metadata": {}}]),
    )
    scores = _write(tmp_path / "scores.jsonl", json.dumps({"unit_id": "u0001", "score": 0.9}) + "\n")

    monkeypatch.setattr(cli, "verify_answer", fake_verify_answer)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "ega",
            "pipeline",
            "--llm-summary-file",
            str(summary),
            "--evidence-json",
            str(evidence),
            "--scores-jsonl",
            str(scores),
        ],
    )

    exit_code = cli.main()
    _ = capsys.readouterr()

    assert exit_code == 0
    config = seen["config"]
    extras = getattr(config, "extras")
    assert extras.get("coverage_config") is None
    assert extras.get("reward_config") is None
    assert extras.get("emit_training_example_path") is None
    assert extras.get("training_example_id") is None
