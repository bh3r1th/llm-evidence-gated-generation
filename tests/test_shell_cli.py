from __future__ import annotations

import io
import json
from pathlib import Path
from uuid import uuid4

import ega.cli as cli
from ega.contract import PolicyConfig
from ega.pipeline import run_pipeline
from ega.types import EvidenceItem, EvidenceSet, VerificationScore


def test_shell_parser_is_registered() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["shell"])
    assert args.command == "shell"


def test_shell_blank_summary_reprompts_and_does_not_call_pipeline(monkeypatch, capsys) -> None:
    calls = {"run_pipeline": 0}

    class FakeVerifier:
        def __init__(self, model_name=None, **_kwargs) -> None:  # type: ignore[no-untyped-def]
            self.model_name = model_name or "fake-shell"

    def fake_run_pipeline(**kwargs):  # type: ignore[no-untyped-def]
        _ = kwargs
        calls["run_pipeline"] += 1
        return {"verified_extract": [], "verified_text": ""}

    answers = iter(["", "exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    monkeypatch.setattr(cli, "NliCrossEncoderVerifier", FakeVerifier)
    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "ega",
            "shell",
            "--threshold-entailment",
            "0.7",
            "--max-contradiction",
            "0.2",
            "--partial-allowed",
        ],
    )

    exit_code = cli.main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "summary path required" in captured.out
    assert calls["run_pipeline"] == 0


def test_shell_stdin_stdout_jsonl_handles_two_requests_with_single_verifier_init(
    monkeypatch, capsys
) -> None:
    calls = {"init": 0}

    class _Score:
        entailment = 1.0
        contradiction = 0.0
        neutral = 0.0
        label = "entailment"
        raw = {}

    class FakeVerifier:
        def __init__(self, model_name=None, **_kwargs) -> None:  # type: ignore[no-untyped-def]
            _ = model_name
            calls["init"] += 1
            self.model_name = "fake-shell"

        def verify_unit(self, unit_text, evidence):  # type: ignore[no-untyped-def]
            _ = unit_text
            _ = evidence
            return _Score()

    req1 = {
        "llm_summary": "Fact one.",
        "evidence": [{"id": "e1", "text": "Fact one.", "metadata": {}}],
        "overrides": {"unitizer_mode": "sentence"},
    }
    req2 = {
        "llm_summary": "Fact two.",
        "evidence": [{"id": "e2", "text": "Fact two.", "metadata": {}}],
        "overrides": {"unitizer_mode": "sentence"},
    }
    stdin_payload = f"{json.dumps(req1)}\n{json.dumps(req2)}\n"

    monkeypatch.setattr(cli, "NliCrossEncoderVerifier", FakeVerifier)
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO(stdin_payload))
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "ega",
            "shell",
            "--stdin-jsonl",
            "--stdout-jsonl",
            "--threshold-entailment",
            "0.7",
            "--max-contradiction",
            "0.2",
            "--partial-allowed",
        ],
    )

    exit_code = cli.main()
    captured = capsys.readouterr()
    lines = [line for line in captured.out.splitlines() if line.strip()]

    assert exit_code == 0
    assert calls["init"] == 1
    assert len(lines) == 2
    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert first["verified_extract"]
    assert second["verified_extract"]


def test_shell_stdin_stdout_jsonl_appends_trace_per_line(monkeypatch, capsys) -> None:
    class FakeVerifier:
        def __init__(self, model_name=None) -> None:
            self.model_name = model_name or "fake-shell"

        def verify_many(self, candidate, evidence):  # type: ignore[no-untyped-def]
            _ = evidence
            return [
                VerificationScore(
                    unit_id=unit.id,
                    entailment=1.0,
                    contradiction=0.0,
                    neutral=0.0,
                    label="entailment",
                    raw={"chosen_evidence_id": "e1", "per_item_probs": []},
                )
                for unit in candidate.units
            ]

        @staticmethod
        def get_last_verify_trace() -> dict[str, float | int]:
            return {
                "preselect_seconds": 0.01,
                "tokenize_seconds": 0.02,
                "forward_seconds": 0.03,
                "post_seconds": 0.01,
                "n_pairs_scored": 2,
            }

    req1 = {"llm_summary": "Fact one.", "evidence": [{"id": "e1", "text": "Fact one."}]}
    req2 = {"llm_summary": "Fact two.", "evidence": [{"id": "e2", "text": "Fact two."}]}
    stdin_payload = f"{json.dumps(req1)}\n{json.dumps(req2)}\n"
    trace_path = Path("data") / f"shell_trace_{uuid4().hex}.jsonl"
    try:
        monkeypatch.setattr(cli, "NliCrossEncoderVerifier", FakeVerifier)
        monkeypatch.setattr(cli.sys, "stdin", io.StringIO(stdin_payload))
        monkeypatch.setattr(
            cli.sys,
            "argv",
            [
                "ega",
                "shell",
                "--stdin-jsonl",
                "--stdout-jsonl",
                "--trace-out",
                str(trace_path),
            ],
        )

        exit_code = cli.main()
        _ = capsys.readouterr()

        assert exit_code == 0
        lines = [line for line in trace_path.read_text(encoding="utf-8").splitlines() if line]
        assert len(lines) == 2
        assert all("verify_compute_seconds" in json.loads(line) for line in lines)
        assert all("n_pairs" in json.loads(line) for line in lines)
    finally:
        trace_path.unlink(missing_ok=True)


def test_shell_trace_verify_compute_within_20_percent_of_pipeline(monkeypatch, capsys) -> None:
    class FakeVerifier:
        def __init__(self, model_name=None, **_kwargs):  # type: ignore[no-untyped-def]
            self.model_name = model_name or "fake-shell"

        def verify_many(self, candidate, evidence):  # type: ignore[no-untyped-def]
            _ = evidence
            return [
                VerificationScore(
                    unit_id=unit.id,
                    entailment=1.0,
                    contradiction=0.0,
                    neutral=0.0,
                    label="entailment",
                    raw={"chosen_evidence_id": "e1", "per_item_probs": []},
                )
                for unit in candidate.units
            ]

        @staticmethod
        def get_last_verify_trace() -> dict[str, float | int]:
            return {
                "preselect_seconds": 0.02,
                "tokenize_seconds": 0.03,
                "forward_seconds": 0.04,
                "post_seconds": 0.01,
                "n_pairs_scored": 2,
                "device": "cpu",
                "dtype": "float32",
                "amp_enabled": False,
            }

    pipeline_trace_path = Path("data") / f"pipeline_trace_compare_{uuid4().hex}.jsonl"
    shell_trace_path = Path("data") / f"shell_trace_compare_{uuid4().hex}.jsonl"
    try:
        monkeypatch.setattr(cli, "NliCrossEncoderVerifier", FakeVerifier)
        evidence = EvidenceSet(items=[EvidenceItem(id="e1", text="Fact one.", metadata={})])
        run_pipeline(
            llm_summary_text="Fact one.",
            evidence=evidence,
            unitizer_mode="sentence",
            policy_config=PolicyConfig(
                threshold_entailment=0.8,
                max_contradiction=0.2,
                partial_allowed=True,
            ),
            use_oss_nli=True,
            verifier=FakeVerifier(),
            trace_out=str(pipeline_trace_path),
        )
        pipeline_row = json.loads(pipeline_trace_path.read_text(encoding="utf-8").splitlines()[0])

        req = {"llm_summary": "Fact one.", "evidence": [{"id": "e1", "text": "Fact one."}]}
        monkeypatch.setattr(cli.sys, "stdin", io.StringIO(json.dumps(req) + "\n"))
        monkeypatch.setattr(
            cli.sys,
            "argv",
            [
                "ega",
                "shell",
                "--stdin-jsonl",
                "--stdout-jsonl",
                "--trace-out",
                str(shell_trace_path),
                "--unitizer",
                "sentence",
            ],
        )
        exit_code = cli.main()
        _ = capsys.readouterr()
        assert exit_code == 0

        shell_row = json.loads(shell_trace_path.read_text(encoding="utf-8").splitlines()[0])
        assert shell_row["n_pairs"] == pipeline_row["n_pairs"]
        if pipeline_row["verify_compute_seconds"] == 0:
            assert shell_row["verify_compute_seconds"] == 0
        else:
            delta = abs(shell_row["verify_compute_seconds"] - pipeline_row["verify_compute_seconds"])
            ratio = delta / pipeline_row["verify_compute_seconds"]
            assert ratio <= 0.2
    finally:
        pipeline_trace_path.unlink(missing_ok=True)
        shell_trace_path.unlink(missing_ok=True)
