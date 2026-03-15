from __future__ import annotations

import io
import json
from pathlib import Path

import ega.cli as cli


def _write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_pipeline_cli_v2_args_flow_into_run_pipeline(tmp_path: Path, monkeypatch, capsys) -> None:
    seen: dict[str, object] = {}

    class FakeReranker:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

    def fake_run_pipeline(**kwargs):  # type: ignore[no-untyped-def]
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
    conformal = _write(tmp_path / "conformal.json", json.dumps({"threshold": 0.5, "meta": {}}))

    monkeypatch.setattr(cli, "CrossEncoderReranker", FakeReranker)
    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)
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
            "--use-reranker",
            "--reranker-model",
            "cross-encoder/test-model",
            "--rerank-topk",
            "7",
            "--conformal-state",
            str(conformal),
            "--use-budget",
            "--latency-budget-ms",
            "123",
            "--budget-max-pairs",
            "42",
            "--accept-threshold",
            "0.05",
            "--render-safe-answer",
        ],
    )

    exit_code = cli.main()
    _ = capsys.readouterr()

    assert exit_code == 0
    assert isinstance(seen["reranker"], FakeReranker)
    assert seen["rerank_topk"] == 7
    assert seen["conformal_state_path"] == str(conformal.resolve())
    assert seen["budget_policy"] is not None
    budget_config = seen["budget_config"]
    assert budget_config is not None
    assert getattr(budget_config, "latency_budget_ms") == 123
    assert getattr(budget_config, "max_pairs_total") == 42
    assert seen["accept_threshold"] == 0.05
    assert seen["render_safe_answer"] is True


def test_v2_eval_cli_render_safe_answer_flag_flows_into_run_v2_eval(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    seen: dict[str, object] = {}

    def fake_run_v2_eval(**kwargs):  # type: ignore[no-untyped-def]
        seen.update(kwargs)
        return {"n_examples": 0, "variants": {}}

    dataset = _write(
        tmp_path / "dataset.jsonl",
        json.dumps({"id": "ex1", "llm_summary_text": "One.", "evidence_json": []}) + "\n",
    )
    out_path = tmp_path / "out.json"

    monkeypatch.setattr(cli, "run_v2_eval", fake_run_v2_eval)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "ega",
            "v2-eval",
            "--dataset",
            str(dataset),
            "--out",
            str(out_path),
            "--render-safe-answer",
        ],
    )

    exit_code = cli.main()
    _ = capsys.readouterr()

    assert exit_code == 0
    assert seen["render_safe_answer"] is True


def test_v2_eval_cli_final_summary_path_runs_release_generation(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    seen: dict[str, object] = {}

    def fake_run_v2_eval(**kwargs):  # type: ignore[no-untyped-def]
        seen["eval_out_path"] = kwargs["out_path"]
        Path(kwargs["out_path"]).write_text(json.dumps({"variants": {}}), encoding="utf-8")
        return {"variants": {"budget_only": {}, "v1_baseline": {}}}

    def fake_build_final_poc_summary(**kwargs):  # type: ignore[no-untyped-def]
        seen["release_source_summary_path"] = kwargs["source_summary_path"]
        seen["release_out_path"] = kwargs["out_path"]
        seen["include_experimental"] = kwargs.get("include_experimental", False)
        Path(kwargs["out_path"]).write_text(json.dumps({"variants": {"v1_baseline": {}}}), encoding="utf-8")
        return {"variants": {"v1_baseline": {"debug": {"verifier_model_name": "fake-nli"}}}}

    def fake_write_poc_results_markdown(**kwargs):  # type: ignore[no-untyped-def]
        seen["report_summary_path"] = kwargs["summary_path"]
        seen["report_out_path"] = kwargs["out_path"]
        Path(kwargs["out_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["out_path"]).write_text("# report\n", encoding="utf-8")
        return "# report\n"

    dataset = _write(
        tmp_path / "dataset.jsonl",
        json.dumps({"id": "ex1", "llm_summary_text": "One.", "evidence_json": []}) + "\n",
    )
    conformal = _write(tmp_path / "conformal.json", json.dumps({"threshold": 0.5, "meta": {}}))
    out_path = tmp_path / "final_poc_summary.json"

    monkeypatch.setattr(cli, "run_v2_eval", fake_run_v2_eval)
    monkeypatch.setattr(cli, "build_final_poc_summary", fake_build_final_poc_summary)
    monkeypatch.setattr(cli, "write_poc_results_markdown", fake_write_poc_results_markdown)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "ega",
            "v2-eval",
            "--dataset",
            str(dataset),
            "--out",
            str(out_path),
            "--conformal-state",
            str(conformal),
            "--accept-threshold",
            "0.05",
            "--render-safe-answer",
        ],
    )

    exit_code = cli.main()
    _ = capsys.readouterr()

    assert exit_code == 0
    assert Path(seen["eval_out_path"]).name == "pilot_threshold_005_recalibrated.json"
    assert seen["release_out_path"] == out_path.resolve()
    assert seen["release_source_summary_path"] == Path(seen["eval_out_path"])
    assert seen["include_experimental"] is False


def test_generate_poc_report_cli_wires_release_helpers(tmp_path: Path, monkeypatch, capsys) -> None:
    seen: dict[str, object] = {}

    def fake_build_final_poc_summary(**kwargs):  # type: ignore[no-untyped-def]
        seen.update(kwargs)
        Path(kwargs["out_path"]).write_text(json.dumps({"variants": {}}), encoding="utf-8")
        return {"variants": {}}

    def fake_write_poc_results_markdown(**kwargs):  # type: ignore[no-untyped-def]
        seen["markdown_summary_path"] = kwargs["summary_path"]
        seen["markdown_out_path"] = kwargs["out_path"]
        Path(kwargs["out_path"]).write_text("# report\n", encoding="utf-8")
        return "# report\n"

    source_summary = _write(tmp_path / "source_summary.json", json.dumps({"variants": {}}))
    dataset = _write(
        tmp_path / "dataset.jsonl",
        json.dumps({"id": "ex1", "llm_summary_text": "One.", "evidence_json": []}) + "\n",
    )
    conformal = _write(tmp_path / "conformal.json", json.dumps({"threshold": 0.5, "meta": {}}))
    summary_out = tmp_path / "final_poc_summary.json"
    report_out = tmp_path / "poc_results.md"

    monkeypatch.setattr(cli, "build_final_poc_summary", fake_build_final_poc_summary)
    monkeypatch.setattr(cli, "write_poc_results_markdown", fake_write_poc_results_markdown)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "ega",
            "generate-poc-report",
            "--source-summary",
            str(source_summary),
            "--dataset",
            str(dataset),
            "--conformal-state",
            str(conformal),
            "--summary-out",
            str(summary_out),
            "--report-out",
            str(report_out),
            "--include-experimental",
        ],
    )

    exit_code = cli.main()
    _ = capsys.readouterr()

    assert exit_code == 0
    assert seen["include_experimental"] is True
    assert seen["out_path"] == summary_out.resolve()
    assert seen["markdown_out_path"] == report_out.resolve()


def test_shell_cli_v2_args_flow_into_run_pipeline(monkeypatch, capsys, tmp_path: Path) -> None:
    seen_calls: list[dict[str, object]] = []

    class FakeVerifier:
        def __init__(self, model_name=None, **_kwargs) -> None:  # type: ignore[no-untyped-def]
            self.model_name = model_name or "fake-shell"

    class FakeReranker:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

    def fake_run_pipeline(**kwargs):  # type: ignore[no-untyped-def]
        seen_calls.append(dict(kwargs))
        return {
            "verified_extract": [],
            "verified_text": "",
            "decision": {"refusal": False, "reason_code": "ok", "summary_stats": {}},
            "stats": {"kept_units": 0, "dropped_units": 0, "model_name": "fake"},
            "polish_status": "skipped",
        }

    conformal = _write(tmp_path / "conformal.json", json.dumps({"threshold": 0.5, "meta": {}}))
    req = {"llm_summary": "Fact one.", "evidence": [{"id": "e1", "text": "Fact one.", "metadata": {}}]}
    monkeypatch.setattr(cli, "NliCrossEncoderVerifier", FakeVerifier)
    monkeypatch.setattr(cli, "CrossEncoderReranker", FakeReranker)
    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO(json.dumps(req) + "\n"))
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "ega",
            "shell",
            "--stdin-jsonl",
            "--stdout-jsonl",
            "--use-reranker",
            "--reranker-model",
            "cross-encoder/test-model",
            "--rerank-topk",
            "8",
            "--conformal-state",
            str(conformal),
            "--use-budget",
            "--latency-budget-ms",
            "50",
            "--budget-max-pairs",
            "12",
        ],
    )

    exit_code = cli.main()
    _ = capsys.readouterr()

    assert exit_code == 0
    assert len(seen_calls) == 1
    seen = seen_calls[0]
    assert isinstance(seen["reranker"], FakeReranker)
    assert seen["rerank_topk"] == 8
    assert seen["conformal_state_path"] == str(conformal.resolve())
    assert seen["budget_policy"] is not None
    budget_config = seen["budget_config"]
    assert budget_config is not None
    assert getattr(budget_config, "latency_budget_ms") == 50
    assert getattr(budget_config, "max_pairs_total") == 12


def test_pipeline_cli_use_reranker_missing_dep_returns_clear_error(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    class FailingReranker:
        def __init__(self, model_name: str) -> None:
            _ = model_name
            raise ImportError("missing dependency")

    summary = _write(tmp_path / "llm_summary.txt", "One.\n")
    evidence = _write(
        tmp_path / "evidence.json",
        json.dumps([{"id": "e1", "text": "One.", "metadata": {}}]),
    )
    scores = _write(tmp_path / "scores.jsonl", json.dumps({"unit_id": "u0001", "score": 0.9}) + "\n")

    monkeypatch.setattr(cli, "CrossEncoderReranker", FailingReranker)
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
            "--use-reranker",
        ],
    )

    exit_code = cli.main()
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "ega[rerank]" in captured.err
