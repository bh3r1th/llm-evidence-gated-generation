from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from ega.cli import main
from ega.contract import PolicyConfig, ReasonCode
from ega.pipeline import run_pipeline
from ega.types import EvidenceItem, EvidenceSet

_FIXTURES_DIR = Path("examples") / "pipeline_demo"


def _evidence() -> EvidenceSet:
    payload = json.loads((_FIXTURES_DIR / "evidence.json").read_text(encoding="utf-8"))
    return EvidenceSet(
        items=[
            EvidenceItem(
                id=str(item["id"]),
                text=str(item["text"]),
                metadata=dict(item.get("metadata", {})),
            )
            for item in payload
        ]
    )


def _summary() -> str:
    return (_FIXTURES_DIR / "llm_summary.txt").read_text(encoding="utf-8")


def _polished(name: str) -> list[dict[str, str]]:
    return json.loads((_FIXTURES_DIR / name).read_text(encoding="utf-8"))


def test_pipeline_scores_jsonl_partial_has_verified_extract() -> None:
    output = run_pipeline(
        llm_summary_text=_summary(),
        evidence=_evidence(),
        unitizer_mode="sentence",
        policy_config=PolicyConfig(
            threshold_entailment=0.8,
            max_contradiction=0.2,
            partial_allowed=True,
        ),
        scores_jsonl_path=str(_FIXTURES_DIR / "scores.jsonl"),
    )

    assert output["verified_extract"]
    assert output["decision"]["reason_code"] == ReasonCode.OK_PARTIAL.value


def test_pipeline_polished_ok_passes_and_sets_polished_text() -> None:
    output = run_pipeline(
        llm_summary_text=_summary(),
        evidence=_evidence(),
        unitizer_mode="sentence",
        policy_config=PolicyConfig(
            threshold_entailment=0.8,
            max_contradiction=0.2,
            partial_allowed=True,
        ),
        scores_jsonl_path=str(_FIXTURES_DIR / "scores.jsonl"),
        polished_json=_polished("polished_ok.json"),
        enable_polish_validation=True,
    )

    assert output["polish_status"] == "passed"
    assert "polished_text" in output
    assert output["polished_text"]


def test_pipeline_polished_bad_fails_and_omits_polished_text() -> None:
    output = run_pipeline(
        llm_summary_text=_summary(),
        evidence=_evidence(),
        unitizer_mode="sentence",
        policy_config=PolicyConfig(
            threshold_entailment=0.8,
            max_contradiction=0.2,
            partial_allowed=True,
        ),
        scores_jsonl_path=str(_FIXTURES_DIR / "scores.jsonl"),
        polished_json=_polished("polished_bad.json"),
        enable_polish_validation=True,
    )

    assert output["polish_status"] == "failed"
    assert "polished_text" not in output
    assert output["polish_fail_reasons"]


def test_pipeline_cli_command_outputs_expected_payload(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "ega",
            "pipeline",
            "--llm-summary-file",
            str(_FIXTURES_DIR / "llm_summary.txt"),
            "--evidence-json",
            str(_FIXTURES_DIR / "evidence.json"),
            "--scores-jsonl",
            str(_FIXTURES_DIR / "scores.jsonl"),
            "--unitizer",
            "sentence",
            "--partial-allowed",
            "--polished-json",
            str(_FIXTURES_DIR / "polished_ok.json"),
        ],
    )

    exit_code = main()
    payload = json.loads(capsys.readouterr().out.strip())

    assert exit_code == 0
    assert payload["decision"]["reason_code"] == ReasonCode.OK_PARTIAL.value
    assert payload["verified_extract"]
    assert payload["polish_status"] == "passed"


def test_pipeline_strips_bom_from_verified_output(tmp_path: Path) -> None:
    scores_path = tmp_path / "scores.jsonl"
    scores_path.write_text(
        json.dumps({"unit_id": "u0001", "score": 0.95}) + "\n",
        encoding="utf-8",
    )

    output = run_pipeline(
        llm_summary_text="\ufeffSupported fact.",
        evidence=EvidenceSet(
            items=[EvidenceItem(id="e1", text="\ufeffSupported fact.", metadata={})]
        ),
        unitizer_mode="sentence",
        policy_config=PolicyConfig(
            threshold_entailment=0.8,
            max_contradiction=0.2,
            partial_allowed=True,
        ),
        scores_jsonl_path=str(scores_path),
    )

    assert "\ufeff" not in output["verified_text"]
    assert "\ufeff" not in output["verified_extract"][0]["text"]


def test_pipeline_trace_out_contains_stage_breakdown() -> None:
    scores_path = Path("data") / f"trace_scores_{uuid4().hex}.jsonl"
    trace_path = Path("data") / f"trace_out_{uuid4().hex}.jsonl"
    try:
        scores_path.write_text(
            json.dumps({"unit_id": "u0001", "score": 0.95}) + "\n",
            encoding="utf-8",
        )

        output = run_pipeline(
            llm_summary_text="Supported fact.",
            evidence=EvidenceSet(items=[EvidenceItem(id="e1", text="Supported fact.", metadata={})]),
            unitizer_mode="sentence",
            policy_config=PolicyConfig(
                threshold_entailment=0.8,
                max_contradiction=0.2,
                partial_allowed=True,
            ),
            scores_jsonl_path=str(scores_path),
            trace_out=str(trace_path),
        )

        assert output["verified_extract"]
        lines = trace_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1

        payload = json.loads(lines[0])
        assert "total_seconds" in payload
        assert "verify_seconds" in payload
        assert "load_seconds" in payload
        assert "verify_compute_seconds" in payload
        assert "n_units" in payload
        assert "n_pairs" in payload
        assert "preselect_seconds" in payload
        assert "tokenize_seconds" in payload
        assert "forward_seconds" in payload
        assert "post_seconds" in payload
        assert "num_batches" in payload
        assert "batch_size_mean" in payload
        assert "batch_size_max" in payload
        assert "seq_len_mean" in payload
        assert "seq_len_p50" in payload
        assert "seq_len_p95" in payload
        assert "tokens_total" in payload
        assert "device" in payload
        assert "dtype" in payload
        assert "amp_enabled" in payload
        assert "compiled_enabled" in payload
        assert "pairs_pruned_stage1" in payload
        assert "pairs_pruned_stage2" in payload
        assert "evidence_truncated_frac" in payload
        assert "evidence_chars_mean_before" in payload
        assert "evidence_chars_mean_after" in payload

        stage_sum = (
            payload["read_seconds"]
            + payload["unitize_seconds"]
            + payload["verify_seconds"]
            + payload["enforce_seconds"]
            + payload["polish_seconds"]
        )
        assert payload["total_seconds"] + 1e-6 >= stage_sum
        assert payload["load_seconds"] >= 0.0
        assert payload["verify_compute_seconds"] >= 0.0
        assert payload["verify_compute_seconds"] <= payload["verify_seconds"]
    finally:
        scores_path.unlink(missing_ok=True)
        trace_path.unlink(missing_ok=True)
