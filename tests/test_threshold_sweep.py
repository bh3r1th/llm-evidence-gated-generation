from __future__ import annotations

import json
from pathlib import Path

import ega.cli as cli
import ega.v2.threshold_sweep as threshold_sweep


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def test_threshold_sweep_lower_threshold_increases_recall(tmp_path: Path, monkeypatch) -> None:
    dataset_path = _write_jsonl(
        tmp_path / "dataset.jsonl",
        [
            {
                "id": "example-1",
                "llm_summary_text": "Alpha. Beta. Gamma.",
                "evidence_json": [{"id": "e1", "text": "stub", "metadata": {}}],
                "gold_units": [
                    {"unit_id": "u0001", "text": "Alpha.", "supported": True, "required_evidence_ids": [], "relevant_evidence_ids": []},
                    {"unit_id": "u0002", "text": "Beta.", "supported": True, "required_evidence_ids": [], "relevant_evidence_ids": []},
                    {"unit_id": "u0003", "text": "Gamma.", "supported": False, "required_evidence_ids": [], "relevant_evidence_ids": []},
                ],
            }
        ],
    )
    out_path = tmp_path / "sweep.json"

    def fake_run_one(**kwargs):  # type: ignore[no-untyped-def]
        assert kwargs["render_safe_answer"] is False
        return (
            {
                "units": [
                    {"unit_id": "u0001", "text": "Alpha."},
                    {"unit_id": "u0002", "text": "Beta."},
                    {"unit_id": "u0003", "text": "Gamma."},
                ],
                "verifier_scores": {
                    "u0001": {"entailment": 0.12, "contradiction": 0.01},
                    "u0002": {"entailment": 0.40, "contradiction": 0.01},
                    "u0003": {"entailment": 0.08, "contradiction": 0.01},
                },
            },
            {},
        )

    monkeypatch.setattr(threshold_sweep, "_run_one", fake_run_one)

    summary = threshold_sweep.run_threshold_sweep(dataset_path=dataset_path, out_path=out_path)

    by_threshold = {row["threshold"]: row for row in summary["sweeps"]}
    assert by_threshold[0.05]["recall"] == 1.0
    assert by_threshold[0.20]["recall"] == 0.5
    assert by_threshold[0.05]["recall"] > by_threshold[0.20]["recall"]


def test_threshold_sweep_output_contains_required_metrics_and_cli_dispatch(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    dataset_path = _write_jsonl(
        tmp_path / "dataset.jsonl",
        [
            {
                "id": "example-2",
                "llm_summary_text": "Alpha. Beta.",
                "evidence_json": [{"id": "e1", "text": "stub", "metadata": {}}],
                "gold_units": [
                    {"unit_id": "u0001", "text": "Alpha.", "supported": True, "required_evidence_ids": [], "relevant_evidence_ids": []},
                    {"unit_id": "u0002", "text": "Beta.", "supported": False, "required_evidence_ids": [], "relevant_evidence_ids": []},
                ],
            }
        ],
    )
    out_path = tmp_path / "sweep.json"

    def fake_run_one(**kwargs):  # type: ignore[no-untyped-def]
        assert kwargs["render_safe_answer"] is False
        return (
            {
                "units": [
                    {"unit_id": "u0001", "text": "Alpha."},
                    {"unit_id": "u0002", "text": "Beta."},
                ],
                "verifier_scores": {
                    "u0001": {"entailment": 0.30, "contradiction": 0.01},
                    "u0002": {"entailment": 0.04, "contradiction": 0.01},
                },
            },
            {},
        )

    monkeypatch.setattr(threshold_sweep, "_run_one", fake_run_one)
    summary = threshold_sweep.run_threshold_sweep(dataset_path=dataset_path, out_path=out_path)
    saved = json.loads(out_path.read_text(encoding="utf-8"))

    required = {
        "threshold",
        "kept_units",
        "dropped_units",
        "precision",
        "recall",
        "f1",
        "unsupported_claim_rate",
        "hallucination_rate",
    }
    assert required.issubset(saved["sweeps"][0])
    assert saved["recommended_threshold"] == summary["recommended_threshold"]

    seen_call: dict[str, object] = {}

    def fake_run_threshold_sweep(**kwargs):  # type: ignore[no-untyped-def]
        seen_call.update(kwargs)
        return {
            "dataset_path": str(kwargs["dataset_path"]),
            "n_examples": 1,
            "accept_threshold": kwargs["accept_threshold"],
            "thresholds": [0.1],
            "sweeps": [
                {
                    "threshold": 0.1,
                    "kept_units": 1,
                    "dropped_units": 0,
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1": 1.0,
                    "unsupported_claim_rate": 0.0,
                    "hallucination_rate": 0.0,
                }
            ],
            "recommended_threshold": 0.1,
            "selection_metric": "f1",
        }

    monkeypatch.setattr(cli, "run_threshold_sweep", fake_run_threshold_sweep)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "ega",
            "threshold-sweep",
            "--dataset",
            str(dataset_path),
            "--out",
            str(out_path),
            "--topk-per-unit",
            "7",
            "--max-pairs-total",
            "11",
            "--accept-threshold",
            "0.05",
        ],
    )

    exit_code = cli.main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert seen_call["topk_per_unit"] == 7
    assert seen_call["max_pairs_total"] == 11
    assert seen_call["accept_threshold"] == 0.05
    assert json.loads(captured.out)["recommended_threshold"] == 0.1
