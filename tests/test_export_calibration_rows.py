from __future__ import annotations

import json
from pathlib import Path

import ega.cli as cli
import ega.v2.export_calibration_rows as export_mod
from ega.v2.calibrate import calibrate_jsonl_to_state, save_conformal_state_json
from ega.v2.conformal import ConformalCalibrator, ConformalConfig


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def test_export_calibration_rows_contain_score_and_supported(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_path = _write_jsonl(
        tmp_path / "dataset.jsonl",
        [
            {
                "id": "example-1",
                "llm_summary_text": "Alpha. Beta.",
                "evidence_json": [{"id": "e1", "text": "stub", "metadata": {}}],
                "gold_units": [
                    {"unit_id": "u0001", "text": "Alpha.", "supported": True},
                    {"unit_id": "u0002", "text": "Beta.", "supported": False},
                ],
            }
        ],
    )
    out_path = tmp_path / "rows.jsonl"

    def fake_run_one(**kwargs):  # type: ignore[no-untyped-def]
        assert kwargs["accept_threshold"] == 0.05
        assert kwargs["render_safe_answer"] is False
        return (
            {
                "accept_threshold": 0.05,
                "verifier_model_name": "nli/test-model",
                "verifier_scores": {
                    "u0001": {
                        "entailment": 0.73,
                        "contradiction": 0.01,
                        "chosen_evidence_id": "e1",
                    },
                    "u0002": {
                        "entailment": 0.12,
                        "contradiction": 0.20,
                        "chosen_evidence_id": "e2",
                    },
                },
            },
            {},
        )

    monkeypatch.setattr(export_mod, "_run_one", fake_run_one)

    summary = export_mod.export_calibration_rows(
        dataset_path=dataset_path,
        out_path=out_path,
        accept_threshold=0.05,
    )
    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]

    assert summary["n_rows"] == 2
    assert rows == [
        {
            "accept_threshold": 0.05,
            "chosen_evidence_id": "e1",
            "example_id": "example-1",
            "score": 0.73,
            "supported": True,
            "unit_id": "u0001",
            "verifier_model_name": "nli/test-model",
        },
        {
            "accept_threshold": 0.05,
            "chosen_evidence_id": "e2",
            "example_id": "example-1",
            "score": 0.12,
            "supported": False,
            "unit_id": "u0002",
            "verifier_model_name": "nli/test-model",
        },
    ]


def test_export_calibration_rows_falls_back_to_entailment_when_conformal_score_is_none(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_path = _write_jsonl(
        tmp_path / "dataset.jsonl",
        [
            {
                "id": "example-1",
                "llm_summary_text": "Alpha.",
                "evidence_json": [{"id": "e1", "text": "stub", "metadata": {}}],
                "gold_units": [{"unit_id": "u0001", "text": "Alpha.", "supported": True}],
            }
        ],
    )
    out_path = tmp_path / "rows.jsonl"

    def fake_run_one(**kwargs):  # type: ignore[no-untyped-def]
        assert kwargs["render_safe_answer"] is False
        return (
            {
                "accept_threshold": 0.05,
                "verifier_model_name": "nli/test-model",
                "verifier_scores": {
                    "u0001": {
                        "conformal_score": None,
                        "entailment": 0.61,
                        "chosen_evidence_id": "e1",
                        "debug_score": None,
                    }
                },
            },
            {},
        )

    monkeypatch.setattr(export_mod, "_run_one", fake_run_one)

    export_mod.export_calibration_rows(
        dataset_path=dataset_path,
        out_path=out_path,
        accept_threshold=0.05,
    )
    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]

    assert rows == [
        {
            "accept_threshold": 0.05,
            "chosen_evidence_id": "e1",
            "example_id": "example-1",
            "score": 0.61,
            "supported": True,
            "unit_id": "u0001",
            "verifier_model_name": "nli/test-model",
        }
    ]
    assert isinstance(rows[0]["score"], float)


def test_export_calibration_rows_uses_zero_when_scores_are_missing(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_path = _write_jsonl(
        tmp_path / "dataset.jsonl",
        [
            {
                "id": "example-1",
                "llm_summary_text": "Alpha.",
                "evidence_json": [{"id": "e1", "text": "stub", "metadata": {}}],
                "gold_units": [{"unit_id": "u0001", "text": "Alpha.", "supported": False}],
            }
        ],
    )
    out_path = tmp_path / "rows.jsonl"

    def fake_run_one(**kwargs):  # type: ignore[no-untyped-def]
        _ = kwargs
        return (
            {
                "accept_threshold": 0.05,
                "verifier_model_name": "nli/test-model",
                "verifier_scores": {
                    "u0001": {
                        "conformal_score": None,
                        "entailment": None,
                        "chosen_evidence_id": None,
                        "debug_score": None,
                    }
                },
            },
            {},
        )

    monkeypatch.setattr(export_mod, "_run_one", fake_run_one)

    export_mod.export_calibration_rows(
        dataset_path=dataset_path,
        out_path=out_path,
        accept_threshold=0.05,
    )
    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]

    assert rows == [
        {
            "accept_threshold": 0.05,
            "chosen_evidence_id": None,
            "example_id": "example-1",
            "score": 0.0,
            "supported": False,
            "unit_id": "u0001",
            "verifier_model_name": "nli/test-model",
        }
    ]
    assert isinstance(rows[0]["score"], float)


def test_exported_rows_can_be_calibrated_and_saved(tmp_path: Path, monkeypatch) -> None:
    dataset_path = _write_jsonl(
        tmp_path / "dataset.jsonl",
        [
            {
                "id": "example-1",
                "llm_summary_text": "A. B. C.",
                "evidence_json": [{"id": "e1", "text": "stub", "metadata": {}}],
                "gold_units": [
                    {"unit_id": "u0001", "text": "A.", "supported": True},
                    {"unit_id": "u0002", "text": "B.", "supported": False},
                    {"unit_id": "u0003", "text": "C.", "supported": False},
                ],
            }
        ],
    )
    rows_path = tmp_path / "rows.jsonl"
    state_path = tmp_path / "state.json"

    def fake_run_one(**kwargs):  # type: ignore[no-untyped-def]
        _ = kwargs
        return (
            {
                "accept_threshold": 0.05,
                "verifier_model_name": "nli/test-model",
                "verifier_scores": {
                    "u0001": {"entailment": 0.91, "chosen_evidence_id": "e1"},
                    "u0002": {"entailment": 0.62, "chosen_evidence_id": "e1"},
                    "u0003": {"entailment": 0.18, "chosen_evidence_id": "e1"},
                },
            },
            {},
        )

    monkeypatch.setattr(export_mod, "_run_one", fake_run_one)

    export_mod.export_calibration_rows(dataset_path=dataset_path, out_path=rows_path)
    state, n_rows = calibrate_jsonl_to_state(in_path=rows_path, epsilon=0.5, min_calib=3)
    save_conformal_state_json(state_path, state)

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert n_rows == 3
    assert isinstance(payload["threshold"], float)
    assert payload["meta"]["n_calib"] == 3


def test_conformal_gate_can_abstain_on_synthetic_scores() -> None:
    calibrator = ConformalCalibrator()
    state = calibrator.fit(
        rows=[
            {"score": 0.91, "supported": True},
            {"score": 0.78, "supported": False},
            {"score": 0.61, "supported": False},
            {"score": 0.59, "supported": True},
            {"score": 0.20, "supported": False},
        ],
        config=ConformalConfig(epsilon=0.5, min_calib=5, abstain_k=1.0),
    )

    assert calibrator.gate(score=state.threshold, state=state) == "abstain"


def test_export_calibration_rows_cli_dispatches(tmp_path: Path, monkeypatch, capsys) -> None:
    dataset_path = _write_jsonl(
        tmp_path / "dataset.jsonl",
        [
            {
                "id": "example-1",
                "llm_summary_text": "Alpha.",
                "evidence_json": [{"id": "e1", "text": "stub", "metadata": {}}],
                "gold_units": [{"unit_id": "u0001", "text": "Alpha.", "supported": True}],
            }
        ],
    )
    out_path = tmp_path / "rows.jsonl"
    seen_call: dict[str, object] = {}

    def fake_export_calibration_rows(**kwargs):  # type: ignore[no-untyped-def]
        seen_call.update(kwargs)
        return {
            "dataset_path": str(kwargs["dataset_path"]),
            "out_path": str(kwargs["out_path"]),
            "n_examples": 1,
            "n_rows": 1,
            "score_definition": export_mod.CALIBRATION_SCORE_DEFINITION,
        }

    monkeypatch.setattr(cli, "export_calibration_rows", fake_export_calibration_rows)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "ega",
            "export-calibration-rows",
            "--dataset",
            str(dataset_path),
            "--out",
            str(out_path),
            "--accept-threshold",
            "0.05",
            "--topk-per-unit",
            "7",
            "--max-pairs-total",
            "11",
        ],
    )

    exit_code = cli.main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert seen_call["accept_threshold"] == 0.05
    assert seen_call["topk_per_unit"] == 7
    assert seen_call["max_pairs_total"] == 11
    assert json.loads(captured.out)["n_rows"] == 1
