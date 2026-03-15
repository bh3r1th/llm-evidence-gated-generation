from __future__ import annotations

import json
from pathlib import Path

from ega.cli import main
from ega.v2.calibrate import calibrate_jsonl_to_state
from ega.v2.conformal import ConformalCalibrator


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    payload = "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    path.write_text(payload, encoding="utf-8")


def test_conformal_calibration_threshold_directionality_and_gate(tmp_path: Path) -> None:
    rows = [
        {"unit_id": "u1", "score": 0.92, "supported": True},
        {"unit_id": "u2", "score": 0.89, "supported": True},
        {"unit_id": "u3", "score": 0.81, "supported": False},
        {"unit_id": "u4", "score": 0.65, "supported": False},
        {"unit_id": "u5", "score": 0.35, "supported": True},
        {"unit_id": "u6", "score": 0.22, "supported": False},
        {"unit_id": "u7", "score": 0.11, "supported": True},
        {"unit_id": "u8", "score": 0.04, "supported": False},
    ]
    data_path = tmp_path / "calib.jsonl"
    _write_rows(data_path, rows)

    state_low_eps, n1 = calibrate_jsonl_to_state(
        in_path=data_path,
        epsilon=0.10,
        min_calib=5,
        abstain_margin=0.05,
    )
    state_high_eps, n2 = calibrate_jsonl_to_state(
        in_path=data_path,
        epsilon=0.50,
        min_calib=5,
        abstain_margin=0.05,
    )

    assert n1 == len(rows)
    assert n2 == len(rows)
    assert state_low_eps.threshold >= state_high_eps.threshold

    calibrator = ConformalCalibrator()
    assert calibrator.gate(score=state_low_eps.threshold, state=state_low_eps) == "abstain"
    assert (
        calibrator.gate(score=min(1.0, state_low_eps.threshold + 0.20), state=state_low_eps)
        == "accept"
    )
    assert (
        calibrator.gate(score=max(0.0, state_low_eps.threshold - 0.20), state=state_low_eps)
        == "reject"
    )


def test_cli_conformal_calibrate_writes_json_and_prints_summary(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    rows = [
        {"unit_id": "u1", "score": 0.90, "supported": True},
        {"unit_id": "u2", "score": 0.80, "supported": False},
        {"unit_id": "u3", "score": 0.70, "supported": True},
        {"unit_id": "u4", "score": 0.60, "supported": False},
        {"unit_id": "u5", "score": 0.50, "supported": True},
    ]
    in_path = tmp_path / "input.jsonl"
    out_path = tmp_path / "state.json"
    _write_rows(in_path, rows)

    monkeypatch.setattr(
        "sys.argv",
        [
            "ega",
            "conformal-calibrate",
            "--in",
            str(in_path),
            "--out",
            str(out_path),
            "--epsilon",
            "0.2",
            "--min-calib",
            "5",
        ],
    )

    exit_code = main()
    captured = capsys.readouterr()
    payload = json.loads(out_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert "n=5" in captured.out
    assert "threshold=" in captured.out
    assert "epsilon=0.200000" in captured.out
    assert isinstance(payload["threshold"], float)
    assert payload["meta"]["epsilon"] == 0.2
    assert payload["meta"]["n_calib"] == 5
