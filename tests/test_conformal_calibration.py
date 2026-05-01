from __future__ import annotations

import json
import math
from pathlib import Path

from ega.v2.calibrate import calibrate_jsonl_to_state, save_conformal_state_json


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    payload = "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    path.write_text(payload, encoding="utf-8")


def _expected_quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    idx = int(math.ceil(q * len(ordered)) - 1)
    idx = max(0, min(idx, len(ordered) - 1))
    return float(ordered[idx])


def test_fit_uses_full_distribution_not_unsupported_subset(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = []
    for i in range(100):
        rows.append({"unit_id": f"s{i}", "score": 0.80 + (i / 1000.0), "supported": True})
    for i in range(100):
        rows.append({"unit_id": f"u{i}", "score": 0.20 + (i / 1000.0), "supported": False})
    data_path = tmp_path / "calib.jsonl"
    _write_rows(data_path, rows)

    epsilon = 0.10
    state, n = calibrate_jsonl_to_state(in_path=data_path, epsilon=epsilon, min_calib=10)

    all_scores = [float(row["score"]) for row in rows]
    unsupported_scores = [float(row["score"]) for row in rows if not bool(row["supported"])]
    expected_full = _expected_quantile(all_scores, 1.0 - epsilon)
    unsupported_only = _expected_quantile(unsupported_scores, 1.0 - epsilon)

    assert n == 200
    assert state.threshold == expected_full
    assert state.threshold != unsupported_only


def test_band_width_is_abstain_k_times_std_all_scores(tmp_path: Path) -> None:
    rows = [
        {"unit_id": f"u{i}", "score": float(i) / 20.0, "supported": bool(i % 2)}
        for i in range(200)
    ]
    data_path = tmp_path / "calib.jsonl"
    _write_rows(data_path, rows)

    abstain_k = 1.7
    state, _ = calibrate_jsonl_to_state(
        in_path=data_path,
        epsilon=0.2,
        min_calib=10,
        abstain_k=abstain_k,
    )
    scores = [float(row["score"]) for row in rows]
    mean = sum(scores) / len(scores)
    std = math.sqrt(sum((value - mean) ** 2 for value in scores) / len(scores))
    expected_band = abstain_k * std

    assert abs(state.band_width - expected_band) <= 1e-6


def test_conformal_state_contains_required_fields(tmp_path: Path) -> None:
    rows = [
        {"unit_id": f"u{i}", "score": 0.1 + (i / 500.0), "supported": bool(i % 3)}
        for i in range(200)
    ]
    data_path = tmp_path / "calib.jsonl"
    _write_rows(data_path, rows)

    state, _ = calibrate_jsonl_to_state(in_path=data_path, epsilon=0.2, min_calib=10)

    assert isinstance(state.threshold, float)
    assert isinstance(state.band_width, float)
    assert isinstance(state.abstain_k, float)
    assert isinstance(state.n_samples, int)
    assert isinstance(state.score_mean, float)
    assert isinstance(state.score_std, float)


def test_saved_conformal_state_round_trips_threshold_and_band_width(tmp_path: Path) -> None:
    rows = [
        {"unit_id": f"u{i}", "score": 0.2 + (i / 1000.0), "supported": bool(i % 2)}
        for i in range(200)
    ]
    data_path = tmp_path / "calib.jsonl"
    out_path = tmp_path / "state.json"
    _write_rows(data_path, rows)

    state, _ = calibrate_jsonl_to_state(in_path=data_path, epsilon=0.15, min_calib=10)
    save_conformal_state_json(out_path, state)
    payload = json.loads(out_path.read_text(encoding="utf-8"))

    assert payload["threshold"] == state.threshold
    assert payload["band_width"] == state.band_width
