from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from ega.contract import PolicyConfig
from ega.pipeline import run_pipeline
from ega.types import EvidenceItem, EvidenceSet
from ega.v2.calibrate import calibrate_jsonl_to_state, save_conformal_state_json
from ega.v2.conformal import ConformalCalibrator, ConformalConfig


def _rows_from_scores(scores: list[float]) -> list[dict[str, object]]:
    return [{"score": float(s), "supported": bool(i % 2)} for i, s in enumerate(scores)]


def _normal_scores(*, mean: float, std: float, n: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(n):
        sample = rng.gauss(mean, std)
        values.append(max(0.0, min(1.0, sample)))
    return values


def _write_score_jsonl(path: Path, scores: list[float]) -> None:
    rows = [{"unit_id": f"u{i+1:04d}", "score": float(score)} for i, score in enumerate(scores)]
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def test_measure_drift_same_distribution_not_flagged() -> None:
    calib_scores = _normal_scores(mean=0.7, std=0.1, n=400, seed=7)
    live_scores = _normal_scores(mean=0.7, std=0.1, n=400, seed=8)
    calibrator = ConformalCalibrator()
    calibrator.fit(rows=_rows_from_scores(calib_scores), config=ConformalConfig(epsilon=0.2, min_calib=10))

    drift = calibrator.measure_drift(live_scores=live_scores)

    assert drift["drift_flagged"] is False


def test_measure_drift_shifted_distribution_flagged() -> None:
    calib_scores = _normal_scores(mean=0.7, std=0.1, n=400, seed=9)
    live_scores = _normal_scores(mean=0.3, std=0.2, n=400, seed=10)
    calibrator = ConformalCalibrator()
    calibrator.fit(rows=_rows_from_scores(calib_scores), config=ConformalConfig(epsilon=0.2, min_calib=10))

    drift = calibrator.measure_drift(live_scores=live_scores)

    assert drift["drift_flagged"] is True
    assert float(drift["ks_statistic"]) > 0.3


def test_measure_drift_before_fit_raises() -> None:
    with pytest.raises(RuntimeError):
        ConformalCalibrator().measure_drift(live_scores=[0.1, 0.2, 0.3])


def test_run_pipeline_with_conformal_state_attaches_distribution_drift(tmp_path: Path) -> None:
    calib_rows_path = tmp_path / "calib_rows.jsonl"
    scores_path = tmp_path / "scores.jsonl"
    state_path = tmp_path / "state.json"

    calib_rows = [
        {"unit_id": f"cal{i}", "score": float(score), "supported": bool(i % 2)}
        for i, score in enumerate(_normal_scores(mean=0.7, std=0.1, n=200, seed=11))
    ]
    calib_rows_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in calib_rows) + "\n",
        encoding="utf-8",
    )
    state, _ = calibrate_jsonl_to_state(in_path=calib_rows_path, epsilon=0.2, min_calib=10)
    save_conformal_state_json(state_path, state)

    _write_score_jsonl(scores_path, _normal_scores(mean=0.7, std=0.1, n=3, seed=12))
    output = run_pipeline(
        llm_summary_text="A. B. C.",
        evidence=EvidenceSet(items=[EvidenceItem(id="e1", text="A.", metadata={})]),
        unitizer_mode="sentence",
        policy_config=PolicyConfig(threshold_entailment=0.0, max_contradiction=0.2, partial_allowed=True),
        scores_jsonl_path=str(scores_path),
        conformal_state_path=str(state_path),
    )

    drift = output.get("trace", {}).get("distribution_drift")
    assert isinstance(drift, dict)
    assert {"mean_delta", "std_delta", "ks_statistic", "ks_p_value", "drift_flagged"}.issubset(drift.keys())


def test_run_pipeline_without_conformal_state_has_no_distribution_drift(tmp_path: Path) -> None:
    scores_path = tmp_path / "scores.jsonl"
    _write_score_jsonl(scores_path, [0.9, 0.8, 0.7])

    output = run_pipeline(
        llm_summary_text="A. B. C.",
        evidence=EvidenceSet(items=[EvidenceItem(id="e1", text="A.", metadata={})]),
        unitizer_mode="sentence",
        policy_config=PolicyConfig(threshold_entailment=0.0, max_contradiction=0.2, partial_allowed=True),
        scores_jsonl_path=str(scores_path),
    )

    assert "distribution_drift" not in output.get("trace", {})
