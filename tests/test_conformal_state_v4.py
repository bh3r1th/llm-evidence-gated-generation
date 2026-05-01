"""V4 ConformalState: calibration_score_min and calibration_score_max."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ega.pipeline import _load_conformal_state
from ega.v2.calibrate import load_conformal_state_json, save_conformal_state_json
from ega.v2.conformal import ConformalCalibrator, ConformalConfig


def _rows(scores: list[float]) -> list[dict[str, object]]:
    return [{"score": s, "supported": True} for s in scores]


def _fit(scores: list[float], min_calib: int = 10):
    rows = _rows(scores)
    return ConformalCalibrator().fit(
        rows=rows,
        config=ConformalConfig(epsilon=0.1, min_calib=min_calib),
    )


# ---------------------------------------------------------------------------
# Test 1 — fit() populates observed min and max
# ---------------------------------------------------------------------------

def test_fit_populates_calibration_score_min_max() -> None:
    scores = [0.10, 0.30, 0.55, 0.70, 0.85, 0.92, 0.45, 0.60, 0.25, 0.78]
    state = _fit(scores)

    assert state.calibration_score_min == pytest.approx(min(scores))
    assert state.calibration_score_max == pytest.approx(max(scores))


def test_fit_min_max_reflect_clipped_scores() -> None:
    # Scores outside [0,1] are clipped; min/max must reflect clipped values.
    scores = [-0.5, 0.4, 0.6, 0.8, 1.5, 0.3, 0.5, 0.7, 0.2, 0.9]
    state = _fit(scores)

    clipped = [max(0.0, min(1.0, s)) for s in scores]
    assert state.calibration_score_min == pytest.approx(min(clipped))
    assert state.calibration_score_max == pytest.approx(max(clipped))


# ---------------------------------------------------------------------------
# Test 2 — save then load round-trips both fields without loss
# ---------------------------------------------------------------------------

def test_round_trip_preserves_calibration_score_min_max(tmp_path: Path) -> None:
    scores = [0.20, 0.35, 0.50, 0.65, 0.80, 0.40, 0.55, 0.70, 0.30, 0.60]
    state = _fit(scores)
    artifact = tmp_path / "state.json"

    save_conformal_state_json(artifact, state)
    loaded = load_conformal_state_json(artifact)

    assert loaded.calibration_score_min == pytest.approx(state.calibration_score_min)
    assert loaded.calibration_score_max == pytest.approx(state.calibration_score_max)


def test_round_trip_fields_are_floats(tmp_path: Path) -> None:
    scores = [0.1 + i * 0.08 for i in range(10)]
    state = _fit(scores)
    artifact = tmp_path / "state.json"

    save_conformal_state_json(artifact, state)
    loaded = load_conformal_state_json(artifact)

    assert isinstance(loaded.calibration_score_min, float)
    assert isinstance(loaded.calibration_score_max, float)


# ---------------------------------------------------------------------------
# Test 3 — pre-V4 artifact missing the fields raises ValueError
# ---------------------------------------------------------------------------

def test_load_pre_v4_missing_calibration_score_min_raises(tmp_path: Path) -> None:
    artifact = tmp_path / "pre_v4.json"
    artifact.write_text(
        json.dumps({
            "threshold": 0.70,
            "band_width": 0.05,
            "abstain_k": 1.0,
            "n_samples": 100,
            "score_mean": 0.65,
            "score_std": 0.10,
            "meta": {},
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="calibration_score_min"):
        load_conformal_state_json(artifact)


def test_load_artifact_missing_only_max_raises(tmp_path: Path) -> None:
    artifact = tmp_path / "partial.json"
    artifact.write_text(
        json.dumps({
            "threshold": 0.70,
            "band_width": 0.05,
            "abstain_k": 1.0,
            "n_samples": 100,
            "score_mean": 0.65,
            "score_std": 0.10,
            "calibration_score_min": 0.10,
            "meta": {},
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="calibration_score_max"):
        load_conformal_state_json(artifact)


# ---------------------------------------------------------------------------
# Test 4 — meta dict written to artifact contains both fields
# ---------------------------------------------------------------------------

def test_meta_contains_calibration_score_min_max() -> None:
    scores = [0.15, 0.35, 0.55, 0.75, 0.90, 0.25, 0.45, 0.65, 0.20, 0.80]
    state = _fit(scores)

    assert "calibration_score_min" in state.meta
    assert "calibration_score_max" in state.meta
    assert state.meta["calibration_score_min"] == pytest.approx(state.calibration_score_min)
    assert state.meta["calibration_score_max"] == pytest.approx(state.calibration_score_max)


def test_saved_artifact_json_contains_both_fields(tmp_path: Path) -> None:
    scores = [0.1 + i * 0.07 for i in range(10)]
    state = _fit(scores)
    artifact = tmp_path / "state.json"

    save_conformal_state_json(artifact, state)
    raw = json.loads(artifact.read_text(encoding="utf-8"))

    assert "calibration_score_min" in raw
    assert "calibration_score_max" in raw
    assert raw["calibration_score_min"] == pytest.approx(state.calibration_score_min)
    assert raw["calibration_score_max"] == pytest.approx(state.calibration_score_max)


# ---------------------------------------------------------------------------
# Test 5 — _load_conformal_state reads fields from runs/v3_calibration artifact
# ---------------------------------------------------------------------------

def test_pipeline_load_conformal_state_reads_calibration_score_min_max() -> None:
    artifact = Path("runs/v3_calibration/conformal_state.json")
    state = _load_conformal_state(artifact)

    assert state.calibration_score_min is not None
    assert state.calibration_score_max is not None
    assert state.calibration_score_min == 0.0
    assert state.calibration_score_max == pytest.approx(0.591796875)
