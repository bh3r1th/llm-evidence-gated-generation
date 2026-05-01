"""Tests for make_unit_authority_decision() and UnitDecisionRecord."""

from __future__ import annotations

import pytest

from ega.core.pipeline_core import UnitDecisionRecord, make_unit_authority_decision
from ega.v2.conformal import ConformalState


def _state(
    threshold: float = 0.70,
    band_width: float = 0.05,
    cal_min: float | None = 0.20,
    cal_max: float | None = 0.95,
) -> ConformalState:
    return ConformalState(
        threshold=threshold,
        band_width=band_width,
        abstain_k=1.0,
        n_samples=100,
        score_mean=0.60,
        score_std=0.15,
        meta={},
        calibration_score_min=cal_min,
        calibration_score_max=cal_max,
    )


# ---------------------------------------------------------------------------
# Test 1 — no conformal state: accept above threshold
# ---------------------------------------------------------------------------

def test_threshold_authority_accept() -> None:
    record = make_unit_authority_decision(
        score=0.85,
        conformal_state=None,
        accept_threshold=0.70,
    )
    assert isinstance(record, UnitDecisionRecord)
    assert record.authority == "threshold"
    assert record.final_decision == "accept"
    assert record.reason_code == "THRESHOLD_ACCEPT"
    assert record.conformal_decision is None
    assert record.conformal_threshold is None
    assert record.calibration_range is None
    assert record.fallback_reason is None


# ---------------------------------------------------------------------------
# Test 2 — no conformal state: reject below threshold
# ---------------------------------------------------------------------------

def test_threshold_authority_reject() -> None:
    record = make_unit_authority_decision(
        score=0.50,
        conformal_state=None,
        accept_threshold=0.70,
    )
    assert record.authority == "threshold"
    assert record.final_decision == "reject"
    assert record.reason_code == "THRESHOLD_REJECT"
    assert record.conformal_decision is None
    assert record.fallback_reason is None


# ---------------------------------------------------------------------------
# Test 3 — OOR low: score < calibration_score_min → auto-reject
# ---------------------------------------------------------------------------

def test_oor_low_auto_reject() -> None:
    state = _state(cal_min=0.30)
    record = make_unit_authority_decision(
        score=0.15,
        conformal_state=state,
        accept_threshold=0.70,
    )
    assert record.authority == "conformal_oor"
    assert record.conformal_decision == "reject"
    assert record.final_decision == "reject"
    assert record.reason_code == "CONFORMAL_OOR_LOW"
    assert record.fallback_reason == "below_calibration_range"
    assert record.calibration_range == pytest.approx((0.30, 0.95))
    assert record.conformal_threshold == pytest.approx(0.70)
    assert record.conformal_band_width == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Test 4 — OOR high: score > calibration_score_max → auto-accept
# ---------------------------------------------------------------------------

def test_oor_high_auto_accept() -> None:
    state = _state(cal_max=0.90)
    record = make_unit_authority_decision(
        score=0.97,
        conformal_state=state,
        accept_threshold=0.70,
    )
    assert record.authority == "conformal_oor"
    assert record.conformal_decision == "accept"
    assert record.final_decision == "accept"
    assert record.reason_code == "CONFORMAL_OOR_HIGH"
    assert record.fallback_reason == "above_calibration_range"
    assert record.calibration_range == pytest.approx((0.20, 0.90))


# ---------------------------------------------------------------------------
# Test 5 — in-range conformal gate decisions
# ---------------------------------------------------------------------------

def test_in_range_conformal_accept() -> None:
    # threshold=0.70, band_width=0.05 → abstain band [0.65, 0.75]
    # score=0.85 → above band → accept
    state = _state(threshold=0.70, band_width=0.05)
    record = make_unit_authority_decision(score=0.85, conformal_state=state, accept_threshold=0.0)
    assert record.authority == "conformal"
    assert record.final_decision == "accept"
    assert record.conformal_decision == "accept"
    assert record.reason_code == "CONFORMAL_ACCEPT"
    assert record.fallback_reason is None


def test_in_range_conformal_reject() -> None:
    # score=0.50 → below band [0.65, 0.75] → reject
    state = _state(threshold=0.70, band_width=0.05)
    record = make_unit_authority_decision(score=0.50, conformal_state=state, accept_threshold=0.0)
    assert record.authority == "conformal"
    assert record.final_decision == "reject"
    assert record.reason_code == "CONFORMAL_REJECT"


def test_in_range_conformal_abstain() -> None:
    # score=0.72 → inside band [0.65, 0.75] → abstain
    state = _state(threshold=0.70, band_width=0.05)
    record = make_unit_authority_decision(score=0.72, conformal_state=state, accept_threshold=0.0)
    assert record.authority == "conformal"
    assert record.final_decision == "abstain"
    assert record.reason_code == "CONFORMAL_ABSTAIN"


# ---------------------------------------------------------------------------
# Test 6 — pre-V4 state (cal_min/max = None) skips OOR, falls to normal gate
# ---------------------------------------------------------------------------

def test_no_cal_range_skips_oor_check() -> None:
    # A score that would be OOR-low if cal_min were set — but cal_min is None.
    # Should fall through to normal conformal gate.
    state = _state(cal_min=None, cal_max=None, threshold=0.70, band_width=0.05)
    # score=0.05 → below any realistic threshold → gate returns "reject"
    record = make_unit_authority_decision(score=0.05, conformal_state=state, accept_threshold=0.0)
    assert record.authority == "conformal"
    assert record.final_decision == "reject"
    assert record.calibration_range is None
    assert record.reason_code == "CONFORMAL_REJECT"
