"""File-driven conformal calibration utilities for EGA v2."""

from __future__ import annotations

import json
from pathlib import Path

from ega.v2.conformal import ConformalCalibrator, ConformalConfig, ConformalState


def load_unit_calibration_jsonl(path: str | Path) -> tuple[list[float], list[bool]]:
    """Load per-unit score/support labels from JSONL."""
    resolved = Path(path)
    scores: list[float] = []
    labels_supported: list[bool] = []

    with resolved.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSONL row at line {line_no}: {exc.msg}."
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Malformed JSONL row at line {line_no}: expected object.")
            if "unit_id" not in payload:
                raise ValueError(f"Row {line_no} missing required field 'unit_id'.")
            if "score" not in payload:
                raise ValueError(f"Row {line_no} missing required field 'score'.")
            if "supported" not in payload:
                raise ValueError(f"Row {line_no} missing required field 'supported'.")
            if not isinstance(payload["supported"], bool):
                raise ValueError(f"Row {line_no} field 'supported' must be bool.")

            _ = str(payload["unit_id"])
            scores.append(float(payload["score"]))
            labels_supported.append(bool(payload["supported"]))

    if not scores:
        raise ValueError("Calibration JSONL contains no usable rows.")

    return scores, labels_supported


def calibrate_jsonl_to_state(
    *,
    in_path: str | Path,
    epsilon: float,
    mode: str = "supported_rate",
    min_calib: int = 50,
    abstain_margin: float = 0.02,
) -> tuple[ConformalState, int]:
    """Fit a conformal state from per-unit JSONL rows."""
    scores, labels_supported = load_unit_calibration_jsonl(in_path)
    calibrator = ConformalCalibrator()
    state = calibrator.fit(
        scores=scores,
        labels_supported=labels_supported,
        config=ConformalConfig(
            epsilon=epsilon,
            mode=mode,
            min_calib=min_calib,
            abstain_margin=abstain_margin,
        ),
    )
    return state, len(scores)


def save_conformal_state_json(path: str | Path, state: ConformalState) -> None:
    """Write threshold/meta state JSON artifact."""
    resolved = Path(path)
    payload = {
        "threshold": float(state.threshold),
        "meta": dict(state.meta),
    }
    resolved.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
