"""File-driven conformal calibration utilities for EGA v2."""

from __future__ import annotations

import json
from pathlib import Path

from ega.v2.conformal import ConformalCalibrator, ConformalConfig, ConformalState


def load_unit_calibration_jsonl(path: str | Path) -> list[dict[str, object]]:
    """Load per-unit score/support labels from JSONL."""
    resolved = Path(path)
    rows: list[dict[str, object]] = []

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
            rows.append(
                {
                    "score": float(payload["score"]),
                    "supported": bool(payload["supported"]),
                }
            )

    if not rows:
        raise ValueError("Calibration JSONL contains no usable rows.")

    return rows


def calibrate_jsonl_to_state(
    *,
    in_path: str | Path,
    epsilon: float,
    mode: str = "supported_rate",
    min_calib: int = 50,
    abstain_k: float = 1.0,
) -> tuple[ConformalState, int]:
    """Fit a conformal state from per-unit JSONL rows."""
    rows = load_unit_calibration_jsonl(in_path)
    calibrator = ConformalCalibrator()
    state = calibrator.fit(
        rows=rows,
        config=ConformalConfig(
            epsilon=epsilon,
            mode=mode,
            min_calib=min_calib,
            abstain_k=abstain_k,
        ),
    )
    return state, len(rows)


def save_conformal_state_json(path: str | Path, state: ConformalState) -> None:
    """Write threshold/meta state JSON artifact."""
    resolved = Path(path)
    payload: dict[str, object] = {
        "threshold": float(state.threshold),
        "band_width": float(state.band_width),
        "abstain_k": float(state.abstain_k),
        "n_samples": int(state.n_samples),
        "score_mean": float(state.score_mean),
        "score_std": float(state.score_std),
        "meta": dict(state.meta),
    }
    if state.calibration_score_min is not None:
        payload["calibration_score_min"] = float(state.calibration_score_min)
    if state.calibration_score_max is not None:
        payload["calibration_score_max"] = float(state.calibration_score_max)
    resolved.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")


def load_conformal_state_json(path: str | Path) -> ConformalState:
    """Load and validate a V4 conformal state JSON artifact.

    Raises ValueError if calibration_score_min or calibration_score_max are absent.
    Pre-V4 artifacts missing these fields are rejected.
    """
    resolved = Path(path)
    payload = json.loads(resolved.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("Conformal state JSON must be an object.")
    for required_field in ("threshold", "calibration_score_min", "calibration_score_max"):
        if required_field not in payload:
            raise ValueError(
                f"Conformal state artifact missing required field '{required_field}'. "
                "Pre-V4 artifacts are not accepted; re-run calibration to generate a V4 artifact."
            )
    threshold = float(payload["threshold"])
    raw_meta = payload.get("meta", {})
    meta = dict(raw_meta) if isinstance(raw_meta, dict) else {}
    band_width = float(
        payload.get("band_width", meta.get("band_width", meta.get("abstain_margin", 0.02)))
    )
    abstain_k = float(payload.get("abstain_k", meta.get("abstain_k", 1.0)))
    n_samples = int(payload.get("n_samples", meta.get("n_samples", meta.get("n_calib", 0))))
    score_mean = float(payload.get("score_mean", meta.get("score_mean", 0.0)))
    score_std = float(payload.get("score_std", meta.get("score_std", 0.0)))
    return ConformalState(
        threshold=threshold,
        band_width=band_width,
        abstain_k=abstain_k,
        n_samples=n_samples,
        score_mean=score_mean,
        score_std=score_std,
        meta=meta,
        calibration_score_min=float(payload["calibration_score_min"]),
        calibration_score_max=float(payload["calibration_score_max"]),
    )
