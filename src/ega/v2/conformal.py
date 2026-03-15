"""Conformal-style calibration and gating helpers for EGA v2."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ConformalConfig:
    """Configuration for conformal-style risk gating."""

    epsilon: float
    mode: str = "supported_rate"
    min_calib: int = 50
    abstain_margin: float = 0.02


@dataclass(frozen=True, slots=True)
class ConformalState:
    """Calibrated decision threshold and metadata."""

    threshold: float
    meta: dict[str, Any] = field(default_factory=dict)


class ConformalCalibrator:
    """Baseline conformal-style calibrator using best entailment scores."""

    def fit(
        self,
        scores: list[float],
        labels_supported: list[bool],
        config: ConformalConfig,
    ) -> ConformalState:
        if len(scores) != len(labels_supported):
            raise ValueError("scores and labels_supported must have equal length.")
        if len(scores) < config.min_calib:
            raise ValueError(
                f"Need at least {config.min_calib} calibration examples; got {len(scores)}."
            )
        if not (0.0 <= config.epsilon <= 1.0):
            raise ValueError("epsilon must be in [0, 1].")

        normalized = [self._clip01(float(value)) for value in scores]
        unsupported_scores = [s for s, supported in zip(normalized, labels_supported, strict=True) if not supported]

        if unsupported_scores:
            q = max(0.0, min(1.0, 1.0 - config.epsilon))
            threshold = self._quantile(unsupported_scores, q)
        else:
            threshold = 0.0

        meta = {
            "mode": config.mode,
            "epsilon": float(config.epsilon),
            "min_calib": int(config.min_calib),
            "n_calib": len(normalized),
            "n_unsupported": len(unsupported_scores),
            "abstain_margin": float(config.abstain_margin),
        }
        return ConformalState(threshold=threshold, meta=meta)

    def gate(self, score: float, state: ConformalState) -> str:
        margin = float(state.meta.get("abstain_margin", 0.02))
        value = self._clip01(float(score))
        threshold = self._clip01(float(state.threshold))

        if abs(value - threshold) <= margin:
            return "abstain"
        if value > threshold:
            return "accept"
        return "reject"

    @staticmethod
    def _clip01(value: float) -> float:
        if math.isnan(value):
            return 0.0
        return max(0.0, min(1.0, value))

    @staticmethod
    def _quantile(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        if len(ordered) == 1:
            return float(ordered[0])
        idx = int(math.ceil(q * len(ordered)) - 1)
        idx = max(0, min(idx, len(ordered) - 1))
        return float(ordered[idx])
