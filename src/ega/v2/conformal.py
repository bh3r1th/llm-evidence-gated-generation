"""Conformal-style calibration and gating helpers for EGA v2."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ConformalConfig:
    """Configuration for conformal-style risk gating."""

    epsilon: float
    mode: str = "supported_rate"
    min_calib: int = 50
    abstain_k: float = 1.0


@dataclass(frozen=True, slots=True)
class ConformalState:
    """Calibrated decision threshold and metadata."""

    threshold: float
    band_width: float
    abstain_k: float
    n_samples: int
    score_mean: float
    score_std: float
    meta: dict[str, Any]
    calibration_score_min: float | None = None
    calibration_score_max: float | None = None


class ConformalCalibrator:
    """Baseline conformal-style calibrator using best entailment scores."""

    def __init__(self) -> None:
        self._calibration_scores: list[float] | None = None

    def fit(
        self,
        rows: list[dict[str, Any]],
        config: ConformalConfig,
    ) -> ConformalState:
        if len(rows) < config.min_calib:
            raise ValueError(
                f"Need at least {config.min_calib} calibration examples; got {len(rows)}."
            )
        if not (0.0 <= config.epsilon <= 1.0):
            raise ValueError("epsilon must be in [0, 1].")
        if config.abstain_k < 0.0:
            raise ValueError("abstain_k must be non-negative.")

        normalized: list[float] = []
        n_unsupported = 0
        for idx, row in enumerate(rows, start=1):
            if "score" not in row:
                raise ValueError(f"Calibration row {idx} missing required field 'score'.")
            if "supported" not in row:
                raise ValueError(f"Calibration row {idx} missing required field 'supported'.")
            if not isinstance(row["supported"], bool):
                raise ValueError(f"Calibration row {idx} field 'supported' must be bool.")
            normalized.append(self._clip01(float(row["score"])))
            if not row["supported"]:
                n_unsupported += 1

        q = max(0.0, min(1.0, 1.0 - config.epsilon))
        threshold = self._quantile(normalized, q)
        score_mean = self._mean(normalized)
        score_std = self._std(normalized, score_mean)
        band_width = float(config.abstain_k) * score_std
        calibration_score_min = min(normalized)
        calibration_score_max = max(normalized)

        meta = {
            "mode": config.mode,
            "epsilon": float(config.epsilon),
            "min_calib": int(config.min_calib),
            "n_calib": len(normalized),
            "n_unsupported": n_unsupported,
            "abstain_margin": band_width,
            "band_width": band_width,
            "abstain_k": float(config.abstain_k),
            "n_samples": len(normalized),
            "score_mean": score_mean,
            "score_std": score_std,
            "calibration_score_min": calibration_score_min,
            "calibration_score_max": calibration_score_max,
        }
        self._calibration_scores = list(normalized)
        return ConformalState(
            threshold=threshold,
            band_width=band_width,
            abstain_k=float(config.abstain_k),
            n_samples=len(normalized),
            score_mean=score_mean,
            score_std=score_std,
            meta=meta,
            calibration_score_min=calibration_score_min,
            calibration_score_max=calibration_score_max,
        )

    def load_reference_from_state(self, state: ConformalState) -> None:
        n = max(2, int(state.n_samples))
        mean = self._clip01(float(state.score_mean))
        std = max(0.0, float(state.score_std))
        if std == 0.0:
            self._calibration_scores = [mean] * n
            return
        from statistics import NormalDist

        nd = NormalDist()
        synthesized: list[float] = []
        for i in range(n):
            q = (i + 0.5) / n
            z = nd.inv_cdf(q)
            synthesized.append(self._clip01(mean + (std * z)))
        self._calibration_scores = synthesized

    def measure_drift(
        self,
        live_scores: list[float],
        *,
        drift_p_threshold: float = 0.05,
    ) -> dict[str, float | bool]:
        if self._calibration_scores is None:
            raise RuntimeError("measure_drift() requires calibration fit/reference to be loaded first.")
        if not (0.0 <= float(drift_p_threshold) <= 1.0):
            raise ValueError("drift_p_threshold must be in [0, 1].")
        if not live_scores:
            raise ValueError("live_scores must not be empty.")
        try:
            from scipy.stats import ks_2samp
        except ImportError as exc:
            raise ImportError(
                "measure_drift() requires scipy; install it to enable KS-based drift checks."
            ) from exc

        live = [self._clip01(float(value)) for value in live_scores]
        calib = list(self._calibration_scores)
        live_mean = self._mean(live)
        live_std = self._std(live, live_mean)
        calib_mean = self._mean(calib)
        calib_std = self._std(calib, calib_mean)
        ks = ks_2samp(calib, live)
        p_value = float(ks.pvalue)
        return {
            "mean_delta": float(live_mean - calib_mean),
            "std_delta": float(live_std - calib_std),
            "ks_statistic": float(ks.statistic),
            "ks_p_value": p_value,
            "drift_flagged": bool(p_value < float(drift_p_threshold)),
        }

    def gate(self, score: float, state: ConformalState) -> str:
        margin = float(state.band_width)
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

    @staticmethod
    def _mean(values: list[float]) -> float:
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    @staticmethod
    def _std(values: list[float], mean: float) -> float:
        if not values:
            return 0.0
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        return float(math.sqrt(max(0.0, variance)))
