"""Reward signal computation for EGA v2."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ega.types import Unit
from ega.v2.coverage import CoverageResult


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Weights and clamps for reward computation."""

    w_support: float = 1.0
    w_hallucination: float = 2.0
    w_abstain: float = 0.5
    w_coverage: float = 1.0
    clamp_min: float = -5.0
    clamp_max: float = 5.0


@dataclass(frozen=True, slots=True)
class UnitReward:
    """Per-unit reward decomposition."""

    unit_id: str
    support_score: float
    hallucination_penalty: float
    abstain_penalty: float
    coverage_score: float
    total_reward: float
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RewardSummary:
    """Aggregate reward statistics."""

    total_reward: float
    avg_reward: float
    avg_support_score: float
    hallucination_rate: float
    abstention_rate: float
    avg_coverage_score: float
    meta: dict[str, Any] = field(default_factory=dict)


class RewardComputer:
    """Convert verification + decisions + coverage into reward signals."""

    def compute(
        self,
        *,
        units: list[Unit],
        verification: dict[str, Any],
        decisions: dict[str, str],
        coverage: dict[str, CoverageResult] | None,
        config: RewardConfig,
    ) -> tuple[dict[str, UnitReward], RewardSummary]:
        out: dict[str, UnitReward] = {}
        total_reward = 0.0
        support_sum = 0.0
        hallucination_sum = 0.0
        abstain_sum = 0.0
        coverage_sum = 0.0

        for unit in units:
            unit_id = unit.id
            support = self._extract_support(verification.get(unit_id))
            decision = str(decisions.get(unit_id, "reject")).strip().lower()
            hallucination = 1.0 if self._is_hallucination(decision) else 0.0
            abstain = 1.0 if self._is_abstain(decision) else 0.0
            coverage_score = self._coverage_for_unit(unit_id=unit_id, coverage=coverage)

            reward = (
                float(config.w_support) * support
                - float(config.w_hallucination) * hallucination
                - float(config.w_abstain) * abstain
                + float(config.w_coverage) * coverage_score
            )
            reward = max(float(config.clamp_min), min(float(config.clamp_max), reward))

            out[unit_id] = UnitReward(
                unit_id=unit_id,
                support_score=support,
                hallucination_penalty=hallucination,
                abstain_penalty=abstain,
                coverage_score=coverage_score,
                total_reward=reward,
                meta={"decision": decision},
            )

            total_reward += reward
            support_sum += support
            hallucination_sum += hallucination
            abstain_sum += abstain
            coverage_sum += coverage_score

        n_units = len(units)
        denom = float(n_units) if n_units > 0 else 1.0
        summary = RewardSummary(
            total_reward=total_reward,
            avg_reward=total_reward / denom if n_units > 0 else 0.0,
            avg_support_score=support_sum / denom if n_units > 0 else 0.0,
            hallucination_rate=hallucination_sum / denom if n_units > 0 else 0.0,
            abstention_rate=abstain_sum / denom if n_units > 0 else 0.0,
            avg_coverage_score=coverage_sum / denom if n_units > 0 else 0.0,
            meta={
                "n_units": n_units,
                "weights": {
                    "w_support": float(config.w_support),
                    "w_hallucination": float(config.w_hallucination),
                    "w_abstain": float(config.w_abstain),
                    "w_coverage": float(config.w_coverage),
                },
                "clamp": {
                    "min": float(config.clamp_min),
                    "max": float(config.clamp_max),
                },
            },
        )
        return out, summary

    @staticmethod
    def _is_abstain(decision: str) -> bool:
        return decision in {"abstain", "conformal_abstain"}

    @classmethod
    def _is_hallucination(cls, decision: str) -> bool:
        if cls._is_abstain(decision):
            return False
        return decision in {
            "reject",
            "unsupported",
            "drop",
            "dropped",
            "refuse",
            "refusal",
        }

    @staticmethod
    def _coverage_for_unit(
        *,
        unit_id: str,
        coverage: dict[str, CoverageResult] | None,
    ) -> float:
        if coverage is None:
            return 0.0
        row = coverage.get(unit_id)
        if row is None:
            return 0.0
        try:
            score = float(row.coverage_score)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, score))

    @staticmethod
    def _extract_support(payload: Any) -> float:
        if payload is None:
            return 0.0

        if hasattr(payload, "entailment"):
            try:
                return max(0.0, min(1.0, float(getattr(payload, "entailment"))))
            except (TypeError, ValueError):
                return 0.0

        if isinstance(payload, dict):
            for key in ("entailment", "support_score", "score"):
                if key in payload:
                    try:
                        return max(0.0, min(1.0, float(payload[key])))
                    except (TypeError, ValueError):
                        return 0.0

        return 0.0
