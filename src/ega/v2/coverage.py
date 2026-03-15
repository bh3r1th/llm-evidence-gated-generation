"""Deterministic evidence coverage analysis utilities for EGA v2."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ega.types import EvidenceSet, Unit


@dataclass(frozen=True, slots=True)
class CoverageConfig:
    """Configuration for evidence coverage computation."""

    pool_topk: int = 20
    relevant_threshold: float = 0.0
    normalize: str = "recall"


@dataclass(frozen=True, slots=True)
class CoverageResult:
    """Coverage details for one unit."""

    unit_id: str
    relevant_evidence_ids: list[str]
    used_evidence_ids: list[str]
    coverage_score: float
    missing_evidence_ids: list[str]
    meta: dict[str, Any] = field(default_factory=dict)


class EvidenceCoverageAnalyzer:
    """Compute deterministic per-unit evidence coverage."""

    def analyze(
        self,
        *,
        units: list[Unit],
        evidence: EvidenceSet,
        pool_candidates: dict[str, list[str]],
        used_evidence: dict[str, list[str]],
        config: CoverageConfig,
    ) -> dict[str, CoverageResult]:
        if config.normalize != "recall":
            raise ValueError("Only normalize='recall' is currently supported.")

        _ = evidence
        out: dict[str, CoverageResult] = {}
        max_pool = max(0, int(config.pool_topk))
        threshold = float(config.relevant_threshold)

        for unit in units:
            raw_pool = pool_candidates.get(unit.id, [])
            relevant_ids = self._relevant_ids(raw_pool=raw_pool, topk=max_pool, threshold=threshold)

            used_raw = used_evidence.get(unit.id, [])
            relevant_set = set(relevant_ids)
            used_filtered: list[str] = []
            seen_used: set[str] = set()
            for evidence_id in used_raw:
                eid = str(evidence_id)
                if eid in relevant_set and eid not in seen_used:
                    seen_used.add(eid)
                    used_filtered.append(eid)

            missing_ids = [eid for eid in relevant_ids if eid not in seen_used]
            coverage = float(len(used_filtered)) / float(max(1, len(relevant_ids)))

            out[unit.id] = CoverageResult(
                unit_id=unit.id,
                relevant_evidence_ids=relevant_ids,
                used_evidence_ids=used_filtered,
                coverage_score=coverage,
                missing_evidence_ids=missing_ids,
                meta={
                    "normalize": config.normalize,
                    "pool_topk": max_pool,
                    "relevant_threshold": threshold,
                    "n_relevant": len(relevant_ids),
                    "n_used": len(used_filtered),
                },
            )

        return out

    @staticmethod
    def _relevant_ids(*, raw_pool: list[Any], topk: int, threshold: float) -> list[str]:
        relevant: list[str] = []
        seen: set[str] = set()

        for item in raw_pool:
            evidence_id, score = EvidenceCoverageAnalyzer._parse_pool_item(item)
            if evidence_id is None:
                continue
            if score is not None and score < threshold:
                continue
            if evidence_id in seen:
                continue
            seen.add(evidence_id)
            relevant.append(evidence_id)
            if len(relevant) >= topk:
                break

        return relevant

    @staticmethod
    def _parse_pool_item(item: Any) -> tuple[str | None, float | None]:
        if isinstance(item, str):
            return item, None

        if isinstance(item, dict):
            raw_id = item.get("evidence_id", item.get("id"))
            if raw_id is None:
                return None, None
            raw_score = item.get("score")
            score: float | None = None
            if raw_score is not None:
                try:
                    score = float(raw_score)
                except (TypeError, ValueError):
                    score = None
            return str(raw_id), score

        if isinstance(item, (list, tuple)) and item:
            evidence_id = str(item[0])
            score: float | None = None
            if len(item) > 1:
                try:
                    score = float(item[1])  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    score = None
            return evidence_id, score

        return None, None
