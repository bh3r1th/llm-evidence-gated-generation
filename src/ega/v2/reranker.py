"""Evidence reranking interfaces for EGA v2."""

from __future__ import annotations

from typing import Protocol

from ega.types import EvidenceSet, Unit


class EvidenceReranker(Protocol):
    """Select and order evidence ids per unit."""

    def rerank(
        self,
        units: list[Unit],
        evidence: EvidenceSet,
        candidates: dict[str, list[str]],
        topk: int,
    ) -> dict[str, list[str]]:
        """Return reranked evidence ids for each unit id."""


class NoopReranker:
    """Pass-through reranker that leaves candidates unchanged."""

    def rerank(
        self,
        units: list[Unit],
        evidence: EvidenceSet,
        candidates: dict[str, list[str]],
        topk: int,
    ) -> dict[str, list[str]]:
        _ = (units, evidence, topk)
        return {unit_id: list(evidence_ids) for unit_id, evidence_ids in candidates.items()}
