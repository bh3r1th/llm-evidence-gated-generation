"""Vendor-neutral score provider interface for precomputed verification scores."""

from __future__ import annotations

from typing import Protocol

from ega.types import AnswerCandidate, EvidenceSet, VerificationScore


class ScoreProvider(Protocol):
    """Protocol for loading unit-level verification scores from external systems."""

    def load_scores(
        self,
        *,
        candidate: AnswerCandidate,
        evidence: EvidenceSet,
    ) -> list[VerificationScore]:
        """Return per-unit scores with `unit_id` and score payload preserved in `raw`."""

