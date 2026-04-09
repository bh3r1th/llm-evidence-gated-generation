"""Adapters for normalizing verifier implementations to the Verifier protocol."""

from __future__ import annotations

from typing import Any

from ega.interfaces import Verifier
from ega.types import AnswerCandidate, EvidenceSet, Unit, VerificationScore


class LegacyVerifierAdapter(Verifier):
    """Adapter that accepts legacy verifier shapes and exposes ``Verifier``."""

    def __init__(self, verifier: Any) -> None:
        self._verifier = verifier

    @property
    def model_name(self) -> str | None:
        return getattr(self._verifier, "model_name", None)

    def verify(self, units: list[Unit], evidence: EvidenceSet) -> list[VerificationScore]:
        verify = getattr(self._verifier, "verify", None)
        if callable(verify):
            return self._normalize_scores(units=units, scores=verify(units, evidence))

        verify_many = getattr(self._verifier, "verify_many", None)
        if callable(verify_many):
            candidate = AnswerCandidate(raw_answer_text="\n".join(unit.text for unit in units), units=units)
            scores = verify_many(candidate, evidence)
            return self._normalize_scores(units=units, scores=scores)

        verify_unit = getattr(self._verifier, "verify_unit", None)
        if callable(verify_unit):
            scores: list[VerificationScore] = []
            for unit in units:
                scores.append(
                    self._as_score(unit_id=unit.id, score=verify_unit(unit.text, evidence))
                )
            return scores

        raise AttributeError("verifier must implement verify, verify_many, or verify_unit")

    def get_last_verify_trace(self) -> dict[str, Any]:
        getter = getattr(self._verifier, "get_last_verify_trace", None)
        if callable(getter):
            payload = getter()
            if isinstance(payload, dict):
                return dict(payload)
        return {}

    def _normalize_scores(self, units: list[Unit], scores: Any) -> list[VerificationScore]:
        rows = list(scores)
        if len(rows) != len(units):
            raise ValueError("verifier returned mismatched number of scores")
        return [self._as_score(unit_id=unit.id, score=score) for unit, score in zip(units, rows, strict=True)]

    @staticmethod
    def _as_score(*, unit_id: str, score: Any) -> VerificationScore:
        raw_payload = getattr(score, "raw", {})
        return VerificationScore(
            unit_id=unit_id,
            entailment=float(getattr(score, "entailment")),
            contradiction=float(getattr(score, "contradiction")),
            neutral=float(getattr(score, "neutral")),
            label=str(getattr(score, "label")),
            raw=dict(raw_payload) if isinstance(raw_payload, dict) else {},
        )
