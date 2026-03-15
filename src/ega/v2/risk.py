"""Baseline risk feature extraction for EGA v2 budget policies."""

from __future__ import annotations

import math
from typing import Mapping

from ega.types import EvidenceSet, Unit


def extract_unit_risks(
    *,
    units: list[Unit],
    evidence: EvidenceSet,
    similarity_by_unit: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Return per-unit risk in [0, 1] from available retrieval-like signals.

    Preference order:
    - top-1 retrieval similarity when available
    - top1/top2 margin when available
    - lexical overlap against current evidence as a cheap fallback
    - unit length as a final tie-breaker fallback
    """
    out: dict[str, float] = {}
    for unit in units:
        similarity = _initial_similarity(unit=unit, similarity_by_unit=similarity_by_unit)
        margin = _initial_margin(unit=unit)
        lexical_overlap = _best_lexical_overlap(unit_text=unit.text, evidence=evidence)
        length_risk = _length_risk(unit_text=unit.text)
        if similarity is not None:
            risk = 1.0 - similarity
            if margin is not None:
                risk = 0.7 * risk + 0.3 * (1.0 - margin)
        elif lexical_overlap is not None:
            risk = 1.0 - lexical_overlap
            if margin is not None:
                risk = 0.75 * risk + 0.25 * (1.0 - margin)
            else:
                risk = 0.85 * risk + 0.15 * length_risk
        else:
            risk = 0.5 if margin is None else 0.7 * (1.0 - margin) + 0.3 * length_risk
        out[unit.id] = max(0.0, min(1.0, risk))
    return out


def _initial_similarity(
    *,
    unit: Unit,
    similarity_by_unit: Mapping[str, float] | None,
) -> float | None:
    if similarity_by_unit is not None and unit.id in similarity_by_unit:
        return _as_valid_prob(similarity_by_unit[unit.id])

    for key in (
        "initial_retrieval_similarity",
        "top1_similarity",
        "retrieval_similarity",
    ):
        if key in unit.metadata:
            return _as_valid_prob(unit.metadata.get(key))
    return None


def _initial_margin(*, unit: Unit) -> float | None:
    for key in (
        "top1_top2_margin",
        "retrieval_margin",
        "initial_retrieval_margin",
    ):
        if key in unit.metadata:
            return _as_valid_prob(unit.metadata.get(key))
    return None


def _best_lexical_overlap(*, unit_text: str, evidence: EvidenceSet) -> float | None:
    unit_tokens = set(_tokenize(unit_text))
    if not unit_tokens:
        return None
    best = 0.0
    found = False
    for item in evidence.items:
        evidence_tokens = set(_tokenize(item.text))
        if not evidence_tokens:
            continue
        found = True
        overlap = float(len(unit_tokens.intersection(evidence_tokens))) / float(len(unit_tokens))
        best = max(best, overlap)
    return best if found else None


def _length_risk(*, unit_text: str) -> float:
    token_count = len(_tokenize(unit_text))
    if token_count <= 6:
        return 0.25
    if token_count >= 20:
        return 0.85
    return max(0.25, min(0.85, 0.25 + (float(token_count - 6) / 14.0) * 0.60))


def _tokenize(text: str) -> list[str]:
    return [token for token in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if token]


def _as_valid_prob(value: object) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return max(0.0, min(1.0, v))
