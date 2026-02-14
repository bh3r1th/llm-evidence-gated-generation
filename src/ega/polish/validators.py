"""Deterministic validators for optional polish edits."""

from __future__ import annotations

import re
from collections import Counter

from ega.polish.types import PolishedUnit
from ega.types import Unit

_NUMERIC_OR_DATE_TOKEN = re.compile(
    r"\b(?:\d{1,4}(?:[/-]\d{1,2}(?:[/-]\d{1,4})?)?|"
    r"(?:Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|"
    r"Sep|Sept|September|Oct|October|Nov|November|Dec|December)\s+\d{1,2}(?:,\s*\d{4})?)\b"
)
_WORD_TOKEN = re.compile(r"[A-Za-z0-9']+")
_CAPITALIZED_TOKEN = re.compile(r"\b[A-Z][a-zA-Z]+\b")


def validate_schema(original_units: list[Unit], polished: list[PolishedUnit]) -> list[str]:
    """Validate one-to-one schema invariants between original and polished units."""

    errors: list[str] = []
    if len(original_units) != len(polished):
        errors.append(
            f"count_mismatch: expected {len(original_units)} units, got {len(polished)}."
        )

    for index, (original, edited) in enumerate(
        zip(original_units, polished, strict=False),
        start=1,
    ):
        if original.id != edited.unit_id:
            errors.append(
                f"unit_id_mismatch_at_{index}: expected {original.id!r}, got {edited.unit_id!r}."
            )
    return errors


def validate_no_new_numbers_dates(original: str, edited: str) -> bool:
    """Return True when number/date-like tokens match exactly between texts."""

    return _NUMERIC_OR_DATE_TOKEN.findall(original) == _NUMERIC_OR_DATE_TOKEN.findall(edited)


def validate_overlap_bounds(
    original: str,
    edited: str,
    *,
    max_expansion_ratio: float = 1.20,
    min_ngram_overlap: float = 0.60,
) -> bool:
    """Check deterministic expansion and 3-gram overlap bounds."""

    original_tokens = _WORD_TOKEN.findall(original.lower())
    edited_tokens = _WORD_TOKEN.findall(edited.lower())

    if not original_tokens:
        return not edited_tokens

    expansion_ratio = len(edited_tokens) / len(original_tokens)
    if expansion_ratio > max_expansion_ratio:
        return False

    if len(original_tokens) < 3 or len(edited_tokens) < 3:
        if not original_tokens:
            return not edited_tokens
        overlap = len(set(original_tokens) & set(edited_tokens)) / len(set(original_tokens))
        return overlap >= min_ngram_overlap

    original_ngrams = Counter(_token_ngrams(original_tokens, 3))
    edited_ngrams = Counter(_token_ngrams(edited_tokens, 3))
    if not original_ngrams:
        return True

    shared = sum((original_ngrams & edited_ngrams).values())
    overlap_ratio = shared / sum(original_ngrams.values())
    return overlap_ratio >= min_ngram_overlap


def validate_no_new_named_entities(original: str, edited: str) -> bool:
    """Return True when edited capitalized tokens are subset of original."""

    original_caps = set(_CAPITALIZED_TOKEN.findall(original))
    edited_caps = set(_CAPITALIZED_TOKEN.findall(edited))
    return edited_caps.issubset(original_caps)


def _token_ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[index : index + n]) for index in range(0, len(tokens) - n + 1)]
