"""Deterministic answer unitization utilities for EGA v1."""

from __future__ import annotations

import re
from typing import Protocol

from ega.types import AnswerCandidate, Unit

_SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+")
_BULLET_PATTERN = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+\S")


class Unitizer(Protocol):
    """Protocol for strategies that split text into deterministic units."""

    def unitize(self, text: str) -> list[Unit]:
        """Split input text into ordered units with stable IDs."""


class SentenceUnitizer:
    """Split text with a simple regex-based sentence boundary rule."""

    def unitize(self, text: str) -> list[Unit]:
        stripped_text = text.strip()
        if not stripped_text:
            return []

        parts = [part.strip() for part in _SENTENCE_BOUNDARY_PATTERN.split(stripped_text)]
        kept_parts = [part for part in parts if part]
        return _build_units(kept_parts)


class MarkdownBulletUnitizer:
    """Split text by markdown-like bullet lines, with sentence fallback."""

    def __init__(self) -> None:
        self._sentence_unitizer = SentenceUnitizer()

    def unitize(self, text: str) -> list[Unit]:
        stripped_text = text.strip()
        if not stripped_text:
            return []

        lines = [line.strip() for line in stripped_text.splitlines()]
        bullet_lines = [line for line in lines if line and _BULLET_PATTERN.match(line)]
        if bullet_lines:
            return _build_units(bullet_lines)

        return self._sentence_unitizer.unitize(stripped_text)


def unitize_answer(text: str, mode: str = "sentence") -> AnswerCandidate:
    """Build an :class:`AnswerCandidate` with units created by the requested mode."""

    unitizers: dict[str, Unitizer] = {
        "sentence": SentenceUnitizer(),
        "markdown_bullet": MarkdownBulletUnitizer(),
    }
    try:
        unitizer = unitizers[mode]
    except KeyError as exc:
        supported_modes = ", ".join(sorted(unitizers))
        msg = f"Unsupported unitization mode: {mode!r}. Supported modes: {supported_modes}."
        raise ValueError(msg) from exc

    return AnswerCandidate(raw_answer_text=text, units=unitizer.unitize(text))


def _build_units(parts: list[str]) -> list[Unit]:
    return [
        Unit(id=f"u{index:04d}", text=part)
        for index, part in enumerate(parts, start=1)
    ]
