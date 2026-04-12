"""Deterministic answer unitization utilities for EGA v1."""

from __future__ import annotations

import re
import json
from collections.abc import Iterator
from typing import Any, Protocol

from ega.text_clean import clean_text
from ega.types import AnswerCandidate, Unit

_SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+")
_BULLET_PATTERN = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+\S")


class Unitizer(Protocol):
    """Protocol for strategies that split text into deterministic units."""

    def unitize(self, text: str) -> list[Unit]:
        """Split input text into ordered units with stable IDs."""


class StructuredUnitizer(Protocol):
    """Protocol for strategies that split structured payloads into units."""

    def unitize(self, payload: Any) -> list[Unit]:
        """Split structured payload into ordered units with stable IDs."""


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


class StructuredFieldUnitizer:
    """Emit deterministic units for scalar leaf fields in structured payloads."""

    def unitize(self, payload: Any) -> list[Unit]:
        if not isinstance(payload, (dict, list)):
            return []
        return list(self._iter_leaf_units(payload, path="$", field_name="$"))

    def _iter_leaf_units(self, value: Any, *, path: str, field_name: str) -> Iterator[Unit]:
        if isinstance(value, dict):
            for key in sorted(value, key=lambda item: str(item)):
                key_str = str(key)
                next_path = _append_object_path(path, key_str)
                yield from self._iter_leaf_units(
                    value[key],
                    path=next_path,
                    field_name=key_str,
                )
            return

        if isinstance(value, list):
            for index, item in enumerate(value):
                yield from self._iter_leaf_units(
                    item,
                    path=f"{path}[{index}]",
                    field_name=field_name,
                )
            return

        normalized = _normalize_scalar(value)
        if normalized is None:
            return

        yield Unit(
            id=path,
            text=normalized,
            metadata={
                "path": path,
                "field_name": field_name,
                "structured_mode": True,
            },
        )


def unitize_answer(text: Any, mode: str = "sentence") -> AnswerCandidate:
    """Build an :class:`AnswerCandidate` with units created by the requested mode."""

    unitizers: dict[str, Unitizer] = {
        "sentence": SentenceUnitizer(),
        "markdown_bullet": MarkdownBulletUnitizer(),
    }
    structured_unitizers: dict[str, StructuredUnitizer] = {
        "structured_field": StructuredFieldUnitizer(),
    }
    use_structured = mode in structured_unitizers
    use_spacy = mode == "spacy_sentence"
    if use_structured:
        structured_unitizer = structured_unitizers[mode]
    elif use_spacy:
        try:
            from ega.unitization_spacy import SpaCySentenceUnitizer
            unitizer: Unitizer = SpaCySentenceUnitizer()
        except ImportError as exc:
            msg = (
                "spaCy is required for mode 'spacy_sentence'. "
                "Install with: pip install 'ega[unitize]'."
            )
            raise ImportError(msg) from exc
    else:
        try:
            unitizer = unitizers[mode]
        except KeyError as exc:
            supported_modes = ", ".join(
                sorted([*unitizers, *structured_unitizers, "spacy_sentence"])
            )
            msg = f"Unsupported unitization mode: {mode!r}. Supported modes: {supported_modes}."
            raise ValueError(msg) from exc

    if use_structured:
        raw_units = structured_unitizer.unitize(text)
        cleaned_text = _normalize_structured_root_text(text)
    else:
        cleaned_text = clean_text(text)
        try:
            raw_units = unitizer.unitize(cleaned_text)
        except ImportError as exc:
            if use_spacy:
                msg = (
                    "spaCy is required for mode 'spacy_sentence'. "
                    "Install with: pip install 'ega[unitize]'."
                )
                raise ImportError(msg) from exc
            raise

    units = [
        Unit(
            id=unit.id,
            text=clean_text(unit.text),
            metadata=dict(unit.metadata),
            source_ids=list(unit.source_ids) if unit.source_ids is not None else None,
        )
        for unit in raw_units
    ]
    return AnswerCandidate(raw_answer_text=cleaned_text, units=units)


def _build_units(parts: list[str]) -> list[Unit]:
    return [
        Unit(id=f"u{index:04d}", text=part, metadata={})
        for index, part in enumerate(parts, start=1)
    ]


def _normalize_scalar(value: Any) -> str | None:
    if isinstance(value, str):
        normalized = clean_text(value)
        return normalized if normalized else ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    return None


def _append_object_path(base_path: str, key: str) -> str:
    if _is_identifier_like(key):
        return f"{base_path}.{key}"
    escaped = key.replace("\\", "\\\\").replace('"', '\\"')
    return f'{base_path}["{escaped}"]'


def _is_identifier_like(value: str) -> bool:
    return bool(value) and (value[0].isalpha() or value[0] == "_") and all(
        char.isalnum() or char == "_" for char in value
    )


def _normalize_structured_root_text(value: Any) -> str:
    if isinstance(value, (dict, list)):
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return clean_text(serialized)
    return clean_text(str(value))
