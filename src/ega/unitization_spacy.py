"""spaCy-backed deterministic sentence unitization for evaluation.

Requires spaCy version range ``>=3.7,<4.0`` (install via ``pip install 'ega[unitize]'``).
"""

from __future__ import annotations

from ega.types import Unit


class SpaCySentenceUnitizer:
    """Split text into sentence units with spaCy sentencizer only."""

    def __init__(self) -> None:
        self._nlp = None

    def _ensure_nlp(self):
        if self._nlp is not None:
            return self._nlp
        import spacy

        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        self._nlp = nlp
        return nlp

    def unitize(self, text: str) -> list[Unit]:
        stripped_text = text.strip()
        if not stripped_text:
            return []

        doc = self._ensure_nlp()(stripped_text)
        parts = [sentence.text.strip() for sentence in doc.sents if sentence.text.strip()]

        units: list[Unit] = []
        for index, part in enumerate(parts, start=1):
            prev_text = parts[index - 2] if index > 1 else ""
            next_text = parts[index] if index < len(parts) else ""
            units.append(
                Unit(
                    id=f"u{index:04d}",
                    text=part,
                    metadata={"context_window": {"prev": prev_text, "next": next_text}},
                )
            )
        return units
