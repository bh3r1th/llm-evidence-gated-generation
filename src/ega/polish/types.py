"""Data types for optional post-enforcement polish lane."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class PolishedUnit:
    """Edited text for a single verified unit."""

    unit_id: str
    edited_text: str


@dataclass(frozen=True, slots=True)
class PolishResult:
    """Container for optional polish model output and metadata."""

    units: list[PolishedUnit]
    model_id: str | None = None
    prompt_version: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

