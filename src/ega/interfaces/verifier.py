"""Verifier interface boundary used by core pipeline."""

from __future__ import annotations

from typing import Any, Protocol

from ega.types import EvidenceSet, Unit, VerificationScore


class Verifier(Protocol):
    """Pluggable verifier interface for scoring units against evidence."""

    model_name: str | None

    def verify(self, units: list[Unit], evidence: EvidenceSet) -> list[VerificationScore]:
        """Return one verification score per input unit."""

    def get_last_verify_trace(self) -> dict[str, Any]:
        """Return implementation-specific metrics from the most recent verify call."""
