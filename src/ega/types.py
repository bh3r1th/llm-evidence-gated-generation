"""Core typed data structures for Evidence-Gated Answering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ega.contract import EGA_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class Unit:
    """A minimal text unit used by the gating and verification layers."""

    id: str
    text: str
    metadata: dict[str, Any]
    source_ids: list[str] | None = None


@dataclass(frozen=True, slots=True)
class AnswerCandidate:
    """A candidate answer and the units it was derived from."""

    raw_answer_text: str
    units: list[Unit]


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    """Canonical evidence item used for verification."""

    id: str
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class EvidenceSet:
    """Collection of evidence items supplied to verifiers."""

    items: list[EvidenceItem]


@dataclass(frozen=True, slots=True)
class VerificationScore:
    """Verifier scores for a single unit."""

    unit_id: str
    entailment: float
    contradiction: float
    neutral: float
    label: str
    raw: dict[str, Any]


@dataclass(frozen=True, slots=True)
class GateDecision:
    """Deterministic decision payload emitted by the gate."""

    allowed_units: list[str]
    dropped_units: list[str]
    refusal: bool
    reason_code: str
    summary_stats: dict[str, Any]


@dataclass(frozen=True, slots=True)
class EnforcementResult:
    """Top-level result produced after policy enforcement."""

    final_text: str | None
    kept_units: list[str]
    dropped_units: list[str]
    refusal_message: str | None
    decision: GateDecision
    scores: list[VerificationScore]
    ega_schema_version: str = EGA_SCHEMA_VERSION
