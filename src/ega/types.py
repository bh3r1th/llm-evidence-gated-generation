"""Core typed data structures for Evidence-Gated Answering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Unit:
    """A minimal text unit used by the gating and verification layers."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    source_ids: list[str] | None = None


@dataclass(frozen=True, slots=True)
class AnswerCandidate:
    """A candidate answer and the units it was derived from."""

    raw_answer_text: str
    units: list[Unit] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    """Canonical evidence item used for verification."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvidenceSet:
    """Collection of evidence items supplied to verifiers."""

    items: list[EvidenceItem] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class VerificationScore:
    """Verifier scores for a single unit."""

    unit_id: str
    entailment: float
    contradiction: float
    neutral: float
    label: str
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GateDecision:
    """Deterministic decision payload emitted by the gate."""

    allowed_units: list[Unit] = field(default_factory=list)
    dropped_units: list[Unit] = field(default_factory=list)
    refusal: bool = False
    reason_code: str = ""
    summary_stats: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EnforcementResult:
    """Top-level result produced after policy enforcement."""

    final_text: str | None
    kept_units: list[Unit] = field(default_factory=list)
    dropped_units: list[Unit] = field(default_factory=list)
    refusal_message: str | None = None
    decision: GateDecision = field(default_factory=GateDecision)
    scores: list[VerificationScore] = field(default_factory=list)
