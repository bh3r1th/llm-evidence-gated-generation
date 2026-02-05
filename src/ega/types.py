"""Core typed data structures for Evidence-Gated Answering.

This module defines neutral, serializable dataclasses and enums used across the
decision and enforcement layers.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DecisionOutcome(str, Enum):
    """High-level decision outcomes produced by the EGA decision layer."""

    ALLOW = "allow"
    ABSTAIN = "abstain"
    BLOCK = "block"


@dataclass(slots=True)
class EvidenceUnit:
    """A minimal evidence unit used for verification.

    TODO: Expand fields once evidence unitization strategy is finalized.
    """

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VerificationResult:
    """Result container for verifier outputs.

    TODO: Define canonical score schema for multi-verifier aggregation.
    """

    verifier_name: str
    score: float
    passed: bool
    rationale: str = ""
