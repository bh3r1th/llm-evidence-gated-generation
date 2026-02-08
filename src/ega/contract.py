"""Canonical public contract for EGA."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

EGA_SCHEMA_VERSION = "1"


class ReasonCode(str, Enum):
    """Stable reason codes emitted by gate decisions."""

    OK_FULL = "OK_FULL"
    OK_PARTIAL = "OK_PARTIAL"
    ALL_DROPPED = "ALL_DROPPED"
    PARTIAL_NOT_ALLOWED = "PARTIAL_NOT_ALLOWED"


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    """Runtime policy thresholds and output constraints."""

    threshold_entailment: float = 0.8
    max_contradiction: float = 0.2
    partial_allowed: bool = True
