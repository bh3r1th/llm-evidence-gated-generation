"""Policy definitions for answer gating behavior.

Policies in EGA define thresholds and constraints that determine whether a
candidate answer can be emitted, abstained, or blocked.
"""

from dataclasses import dataclass


@dataclass(slots=True)
class GatingPolicy:
    """Top-level policy object used by the enforcement pipeline.

    TODO: Add policy fields for verifier routing and evidence requirements.
    """

    minimum_support_score: float = 0.8
    allow_abstain: bool = True
