"""Evidence unitization utilities.

Unitization breaks raw context into chunks suitable for downstream verifier
consumption. This module only includes scaffolding for now.
"""

from ega.types import EvidenceUnit


def unitize(text: str) -> list[EvidenceUnit]:
    """Convert raw text into evidence units.

    TODO: Replace with configurable segmentation strategy.
    """
    if not text.strip():
        return []
    return [EvidenceUnit(text=text.strip())]
