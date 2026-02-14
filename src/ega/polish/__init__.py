"""Optional post-enforcement polish lane interfaces and deterministic gate."""

from ega.polish.gate import PolishGateConfig, apply_polish_gate, gate_polish
from ega.polish.types import PolishedUnit, PolishResult

__all__ = [
    "PolishedUnit",
    "PolishGateConfig",
    "PolishResult",
    "apply_polish_gate",
    "gate_polish",
]

