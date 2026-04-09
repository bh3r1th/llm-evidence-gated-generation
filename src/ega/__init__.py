"""Evidence-Gated Answering (EGA) package public API."""

from ega.api import verify_answer
from ega.config import OutputConfig, PipelineConfig, RerankerConfig, VerifierConfig
from ega.contract import PolicyConfig

__all__ = [
    "verify_answer",
    "PipelineConfig",
    "PolicyConfig",
    "VerifierConfig",
    "RerankerConfig",
    "OutputConfig",
]

__version__ = "0.1.0"
