"""Evidence-Gated Answering (EGA) package public API.

Only ``verify_answer`` and ``PipelineConfig`` are stability-guaranteed package
entry points (plus ``PolicyConfig``, which is required to build
``PipelineConfig``).
"""

from ega.api import verify_answer
from ega.config import PipelineConfig
from ega.contract import PolicyConfig
from ega.utils.summarize import summarize_result
from ega.version import __version__

__all__ = [
    "verify_answer",
    "PipelineConfig",
    "PolicyConfig",
    "summarize_result",
]
