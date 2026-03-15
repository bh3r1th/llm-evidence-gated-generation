"""Evidence-Gated Answering (EGA) package.

EGA focuses on policy enforcement and final decision gating for answers that
must be supported by acceptable evidence.
"""

from ega.contract import EGA_SCHEMA_VERSION, PolicyConfig, ReasonCode
from ega.verifiers.nli_cross_encoder import DEFAULT_MODEL_NAME

__all__ = [
    "DEFAULT_MODEL_NAME",
    "EGA_SCHEMA_VERSION",
    "PolicyConfig",
    "ReasonCode",
    "decision",
    "enforcer",
    "policy",
    "serialization",
    "types",
    "unitization",
    "v2",
]

__version__ = "0.1.0"
