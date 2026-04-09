"""Verifier integrations for EGA."""

from ega.verifiers.adapter import LegacyVerifierAdapter
from ega.verifiers.nli_cross_encoder import NliCrossEncoderVerifier

__all__ = ["LegacyVerifierAdapter", "NliCrossEncoderVerifier"]
