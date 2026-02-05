"""Serialization helpers for EGA artifacts.

This module provides placeholders for stable persistence and interchange of
policy, verification, and decision outputs.
"""

import json

from ega.types import VerificationResult


def verification_to_json(result: VerificationResult) -> str:
    """Serialize a verification result to JSON.

    TODO: Add schema versioning and richer payload support.
    """
    return json.dumps(
        {
            "verifier_name": result.verifier_name,
            "score": result.score,
            "passed": result.passed,
            "rationale": result.rationale,
        },
        sort_keys=True,
    )
