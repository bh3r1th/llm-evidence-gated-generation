"""Minimal EGA usage example.

This example demonstrates the current scaffold API shape only.
"""

from ega.enforcer import Enforcer
from ega.types import VerificationResult


def run() -> None:
    enforcer = Enforcer()
    result = VerificationResult(verifier_name="demo", score=0.4, passed=False)
    decision = enforcer.enforce(result)
    print(f"Decision: {decision.value}")


if __name__ == "__main__":
    run()
