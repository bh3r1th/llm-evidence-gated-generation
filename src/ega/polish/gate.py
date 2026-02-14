"""Optional deterministic gate for post-enforcement polish outputs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Protocol

from ega.polish.types import PolishedUnit
from ega.polish.validators import (
    validate_no_new_named_entities,
    validate_no_new_numbers_dates,
    validate_overlap_bounds,
    validate_schema,
)
from ega.types import EnforcementResult, EvidenceItem, EvidenceSet, Unit


class _UnitEntailmentVerifier(Protocol):
    def verify(self, *, unit_text: str, unit_id: str, evidence: EvidenceSet) -> Any:
        """Return score-like object with an entailment float attribute."""


@dataclass(frozen=True, slots=True)
class PolishGateConfig:
    """Config for deterministic polish gating checks."""

    max_expansion_ratio: float = 1.20
    min_ngram_overlap: float = 0.60
    enable_nli_check: bool = False
    nli_entailment_threshold: float = 0.80
    verifier: _UnitEntailmentVerifier | None = None


def gate_polish(
    original_units: list[Unit],
    polished_units: list[PolishedUnit],
    config: PolishGateConfig,
) -> tuple[bool, list[str]]:
    """Return gate pass/fail and deterministic validation reasons."""

    errors = validate_schema(original_units, polished_units)
    if errors:
        return False, errors

    for original, polished in zip(original_units, polished_units, strict=False):
        if not validate_no_new_numbers_dates(original.text, polished.edited_text):
            errors.append(f"{original.id}: numbers_or_dates_changed")
        if not validate_overlap_bounds(
            original.text,
            polished.edited_text,
            max_expansion_ratio=config.max_expansion_ratio,
            min_ngram_overlap=config.min_ngram_overlap,
        ):
            errors.append(f"{original.id}: overlap_bounds_failed")
        if not validate_no_new_named_entities(original.text, polished.edited_text):
            errors.append(f"{original.id}: new_named_entity_proxy_detected")

        if config.enable_nli_check:
            if config.verifier is None:
                errors.append(f"{original.id}: nli_check_requested_without_verifier")
                continue
            try:
                score = config.verifier.verify(
                    unit_text=polished.edited_text,
                    unit_id=original.id,
                    evidence=EvidenceSet(
                        items=[EvidenceItem(id=original.id, text=original.text, metadata={})]
                    ),
                )
            except ImportError as exc:
                raise ImportError(
                    "NLI check requested but verifier dependencies are missing."
                ) from exc
            entailment = float(score.entailment)
            if entailment < config.nli_entailment_threshold:
                errors.append(f"{original.id}: nli_entailment_below_threshold")

    return (len(errors) == 0), errors


def apply_polish_gate(
    *,
    result: EnforcementResult,
    original_units: list[Unit],
    polished_units: list[PolishedUnit],
    config: PolishGateConfig,
) -> EnforcementResult:
    """Attach optional polish lane payload to an enforcement result."""

    passed, errors = gate_polish(original_units, polished_units, config)
    if not passed:
        return replace(
            result,
            polished_units=None,
            polish_status="failed",
            polish_fail_reasons=errors,
        )
    return replace(
        result,
        polished_units=[
            {"unit_id": unit.unit_id, "edited_text": unit.edited_text}
            for unit in polished_units
        ],
        polish_status="passed",
        polish_fail_reasons=[],
    )
