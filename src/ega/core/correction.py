"""Bounded correction loop utilities for failed-unit regeneration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ega.text_clean import clean_text
from ega.types import Unit
from ega.unitization import unitize_answer


Generator = Callable[[list[Unit], Any, int], dict[str, str] | None]
Verifier = Callable[[str], dict[str, Any]]


@dataclass(frozen=True, slots=True)
class CorrectionConfig:
    """Runtime options for bounded correction retries."""

    enable_correction: bool = False
    max_retries: int = 1
    unitizer_mode: str = "sentence"


def run_correction_loop(
    core_output: dict[str, Any],
    generator: Generator | None,
    verifier: Verifier,
    config: CorrectionConfig,
) -> dict[str, Any]:
    """Retry failed units only with bounded regeneration + re-verification."""
    if not config.enable_correction or generator is None:
        return core_output

    max_retries = max(0, int(config.max_retries))
    if max_retries <= 0:
        return core_output

    current_output = core_output
    retries_attempted = 0
    corrected_unit_ids: set[str] = set()
    stopped_reason = "retry_limit_reached"
    for retry_idx in range(max_retries):
        failed_units_for_retry = _failed_units(current_output)
        if not failed_units_for_retry:
            stopped_reason = "no_failed_units"
            break

        generated_replacements = generator(
            failed_units_for_retry,
            current_output["intermediate_stats"]["cleaned_evidence"],
            retry_idx,
        )
        replacement_text_by_failed_unit = {
            unit.id: generated_replacements[unit.id]
            for unit in failed_units_for_retry
            if generated_replacements is not None and unit.id in generated_replacements
        }
        if not replacement_text_by_failed_unit:
            # No correction candidate was produced for currently failed units.
            # Keep loop deterministic and bounded by max_retries.
            continue

        failed_unit_ids_targeted = set(replacement_text_by_failed_unit.keys())
        next_summary = _apply_failed_unit_rewrites(
            units=current_output["intermediate_stats"]["candidate"].units,
            replacements=replacement_text_by_failed_unit,
            unitizer_mode=config.unitizer_mode,
            retry_index=retry_idx,
        )
        retries_attempted += 1
        current_output = verifier(next_summary)
        failed_unit_ids_after_retry = {unit.id for unit in _failed_units(current_output)}
        corrected_unit_ids.update(
            unit_id
            for unit_id in failed_unit_ids_targeted
            if unit_id not in failed_unit_ids_after_retry
        )

    final_failed = _failed_units(current_output)
    if not final_failed:
        stopped_reason = "all_corrected" if retries_attempted > 0 else "no_failed_units"
    elif retries_attempted >= max_retries:
        stopped_reason = "retry_limit_reached"

    current_output["correction"] = {
        "enabled": True,
        "attempts": retries_attempted,
        "max_retries": max_retries,
        "retries_attempted": retries_attempted,
        "corrected_unit_count": int(len(corrected_unit_ids)),
        "still_failed_count": int(len(final_failed)),
        "reverify_occurred": bool(retries_attempted > 0),
        "stopped_reason": stopped_reason,
    }
    return current_output


def _failed_units(core_output: dict[str, Any]) -> list[Unit]:
    candidate = core_output["intermediate_stats"]["candidate"]
    decisions = core_output.get("decisions", {})
    failure_class_by_unit = core_output.get("failure_class_by_unit")
    if failure_class_by_unit is not None:
        return [
            unit
            for unit in candidate.units
            if failure_class_by_unit.get(unit.id) == "UNSUPPORTED_CLAIM"
        ]
    return [unit for unit in candidate.units if decisions.get(unit.id) != "accept"]


def _apply_failed_unit_rewrites(
    *,
    units: list[Unit],
    replacements: dict[str, str],
    unitizer_mode: str,
    retry_index: int,
) -> str:
    next_units: list[str] = []
    for unit in units:
        replacement = replacements.get(unit.id)
        if replacement is None:
            next_units.append(clean_text(unit.text))
            continue

        regenerated = _unitize_regenerated_text(
            unit_id=unit.id,
            text=replacement,
            mode=unitizer_mode,
            retry_index=retry_index,
        )
        next_units.extend(regenerated)
    return clean_text("\n".join(text for text in next_units if text))


def _unitize_regenerated_text(
    *,
    unit_id: str,
    text: str,
    mode: str,
    retry_index: int,
) -> list[str]:
    cleaned = clean_text(text)
    if not cleaned:
        return []

    try:
        candidate = unitize_answer(cleaned, mode=("markdown_bullet" if mode == "bullets" else mode))
    except ValueError:
        candidate = None

    if candidate is None or not candidate.units:
        return [cleaned]

    rewritten: list[str] = []
    for _idx, unit in enumerate(candidate.units, start=1):
        text_value = clean_text(unit.text)
        if text_value:
            rewritten.append(text_value)

    if not rewritten:
        return [cleaned]
    _ = unit_id, retry_index
    return rewritten
