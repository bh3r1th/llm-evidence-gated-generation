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

    retries = max(0, int(config.max_retries))
    if retries <= 0:
        return core_output

    current_output = core_output
    attempts = 0
    for retry_idx in range(retries):
        failed_units = _failed_units(current_output)
        if not failed_units:
            break

        replacements = generator(
            failed_units,
            current_output["intermediate_stats"]["cleaned_evidence"],
            retry_idx,
        )
        if not replacements:
            break

        next_summary = _apply_failed_unit_rewrites(
            units=current_output["intermediate_stats"]["candidate"].units,
            replacements=replacements,
            unitizer_mode=config.unitizer_mode,
            retry_index=retry_idx,
        )
        attempts += 1
        current_output = verifier(next_summary)

    current_output["correction"] = {
        "enabled": True,
        "attempts": attempts,
        "max_retries": retries,
    }
    return current_output


def _failed_units(core_output: dict[str, Any]) -> list[Unit]:
    candidate = core_output["intermediate_stats"]["candidate"]
    decisions = core_output.get("decisions", {})
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
