"""Regression anchor and import-identity tests for deduplicated pipeline functions.

Test 1 – regression anchor: run_pipeline() output must be structurally stable
    and deterministic before and after deduplication.

Test 2 – import identity: _apply_conformal_gate and _normalize_unit_decisions
    must be the *same object* when imported from either module, proving that
    core/pipeline_core.py no longer owns a separate duplicate body.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ega.contract import PolicyConfig
from ega.pipeline import run_pipeline
from ega.types import EvidenceItem, EvidenceSet


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_LLM_TEXT = "Paris is in France. It is a beautiful city."
# Sentence unitizer splits on ". " → ["Paris is in France.", "It is a beautiful city."]
# → unit ids u0001, u0002


def _make_evidence() -> EvidenceSet:
    return EvidenceSet(
        items=[
            EvidenceItem(id="e1", text="Paris is the capital of France.", metadata={})
        ]
    )


def _make_policy() -> PolicyConfig:
    return PolicyConfig(
        threshold_entailment=0.5,
        max_contradiction=0.9,
        partial_allowed=True,
    )


def _write_scores(tmp_path: Path) -> str:
    """Two rows, one per unit, both scoring 0.9 (well above 0.5 threshold)."""
    rows = [
        {"unit_id": "u0001", "score": 0.9, "label": "entailment"},
        {"unit_id": "u0002", "score": 0.9, "label": "entailment"},
    ]
    path = tmp_path / "scores.jsonl"
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )
    return str(path)


# ---------------------------------------------------------------------------
# Test 1 – regression anchor
# ---------------------------------------------------------------------------

def test_run_pipeline_regression_anchor(tmp_path: Path) -> None:
    """run_pipeline() must return the expected structure with all units accepted.

    This test is the behavioral anchor: if deduplication changes anything
    observable it will fail here first.
    """
    scores_path = _write_scores(tmp_path)

    out = run_pipeline(
        llm_summary_text=_LLM_TEXT,
        evidence=_make_evidence(),
        policy_config=_make_policy(),
        scores_jsonl_path=scores_path,
    )

    assert isinstance(out, dict)

    # Both units score 0.9 > 0.5 threshold → all accepted.
    assert out["payload_status"] == "ACCEPT", out.get("payload_status")
    assert out["payload_action"] == "EMIT", out.get("payload_action")

    verified = out["verified_extract"]
    assert isinstance(verified, list)
    assert len(verified) == 2, f"expected 2 verified units, got {len(verified)}"

    # Decision map: both accepted.
    decisions = out["decisions"]
    assert all(v == "accept" for v in decisions.values()), decisions

    # Trace is always present and schema-versioned.
    trace = out["trace"]
    assert isinstance(trace, dict)
    assert trace["trace_schema_version"] == 1
    assert trace["n_units"] == 2
    assert trace["kept_units"] == 2
    assert trace["dropped_units"] == 0


# ---------------------------------------------------------------------------
# Test 2 – import identity
# ---------------------------------------------------------------------------

def test_apply_conformal_gate_import_identity() -> None:
    """_apply_conformal_gate must be the same object in both modules.

    After deduplication, pipeline.py re-exports the function via an import
    from core.pipeline_core rather than defining its own copy.
    """
    from ega.pipeline import _apply_conformal_gate as gate_pipeline
    from ega.core.pipeline_core import _apply_conformal_gate as gate_core

    assert gate_pipeline is gate_core, (
        "_apply_conformal_gate is a separate object in pipeline.py — "
        "deduplication has not been applied yet, or was applied incorrectly."
    )


def test_normalize_unit_decisions_import_identity() -> None:
    """_normalize_unit_decisions must be the same object in both modules."""
    from ega.pipeline import _normalize_unit_decisions as norm_pipeline
    from ega.core.pipeline_core import _normalize_unit_decisions as norm_core

    assert norm_pipeline is norm_core, (
        "_normalize_unit_decisions is a separate object in pipeline.py — "
        "deduplication has not been applied yet, or was applied incorrectly."
    )
