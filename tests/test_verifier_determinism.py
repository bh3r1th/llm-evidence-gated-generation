"""Determinism tests for NliCrossEncoderVerifier.

Test 1 – identical inputs, two consecutive calls → identical output.
Test 2 – shuffle evidence item order → identical per-unit results.
Test 3 – shuffle unit order → identical per-unit results.
"""
from __future__ import annotations

import random

import pytest

from ega.types import AnswerCandidate, EvidenceItem, EvidenceSet, Unit
from ega.verifiers.nli_cross_encoder import NliCrossEncoderVerifier


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_UNITS = [
    Unit(id="uA", text="The Eiffel Tower is in Paris.", metadata={}),
    Unit(id="uB", text="Mount Everest is the tallest mountain.", metadata={}),
    Unit(id="uC", text="The Amazon river is the longest river.", metadata={}),
]

_EVIDENCE = [
    EvidenceItem(id="e1", text="Paris hosts the Eiffel Tower.", metadata={}),
    EvidenceItem(id="e2", text="Everest stands at 8849 metres above sea level.", metadata={}),
    EvidenceItem(id="e3", text="The Amazon is the world's longest river.", metadata={}),
    EvidenceItem(id="e4", text="France is a country in Europe.", metadata={}),
]

# Fixed per-(unit_text, evidence_text) scores so the predictor is
# content-addressed and batch-order-independent.
_SCORE_TABLE: dict[tuple[str, str], dict[str, float]] = {
    (_UNITS[0].text, _EVIDENCE[0].text): {"entailment": 0.82, "contradiction": 0.08, "neutral": 0.10},
    (_UNITS[0].text, _EVIDENCE[1].text): {"entailment": 0.05, "contradiction": 0.10, "neutral": 0.85},
    (_UNITS[0].text, _EVIDENCE[2].text): {"entailment": 0.06, "contradiction": 0.07, "neutral": 0.87},
    (_UNITS[0].text, _EVIDENCE[3].text): {"entailment": 0.71, "contradiction": 0.10, "neutral": 0.19},
    (_UNITS[1].text, _EVIDENCE[0].text): {"entailment": 0.09, "contradiction": 0.12, "neutral": 0.79},
    (_UNITS[1].text, _EVIDENCE[1].text): {"entailment": 0.78, "contradiction": 0.07, "neutral": 0.15},
    (_UNITS[1].text, _EVIDENCE[2].text): {"entailment": 0.11, "contradiction": 0.08, "neutral": 0.81},
    (_UNITS[1].text, _EVIDENCE[3].text): {"entailment": 0.14, "contradiction": 0.09, "neutral": 0.77},
    (_UNITS[2].text, _EVIDENCE[0].text): {"entailment": 0.07, "contradiction": 0.11, "neutral": 0.82},
    (_UNITS[2].text, _EVIDENCE[1].text): {"entailment": 0.10, "contradiction": 0.09, "neutral": 0.81},
    (_UNITS[2].text, _EVIDENCE[2].text): {"entailment": 0.88, "contradiction": 0.06, "neutral": 0.06},
    (_UNITS[2].text, _EVIDENCE[3].text): {"entailment": 0.13, "contradiction": 0.10, "neutral": 0.77},
}
_DEFAULT_PROB = {"entailment": 0.05, "contradiction": 0.10, "neutral": 0.85}


def _predictor(pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
    return [dict(_SCORE_TABLE.get(p, _DEFAULT_PROB)) for p in pairs]


def _make_verifier() -> NliCrossEncoderVerifier:
    return NliCrossEncoderVerifier(pair_predictor=_predictor, topk_per_unit=4)


def _fingerprint(scores: list) -> list[tuple[str, float, str | None]]:
    return [
        (s.unit_id, s.entailment, s.raw.get("chosen_evidence_id"))
        for s in scores
    ]


# ---------------------------------------------------------------------------
# Test 1 – two identical calls produce identical output
# ---------------------------------------------------------------------------

def test_identical_inputs_produce_identical_output() -> None:
    """Two consecutive verify() calls on the same inputs must return the same list."""
    verifier = _make_verifier()
    evidence = EvidenceSet(items=list(_EVIDENCE))

    scores_a = verifier.verify(list(_UNITS), evidence)
    scores_b = verifier.verify(list(_UNITS), evidence)

    assert _fingerprint(scores_a) == _fingerprint(scores_b), (
        "Second call produced different results for identical inputs"
    )


# ---------------------------------------------------------------------------
# Test 2 – evidence order does not affect per-unit results
# ---------------------------------------------------------------------------

def test_shuffled_evidence_produces_identical_per_unit_results() -> None:
    """Shuffling evidence items must not change which evidence is chosen or its score."""
    rng = random.Random(42)
    verifier = _make_verifier()

    evidence_ordered = EvidenceSet(items=list(_EVIDENCE))
    shuffled_items = list(_EVIDENCE)
    rng.shuffle(shuffled_items)
    evidence_shuffled = EvidenceSet(items=shuffled_items)

    # Sanity-check: the two EvidenceSets contain the same items in different orders.
    assert sorted(i.id for i in evidence_ordered.items) == sorted(i.id for i in evidence_shuffled.items)
    assert [i.id for i in evidence_ordered.items] != [i.id for i in evidence_shuffled.items]

    scores_ordered = verifier.verify(list(_UNITS), evidence_ordered)
    scores_shuffled = verifier.verify(list(_UNITS), evidence_shuffled)

    by_id_ordered = {s.unit_id: s for s in scores_ordered}
    by_id_shuffled = {s.unit_id: s for s in scores_shuffled}

    assert set(by_id_ordered) == set(by_id_shuffled), "Different unit_ids returned"

    for uid in by_id_ordered:
        o = by_id_ordered[uid]
        s = by_id_shuffled[uid]
        assert o.entailment == pytest.approx(s.entailment), (
            f"unit {uid}: entailment differs after evidence shuffle "
            f"({o.entailment} vs {s.entailment})"
        )
        assert o.raw.get("chosen_evidence_id") == s.raw.get("chosen_evidence_id"), (
            f"unit {uid}: chosen_evidence_id differs after evidence shuffle"
        )


# ---------------------------------------------------------------------------
# Test 3 – unit input order does not affect per-unit results
# ---------------------------------------------------------------------------

def test_shuffled_units_produce_identical_per_unit_results() -> None:
    """Shuffling the unit list must not change any individual unit's VerificationScore."""
    rng = random.Random(99)
    verifier = _make_verifier()
    evidence = EvidenceSet(items=list(_EVIDENCE))

    units_ordered = list(_UNITS)
    units_shuffled = list(_UNITS)
    rng.shuffle(units_shuffled)

    # Sanity-check: shuffled order differs.
    assert [u.id for u in units_ordered] != [u.id for u in units_shuffled]

    scores_ordered = verifier.verify(units_ordered, evidence)
    scores_shuffled = verifier.verify(units_shuffled, evidence)

    by_id_ordered = {s.unit_id: s for s in scores_ordered}
    by_id_shuffled = {s.unit_id: s for s in scores_shuffled}

    assert set(by_id_ordered) == set(by_id_shuffled), "Different unit_ids returned"

    for uid in by_id_ordered:
        o = by_id_ordered[uid]
        s = by_id_shuffled[uid]
        assert o.entailment == pytest.approx(s.entailment), (
            f"unit {uid}: entailment differs after unit shuffle "
            f"({o.entailment} vs {s.entailment})"
        )
        assert o.raw.get("chosen_evidence_id") == s.raw.get("chosen_evidence_id"), (
            f"unit {uid}: chosen_evidence_id differs after unit shuffle"
        )


# ---------------------------------------------------------------------------
# Test 4 – output list is always sorted by unit_id ASC
# ---------------------------------------------------------------------------

def test_output_is_sorted_by_unit_id() -> None:
    """verify() must return scores sorted by unit_id regardless of input order."""
    verifier = _make_verifier()
    evidence = EvidenceSet(items=list(_EVIDENCE))

    # Reverse unit order so input order != sorted order.
    units_reversed = list(reversed(_UNITS))
    scores = verifier.verify(units_reversed, evidence)

    unit_ids = [s.unit_id for s in scores]
    assert unit_ids == sorted(unit_ids), (
        f"Output not sorted by unit_id: {unit_ids}"
    )
