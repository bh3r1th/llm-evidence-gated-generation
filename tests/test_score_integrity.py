"""Score integrity tests.

Test 1 – entailment preserved after conformal gate rejects a unit.
Test 2 – conformal_decision populated on every unit after gate runs.
Test 3 – conformal_raw_score matches the score that triggered the decision.
Test 4 – nli_score populated by NliCrossEncoderVerifier on every VerificationScore.
"""
from __future__ import annotations

import pytest

from ega.core.pipeline_core import _apply_conformal_gate
from ega.types import EvidenceItem, EvidenceSet, Unit, VerificationScore
from ega.v2.conformal import ConformalState
from ega.verifiers.nli_cross_encoder import NliCrossEncoderVerifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_score(unit_id: str, entailment: float, contradiction: float = 0.1) -> VerificationScore:
    return VerificationScore(
        unit_id=unit_id,
        entailment=entailment,
        contradiction=contradiction,
        neutral=round(1.0 - entailment - contradiction, 6),
        label="entailment" if entailment > 0.5 else "neutral",
        raw={},
    )


def _state(threshold: float, margin: float = 0.0) -> ConformalState:
    return ConformalState(
        threshold=threshold,
        band_width=margin,
        abstain_k=1.0,
        n_samples=0,
        score_mean=0.0,
        score_std=0.0,
        meta={"abstain_margin": margin},
    )


# ---------------------------------------------------------------------------
# Test 1 – entailment is preserved after gate rejects a unit
# ---------------------------------------------------------------------------

def test_original_entailment_preserved_on_rejected_unit() -> None:
    """_apply_conformal_gate must not overwrite entailment/contradiction on rejected units."""
    original_entailment = 0.15
    original_contradiction = 0.75

    score = VerificationScore(
        unit_id="u1",
        entailment=original_entailment,
        contradiction=original_contradiction,
        neutral=0.10,
        label="contradiction",
        raw={},
    )
    # threshold=0.6, margin=0.0 → 0.15 < 0.6 → reject
    gated, _abstain_count, _meta = _apply_conformal_gate(
        scores=[score],
        state=_state(threshold=0.6, margin=0.0),
    )

    assert len(gated) == 1
    result = gated[0]
    assert result.conformal_decision == "reject"
    assert result.entailment == pytest.approx(original_entailment), (
        f"entailment was overwritten: expected {original_entailment}, got {result.entailment}"
    )
    assert result.contradiction == pytest.approx(original_contradiction), (
        f"contradiction was overwritten: expected {original_contradiction}, got {result.contradiction}"
    )


# ---------------------------------------------------------------------------
# Test 2 – conformal_decision populated on every unit
# ---------------------------------------------------------------------------

def test_conformal_decision_populated_on_all_units() -> None:
    """Every VerificationScore returned by _apply_conformal_gate must have conformal_decision set."""
    scores = [
        _make_score("u1", entailment=0.90),   # accept
        _make_score("u2", entailment=0.20),   # reject
        _make_score("u3", entailment=0.60),   # abstain (within margin ±0.05 of threshold=0.60)
    ]
    state = _state(threshold=0.60, margin=0.05)

    gated, _, _ = _apply_conformal_gate(scores=scores, state=state)

    assert len(gated) == len(scores)
    for s in gated:
        assert s.conformal_decision is not None, (
            f"unit {s.unit_id} has conformal_decision=None"
        )
    decisions = {s.unit_id: s.conformal_decision for s in gated}
    assert decisions["u1"] == "accept"
    assert decisions["u2"] == "reject"
    assert decisions["u3"] == "abstain"


# ---------------------------------------------------------------------------
# Test 3 – conformal_raw_score matches the triggering score
# ---------------------------------------------------------------------------

def test_conformal_raw_score_matches_triggering_score() -> None:
    """conformal_raw_score must equal the entailment value that was passed to gate()."""
    entailment_values = [0.82, 0.11, 0.58]
    scores = [_make_score(f"u{i+1}", e) for i, e in enumerate(entailment_values)]
    # threshold=0.55, margin=0.05 → u3 (0.58) falls within [0.50, 0.60] → abstain
    state = _state(threshold=0.55, margin=0.05)

    gated, _, _ = _apply_conformal_gate(scores=scores, state=state)

    for original, result in zip(scores, gated):
        assert result.conformal_raw_score == pytest.approx(original.entailment), (
            f"unit {result.unit_id}: conformal_raw_score {result.conformal_raw_score} "
            f"!= original entailment {original.entailment}"
        )


# ---------------------------------------------------------------------------
# Test 4 – nli_score populated by NliCrossEncoderVerifier
# ---------------------------------------------------------------------------

def test_nli_score_populated_by_nli_cross_encoder_verifier() -> None:
    """NliCrossEncoderVerifier must set nli_score on every returned VerificationScore."""
    predictor_probs = [
        {"entailment": 0.72, "contradiction": 0.15, "neutral": 0.13},
        {"entailment": 0.31, "contradiction": 0.55, "neutral": 0.14},
    ]

    def _predictor(pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        return [predictor_probs[i % len(predictor_probs)] for i in range(len(pairs))]

    verifier = NliCrossEncoderVerifier(pair_predictor=_predictor)
    units = [
        Unit(id="u1", text="Paris is the capital of France.", metadata={}),
        Unit(id="u2", text="The Eiffel Tower is in Paris.", metadata={}),
    ]
    evidence = EvidenceSet(
        items=[
            EvidenceItem(id="e1", text="Paris is in France.", metadata={}),
            EvidenceItem(id="e2", text="The Eiffel Tower stands in Paris.", metadata={}),
        ]
    )

    scores = verifier.verify(units, evidence)

    assert len(scores) == 2
    for score in scores:
        assert score.nli_score is not None, (
            f"unit {score.unit_id}: nli_score is None — NliCrossEncoderVerifier did not populate it"
        )
        # nli_score must equal the entailment selected by max_entailment aggregation
        assert score.nli_score == pytest.approx(score.entailment), (
            f"unit {score.unit_id}: nli_score {score.nli_score} != entailment {score.entailment}"
        )
