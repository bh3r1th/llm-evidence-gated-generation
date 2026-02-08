from __future__ import annotations

import os

import pytest

from ega.types import AnswerCandidate, EvidenceItem, EvidenceSet, Unit
from ega.verifiers.nli_cross_encoder import DEFAULT_MODEL_NAME, NliCrossEncoderVerifier


def test_label_index_resolution_handles_uppercase_id2label() -> None:
    resolved = NliCrossEncoderVerifier._resolve_label_indices(
        id2label={0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"},
        num_labels=3,
    )

    assert resolved == {"contradiction": 0, "neutral": 1, "entailment": 2}


def test_verify_unit_aggregates_by_max_entailment() -> None:
    evidence = EvidenceSet(
        items=[
            EvidenceItem(id="e1", text="Earth has one moon.", metadata={}),
            EvidenceItem(id="e2", text="Earth has two moons.", metadata={}),
        ]
    )

    verifier = NliCrossEncoderVerifier(
        pair_predictor=lambda pairs: [
            {"entailment": 0.2, "contradiction": 0.7, "neutral": 0.1},
            {"entailment": 0.8, "contradiction": 0.1, "neutral": 0.1},
        ],
    )

    score = verifier.verify_unit("Earth has two moons.", evidence)

    assert len(score.raw["per_item_probs"]) == 2
    assert score.entailment == pytest.approx(0.8)
    assert score.label == "entailment"
    assert score.raw["chosen_evidence_id"] == "e2"


def test_verify_returns_scores_for_each_candidate_unit() -> None:
    candidate = AnswerCandidate(
        raw_answer_text="u1\nu2",
        units=[Unit(id="u1", text="A", metadata={}), Unit(id="u2", text="B", metadata={})],
    )
    evidence = EvidenceSet(items=[EvidenceItem(id="e1", text="A", metadata={})])

    verifier = NliCrossEncoderVerifier(
        pair_predictor=lambda pairs: [
            {"entailment": 0.6, "contradiction": 0.2, "neutral": 0.2}
            for _ in pairs
        ]
    )

    scores = verifier.verify(candidate, evidence)

    assert [score.unit_id for score in scores] == ["u1", "u2"]
    assert all(score.raw["chosen_evidence_id"] == "e1" for score in scores)


def test_default_model_name_is_applied_when_none() -> None:
    verifier = NliCrossEncoderVerifier(
        model_name=None,
        pair_predictor=lambda pairs: [
            {"entailment": 0.7, "contradiction": 0.2, "neutral": 0.1}
            for _ in pairs
        ],
    )
    evidence = EvidenceSet(items=[EvidenceItem(id="e1", text="A", metadata={})])

    score = verifier.verify_unit("A", evidence)

    assert verifier.model_name == DEFAULT_MODEL_NAME
    assert score.raw["model_name"] == DEFAULT_MODEL_NAME


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("EGA_RUN_NLI_INTEGRATION") != "1",
    reason="Set EGA_RUN_NLI_INTEGRATION=1 to run real-model test.",
)
def test_integration_real_model_smoke() -> None:
    verifier = NliCrossEncoderVerifier(batch_size=2)
    evidence = EvidenceSet(
        items=[
            EvidenceItem(id="e1", text="Paris is the capital of France.", metadata={}),
            EvidenceItem(id="e2", text="Berlin is in Germany.", metadata={}),
        ]
    )

    score = verifier.verify_unit("Paris is France's capital.", evidence)

    assert score.raw["chosen_evidence_id"] in {"e1", "e2"}
    assert 0.0 <= score.entailment <= 1.0
