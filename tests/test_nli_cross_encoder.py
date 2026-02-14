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


def test_verify_many_stage1_topk_is_deterministic_on_ties() -> None:
    seen_pairs: list[tuple[str, str]] = []

    def pair_predictor(pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        seen_pairs.extend(pairs)
        return [
            {"entailment": 0.5, "contradiction": 0.3, "neutral": 0.2}
            for _ in pairs
        ]

    verifier = NliCrossEncoderVerifier(
        pair_predictor=pair_predictor,
        topk_per_unit=2,
        max_pairs_total=100,
    )
    candidate = AnswerCandidate(
        raw_answer_text="z",
        units=[Unit(id="u1", text="zzz", metadata={})],
    )
    evidence = EvidenceSet(
        items=[
            EvidenceItem(id="e0", text="alpha", metadata={}),
            EvidenceItem(id="e1", text="beta", metadata={}),
            EvidenceItem(id="e2", text="gamma", metadata={}),
        ]
    )

    scores = verifier.verify_many(candidate, evidence)

    assert seen_pairs == [("zzz", "alpha"), ("zzz", "beta")]
    assert [row["evidence_id"] for row in scores[0].raw["per_item_probs"]] == ["e0", "e1"]


def test_verify_many_caps_reduce_pairs_and_trace_counts() -> None:
    verifier = NliCrossEncoderVerifier(
        pair_predictor=lambda pairs: [
            {"entailment": 0.5, "contradiction": 0.3, "neutral": 0.2}
            for _ in pairs
        ],
        topk_per_unit=3,
        max_pairs_total=4,
    )
    candidate = AnswerCandidate(
        raw_answer_text="u1 u2 u3",
        units=[
            Unit(id="u1", text="x", metadata={}),
            Unit(id="u2", text="y", metadata={}),
            Unit(id="u3", text="z", metadata={}),
        ],
    )
    evidence = EvidenceSet(
        items=[
            EvidenceItem(id="e0", text="a", metadata={}),
            EvidenceItem(id="e1", text="b", metadata={}),
            EvidenceItem(id="e2", text="c", metadata={}),
            EvidenceItem(id="e3", text="d", metadata={}),
        ]
    )

    scores = verifier.verify_many(candidate, evidence)
    trace = verifier.get_last_verify_trace()

    assert trace["pairs_pruned_stage1"] == 3
    assert trace["pairs_pruned_stage2"] == 5
    assert trace["n_pairs_scored"] == 4
    assert sum(len(score.raw["per_item_probs"]) for score in scores) == 4


def test_verify_many_token_budget_batches_and_preserves_pair_mapping() -> None:
    batch_sizes: list[int] = []
    pair_to_entailment = {
        ("u short", "e one"): 0.2,
        ("u short", "e much longer evidence text"): 0.9,
        ("u much longer unit text", "e one"): 0.7,
        ("u much longer unit text", "e much longer evidence text"): 0.1,
    }

    def pair_predictor(pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        batch_sizes.append(len(pairs))
        return [
            {
                "entailment": pair_to_entailment[pair],
                "contradiction": 1.0 - pair_to_entailment[pair],
                "neutral": 0.0,
            }
            for pair in pairs
        ]

    verifier = NliCrossEncoderVerifier(
        pair_predictor=pair_predictor,
        topk_per_unit=2,
        max_pairs_total=10,
        max_batch_tokens=6,
    )
    candidate = AnswerCandidate(
        raw_answer_text="u1 u2",
        units=[
            Unit(id="u1", text="u short", metadata={}),
            Unit(id="u2", text="u much longer unit text", metadata={}),
        ],
    )
    evidence = EvidenceSet(
        items=[
            EvidenceItem(id="e1", text="e one", metadata={}),
            EvidenceItem(id="e2", text="e much longer evidence text", metadata={}),
        ]
    )

    scores = verifier.verify_many(candidate, evidence)
    trace = verifier.get_last_verify_trace()

    assert trace["num_batches"] >= 2
    assert sum(batch_sizes) == 4
    assert scores[0].raw["chosen_evidence_id"] == "e2"
    assert scores[1].raw["chosen_evidence_id"] == "e1"


def test_evidence_truncation_is_deterministic_and_applied_before_scoring() -> None:
    seen_pairs: list[tuple[str, str]] = []

    def pair_predictor(pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        seen_pairs.extend(pairs)
        return [
            {"entailment": 0.5, "contradiction": 0.3, "neutral": 0.2}
            for _ in pairs
        ]

    verifier = NliCrossEncoderVerifier(
        pair_predictor=pair_predictor,
        evidence_max_sentences=1,
        evidence_max_chars=20,
        topk_per_unit=1,
        max_pairs_total=10,
    )
    candidate = AnswerCandidate(
        raw_answer_text="u1",
        units=[Unit(id="u1", text="Unit text.", metadata={})],
    )
    evidence = EvidenceSet(
        items=[
            EvidenceItem(id="e1", text="First sentence. Second sentence here.", metadata={}),
            EvidenceItem(id="e2", text="Another start. Another second.", metadata={}),
        ]
    )

    verifier.verify_many(candidate, evidence)
    trace = verifier.get_last_verify_trace()

    assert seen_pairs
    assert seen_pairs[0][1] == "First sentence."
    assert trace["evidence_truncated_frac"] > 0.0
    assert trace["evidence_chars_mean_after"] <= trace["evidence_chars_mean_before"]


def _fake_torch(*, cuda_available: bool):  # type: ignore[no-untyped-def]
    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return cuda_available

    class _Torch:
        cuda = _Cuda()
        float16 = "float16"
        float32 = "float32"
        bfloat16 = "bfloat16"

    return _Torch


def test_runtime_cpu_auto_dtype_resolves_to_float32(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        NliCrossEncoderVerifier,
        "_import_torch",
        staticmethod(lambda: _fake_torch(cuda_available=False)),
    )
    verifier = NliCrossEncoderVerifier(
        pair_predictor=lambda pairs: [
            {"entailment": 0.6, "contradiction": 0.2, "neutral": 0.2}
            for _ in pairs
        ],
        device="cpu",
        dtype="auto",
    )
    verifier.verify_many(
        AnswerCandidate(raw_answer_text="a", units=[Unit(id="u1", text="a", metadata={})]),
        EvidenceSet(items=[EvidenceItem(id="e1", text="a", metadata={})]),
    )
    trace = verifier.get_last_verify_trace()

    assert verifier.device == "cpu"
    assert verifier.dtype == "float32"
    assert verifier.amp_enabled is False
    assert trace["dtype"] == "float32"
    assert trace["amp_enabled"] is False


def test_runtime_cpu_float16_is_overridden_to_float32(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        NliCrossEncoderVerifier,
        "_import_torch",
        staticmethod(lambda: _fake_torch(cuda_available=False)),
    )
    verifier = NliCrossEncoderVerifier(
        pair_predictor=lambda pairs: [
            {"entailment": 0.6, "contradiction": 0.2, "neutral": 0.2}
            for _ in pairs
        ],
        device="cpu",
        dtype="float16",
    )
    verifier.verify_many(
        AnswerCandidate(raw_answer_text="a", units=[Unit(id="u1", text="a", metadata={})]),
        EvidenceSet(items=[EvidenceItem(id="e1", text="a", metadata={})]),
    )
    trace = verifier.get_last_verify_trace()

    assert verifier.dtype == "float32"
    assert verifier.dtype_overridden is True
    assert verifier.amp_enabled is False
    assert trace["dtype"] == "float32"
    assert trace["dtype_overridden"] is True
    assert trace["amp_enabled"] is False


def test_runtime_auto_device_resolves_to_cpu_when_cuda_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        NliCrossEncoderVerifier,
        "_import_torch",
        staticmethod(lambda: _fake_torch(cuda_available=False)),
    )
    verifier = NliCrossEncoderVerifier(
        pair_predictor=lambda pairs: [
            {"entailment": 0.6, "contradiction": 0.2, "neutral": 0.2}
            for _ in pairs
        ],
        device="auto",
        dtype="auto",
    )

    assert verifier.device == "cpu"
    assert verifier.dtype == "float32"
    assert verifier.amp_enabled is False


def test_runtime_auto_device_resolves_to_cuda_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        NliCrossEncoderVerifier,
        "_import_torch",
        staticmethod(lambda: _fake_torch(cuda_available=True)),
    )
    verifier = NliCrossEncoderVerifier(
        pair_predictor=lambda pairs: [
            {"entailment": 0.6, "contradiction": 0.2, "neutral": 0.2}
            for _ in pairs
        ],
        device="auto",
        dtype="auto",
    )

    assert verifier.device == "cuda"
    assert verifier.dtype == "float16"
    assert verifier.amp_enabled is True


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
