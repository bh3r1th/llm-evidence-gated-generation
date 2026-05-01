from __future__ import annotations

import types
from typing import Callable

import pytest

from ega.types import AnswerCandidate, EvidenceItem, EvidenceSet, Unit
from ega.verifiers.nli_cross_encoder import NliCrossEncoderVerifier


def _install_fake_bm25(
    monkeypatch: pytest.MonkeyPatch,
    scorer: Callable[[list[str], int], list[float]],
) -> list[list[str]]:
    seen_queries: list[list[str]] = []

    class _FakeBM25Okapi:
        def __init__(self, tokenized_corpus: list[list[str]]) -> None:
            self._n = len(tokenized_corpus)

        def get_scores(self, query_tokens: list[str]) -> list[float]:
            seen_queries.append(list(query_tokens))
            return scorer(query_tokens, self._n)

    fake_module = types.SimpleNamespace(BM25Okapi=_FakeBM25Okapi)
    monkeypatch.setitem(__import__("sys").modules, "rank_bm25", fake_module)
    return seen_queries


def _default_verifier() -> NliCrossEncoderVerifier:
    return NliCrossEncoderVerifier(
        pair_predictor=lambda pairs: [
            {"entailment": 0.6, "contradiction": 0.2, "neutral": 0.2} for _ in pairs
        ],
        topk_per_unit=2,
        max_pairs_total=100,
    )


def _evidence() -> EvidenceSet:
    return EvidenceSet(
        items=[
            EvidenceItem(id="e1", text="product price value 123 2024-01-01", metadata={}),
            EvidenceItem(id="e2", text="other context", metadata={}),
        ]
    )


def _run_single_unit(verifier: NliCrossEncoderVerifier, unit: Unit) -> None:
    candidate = AnswerCandidate(raw_answer_text=unit.text, units=[unit])
    verifier.verify_many(candidate, _evidence())


def test_string_structured_query_contains_field_name_path_and_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _install_fake_bm25(monkeypatch, lambda _q, n: [1.0] + [0.0] * (n - 1))
    verifier = _default_verifier()
    unit = Unit(
        id="$.product.name",
        text="wireless mouse",
        metadata={"field_path": "$.product.name", "field_name": "name", "field_type": "string"},
    )

    _run_single_unit(verifier, unit)

    tokens = seen[0]
    assert "name" in tokens
    assert "$.product.name" in tokens
    assert "wireless" in tokens
    assert "mouse" in tokens


def test_number_structured_query_contains_numeric_value_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _install_fake_bm25(monkeypatch, lambda _q, n: [1.0] + [0.0] * (n - 1))
    verifier = _default_verifier()
    unit = Unit(
        id="$.product.price",
        text="123.45",
        metadata={"field_path": "$.product.price", "field_name": "price", "field_type": "number"},
    )

    _run_single_unit(verifier, unit)

    tokens = seen[0]
    assert "price" in tokens
    assert "$.product.price" in tokens
    assert "123.45" in tokens


def test_date_structured_query_contains_date_value_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _install_fake_bm25(monkeypatch, lambda _q, n: [1.0] + [0.0] * (n - 1))
    verifier = _default_verifier()
    unit = Unit(
        id="$.order.date",
        text="2025-03-14",
        metadata={"field_path": "$.order.date", "field_name": "date", "field_type": "date"},
    )

    _run_single_unit(verifier, unit)

    tokens = seen[0]
    assert "date" in tokens
    assert "$.order.date" in tokens
    assert "2025-03-14" in tokens


def test_structured_zero_candidates_falls_back_to_value_query_and_sets_trace_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def scorer(query_tokens: list[str], n: int) -> list[float]:
        if query_tokens == ["fallback-value"]:
            return [1.0] + [0.0] * (n - 1)
        return [0.0] * n

    seen = _install_fake_bm25(monkeypatch, scorer)
    verifier = _default_verifier()
    unit = Unit(
        id="$.status",
        text="fallback-value",
        metadata={"field_path": "$.status", "field_name": "status", "field_type": "string"},
    )

    _run_single_unit(verifier, unit)
    trace = verifier.get_last_verify_trace()

    assert len(seen) >= 2
    assert seen[1] == ["fallback-value"]
    assert trace["per_unit_preselect"]["$.status"]["field_query_fallback"] is True


def test_unstructured_query_is_unchanged_and_no_fallback_trace_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _install_fake_bm25(monkeypatch, lambda _q, n: [1.0] + [0.0] * (n - 1))
    verifier = _default_verifier()
    unit = Unit(id="u0001", text="plain sentence", metadata={})

    _run_single_unit(verifier, unit)
    trace = verifier.get_last_verify_trace()

    assert seen[0] == ["plain", "sentence"]
    assert "per_unit_preselect" not in trace


def test_structured_bm25_candidate_selection_is_deterministic_across_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def scorer(query_tokens: list[str], n: int) -> list[float]:
        if "name" in query_tokens and "$.product.name" in query_tokens:
            return [0.1, 0.9][:n]
        return [0.0] * n

    _install_fake_bm25(monkeypatch, scorer)
    verifier = _default_verifier()
    unit = Unit(
        id="$.product.name",
        text="wireless",
        metadata={"field_path": "$.product.name", "field_name": "name", "field_type": "string"},
    )
    candidate = AnswerCandidate(raw_answer_text=unit.text, units=[unit])
    evidence = _evidence()

    first = verifier.verify_many(candidate, evidence)[0]
    second = verifier.verify_many(candidate, evidence)[0]

    first_ids = [row["evidence_id"] for row in first.raw["per_item_probs"]]
    second_ids = [row["evidence_id"] for row in second.raw["per_item_probs"]]
    assert first_ids == second_ids
