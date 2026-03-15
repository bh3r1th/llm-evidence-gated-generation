from __future__ import annotations

import importlib.util

import pytest

from ega.types import EvidenceItem, EvidenceSet, Unit
from ega.v2.cross_encoder_reranker import CrossEncoderReranker


class _FakeCrossEncoder:
    def predict(self, pairs, batch_size=32):  # type: ignore[no-untyped-def]
        _ = batch_size
        return [0.5 for _ in pairs]


def test_cross_encoder_reranker_smoke_optional_dependency() -> None:
    if importlib.util.find_spec("sentence_transformers") is None:
        pytest.xfail("sentence-transformers not installed; skipping optional smoke test")

    reranker = CrossEncoderReranker(cross_encoder=_FakeCrossEncoder(), batch_size=2, max_pairs=10)
    units = [Unit(id="u1", text="alpha unit", metadata={})]
    evidence = EvidenceSet(
        items=[
            EvidenceItem(id="e2", text="second evidence", metadata={}),
            EvidenceItem(id="e1", text="first evidence", metadata={}),
        ]
    )
    candidates = {"u1": ["e2", "e1"]}

    ranked, stats = reranker.rerank_with_stats(
        units=units,
        evidence=evidence,
        candidates=candidates,
        topk=2,
    )

    assert ranked["u1"] == ["e1", "e2"]
    assert stats.n_pairs_scored == 2
    assert stats.seconds >= 0.0
