from __future__ import annotations

import builtins

import pytest

from ega.v2.cross_encoder_reranker import CrossEncoderReranker


def test_cross_encoder_reranker_missing_dependency_raises_helpful_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def guarded_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "sentence_transformers":
            raise ImportError("No module named 'sentence_transformers'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(ImportError, match=r"ega\[rerank\]"):
        CrossEncoderReranker()
