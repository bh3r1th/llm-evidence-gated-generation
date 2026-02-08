from __future__ import annotations

import builtins
import importlib
import sys

import pytest


def test_importing_ega_core_does_not_require_optional_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    blocked = {"torch", "transformers", "wandb"}
    original_import = builtins.__import__

    def guarded_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        root_name = name.split(".", 1)[0]
        if root_name in blocked:
            raise ImportError(f"blocked optional dependency: {root_name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    sys.modules.pop("ega", None)

    ega = importlib.import_module("ega")

    assert ega.__version__


def test_nli_verifier_raises_clear_error_when_transformers_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ega.verifiers.nli_cross_encoder import NliCrossEncoderVerifier

    original_import = builtins.__import__

    class _TorchStub:
        @staticmethod
        def manual_seed(seed: int) -> None:
            _ = seed

    def guarded_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "torch":
            return _TorchStub
        if name == "transformers":
            raise ImportError("No module named 'transformers'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(ImportError, match=r"pip install 'ega\[nli\]'"):
        NliCrossEncoderVerifier()


def test_wandb_sink_raises_clear_error_when_wandb_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    from ega.adapters.wandb_sink import make_wandb_sink

    original_import = builtins.__import__

    def guarded_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "wandb":
            raise ImportError("No module named 'wandb'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    sink = make_wandb_sink(project="proj")
    with pytest.raises(ImportError, match=r"pip install ega\[wandb\]"):
        sink({"summary_stats": {}})
