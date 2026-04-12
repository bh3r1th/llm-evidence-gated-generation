import builtins
import sys
import types

import pytest

from ega.types import AnswerCandidate
from ega.unitization import MarkdownBulletUnitizer, SentenceUnitizer, unitize_answer


def test_sentence_unitizer_has_stable_ids_and_order() -> None:
    unitizer = SentenceUnitizer()

    units = unitizer.unitize("First sentence. Second sentence? Third sentence!")

    assert [unit.id for unit in units] == ["u0001", "u0002", "u0003"]
    assert [unit.text for unit in units] == [
        "First sentence.",
        "Second sentence?",
        "Third sentence!",
    ]


def test_sentence_unitizer_is_deterministic() -> None:
    text = "A. B. C."
    unitizer = SentenceUnitizer()

    first = unitizer.unitize(text)
    second = unitizer.unitize(text)

    assert first == second


def test_markdown_bullet_unitizer_prefers_bullet_lines() -> None:
    unitizer = MarkdownBulletUnitizer()
    text = """
    - first bullet
    1. second bullet
    * third bullet
    """

    units = unitizer.unitize(text)

    assert [unit.id for unit in units] == ["u0001", "u0002", "u0003"]
    assert [unit.text for unit in units] == [
        "- first bullet",
        "1. second bullet",
        "* third bullet",
    ]


def test_markdown_bullet_unitizer_falls_back_to_sentence_split() -> None:
    unitizer = MarkdownBulletUnitizer()

    units = unitizer.unitize("One sentence. Another sentence.")

    assert [unit.text for unit in units] == ["One sentence.", "Another sentence."]


def test_unitize_answer_uses_requested_mode() -> None:
    candidate = unitize_answer("- item one\n- item two", mode="markdown_bullet")

    assert isinstance(candidate, AnswerCandidate)
    assert [unit.id for unit in candidate.units] == ["u0001", "u0002"]
    assert [unit.text for unit in candidate.units] == ["- item one", "- item two"]


def test_unitize_answer_spacy_mode_raises_clear_error_when_dependency_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def guarded_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "spacy":
            raise ImportError("No module named 'spacy'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(ImportError, match=r"pip install 'ega\[unitize\]'"):
        unitize_answer("First. Second.", mode="spacy_sentence")


def test_unitize_answer_default_uses_lightweight_sentence_mode_without_spacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def guarded_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "spacy":
            raise AssertionError("spacy should not be imported for default unitizer mode")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    candidate = unitize_answer("One. Two.")

    assert [unit.text for unit in candidate.units] == ["One.", "Two."]


def test_unitize_answer_spacy_mode_uses_spacy_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeSent:
        def __init__(self, text: str) -> None:
            self.text = text

    class _FakeDoc:
        def __init__(self, text: str) -> None:
            parts = [part.strip() for part in text.replace("?", ".").replace("!", ".").split(".") if part.strip()]
            self.sents = [_FakeSent(f"{part}.") for part in parts]

    class _FakeNLP:
        def add_pipe(self, _name: str) -> None:
            return None

        def __call__(self, text: str) -> _FakeDoc:
            return _FakeDoc(text)

    fake_spacy = types.SimpleNamespace(blank=lambda _lang: _FakeNLP())
    monkeypatch.setitem(sys.modules, "spacy", fake_spacy)

    candidate = unitize_answer("One. Two?", mode="spacy_sentence")

    assert isinstance(candidate, AnswerCandidate)
    assert [unit.id for unit in candidate.units] == ["u0001", "u0002"]
    assert [unit.text for unit in candidate.units] == ["One.", "Two."]


def test_unitize_answer_structured_field_mode_emits_path_based_ids() -> None:
    candidate = unitize_answer(
        {
            "invoice": {"total": 42.5},
            "line_items": [{"sku": "ABC"}, {"sku": "XYZ"}],
            "tags": ["priority", "paid"],
        },
        mode="structured_field",
    )

    assert [unit.id for unit in candidate.units] == [
        "$.invoice.total",
        "$.line_items[0].sku",
        "$.line_items[1].sku",
        "$.tags[0]",
        "$.tags[1]",
    ]
    assert [unit.text for unit in candidate.units] == [
        "42.5",
        "ABC",
        "XYZ",
        "priority",
        "paid",
    ]
    assert all(unit.metadata["structured_mode"] is True for unit in candidate.units)


def test_unitize_answer_structured_field_mode_skips_non_scalar_leaves() -> None:
    candidate = unitize_answer(
        {
            "a": object(),
            "b": [{"ok": True}, {"x": {"nested": []}}],
            "c": [],
            "d": {},
        },
        mode="structured_field",
    )

    assert [unit.id for unit in candidate.units] == ["$.b[0].ok"]
