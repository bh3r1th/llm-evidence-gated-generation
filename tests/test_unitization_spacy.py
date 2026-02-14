import pytest

from ega.unitization_spacy import SpaCySentenceUnitizer

spacy = pytest.importorskip("spacy")


def test_spacy_sentence_unitizer_is_deterministic_and_stable() -> None:
    _ = spacy
    unitizer = SpaCySentenceUnitizer()
    text = "First sentence. Second sentence? Third sentence!"

    first = unitizer.unitize(text)
    second = unitizer.unitize(text)

    assert first == second
    assert [unit.id for unit in first] == ["u0001", "u0002", "u0003"]
    assert [unit.text for unit in first] == [
        "First sentence.",
        "Second sentence?",
        "Third sentence!",
    ]


def test_spacy_sentence_unitizer_attaches_context_window_metadata() -> None:
    _ = spacy
    unitizer = SpaCySentenceUnitizer()
    units = unitizer.unitize("One. Two. Three.")

    assert units[0].metadata["context_window"] == {"prev": "", "next": "Two."}
    assert units[1].metadata["context_window"] == {"prev": "One.", "next": "Three."}
    assert units[2].metadata["context_window"] == {"prev": "Two.", "next": ""}

