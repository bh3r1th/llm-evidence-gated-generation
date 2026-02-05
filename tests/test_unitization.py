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
