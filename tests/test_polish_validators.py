from ega.polish.types import PolishedUnit
from ega.polish.validators import (
    validate_no_new_named_entities,
    validate_no_new_numbers_dates,
    validate_overlap_bounds,
    validate_schema,
)
from ega.types import Unit


def _unit(unit_id: str, text: str) -> Unit:
    return Unit(id=unit_id, text=text, metadata={})


def test_validate_schema_fails_on_count_and_order() -> None:
    original = [_unit("u1", "A"), _unit("u2", "B")]
    polished = [PolishedUnit(unit_id="u2", edited_text="B")]

    errors = validate_schema(original, polished)

    assert any("count_mismatch" in error for error in errors)
    assert any("unit_id_mismatch" in error for error in errors)


def test_validate_no_new_numbers_dates_fails_when_number_changes() -> None:
    assert (
        validate_no_new_numbers_dates(
            "Revenue was 2024 and date 10/31/2024.",
            "Revenue was 2025 and date 10/31/2024.",
        )
        is False
    )


def test_validate_overlap_bounds_fails_on_large_expansion() -> None:
    assert (
        validate_overlap_bounds(
            "alpha beta gamma",
            "alpha beta gamma delta epsilon zeta eta theta",
            max_expansion_ratio=1.20,
            min_ngram_overlap=0.60,
        )
        is False
    )


def test_validate_overlap_bounds_fails_on_low_ngram_overlap() -> None:
    assert (
        validate_overlap_bounds(
            "alpha beta gamma delta epsilon",
            "zeta eta theta iota kappa",
            max_expansion_ratio=2.0,
            min_ngram_overlap=0.60,
        )
        is False
    )


def test_validate_no_new_named_entities_fails_when_capitalized_token_added() -> None:
    assert validate_no_new_named_entities("Paris is in France.", "Paris is in Germany.") is False

