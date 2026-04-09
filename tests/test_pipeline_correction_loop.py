from __future__ import annotations

from ega.contract import PolicyConfig
from ega.pipeline import run_pipeline
from ega.types import EvidenceItem, EvidenceSet, Unit, VerificationScore


class _CountingConditionalVerifier:
    model_name = "fake-nli"

    def __init__(self) -> None:
        self.verify_calls: list[list[str]] = []

    def verify_many(self, candidate, evidence):  # type: ignore[no-untyped-def]
        del evidence
        self.verify_calls.append([unit.text for unit in candidate.units])
        scores: list[VerificationScore] = []
        for unit in candidate.units:
            is_supported = "good" in unit.text.lower() or "fixed" in unit.text.lower()
            entail = 0.95 if is_supported else 0.05
            scores.append(
                VerificationScore(
                    unit_id=unit.id,
                    entailment=entail,
                    contradiction=0.0 if is_supported else 0.9,
                    neutral=1.0 - entail,
                    label="entailment" if is_supported else "contradiction",
                    raw={"chosen_evidence_id": "e1", "per_item_probs": []},
                )
            )
        return scores

    @staticmethod
    def get_last_verify_trace() -> dict[str, float | int]:
        return {"n_pairs_scored": 1, "forward_seconds": 0.0}


def _evidence() -> EvidenceSet:
    return EvidenceSet(items=[EvidenceItem(id="e1", text="Good fact. Fixed fact.", metadata={})])


def _verified_text_sequence(verifier: _CountingConditionalVerifier) -> list[str]:
    return [texts[0] for texts in verifier.verify_calls]


def _run(
    *,
    llm_summary_text: str,
    enable_correction: bool,
    max_retries: int,
    correction_generator,
    verifier: _CountingConditionalVerifier | None = None,
):
    active_verifier = verifier or _CountingConditionalVerifier()
    output = run_pipeline(
        llm_summary_text=llm_summary_text,
        evidence=_evidence(),
        policy_config=PolicyConfig(
            threshold_entailment=0.5,
            max_contradiction=0.2,
            partial_allowed=True,
        ),
        use_oss_nli=True,
        verifier=active_verifier,
        enable_correction=enable_correction,
        max_retries=max_retries,
        correction_generator=correction_generator,
    )
    return output, active_verifier


def test_correction_disabled_failed_units_remain_and_no_retries() -> None:
    called = False

    def _generator(_failed_units: list[Unit], _evidence: EvidenceSet, _retry: int) -> dict[str, str]:
        nonlocal called
        called = True
        return {}

    output, verifier = _run(
        llm_summary_text="Bad fact. Good fact.",
        enable_correction=False,
        max_retries=2,
        correction_generator=_generator,
    )

    assert called is False
    assert len(verifier.verify_calls) == 2
    assert _verified_text_sequence(verifier) == ["Bad fact.", "Good fact."]
    assert output["stats"]["kept_units"] == 1
    assert output["correction"]["retries_attempted"] == 0
    assert output["correction"]["still_failed_count"] == 1
    assert output["trace"]["correction_stopped_reason"] == "correction_disabled"


def test_correction_enabled_failed_unit_only_is_corrected_and_reverified() -> None:
    correction_calls: list[list[str]] = []

    def _generator(failed_units: list[Unit], _evidence: EvidenceSet, _retry: int) -> dict[str, str]:
        correction_calls.append([unit.text for unit in failed_units])
        return {failed_units[0].id: "Fixed fact."}

    output, verifier = _run(
        llm_summary_text="Bad fact. Good fact.",
        enable_correction=True,
        max_retries=1,
        correction_generator=_generator,
    )

    assert correction_calls == [["Bad fact."]]
    assert len(verifier.verify_calls) == 4
    assert _verified_text_sequence(verifier) == [
        "Bad fact.",
        "Good fact.",
        "Fixed fact.",
        "Good fact.",
    ]
    assert output["verified_extract"] == [
        {"unit_id": "u0001", "text": "Fixed fact."},
        {"unit_id": "u0002", "text": "Good fact."},
    ]
    assert output["verified_text"] == "Fixed fact.\nGood fact."
    assert output["decision"]["dropped_units"] == []
    assert output["correction"]["reverify_occurred"] is True
    assert output["trace"]["correction_stopped_reason"] == "all_corrected"


def test_failed_unit_still_fails_stops_at_max_retries_without_extra_retry() -> None:
    retry_indices: list[int] = []

    def _generator(failed_units: list[Unit], _evidence: EvidenceSet, retry: int) -> dict[str, str]:
        retry_indices.append(retry)
        return {failed_units[0].id: "Still bad fact."}

    output, verifier = _run(
        llm_summary_text="Bad fact. Good fact.",
        enable_correction=True,
        max_retries=2,
        correction_generator=_generator,
    )

    assert retry_indices == [0, 1]
    assert len(verifier.verify_calls) == 6
    assert _verified_text_sequence(verifier) == [
        "Bad fact.",
        "Good fact.",
        "Still bad fact.",
        "Good fact.",
        "Still bad fact.",
        "Good fact.",
    ]
    assert output["correction"]["retries_attempted"] == 2
    assert output["correction"]["retries_attempted"] <= output["correction"]["max_retries"]
    assert output["correction"]["still_failed_count"] == 1
    assert output["decision"]["dropped_units"] == ["u0001"]
    assert output["trace"]["correction_stopped_reason"] == "retry_limit_reached"


def test_multiple_failed_units_targets_only_failed_subset() -> None:
    correction_calls: list[list[str]] = []

    def _generator(failed_units: list[Unit], _evidence: EvidenceSet, _retry: int) -> dict[str, str]:
        correction_calls.append([unit.text for unit in failed_units])
        return {failed_units[0].id: "Fixed fact."}

    output, verifier = _run(
        llm_summary_text="Bad fact one. Good fact. Bad fact two.",
        enable_correction=True,
        max_retries=1,
        correction_generator=_generator,
    )

    assert correction_calls == [["Bad fact one.", "Bad fact two."]]
    assert len(verifier.verify_calls) == 6
    assert _verified_text_sequence(verifier) == [
        "Bad fact one.",
        "Good fact.",
        "Bad fact two.",
        "Fixed fact.",
        "Good fact.",
        "Bad fact two.",
    ]
    assert output["verified_extract"] == [
        {"unit_id": "u0001", "text": "Fixed fact."},
        {"unit_id": "u0002", "text": "Good fact."},
    ]
    assert output["decision"]["dropped_units"] == ["u0003"]


def test_max_retries_enforced_when_generator_returns_no_replacements() -> None:
    retry_indices: list[int] = []

    def _generator(_failed_units: list[Unit], _evidence: EvidenceSet, retry: int) -> dict[str, str]:
        retry_indices.append(retry)
        return {}

    output, verifier = _run(
        llm_summary_text="Bad fact. Good fact.",
        enable_correction=True,
        max_retries=3,
        correction_generator=_generator,
    )

    assert retry_indices == [0, 1, 2]
    assert len(verifier.verify_calls) == 2
    assert output["correction"]["retries_attempted"] == 0
    assert output["correction"]["max_retries"] == 3
    assert output["correction"]["still_failed_count"] == 1
    assert output["trace"]["correction_stopped_reason"] == "retry_limit_reached"


def test_no_full_answer_retry_regression_guard() -> None:
    def _generator(failed_units: list[Unit], _evidence: EvidenceSet, _retry: int) -> dict[str, str]:
        # A full-answer retry bug would pass all units here; contract requires failed-units only.
        assert [unit.text for unit in failed_units] == ["Bad fact."]
        return {failed_units[0].id: "Fixed fact."}

    output, verifier = _run(
        llm_summary_text="Bad fact. Good fact.",
        enable_correction=True,
        max_retries=1,
        correction_generator=_generator,
    )

    assert _verified_text_sequence(verifier)[:2] == ["Bad fact.", "Good fact."]
    assert _verified_text_sequence(verifier)[2:] == ["Fixed fact.", "Good fact."]
    assert output["correction"]["retries_attempted"] == 1
