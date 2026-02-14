import pytest

from ega.polish.gate import PolishGateConfig, apply_polish_gate, gate_polish
from ega.polish.types import PolishedUnit
from ega.types import EnforcementResult, GateDecision, Unit, VerificationScore


def _unit(unit_id: str, text: str) -> Unit:
    return Unit(id=unit_id, text=text, metadata={})


def _result() -> EnforcementResult:
    return EnforcementResult(
        final_text="Paris is in France.",
        kept_units=["u1"],
        dropped_units=[],
        refusal_message=None,
        decision=GateDecision(
            allowed_units=["u1"],
            dropped_units=[],
            refusal=False,
            reason_code="OK_FULL",
            summary_stats={},
        ),
        scores=[
            VerificationScore(
                unit_id="u1",
                entailment=0.95,
                contradiction=0.01,
                neutral=0.04,
                label="entailment",
                raw={},
            )
        ],
        verified_units=[{"unit_id": "u1", "text": "Paris is in France."}],
    )


def test_gate_polish_failure_suppresses_polished_units_and_reports_reasons() -> None:
    original_units = [_unit("u1", "Paris is in France.")]
    polished_units = [PolishedUnit(unit_id="u1", edited_text="Paris is in Germany.")]

    updated = apply_polish_gate(
        result=_result(),
        original_units=original_units,
        polished_units=polished_units,
        config=PolishGateConfig(),
    )

    assert updated.polish_status == "failed"
    assert updated.polished_units is None
    assert updated.polish_fail_reasons


def test_gate_polish_passes_and_attaches_polished_units() -> None:
    original_units = [_unit("u1", "Paris is in France.")]
    polished_units = [PolishedUnit(unit_id="u1", edited_text="Paris is in France")]

    updated = apply_polish_gate(
        result=_result(),
        original_units=original_units,
        polished_units=polished_units,
        config=PolishGateConfig(),
    )

    assert updated.polish_status == "passed"
    assert updated.polish_fail_reasons == []
    assert updated.polished_units == [{"unit_id": "u1", "edited_text": "Paris is in France"}]


def test_gate_polish_nli_import_error_only_when_enabled() -> None:
    class _ImportFailVerifier:
        def verify(self, *, unit_text: str, unit_id: str, evidence):  # type: ignore[no-untyped-def]
            _ = (unit_text, unit_id, evidence)
            raise ImportError("missing optional dependency")

    original_units = [_unit("u1", "Paris is in France.")]
    polished_units = [PolishedUnit(unit_id="u1", edited_text="Paris is in France.")]

    passed, errors = gate_polish(
        original_units,
        polished_units,
        PolishGateConfig(enable_nli_check=False, verifier=_ImportFailVerifier()),
    )
    assert passed is True
    assert errors == []

    with pytest.raises(ImportError, match="NLI check requested"):
        gate_polish(
            original_units,
            polished_units,
            PolishGateConfig(enable_nli_check=True, verifier=_ImportFailVerifier()),
        )

