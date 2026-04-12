from __future__ import annotations

import pytest

from ega.contract import PolicyConfig
from ega.core.correction import CorrectionConfig, run_correction_loop
from ega.pipeline import _derive_workflow_contract, run_pipeline
from ega.types import AnswerCandidate, EvidenceItem, EvidenceSet, Unit, VerificationScore
from ega.unitization import unitize_answer


class _RoutingVerifier:
    model_name = "fake-nli"

    def __init__(self, score_by_text: dict[str, dict[str, object]]) -> None:
        self.score_by_text = score_by_text
        self.calls: list[list[Unit]] = []

    def verify_many(self, candidate, evidence):  # type: ignore[no-untyped-def]
        del evidence
        self.calls.append(list(candidate.units))
        rows: list[VerificationScore] = []
        for unit in candidate.units:
            config = self.score_by_text[unit.text]
            rows.append(
                VerificationScore(
                    unit_id=unit.id,
                    entailment=float(config["entailment"]),
                    contradiction=float(config["contradiction"]),
                    neutral=0.0,
                    label=str(config.get("label", "neutral")),
                    raw={
                        "chosen_evidence_id": config.get("chosen_evidence_id"),
                        "per_item_probs": config.get("per_item_probs", []),
                    },
                )
            )
        return rows

    @staticmethod
    def get_last_verify_trace() -> dict[str, float | int]:
        return {"n_pairs_scored": 1, "forward_seconds": 0.0}



def _policy() -> PolicyConfig:
    return PolicyConfig(threshold_entailment=0.5, max_contradiction=0.2, partial_allowed=True)



def _evidence() -> EvidenceSet:
    return EvidenceSet(items=[EvidenceItem(id="e1", text="support", metadata={})])



def test_text_mode_regression_and_fallback_retry_all_without_failure_classes() -> None:
    verifier = _RoutingVerifier(
        {
            "Good fact.": {
                "entailment": 0.9,
                "contradiction": 0.0,
                "label": "entailment",
                "chosen_evidence_id": "e1",
            },
            "Bad fact.": {
                "entailment": 0.1,
                "contradiction": 0.1,
                "label": "neutral",
                "chosen_evidence_id": "e1",
            },
        }
    )
    output = run_pipeline(
        llm_summary_text="Good fact. Bad fact.",
        evidence=_evidence(),
        policy_config=_policy(),
        use_oss_nli=True,
        verifier=verifier,
    )

    assert [row["unit_id"] for row in output["units"]] == ["u0001", "u0002"]

    core_output = {
        "intermediate_stats": {
            "candidate": AnswerCandidate(
                raw_answer_text="Bad fact. Good fact.",
                units=[
                    Unit(id="u0001", text="Bad fact.", metadata={}),
                    Unit(id="u0002", text="Good fact.", metadata={}),
                ],
            ),
            "cleaned_evidence": _evidence(),
        },
        "decisions": {"u0001": "reject", "u0002": "abstain"},
    }
    seen: list[list[str]] = []

    def _generator(failed_units: list[Unit], _evidence: EvidenceSet, _retry: int) -> dict[str, str]:
        seen.append([unit.id for unit in failed_units])
        return {}

    run_correction_loop(
        core_output=core_output,
        generator=_generator,
        verifier=lambda summary: {"intermediate_stats": core_output["intermediate_stats"], "decisions": {}},
        config=CorrectionConfig(enable_correction=True, max_retries=1, unitizer_mode="sentence"),
    )
    assert seen == [["u0001", "u0002"]]



def test_failure_classification_and_payload_aggregation_states() -> None:
    verifier = _RoutingVerifier(
        {
            "Supported.": {
                "entailment": 0.9,
                "contradiction": 0.0,
                "label": "entailment",
                "chosen_evidence_id": "e1",
                "per_item_probs": [{"evidence_id": "e1", "entailment": 0.9, "contradiction": 0.0}],
            },
            "Missing.": {
                "entailment": 0.1,
                "contradiction": 0.2,
                "label": "neutral",
                "chosen_evidence_id": None,
                "per_item_probs": [],
            },
            "Unsupported.": {
                "entailment": 0.1,
                "contradiction": 0.9,
                "label": "contradiction",
                "chosen_evidence_id": "e1",
                "per_item_probs": [{"evidence_id": "e1", "entailment": 0.1, "contradiction": 0.9}],
            },
            "Ambiguous.": {
                "entailment": 0.4,
                "contradiction": 0.2,
                "label": "neutral",
                "chosen_evidence_id": "e1",
                "per_item_probs": [{"evidence_id": "e1", "entailment": 0.4, "contradiction": 0.2}],
            },
        }
    )

    accepted = run_pipeline(
        llm_summary_text="Supported.",
        evidence=_evidence(),
        policy_config=_policy(),
        use_oss_nli=True,
        verifier=verifier,
    )
    assert accepted["payload_status"] == "ACCEPT"
    assert accepted["route_status"] == "READY"
    assert accepted["business_payload_emitted"] is True
    assert accepted["workflow_status"] == "COMPLETED"
    assert accepted["handoff_required"] is False
    assert accepted["handoff_reason"] is None
    assert accepted["tracking_id"] is None
    assert "adapter_payload" not in accepted

    rejected_repair = run_pipeline(
        llm_summary_text="Unsupported.",
        evidence=_evidence(),
        policy_config=_policy(),
        use_oss_nli=True,
        verifier=verifier,
        enable_correction=True,
        max_retries=1,
    )
    assert rejected_repair["payload_status"] == "REPAIR"
    assert rejected_repair["route_status"] == "REPAIR_PENDING"
    assert rejected_repair["business_payload_emitted"] is False
    assert rejected_repair["workflow_status"] == "PENDING"
    assert rejected_repair["handoff_required"] is True
    assert rejected_repair["handoff_reason"] == "BOUNDED_REPAIR"
    assert isinstance(rejected_repair["tracking_id"], str) and rejected_repair["tracking_id"]
    assert "adapter_payload" not in rejected_repair

    rejected_missing = run_pipeline(
        llm_summary_text="Missing. Ambiguous.",
        evidence=_evidence(),
        policy_config=_policy(),
        use_oss_nli=True,
        verifier=verifier,
        enable_correction=True,
        max_retries=1,
    )
    assert rejected_missing["payload_status"] == "REJECT"
    assert rejected_missing["route_status"] == "REJECTED"
    assert rejected_missing["business_payload_emitted"] is False
    assert rejected_missing["workflow_status"] == "COMPLETED"
    assert rejected_missing["handoff_required"] is False
    assert rejected_missing["handoff_reason"] is None
    assert rejected_missing["tracking_id"] is None
    assert rejected_missing["payload_failure_summary"] == {
        "supported": 0,
        "unsupported_claim": 0,
        "missing_in_source": 1,
        "ambiguous_source": 1,
    }
    assert "adapter_payload" not in rejected_missing


def test_adapter_mode_emits_partial_payload_for_mixed_reject_without_rejected_content() -> None:
    verifier = _RoutingVerifier(
        {
            "Supported.": {
                "entailment": 0.9,
                "contradiction": 0.0,
                "label": "entailment",
                "chosen_evidence_id": "e1",
                "per_item_probs": [{"evidence_id": "e1", "entailment": 0.9, "contradiction": 0.0}],
            },
            "Missing.": {
                "entailment": 0.1,
                "contradiction": 0.2,
                "label": "neutral",
                "chosen_evidence_id": None,
                "per_item_probs": [],
            },
        }
    )

    output = run_pipeline(
        llm_summary_text="Supported. Missing.",
        evidence=_evidence(),
        policy_config=_policy(),
        use_oss_nli=True,
        verifier=verifier,
        downstream_compatibility_mode="ADAPTER",
    )

    assert output["payload_status"] == "REJECT"
    assert output["business_payload_emitted"] is True
    assert output["adapter_payload"] == [{"unit_id": "u0001", "text": "Supported."}]
    assert all(row["text"] != "Missing." for row in output["adapter_payload"])
    assert output["adapter_summary"] == {
        "total_units": 2,
        "accepted_units": 1,
        "rejected_units": 1,
        "supported_count": 1,
        "unsupported_claim_count": 0,
        "missing_in_source_count": 1,
        "ambiguous_source_count": 0,
    }


def test_adapter_mode_repair_keeps_pending_and_does_not_emit_completed_payload() -> None:
    verifier = _RoutingVerifier(
        {
            "Unsupported.": {
                "entailment": 0.1,
                "contradiction": 0.9,
                "label": "contradiction",
                "chosen_evidence_id": "e1",
                "per_item_probs": [{"evidence_id": "e1", "entailment": 0.1, "contradiction": 0.9}],
            },
        }
    )
    output = run_pipeline(
        llm_summary_text="Unsupported.",
        evidence=_evidence(),
        policy_config=_policy(),
        use_oss_nli=True,
        verifier=verifier,
        enable_correction=True,
        max_retries=1,
        downstream_compatibility_mode="ADAPTER",
    )

    assert output["payload_status"] == "REPAIR"
    assert output["workflow_status"] == "PENDING"
    assert output["business_payload_emitted"] is False
    assert output["adapter_payload"] is None


def test_adapter_mode_zero_supported_reject_does_not_emit_business_payload() -> None:
    verifier = _RoutingVerifier(
        {
            "Missing.": {
                "entailment": 0.1,
                "contradiction": 0.2,
                "label": "neutral",
                "chosen_evidence_id": None,
                "per_item_probs": [],
            },
        }
    )
    output = run_pipeline(
        llm_summary_text="Missing.",
        evidence=_evidence(),
        policy_config=_policy(),
        use_oss_nli=True,
        verifier=verifier,
        downstream_compatibility_mode="ADAPTER",
    )

    assert output["payload_status"] == "REJECT"
    assert output["adapter_payload"] is None
    assert output["business_payload_emitted"] is False



def test_repair_gating_retries_only_unsupported_claim_units() -> None:
    score_by_text = {
        "Unsupported.": {
            "entailment": 0.1,
            "contradiction": 0.9,
            "label": "contradiction",
            "chosen_evidence_id": "e1",
            "per_item_probs": [{"evidence_id": "e1", "entailment": 0.1, "contradiction": 0.9}],
        },
        "Missing.": {
            "entailment": 0.1,
            "contradiction": 0.2,
            "label": "neutral",
            "chosen_evidence_id": None,
            "per_item_probs": [],
        },
        "Ambiguous.": {
            "entailment": 0.4,
            "contradiction": 0.2,
            "label": "neutral",
            "chosen_evidence_id": "e1",
            "per_item_probs": [{"evidence_id": "e1", "entailment": 0.4, "contradiction": 0.2}],
        },
        "Fixed.": {
            "entailment": 0.9,
            "contradiction": 0.0,
            "label": "entailment",
            "chosen_evidence_id": "e1",
            "per_item_probs": [{"evidence_id": "e1", "entailment": 0.9, "contradiction": 0.0}],
        },
    }
    verifier = _RoutingVerifier(score_by_text)
    generator_calls: list[list[str]] = []

    def _generator(failed_units: list[Unit], _evidence: EvidenceSet, _retry: int) -> dict[str, str]:
        generator_calls.append([unit.text for unit in failed_units])
        return {failed_units[0].id: "Fixed."}

    output = run_pipeline(
        llm_summary_text="Unsupported. Missing. Ambiguous.",
        evidence=_evidence(),
        policy_config=_policy(),
        use_oss_nli=True,
        verifier=verifier,
        enable_correction=True,
        max_retries=1,
        correction_generator=_generator,
    )

    assert generator_calls == [["Unsupported."]]
    assert output["correction"]["retries_attempted"] == 1



def test_structured_mode_runtime_wiring_and_validation() -> None:
    verifier = _RoutingVerifier(
        {
            "42": {"entailment": 0.9, "contradiction": 0.0, "label": "entailment", "chosen_evidence_id": "e1"},
            "x": {"entailment": 0.9, "contradiction": 0.0, "label": "entailment", "chosen_evidence_id": "e1"},
            "Plain sentence.": {"entailment": 0.9, "contradiction": 0.0, "label": "entailment", "chosen_evidence_id": "e1"},
        }
    )
    structured = run_pipeline(
        llm_summary_text="Ignored summary text.",
        structured_candidate_payload={"a": 42, "b": ["x"]},
        evidence=_evidence(),
        policy_config=_policy(),
        unitizer_mode="structured_field",
        use_oss_nli=True,
        verifier=verifier,
    )
    assert [row["unit_id"] for row in structured["units"]] == ["$.a", "$.b[0]"]

    legacy = run_pipeline(
        llm_summary_text="Plain sentence.",
        structured_candidate_payload={"a": 42},
        evidence=_evidence(),
        policy_config=_policy(),
        unitizer_mode="sentence",
        use_oss_nli=True,
        verifier=verifier,
    )
    assert [row["unit_id"] for row in legacy["units"]] == ["u0001"]

    with pytest.raises(ValueError, match="structured_candidate_payload"):
        run_pipeline(
            llm_summary_text="Ignored",
            evidence=_evidence(),
            policy_config=_policy(),
            unitizer_mode="structured_field",
            use_oss_nli=True,
            verifier=verifier,
        )



def test_unitize_answer_structured_field_mode_root_list_and_empty_payload() -> None:
    candidate = unitize_answer(["a", 2, {"x": True}], mode="structured_field")
    assert [unit.id for unit in candidate.units] == ["$[0]", "$[1]", "$[2].x"]
    assert [unit.text for unit in candidate.units] == ["a", "2", "true"]

    empty = unitize_answer({}, mode="structured_field")
    assert empty.units == []


def test_reject_review_route_sets_pending_handoff_contract() -> None:
    candidate = AnswerCandidate(
        raw_answer_text="Needs review.",
        units=[Unit(id="u0001", text="Needs review.", metadata={})],
    )
    contract = _derive_workflow_contract(
        payload_status="REJECT",
        payload_action="HOLD_REVIEW",
        candidate=candidate,
        decisions_by_unit={"u0001": "reject"},
    )

    assert contract["workflow_status"] == "PENDING"
    assert contract["handoff_required"] is True
    assert contract["handoff_reason"] == "HOLD_REVIEW"
    assert isinstance(contract["tracking_id"], str) and contract["tracking_id"]


def test_tracking_id_is_deterministic_for_same_context() -> None:
    candidate = AnswerCandidate(
        raw_answer_text="Unsupported.",
        units=[Unit(id="u0001", text="Unsupported.", metadata={})],
    )

    left = _derive_workflow_contract(
        payload_status="REPAIR",
        payload_action="BOUNDED_REPAIR",
        candidate=candidate,
        decisions_by_unit={"u0001": "reject"},
    )
    right = _derive_workflow_contract(
        payload_status="REPAIR",
        payload_action="BOUNDED_REPAIR",
        candidate=candidate,
        decisions_by_unit={"u0001": "reject"},
    )

    assert left["tracking_id"] == right["tracking_id"]
