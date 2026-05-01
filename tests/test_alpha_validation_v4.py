from __future__ import annotations

from ega.contract import PolicyConfig
from ega.pipeline import _derive_workflow_contract, run_pipeline
from ega.types import AnswerCandidate, EvidenceItem, EvidenceSet, Unit, VerificationScore


class _AlphaValidationVerifier:
    model_name = "fake-nli"

    def __init__(self, score_by_text: dict[str, dict[str, object]]) -> None:
        self.score_by_text = score_by_text

    def verify_many(self, candidate, evidence):  # type: ignore[no-untyped-def]
        del evidence
        return [
            VerificationScore(
                unit_id=unit.id,
                entailment=float(self.score_by_text[unit.text]["entailment"]),
                contradiction=float(self.score_by_text[unit.text]["contradiction"]),
                neutral=0.0,
                label=str(self.score_by_text[unit.text].get("label", "neutral")),
                raw={
                    "chosen_evidence_id": self.score_by_text[unit.text].get("chosen_evidence_id"),
                    "per_item_probs": self.score_by_text[unit.text].get("per_item_probs", []),
                },
            )
            for unit in candidate.units
        ]

    @staticmethod
    def get_last_verify_trace() -> dict[str, float | int]:
        return {"n_pairs_scored": 1, "forward_seconds": 0.0}


def _policy() -> PolicyConfig:
    return PolicyConfig(threshold_entailment=0.5, max_contradiction=0.2, partial_allowed=True)


def _evidence() -> EvidenceSet:
    return EvidenceSet(items=[EvidenceItem(id="e1", text="support", metadata={})])


def test_alpha_scenario_1_legacy_text_mode_accepted_flow_unchanged() -> None:
    verifier = _AlphaValidationVerifier(
        {
            "Legacy mode works.": {
                "entailment": 0.95,
                "contradiction": 0.01,
                "label": "entailment",
                "chosen_evidence_id": "e1",
            }
        }
    )

    output = run_pipeline(
        llm_summary_text="Legacy mode works.",
        evidence=_evidence(),
        policy_config=_policy(),
        use_oss_nli=True,
        verifier=verifier,
    )

    assert output["payload_status"] == "ACCEPT"
    assert output["business_payload_emitted"] is True
    assert output["route_status"] == "READY"
    assert output["workflow_status"] == "COMPLETED"
    assert "adapter_payload" not in output


def test_alpha_scenario_2_structured_strict_passthrough_blocks_partial_success_payload() -> None:
    verifier = _AlphaValidationVerifier(
        {
            "approved": {
                "entailment": 0.98,
                "contradiction": 0.0,
                "label": "entailment",
                "chosen_evidence_id": "e1",
            },
            "": {
                "entailment": 0.0,
                "contradiction": 1.0,
                "label": "contradiction",
                "chosen_evidence_id": None,
            },
        }
    )

    output = run_pipeline(
        llm_summary_text="ignored in structured_field mode",
        structured_candidate_payload={"status": "approved", "required_field": ""},
        unitizer_mode="structured_field",
        evidence=_evidence(),
        policy_config=_policy(),
        use_oss_nli=True,
        verifier=verifier,
    )

    assert output["payload_status"] == "REJECT"
    assert output["business_payload_emitted"] is False
    assert output["passthrough_mode"] == "STRICT"
    assert output["route_status"] == "REJECTED"
    assert "adapter_payload" not in output


def test_alpha_scenario_3_structured_adapter_mode_filters_rejected_content() -> None:
    verifier = _AlphaValidationVerifier(
        {
            "approved": {
                "entailment": 0.98,
                "contradiction": 0.0,
                "label": "entailment",
                "chosen_evidence_id": "e1",
            },
            "maybe": {
                "entailment": 0.3,
                "contradiction": 0.2,
                "label": "neutral",
                "chosen_evidence_id": None,
            },
        }
    )

    output = run_pipeline(
        llm_summary_text="ignored in structured_field mode",
        structured_candidate_payload={"status": "approved", "review_notes": "maybe"},
        unitizer_mode="structured_field",
        evidence=_evidence(),
        policy_config=_policy(),
        use_oss_nli=True,
        verifier=verifier,
        downstream_compatibility_mode="ADAPTER",
    )

    assert output["payload_status"] == "REJECT"
    assert output["business_payload_emitted"] is True
    assert output["adapter_payload"] == [{"unit_id": "$.status", "text": "approved"}]
    assert all(row["text"] != "maybe" for row in output["adapter_payload"])
    assert all(row["decision"] != "accept" for row in output["adapter_rejected_units"])
    assert any(
        row["text"] == "maybe" and row["failure_class"] == "MISSING_IN_SOURCE"
        for row in output["adapter_rejected_units"]
    )


def test_alpha_scenario_4_only_unsupported_claim_triggers_repair_path() -> None:
    verifier = _AlphaValidationVerifier(
        {
            "Unsupported.": {
                "entailment": 0.1,
                "contradiction": 0.9,
                "label": "contradiction",
                "chosen_evidence_id": "e1",
                "per_item_probs": [{"evidence_id": "e1", "entailment": 0.1, "contradiction": 0.9}],
            },
            "Missing.": {
                "entailment": 0.2,
                "contradiction": 0.1,
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
        }
    )

    unsupported_output = run_pipeline(
        llm_summary_text="Unsupported.",
        evidence=_evidence(),
        policy_config=_policy(),
        use_oss_nli=True,
        verifier=verifier,
        enable_correction=True,
        max_retries=1,
    )
    assert unsupported_output["payload_status"] == "PENDING"
    assert unsupported_output["route_status"] == "REPAIR_PENDING"

    non_unsupported_output = run_pipeline(
        llm_summary_text="Missing. Ambiguous.",
        evidence=_evidence(),
        policy_config=_policy(),
        use_oss_nli=True,
        verifier=verifier,
        enable_correction=True,
        max_retries=1,
    )
    assert non_unsupported_output["payload_status"] == "REJECT"
    assert non_unsupported_output["route_status"] == "REJECTED"


def test_alpha_scenario_5_pending_handoff_paths_are_not_sync_completed() -> None:
    contract = _derive_workflow_contract(
        payload_status="REJECT",
        payload_action="REVIEW",
        candidate=AnswerCandidate(raw_answer_text="x", units=[Unit(id="u0001", text="x", metadata={})]),
        decisions_by_unit={"u0001": "reject"},
    )

    assert contract["workflow_status"] == "PENDING"
    assert contract["handoff_required"] is True
    assert contract["handoff_reason"] == "REVIEW"
    assert isinstance(contract["tracking_id"], str) and contract["tracking_id"]
