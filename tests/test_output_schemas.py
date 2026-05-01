from __future__ import annotations

from ega.contract import PolicyConfig
from ega.pipeline import run_pipeline
from ega.types import EvidenceItem, EvidenceSet, VerificationScore


class _SchemaVerifier:
    model_name = "fake-nli"

    def __init__(self, score_by_text: dict[str, dict[str, object]]) -> None:
        self.score_by_text = score_by_text

    def verify_many(self, candidate, evidence):  # type: ignore[no-untyped-def]
        del evidence
        rows: list[VerificationScore] = []
        for unit in candidate.units:
            cfg = self.score_by_text[unit.text]
            rows.append(
                VerificationScore(
                    unit_id=unit.id,
                    entailment=float(cfg["entailment"]),
                    contradiction=float(cfg["contradiction"]),
                    neutral=float(cfg.get("neutral", 0.0)),
                    label=str(cfg.get("label", "neutral")),
                    raw={
                        "chosen_evidence_id": cfg.get("chosen_evidence_id"),
                        "per_item_probs": cfg.get("per_item_probs", []),
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


def test_strict_mode_all_units_accepted_emits_strict_accepted_shape() -> None:
    verifier = _SchemaVerifier(
        {
            "Supported one.": {"entailment": 0.9, "contradiction": 0.0, "label": "entailment", "chosen_evidence_id": "e1"},
            "Supported two.": {"entailment": 0.8, "contradiction": 0.0, "label": "entailment", "chosen_evidence_id": "e1"},
        }
    )
    output = run_pipeline(
        llm_summary_text="Supported one. Supported two.",
        evidence=_evidence(),
        policy_config=_policy(),
        use_oss_nli=True,
        verifier=verifier,
        output_mode="strict",
    )

    assert output["payload_status"] == "ACCEPT"
    assert output["v4_response"]["payload_status"] == "ACCEPT"
    assert "audit" in output["v4_response"]
    assert "verified_text" in output["v4_response"]
    assert "verified_units" in output["v4_response"]


def test_strict_mode_reject_emits_failed_unit_ids_and_no_verified_text_in_v4_shape() -> None:
    verifier = _SchemaVerifier(
        {
            "Supported.": {"entailment": 0.9, "contradiction": 0.0, "label": "entailment", "chosen_evidence_id": "e1"},
            "Missing.": {"entailment": 0.1, "contradiction": 0.1, "label": "neutral", "chosen_evidence_id": None},
        }
    )
    output = run_pipeline(
        llm_summary_text="Supported. Missing.",
        evidence=_evidence(),
        policy_config=_policy(),
        use_oss_nli=True,
        verifier=verifier,
        output_mode="strict",
    )

    assert output["payload_status"] == "REJECT"
    assert output["v4_response"]["payload_status"] == "REJECT"
    assert output["v4_response"]["failed_unit_ids"]
    assert "verified_text" not in output["v4_response"]


def test_adapter_mode_mixed_units_emits_adapter_envelope_and_full_field_status_map() -> None:
    verifier = _SchemaVerifier(
        {
            "Supported.": {"entailment": 0.9, "contradiction": 0.0, "label": "entailment", "chosen_evidence_id": "e1"},
            "Missing.": {"entailment": 0.1, "contradiction": 0.1, "label": "neutral", "chosen_evidence_id": None},
        }
    )
    output = run_pipeline(
        llm_summary_text="Supported. Missing.",
        evidence=_evidence(),
        policy_config=_policy(),
        use_oss_nli=True,
        verifier=verifier,
        output_mode="adapter",
    )

    assert output["payload_status"] == "REJECT"
    assert output["v4_response"]["accepted_fields"]
    assert output["v4_response"]["rejected_fields"]
    all_ids = {row["unit_id"] for row in output["units"]}
    assert set(output["v4_response"]["field_status_map"].keys()) == all_ids


def test_field_status_map_uses_authority_key() -> None:
    verifier = _SchemaVerifier(
        {
            "Supported.": {"entailment": 0.9, "contradiction": 0.0, "label": "entailment", "chosen_evidence_id": "e1"},
        }
    )
    output = run_pipeline(
        llm_summary_text="Supported.",
        evidence=_evidence(),
        policy_config=_policy(),
        use_oss_nli=True,
        verifier=verifier,
        output_mode="adapter",
    )

    for row in output["v4_response"]["field_status_map"].values():
        assert "authority" in row
        assert "decision_authority" not in row


def test_repair_payload_status_is_pending_with_bounded_repair_route_reason() -> None:
    verifier = _SchemaVerifier(
        {
            "Unsupported.": {"entailment": 0.1, "contradiction": 0.9, "label": "contradiction", "chosen_evidence_id": "e1"},
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
        output_mode="strict",
    )

    assert output["payload_status"] == "PENDING"
    assert output["v4_response"]["payload_status"] == "PENDING"
    assert output["v4_response"]["route_reason"] == "BOUNDED_REPAIR"


def test_v3_keys_remain_present_in_strict_mode_alongside_v4_fields() -> None:
    verifier = _SchemaVerifier(
        {
            "Supported.": {"entailment": 0.9, "contradiction": 0.0, "label": "entailment", "chosen_evidence_id": "e1"},
        }
    )
    output = run_pipeline(
        llm_summary_text="Supported.",
        evidence=_evidence(),
        policy_config=_policy(),
        use_oss_nli=True,
        verifier=verifier,
        output_mode="strict",
    )

    assert "verified_text" in output
    assert "verified_extract" in output
    assert "decision" in output and "dropped_units" in output["decision"]
    assert "trace" in output
    assert "audit" in output
