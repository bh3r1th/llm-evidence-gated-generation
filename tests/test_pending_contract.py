from __future__ import annotations

import uuid

from ega.api import verify_answer
from ega.config import PipelineConfig, VerifierConfig
from ega.contract import PolicyConfig
from ega.types import VerificationScore


class _PendingVerifier:
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
                    raw={"chosen_evidence_id": cfg.get("chosen_evidence_id"), "per_item_probs": []},
                )
            )
        return rows

    @staticmethod
    def get_last_verify_trace() -> dict[str, float | int]:
        return {"n_pairs_scored": 1, "forward_seconds": 0.0}


def _policy() -> PolicyConfig:
    return PolicyConfig(threshold_entailment=0.5, max_contradiction=0.2, partial_allowed=True)


def _run(
    *,
    llm_output: str,
    verifier: _PendingVerifier,
    tracking_id: str | None,
    output_mode: str = "strict",
    enable_correction: bool = False,
    max_retries: int = 1,
    pending_expires_at: str | None = None,
    extras: dict[str, object] | None = None,
) -> dict[str, object]:
    config = PipelineConfig(
        policy=_policy(),
        verifier=VerifierConfig(verifier=verifier),
        output_mode=output_mode,
        tracking_id=tracking_id,
        enable_correction=enable_correction,
        max_retries=max_retries,
        pending_expires_at=pending_expires_at,
        extras={} if extras is None else extras,
    )
    return verify_answer(
        llm_output=llm_output,
        source_text="supporting evidence",
        config=config,
        return_pipeline_output=True,
    )


def test_tracking_id_from_config_is_preserved_in_output_and_trace() -> None:
    output = _run(
        llm_output="Supported.",
        verifier=_PendingVerifier(
            {"Supported.": {"entailment": 0.9, "contradiction": 0.0, "label": "entailment", "chosen_evidence_id": "source"}}
        ),
        tracking_id="track-123",
    )

    assert output["tracking_id"] == "track-123"
    assert output["trace"]["tracking_id"] == "track-123"


def test_tracking_id_none_generates_uuid4_and_is_in_output_and_trace() -> None:
    output = _run(
        llm_output="Supported.",
        verifier=_PendingVerifier(
            {"Supported.": {"entailment": 0.9, "contradiction": 0.0, "label": "entailment", "chosen_evidence_id": "source"}}
        ),
        tracking_id=None,
    )

    generated = output["tracking_id"]
    assert isinstance(generated, str) and generated
    assert str(uuid.UUID(generated)) == generated
    assert output["trace"]["tracking_id"] == generated


def test_correction_in_progress_emits_pending_with_bounded_repair_route_reason() -> None:
    output = _run(
        llm_output="Unsupported.",
        verifier=_PendingVerifier(
            {"Unsupported.": {"entailment": 0.1, "contradiction": 0.9, "label": "contradiction", "chosen_evidence_id": "source"}}
        ),
        tracking_id="track-pending",
        enable_correction=True,
        max_retries=1,
        extras={"correction_generator": lambda _failed, _evidence, _retry: {}},
    )

    assert output["payload_status"] == "PENDING"
    assert output["route_reason"] == "BOUNDED_REPAIR"


def test_pending_expires_at_is_passthrough_string_on_pending_response() -> None:
    pending_ts = "2026-05-01T00:00:00Z"
    output = _run(
        llm_output="Unsupported.",
        verifier=_PendingVerifier(
            {"Unsupported.": {"entailment": 0.1, "contradiction": 0.9, "label": "contradiction", "chosen_evidence_id": "source"}}
        ),
        tracking_id="track-expiry",
        enable_correction=True,
        max_retries=1,
        pending_expires_at=pending_ts,
        extras={"correction_generator": lambda _failed, _evidence, _retry: {}},
    )

    assert output["payload_status"] == "PENDING"
    assert output["pending_expires_at"] == pending_ts
    assert output["v4_response"]["pending_expires_at"] == pending_ts


def test_pending_expires_at_none_emits_null_without_error() -> None:
    output = _run(
        llm_output="Unsupported.",
        verifier=_PendingVerifier(
            {"Unsupported.": {"entailment": 0.1, "contradiction": 0.9, "label": "contradiction", "chosen_evidence_id": "source"}}
        ),
        tracking_id="track-null-expiry",
        enable_correction=True,
        max_retries=1,
        pending_expires_at=None,
        extras={"correction_generator": lambda _failed, _evidence, _retry: {}},
    )

    assert output["payload_status"] == "PENDING"
    assert output["pending_expires_at"] is None
    assert output["v4_response"]["pending_expires_at"] is None


def test_tracking_id_present_across_strict_accept_strict_reject_adapter_and_pending() -> None:
    strict_accept = _run(
        llm_output="Supported.",
        verifier=_PendingVerifier(
            {"Supported.": {"entailment": 0.9, "contradiction": 0.0, "label": "entailment", "chosen_evidence_id": "source"}}
        ),
        tracking_id="tid-accept",
        output_mode="strict",
    )
    strict_reject = _run(
        llm_output="Missing.",
        verifier=_PendingVerifier(
            {"Missing.": {"entailment": 0.1, "contradiction": 0.1, "label": "neutral", "chosen_evidence_id": None}}
        ),
        tracking_id="tid-reject",
        output_mode="strict",
    )
    adapter_envelope = _run(
        llm_output="Supported. Missing.",
        verifier=_PendingVerifier(
            {
                "Supported.": {"entailment": 0.9, "contradiction": 0.0, "label": "entailment", "chosen_evidence_id": "source"},
                "Missing.": {"entailment": 0.1, "contradiction": 0.1, "label": "neutral", "chosen_evidence_id": None},
            }
        ),
        tracking_id="tid-adapter",
        output_mode="adapter",
    )
    pending = _run(
        llm_output="Unsupported.",
        verifier=_PendingVerifier(
            {"Unsupported.": {"entailment": 0.1, "contradiction": 0.9, "label": "contradiction", "chosen_evidence_id": "source"}}
        ),
        tracking_id="tid-pending",
        output_mode="strict",
        enable_correction=True,
        max_retries=1,
        extras={"correction_generator": lambda _failed, _evidence, _retry: {}},
    )

    assert strict_accept["v4_response"]["tracking_id"] == "tid-accept"
    assert strict_reject["v4_response"]["tracking_id"] == "tid-reject"
    assert adapter_envelope["v4_response"]["tracking_id"] == "tid-adapter"
    assert pending["v4_response"]["tracking_id"] == "tid-pending"
