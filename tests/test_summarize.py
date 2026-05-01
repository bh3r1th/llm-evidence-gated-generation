from __future__ import annotations


def test_authority_distribution_and_abstain_count() -> None:
    result = {
        "tracking_id": "abc",
        "payload_status": "accepted",
        "audit": [
            {"authority": "high", "final_decision": "accept"},
            {"authority": "high", "final_decision": "abstain"},
            {"authority": "low", "final_decision": "accept"},
        ],
    }
    from ega.utils.summarize import summarize_result

    summary = summarize_result(result)

    assert summary["authority_distribution"] == {"high": 2, "low": 1}
    assert summary["abstain_count"] == 1


def test_missing_audit_drift_trace_returns_safe_defaults() -> None:
    from ega.utils.summarize import summarize_result

    summary = summarize_result({"tracking_id": "x", "payload_status": "rejected"})

    assert "authority_distribution" in summary
    assert summary["authority_distribution"] == {}
    assert summary["abstain_count"] == 0
    assert summary["drift_flagged"] is None
    assert summary["field_query_fallback_count"] == 0


def test_summarize_result_importable_from_ega_top_level() -> None:
    from ega import summarize_result  # noqa: F401

    assert callable(summarize_result)
