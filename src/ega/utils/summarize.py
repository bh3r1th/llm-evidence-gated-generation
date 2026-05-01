from __future__ import annotations


def summarize_result(result: dict) -> dict:
    audit = result.get("audit", []) or []

    authority_distribution: dict[str, int] = {}
    abstain_count = 0
    for entry in audit:
        authority = entry.get("authority")
        if authority is not None:
            authority_distribution[str(authority)] = (
                authority_distribution.get(str(authority), 0) + 1
            )
        if entry.get("final_decision") == "abstain":
            abstain_count += 1

    drift_info = result.get("distribution_drift")
    drift_flagged = drift_info.get("drift_flagged") if isinstance(drift_info, dict) else None

    trace = result.get("trace", {}) or {}
    fallback_per_unit = trace.get("field_query_fallback_per_unit", {}) or {}
    field_query_fallback_count = sum(1 for v in fallback_per_unit.values() if v is True)

    return {
        "tracking_id": result.get("tracking_id"),
        "payload_status": result.get("payload_status"),
        "authority_distribution": authority_distribution,
        "abstain_count": abstain_count,
        "drift_flagged": drift_flagged,
        "field_query_fallback_count": field_query_fallback_count,
    }
