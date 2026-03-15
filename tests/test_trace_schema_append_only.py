from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from uuid import uuid4


def _load_validator_module():
    module_path = Path("scripts") / "validate_trace_jsonl.py"
    spec = importlib.util.spec_from_file_location("validate_trace_jsonl", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _baseline_trace_row() -> dict[str, object]:
    return {
        "total_seconds": 0.2,
        "read_seconds": 0.01,
        "unitize_seconds": 0.01,
        "verify_seconds": 0.15,
        "load_seconds": 0.05,
        "verify_compute_seconds": 0.1,
        "enforce_seconds": 0.02,
        "polish_seconds": 0.0,
        "preselect_seconds": 0.01,
        "tokenize_seconds": 0.02,
        "forward_seconds": 0.03,
        "post_seconds": 0.04,
        "num_batches": 1,
        "batch_size_mean": 2.0,
        "batch_size_max": 2,
        "seq_len_mean": 100.0,
        "seq_len_p50": 90.0,
        "seq_len_p95": 120.0,
        "tokens_total": 200,
        "device": "cpu",
        "dtype": "float32",
        "amp_enabled": False,
        "compiled_enabled": False,
        "pairs_pruned_stage1": 0,
        "pairs_pruned_stage2": 0,
        "dtype_overridden": False,
        "evidence_truncated_frac": 0.1,
        "evidence_chars_mean_before": 300.0,
        "evidence_chars_mean_after": 270.0,
        "n_units": 2,
        "n_evidence": 3,
        "n_pairs": 4,
        "kept_units": 1,
        "dropped_units": 1,
        "refusal": False,
        "model_name": "demo",
    }


def test_trace_schema_append_only_fields_are_optional() -> None:
    schema = json.loads(Path("trace_schema.json").read_text(encoding="utf-8"))
    required = set(schema.get("required", []))
    props = dict(schema.get("properties", {}))

    new_fields = {
        "coverage_pool_topk",
        "coverage_avg_score",
        "coverage_unit_scores",
        "coverage_missing_total",
        "reward_total",
        "reward_avg",
        "reward_avg_support",
        "reward_hallucination_rate",
        "reward_abstention_rate",
        "reward_avg_coverage",
        "reward_unit_totals",
    }
    assert new_fields.issubset(set(props.keys()))
    assert required.isdisjoint(new_fields)


def test_trace_schema_still_accepts_legacy_row_without_new_fields() -> None:
    trace_path = Path("data") / f"trace_schema_append_only_legacy_{uuid4().hex}.jsonl"
    try:
        trace_path.write_text(json.dumps(_baseline_trace_row()) + "\n", encoding="utf-8")
        validator = _load_validator_module()
        errors = validator.validate_trace_jsonl(
            trace_path=trace_path,
            schema_path=Path("trace_schema.json"),
        )
        assert errors == []
    finally:
        trace_path.unlink(missing_ok=True)
