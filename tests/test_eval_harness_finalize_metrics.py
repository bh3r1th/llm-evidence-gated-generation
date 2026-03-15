from __future__ import annotations

import json
from pathlib import Path

import ega.v2.eval_harness as harness


def test_run_v2_eval_writes_json_when_finalize_metrics_is_reused_for_skipped_variants(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "llm_summary_text": "A fact.",
                "evidence_json": [{"id": "e1", "text": "A fact.", "metadata": {}}],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    out_path = tmp_path / "summary.json"

    class MissingReranker:
        def __init__(self, model_name: str) -> None:
            raise ImportError("reranker unavailable")

    def fake_run_pipeline(**kwargs):  # type: ignore[no-untyped-def]
        trace_out = kwargs.get("trace_out")
        if isinstance(trace_out, str):
            Path(trace_out).write_text(
                json.dumps(
                    {
                        "total_seconds": 0.01,
                        "n_units": 1,
                        "n_pairs": 1,
                        "rerank_pairs_scored": 0,
                        "conformal_abstain_units": 0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        return {
            "verified_extract": [{"unit_id": "u0001", "text": "u0001"}],
            "verified_text": "u0001",
            "decision": {"refusal": False, "reason_code": "OK_PARTIAL", "summary_stats": {}},
            "stats": {"kept_units": 1, "dropped_units": 0},
        }

    monkeypatch.setattr(harness, "CrossEncoderReranker", MissingReranker)
    monkeypatch.setattr(harness, "run_pipeline", fake_run_pipeline)

    summary = harness.run_v2_eval(dataset_path=dataset_path, out_path=out_path)

    assert out_path.exists()
    assert json.loads(out_path.read_text(encoding="utf-8")) == summary
    assert summary["variants"]["v1_baseline"]["metrics"]["n_examples"] == 1
    assert summary["variants"]["budget_only"]["metrics"]["n_examples"] == 1
    assert summary["variants"]["conformal_only"]["metrics"]["n_examples"] == 0
    assert summary["variants"]["rerank_only"]["metrics"]["n_examples"] == 0
    assert summary["variants"]["combined"]["metrics"]["n_examples"] == 0
