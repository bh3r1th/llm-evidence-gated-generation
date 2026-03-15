from __future__ import annotations

import json
from pathlib import Path

import ega.v2.eval_harness as harness


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return path


def test_eval_harness_reports_coverage_and_reward_metrics_per_variant(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_path = _write_jsonl(
        tmp_path / "dataset.jsonl",
        [
            {
                "llm_summary_text": "A fact. B claim.",
                "evidence_json": [{"id": "e1", "text": "A fact.", "metadata": {}}],
                "scores_jsonl_path": "dummy_scores.jsonl",
            }
        ],
    )
    out_path = tmp_path / "summary.json"
    conformal_path = tmp_path / "conformal.json"
    conformal_path.write_text(json.dumps({"threshold": 0.5, "meta": {}}), encoding="utf-8")

    class FakeReranker:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

    calls: list[dict[str, object]] = []

    def fake_run_pipeline(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(dict(kwargs))
        use_budget = kwargs.get("budget_policy") is not None
        use_rerank = kwargs.get("reranker") is not None
        use_conformal = bool(kwargs.get("conformal_state_path"))

        if use_budget and use_rerank and use_conformal:
            coverage_avg = 0.9
            reward_total = 2.0
            reward_avg = 1.0
            reward_hall_rate = 0.0
            reward_abs_rate = 0.1
            n_pairs = 4
            rerank_pairs = 3
            abstain_units = 1
        elif use_budget:
            coverage_avg = 0.7
            reward_total = 1.2
            reward_avg = 0.6
            reward_hall_rate = 0.1
            reward_abs_rate = 0.0
            n_pairs = 6
            rerank_pairs = 0
            abstain_units = 0
        elif use_rerank:
            coverage_avg = 0.8
            reward_total = 1.6
            reward_avg = 0.8
            reward_hall_rate = 0.05
            reward_abs_rate = 0.0
            n_pairs = 8
            rerank_pairs = 3
            abstain_units = 0
        elif use_conformal:
            coverage_avg = 0.6
            reward_total = 0.8
            reward_avg = 0.4
            reward_hall_rate = 0.0
            reward_abs_rate = 0.2
            n_pairs = 10
            rerank_pairs = 0
            abstain_units = 1
        else:
            coverage_avg = 0.5
            reward_total = 0.4
            reward_avg = 0.2
            reward_hall_rate = 0.2
            reward_abs_rate = 0.0
            n_pairs = 10
            rerank_pairs = 0
            abstain_units = 0

        trace_out = kwargs.get("trace_out")
        if isinstance(trace_out, str):
            trace = {
                "total_seconds": 0.1,
                "n_units": 2,
                "n_pairs": n_pairs,
                "rerank_pairs_scored": rerank_pairs,
                "conformal_abstain_units": abstain_units,
            }
            Path(trace_out).write_text(json.dumps(trace) + "\n", encoding="utf-8")

        return {
            "verified_extract": [{"unit_id": "u0001", "text": "u0001"}],
            "verified_text": "u0001",
            "decision": {"refusal": False, "reason_code": "OK_PARTIAL", "summary_stats": {}},
            "stats": {
                "kept_units": 1,
                "dropped_units": 1,
                "coverage_avg_score": coverage_avg,
                "reward_total": reward_total,
                "reward_avg": reward_avg,
                "reward_hallucination_rate": reward_hall_rate,
                "reward_abstention_rate": reward_abs_rate,
            },
        }

    monkeypatch.setattr(harness, "CrossEncoderReranker", FakeReranker)
    monkeypatch.setattr(harness, "run_pipeline", fake_run_pipeline)

    summary = harness.run_v2_eval(
        dataset_path=dataset_path,
        out_path=out_path,
        conformal_state_path=str(conformal_path),
    )

    assert out_path.exists()
    assert len(calls) == 5
    assert all(call.get("coverage_config") is not None for call in calls)
    assert all(call.get("reward_config") is not None for call in calls)

    for variant in summary["variants"].values():
        metrics = variant["metrics"]
        assert "avg_coverage_score" in metrics
        assert "reward_total" in metrics
        assert "avg_reward" in metrics
        assert "hallucination_rate" in metrics
        assert "abstention_rate" in metrics
