from __future__ import annotations

import json
from pathlib import Path

import ega.v2.eval_harness as harness
from ega.v2.poc_release import build_final_poc_summary, write_poc_results_markdown


def test_poc_release_generates_summary_and_markdown(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source_summary = {
        "variants": {
            "v1_baseline": {
                "metrics": {
                    "kept_units": 7,
                    "dropped_units": 3,
                    "unsupported_claim_rate": 0.0,
                    "hallucination_rate": 0.8,
                    "abstention_rate": 0.0,
                    "gold_coverage_recall": 0.38,
                    "avg_reward": -1.4,
                    "verifier_calls_proxy": 40,
                    "verifier_cost": 40,
                    "reranker_cost": 0,
                    "cost_proxy": 40,
                    "p50_total_seconds": 1.0,
                    "p95_total_seconds": 2.0,
                }
            },
            "rerank_only": {
                "metrics": {
                    "kept_units": 7,
                    "dropped_units": 3,
                    "unsupported_claim_rate": 0.0,
                    "hallucination_rate": 0.8,
                    "abstention_rate": 0.0,
                    "gold_coverage_recall": 0.40,
                    "avg_reward": -1.3,
                    "verifier_calls_proxy": 30,
                    "verifier_cost": 30,
                    "reranker_cost": 12,
                    "cost_proxy": 42,
                    "p50_total_seconds": 1.1,
                    "p95_total_seconds": 2.1,
                }
            },
            "conformal_only": {
                "metrics": {
                    "kept_units": 5,
                    "dropped_units": 5,
                    "unsupported_claim_rate": 0.0,
                    "hallucination_rate": 0.2,
                    "abstention_rate": 0.5,
                    "gold_coverage_recall": 0.38,
                    "avg_reward": -0.4,
                    "verifier_calls_proxy": 40,
                    "verifier_cost": 40,
                    "reranker_cost": 0,
                    "cost_proxy": 40,
                    "p50_total_seconds": 1.2,
                    "p95_total_seconds": 2.2,
                }
            },
            "combined": {
                "metrics": {
                    "kept_units": 5,
                    "dropped_units": 5,
                    "unsupported_claim_rate": 0.0,
                    "hallucination_rate": 0.2,
                    "abstention_rate": 0.5,
                    "gold_coverage_recall": 0.41,
                    "avg_reward": -0.3,
                    "verifier_calls_proxy": 30,
                    "verifier_cost": 30,
                    "reranker_cost": 12,
                    "cost_proxy": 42,
                    "p50_total_seconds": 1.3,
                    "p95_total_seconds": 2.3,
                }
            },
            "budget_only": {
                "metrics": {
                    "kept_units": 6,
                    "dropped_units": 4,
                    "unsupported_claim_rate": 0.0,
                    "hallucination_rate": 0.8,
                    "abstention_rate": 0.0,
                    "gold_coverage_recall": 0.35,
                    "avg_reward": -1.4,
                    "verifier_calls_proxy": 40,
                    "verifier_cost": 40,
                    "reranker_cost": 0,
                    "cost_proxy": 40,
                    "p50_total_seconds": 1.0,
                    "p95_total_seconds": 2.0,
                }
            },
        }
    }
    source_path = tmp_path / "source_eval.json"
    summary_path = tmp_path / "final_poc_summary.json"
    report_path = tmp_path / "poc_results.md"
    source_path.write_text(json.dumps(source_summary, sort_keys=True), encoding="utf-8")

    summary = build_final_poc_summary(
        source_summary_path=source_path,
        out_path=summary_path,
        dataset_path="examples/v2/eval_dataset_tiny.jsonl",
        conformal_state_path="runs/v2_compare/calibration/pilot_conformal_state.json",
    )
    markdown = write_poc_results_markdown(summary_path=summary_path, out_path=report_path)

    assert summary_path.exists()
    assert report_path.exists()
    assert summary["variants"]["v1_baseline"]["config"]["accept_threshold"] == 0.05
    assert "budget_only" not in summary["variants"]
    assert summary["variants"]["combined"]["recommended"] is True
    assert (
        summary["variants"]["v1_baseline"]["debug"]["verifier_model_name"]
        == summary["verifier_model_name"]
    )
    assert "`combined`" in markdown
    assert "budget_only" not in markdown


def test_publish_workflow_smoke_from_eval_summary(tmp_path: Path, monkeypatch) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    conformal_path = tmp_path / "conformal.json"
    eval_out_path = tmp_path / "pilot_eval.json"
    final_summary_path = tmp_path / "final_poc_summary.json"
    report_path = tmp_path / "poc_results.md"
    dataset_path.write_text(
        json.dumps(
            {
                "id": "ex1",
                "llm_summary_text": "A fact.",
                "evidence_json": [{"id": "e1", "text": "A fact.", "metadata": {}}],
                "gold_units": [{"unit_id": "u0001", "text": "A fact.", "supported": True}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    conformal_path.write_text(json.dumps({"threshold": 0.02, "meta": {}}), encoding="utf-8")

    class FakeReranker:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

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
            "verified_extract": [{"unit_id": "u0001", "text": "A fact."}],
            "verified_text": "A fact.",
            "decision": {"refusal": False, "reason_code": "OK_PARTIAL", "summary_stats": {}},
            "stats": {
                "kept_units": 1,
                "dropped_units": 0,
                "reward_avg": 0.1,
                "reward_total": 0.1,
                "reward_hallucination_rate": 0.0,
                "reward_abstention_rate": 0.0,
                "coverage_avg_score": 1.0,
            },
        }

    monkeypatch.setattr(harness, "CrossEncoderReranker", FakeReranker)
    monkeypatch.setattr(harness, "run_pipeline", fake_run_pipeline)

    harness.run_v2_eval(
        dataset_path=dataset_path,
        out_path=eval_out_path,
        conformal_state_path=str(conformal_path),
        accept_threshold=0.05,
    )
    summary = build_final_poc_summary(
        source_summary_path=eval_out_path,
        out_path=final_summary_path,
        dataset_path=dataset_path,
        conformal_state_path=conformal_path,
    )
    markdown = write_poc_results_markdown(summary_path=final_summary_path, out_path=report_path)

    assert summary["accept_threshold"] == 0.05
    assert summary["variants"]["conformal_only"]["config"]["conformal_state_path"] == str(conformal_path)
    assert final_summary_path.exists()
    assert report_path.exists()
    assert "# EGA v2 POC Results" in markdown
    assert summary["variants"]["v1_baseline"]["status"] == "recommended"
    assert summary["variants"]["v1_baseline"]["kept_units"] == 1
    assert (
        summary["variants"]["v1_baseline"]["debug"]["verifier_model_name"]
        == summary["verifier_model_name"]
    )


def test_poc_release_can_include_experimental_budget_variant(tmp_path: Path) -> None:
    source_summary = {
        "variants": {
            "v1_baseline": {"status": "ok", "metrics": {}},
            "rerank_only": {"status": "ok", "metrics": {}},
            "conformal_only": {"status": "ok", "metrics": {}},
            "combined": {"status": "ok", "metrics": {}},
            "budget_only": {"status": "ok", "metrics": {"verifier_calls_proxy": 3}},
        }
    }
    source_path = tmp_path / "source_eval.json"
    dataset_path = tmp_path / "dataset.jsonl"
    conformal_path = tmp_path / "conformal.json"
    out_path = tmp_path / "final_poc_summary.json"
    source_path.write_text(json.dumps(source_summary, sort_keys=True), encoding="utf-8")
    dataset_path.write_text(
        json.dumps({"id": "ex1", "llm_summary_text": "One.", "evidence_json": []}) + "\n",
        encoding="utf-8",
    )
    conformal_path.write_text(json.dumps({"threshold": 0.1, "meta": {}}), encoding="utf-8")

    summary = build_final_poc_summary(
        source_summary_path=source_path,
        out_path=out_path,
        dataset_path=dataset_path,
        conformal_state_path=conformal_path,
        include_experimental=True,
    )

    assert summary["experimental_variants"] == ["budget_only"]
    assert summary["variants"]["budget_only"]["status"] == "experimental_not_recommended"
