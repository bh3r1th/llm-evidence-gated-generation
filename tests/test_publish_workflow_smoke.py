from __future__ import annotations

import json
from pathlib import Path

import ega.v2.eval_harness as harness
import ega.v2.export_calibration_rows as export_mod
import ega.v2.threshold_sweep as sweep_mod
from ega.v2.calibrate import calibrate_jsonl_to_state, save_conformal_state_json
from ega.v2.poc_release import build_final_poc_summary, write_poc_results_markdown


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return path


def test_publish_workflow_smoke(tmp_path: Path, monkeypatch) -> None:
    dataset_path = _write_jsonl(
        tmp_path / "dataset.jsonl",
        [
            {
                "id": "ex1",
                "llm_summary_text": "Alpha. Beta.",
                "evidence_json": [
                    {"id": "e1", "text": "Alpha.", "metadata": {}},
                    {"id": "e2", "text": "Beta.", "metadata": {}},
                ],
                "gold_units": [
                    {"unit_id": "u0001", "text": "Alpha.", "supported": True},
                    {"unit_id": "u0002", "text": "Beta.", "supported": False},
                ],
                "scores_jsonl_path": "dummy_scores.jsonl",
            }
        ],
    )
    threshold_out = tmp_path / "threshold_sweep.json"
    rows_out = tmp_path / "calibration_rows.jsonl"
    conformal_out = tmp_path / "conformal_state.json"
    eval_out = tmp_path / "pilot_eval.json"
    final_summary_out = tmp_path / "final_poc_summary.json"
    report_out = tmp_path / "poc_results.md"

    def fake_export_run_one(**kwargs):  # type: ignore[no-untyped-def]
        threshold = kwargs.get("accept_threshold")
        return (
            {
                "accept_threshold": threshold,
                "verifier_model_name": "fake-nli",
                "verifier_scores": {
                    "u0001": {"entailment": 0.90, "contradiction": 0.01, "chosen_evidence_id": "e1"},
                    "u0002": {"entailment": 0.02, "contradiction": 0.70, "chosen_evidence_id": "e2"},
                },
                "units": [
                    {"unit_id": "u0001", "text": "Alpha."},
                    {"unit_id": "u0002", "text": "Beta."},
                ],
            },
            {},
        )

    class FakeReranker:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

    def fake_eval_run_pipeline(**kwargs):  # type: ignore[no-untyped-def]
        use_rerank = kwargs.get("reranker") is not None
        use_conformal = bool(kwargs.get("conformal_state_path"))
        trace_out = kwargs.get("trace_out")
        if isinstance(trace_out, str):
            payload = {
                "total_seconds": 0.01,
                "n_units": 2,
                "n_pairs": 2 if not use_rerank else 1,
                "rerank_pairs_scored": 0 if not use_rerank else 1,
                "conformal_abstain_units": 1 if use_conformal else 0,
            }
            Path(trace_out).write_text(json.dumps(payload) + "\n", encoding="utf-8")
        kept_units = 2 if not use_conformal else 1
        return {
            "accept_threshold": kwargs.get("accept_threshold"),
            "verifier_model_name": "fake-nli",
            "verified_extract": [
                {"unit_id": "u0001", "text": "Alpha."},
                *([] if use_conformal else [{"unit_id": "u0002", "text": "Beta."}]),
            ],
            "verified_text": "Alpha.",
            "decision": {"refusal": False, "reason_code": "OK_PARTIAL", "summary_stats": {}},
            "stats": {
                "kept_units": kept_units,
                "dropped_units": 2 - kept_units,
                "coverage_avg_score": 1.0,
                "reward_total": 0.2 if use_conformal else -0.1,
                "reward_avg": 0.1 if use_conformal else -0.05,
                "reward_hallucination_rate": 0.0 if use_conformal else 0.5,
                "reward_abstention_rate": 0.5 if use_conformal else 0.0,
                "planned_pairs_total": 2 if not use_rerank else 1,
                "evaluated_pairs_total": 2 if not use_rerank else 1,
                "pruned_pairs_total": 0,
            },
        }

    monkeypatch.setattr(sweep_mod, "_run_one", fake_export_run_one)
    monkeypatch.setattr(export_mod, "_run_one", fake_export_run_one)
    monkeypatch.setattr(harness, "CrossEncoderReranker", FakeReranker)
    monkeypatch.setattr(harness, "run_pipeline", fake_eval_run_pipeline)

    sweep_summary = sweep_mod.run_threshold_sweep(
        dataset_path=dataset_path,
        out_path=threshold_out,
        accept_threshold=0.05,
    )
    export_summary = export_mod.export_calibration_rows(
        dataset_path=dataset_path,
        out_path=rows_out,
        accept_threshold=0.05,
    )
    state, _ = calibrate_jsonl_to_state(
        in_path=rows_out,
        epsilon=0.05,
        min_calib=1,
    )
    save_conformal_state_json(conformal_out, state)
    eval_summary = harness.run_v2_eval(
        dataset_path=dataset_path,
        out_path=eval_out,
        conformal_state_path=str(conformal_out),
        accept_threshold=0.05,
        render_safe_answer=True,
    )
    final_summary = build_final_poc_summary(
        source_summary_path=eval_out,
        out_path=final_summary_out,
        dataset_path=dataset_path,
        conformal_state_path=conformal_out,
    )
    markdown = write_poc_results_markdown(summary_path=final_summary_out, out_path=report_out)

    assert sweep_summary["recommended_threshold"] is not None
    assert export_summary["n_rows"] == 2
    assert eval_summary["variants"]["combined"]["debug"]["reranker_enabled"] is True
    assert final_summary["variants"]["combined"]["status"] == "recommended"
    assert "budget_only" not in final_summary["variants"]
    assert all(
        final_summary["variants"][name]["debug"]["verifier_model_name"] is not None
        for name in ("v1_baseline", "rerank_only", "conformal_only", "combined")
    )
    assert "`v1_baseline`" in markdown
    assert report_out.exists()
