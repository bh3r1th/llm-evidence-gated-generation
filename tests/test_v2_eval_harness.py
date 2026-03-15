from __future__ import annotations

import json
from pathlib import Path

import ega.cli as cli
import ega.v2.eval_harness as harness


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return path


def test_run_v2_eval_aggregates_variant_metrics_without_hf(tmp_path: Path, monkeypatch) -> None:
    dataset_path = _write_jsonl(
        tmp_path / "dataset.jsonl",
        [
            {
                "llm_summary_text": "A fact. B claim.",
                "evidence_json": [{"id": "e1", "text": "A fact.", "metadata": {}}],
                "gold_unit_labels": {"u0001": True, "u0002": False},
                "scores_jsonl_path": "dummy_scores.jsonl",
            },
            {
                "llm_summary_text": "A fact. B claim.",
                "evidence_json": [{"id": "e1", "text": "A fact.", "metadata": {}}],
                "gold_unit_labels": {"u0001": True, "u0002": False},
                "scores_jsonl_path": "dummy_scores.jsonl",
            },
        ],
    )
    out_path = tmp_path / "summary.json"
    conformal_path = tmp_path / "conformal.json"
    conformal_path.write_text(json.dumps({"threshold": 0.5, "meta": {}}), encoding="utf-8")

    class FakeReranker:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

    def fake_run_pipeline(**kwargs):  # type: ignore[no-untyped-def]
        use_budget = kwargs.get("budget_policy") is not None
        use_rerank = kwargs.get("reranker") is not None
        use_conformal = bool(kwargs.get("conformal_state_path"))
        if use_rerank and use_conformal:
            kept_ids = ["u0001"]
            n_pairs = 5
            rerank_pairs = 4
            abstain = 1
            seconds = 0.08
        elif use_budget:
            kept_ids = ["u0001", "u0002"]
            n_pairs = 6
            rerank_pairs = 0
            abstain = 0
            seconds = 0.06
        elif use_rerank:
            kept_ids = ["u0001", "u0002"]
            n_pairs = 9
            rerank_pairs = 4
            abstain = 0
            seconds = 0.09
        elif use_conformal:
            kept_ids = ["u0001"]
            n_pairs = 10
            rerank_pairs = 0
            abstain = 1
            seconds = 0.1
        else:
            kept_ids = ["u0001", "u0002"]
            n_pairs = 10
            rerank_pairs = 0
            abstain = 0
            seconds = 0.1

        trace_out = kwargs.get("trace_out")
        if isinstance(trace_out, str):
            payload = {
                "total_seconds": seconds,
                "n_units": 2,
                "n_pairs": n_pairs,
                "rerank_pairs_scored": rerank_pairs,
                "conformal_abstain_units": abstain,
            }
            Path(trace_out).write_text(json.dumps(payload) + "\n", encoding="utf-8")

        return {
            "verified_extract": [{"unit_id": unit_id, "text": unit_id} for unit_id in kept_ids],
            "verified_text": "\n".join(kept_ids),
            "decision": {"refusal": False, "reason_code": "OK_PARTIAL", "summary_stats": {}},
            "stats": {"kept_units": len(kept_ids), "dropped_units": 2 - len(kept_ids)},
        }

    monkeypatch.setattr(harness, "CrossEncoderReranker", FakeReranker)
    monkeypatch.setattr(harness, "run_pipeline", fake_run_pipeline)

    summary = harness.run_v2_eval(
        dataset_path=dataset_path,
        out_path=out_path,
        conformal_state_path=str(conformal_path),
    )

    assert out_path.exists()
    assert set(summary["variants"].keys()) == {
        "v1_baseline",
        "budget_only",
        "conformal_only",
        "rerank_only",
        "combined",
    }
    baseline = summary["variants"]["v1_baseline"]["metrics"]
    conformal = summary["variants"]["conformal_only"]["metrics"]
    combined = summary["variants"]["combined"]["metrics"]

    assert baseline["unsupported_claim_rate"] == 0.5
    assert conformal["unsupported_claim_rate"] == 0.0
    assert combined["abstention_rate"] > 0.0
    assert combined["verifier_calls_proxy"] < baseline["verifier_calls_proxy"]
    assert combined["verifier_cost"] == 10
    assert combined["reranker_cost"] == 8
    assert combined["cost_proxy"] == 18
    assert summary["variants"]["combined"]["metrics"]["budget_active"] is False
    assert summary["variants"]["v1_baseline"]["debug"]["reranker_enabled"] is False
    assert summary["variants"]["rerank_only"]["debug"]["reranker_enabled"] is True
    assert summary["variants"]["rerank_only"]["debug"]["reranker_model_name"] == "cross-encoder/ms-marco-MiniLM-L-6-v2"


def test_run_v2_eval_baseline_does_not_load_reranker_model(tmp_path: Path, monkeypatch) -> None:
    dataset_path = _write_jsonl(
        tmp_path / "dataset.jsonl",
        [
            {
                "llm_summary_text": "A fact.",
                "evidence_json": [{"id": "e1", "text": "A fact.", "metadata": {}}],
                "scores_jsonl_path": "dummy_scores.jsonl",
            }
        ],
    )
    out_path = tmp_path / "summary.json"
    events: list[str] = []

    class FakeReranker:
        def __init__(self, model_name: str) -> None:
            events.append(f"reranker:{model_name}")
            self.model_name = model_name

    def fake_run_pipeline(**kwargs):  # type: ignore[no-untyped-def]
        if kwargs.get("reranker") is None:
            events.append("baseline_call_without_reranker")
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
            "decision": {"refusal": False, "reason_code": "OK_FULL", "summary_stats": {}},
            "stats": {"kept_units": 1, "dropped_units": 0},
        }

    monkeypatch.setattr(harness, "CrossEncoderReranker", FakeReranker)
    monkeypatch.setattr(harness, "run_pipeline", fake_run_pipeline)

    summary = harness.run_v2_eval(dataset_path=dataset_path, out_path=out_path)

    assert events[0] == "baseline_call_without_reranker"
    assert summary["variants"]["v1_baseline"]["debug"]["reranker_enabled"] is False


def test_run_v2_eval_rerank_only_loads_both_models_correctly(tmp_path: Path, monkeypatch) -> None:
    dataset_path = _write_jsonl(
        tmp_path / "dataset.jsonl",
        [
            {
                "llm_summary_text": "A fact.",
                "evidence_json": [{"id": "e1", "text": "A fact.", "metadata": {}}],
                "scores_jsonl_path": "dummy_scores.jsonl",
            }
        ],
    )
    out_path = tmp_path / "summary.json"
    seen_calls: dict[str, dict[str, object]] = {}

    class FakeReranker:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

    def fake_run_pipeline(**kwargs):  # type: ignore[no-untyped-def]
        key = "rerank_only" if kwargs.get("reranker") is not None and kwargs.get("budget_policy") is None else "other"
        seen_calls[key] = dict(kwargs)
        trace_out = kwargs.get("trace_out")
        if isinstance(trace_out, str):
            Path(trace_out).write_text(
                json.dumps(
                    {
                        "total_seconds": 0.01,
                        "n_units": 1,
                        "n_pairs": 1,
                        "rerank_pairs_scored": 1,
                        "conformal_abstain_units": 0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        return {
            "verified_extract": [{"unit_id": "u0001", "text": "A fact."}],
            "verified_text": "A fact.",
            "decision": {"refusal": False, "reason_code": "OK_FULL", "summary_stats": {}},
            "stats": {"kept_units": 1, "dropped_units": 0},
        }

    monkeypatch.setattr(harness, "CrossEncoderReranker", FakeReranker)
    monkeypatch.setattr(harness, "run_pipeline", fake_run_pipeline)

    summary = harness.run_v2_eval(
        dataset_path=dataset_path,
        out_path=out_path,
        nli_model_name="nli/test-model",
        reranker_model="cross-encoder/test-reranker",
    )

    rerank_call = seen_calls["rerank_only"]
    assert rerank_call["nli_model_name"] == "nli/test-model"
    assert getattr(rerank_call["reranker"], "model_name") == "cross-encoder/test-reranker"
    debug = summary["variants"]["rerank_only"]["debug"]
    assert debug["verifier_model_name"] == "nli/test-model"
    assert debug["reranker_model_name"] == "cross-encoder/test-reranker"


def test_run_v2_eval_uses_gold_units_for_unsupported_rate(tmp_path: Path, monkeypatch) -> None:
    dataset_path = _write_jsonl(
        tmp_path / "dataset.jsonl",
        [
            {
                "id": "tiny-gold-units",
                "prompt": "Summarize the note.",
                "llm_summary_text": "Claim one. Claim two.",
                "evidence_json": [
                    {"id": "e1", "text": "Claim one is supported.", "metadata": {}},
                    {"id": "e2", "text": "Distractor.", "metadata": {}},
                ],
                "gold_units": [
                    {
                        "unit_id": "u0001",
                        "text": "Claim one.",
                        "supported": True,
                        "required_evidence_ids": ["e1"],
                        "relevant_evidence_ids": ["e1"],
                    },
                    {
                        "unit_id": "u0002",
                        "text": "Claim two.",
                        "supported": False,
                        "required_evidence_ids": [],
                        "relevant_evidence_ids": ["e2"],
                    },
                ],
            }
        ],
    )
    out_path = tmp_path / "summary.json"

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
                        "n_units": 2,
                        "n_pairs": 2,
                        "rerank_pairs_scored": 0,
                        "conformal_abstain_units": 0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        return {
            "verified_extract": [
                {"unit_id": "u0001", "text": "Claim one."},
                {"unit_id": "u0002", "text": "Claim two."},
            ],
            "verified_text": "Claim one.\nClaim two.",
            "used_evidence": {"u0001": ["e1"], "u0002": ["e2"]},
            "decision": {"refusal": False, "reason_code": "OK_PARTIAL", "summary_stats": {}},
            "stats": {"kept_units": 2, "dropped_units": 0, "coverage_avg_score": 0.5},
        }

    monkeypatch.setattr(harness, "CrossEncoderReranker", FakeReranker)
    monkeypatch.setattr(harness, "run_pipeline", fake_run_pipeline)

    summary = harness.run_v2_eval(dataset_path=dataset_path, out_path=out_path)

    baseline = summary["variants"]["v1_baseline"]["metrics"]
    assert baseline["unsupported_claim_rate"] == 0.5
    assert baseline["gold_coverage_recall"] == 1.0


def test_run_v2_eval_no_kept_claims_returns_null_unsupported_rate_and_non_null_gold_recall(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_path = _write_jsonl(
        tmp_path / "dataset.jsonl",
        [
            {
                "id": "gold-with-no-keeps",
                "llm_summary_text": "Supported claim. Unsupported claim.",
                "evidence_json": [{"id": "e1", "text": "Supported claim.", "metadata": {}}],
                "gold_units": [
                    {
                        "unit_id": "u0001",
                        "text": "Supported claim.",
                        "supported": True,
                        "required_evidence_ids": ["e1"],
                        "relevant_evidence_ids": ["e1"],
                    },
                    {
                        "unit_id": "u0002",
                        "text": "Unsupported claim.",
                        "supported": False,
                        "required_evidence_ids": [],
                        "relevant_evidence_ids": ["e1"],
                    },
                ],
            }
        ],
    )
    out_path = tmp_path / "summary.json"

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
                        "n_units": 2,
                        "n_pairs": 2,
                        "rerank_pairs_scored": 0,
                        "conformal_abstain_units": 0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        return {
            "verified_extract": [],
            "verified_text": "",
            "used_evidence": {"u0001": ["e1"], "u0002": []},
            "decision": {"refusal": True, "reason_code": "ALL_DROPPED", "summary_stats": {}},
            "stats": {"kept_units": 0, "dropped_units": 2, "coverage_avg_score": 0.5},
        }

    monkeypatch.setattr(harness, "CrossEncoderReranker", FakeReranker)
    monkeypatch.setattr(harness, "run_pipeline", fake_run_pipeline)

    summary = harness.run_v2_eval(dataset_path=dataset_path, out_path=out_path)

    baseline = summary["variants"]["v1_baseline"]["metrics"]
    assert baseline["unsupported_claim_rate"] is None
    assert baseline["gold_coverage_recall"] == 1.0


def test_run_v2_eval_emits_debug_dump_and_conformal_metadata(tmp_path: Path, monkeypatch) -> None:
    dataset_path = _write_jsonl(
        tmp_path / "dataset.jsonl",
        [
            {
                "id": "debug-row",
                "llm_summary_text": "Supported claim.",
                "evidence_json": [{"id": "e1", "text": "Supported claim.", "metadata": {}}],
            }
        ],
    )
    out_path = tmp_path / "summary.json"
    debug_path = tmp_path / "pilot_debug_examples.jsonl"
    conformal_path = tmp_path / "conformal.json"
    conformal_path.write_text(
        json.dumps({"threshold": 0.6, "meta": {"abstain_margin": 0.05}}),
        encoding="utf-8",
    )

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
                        "n_pairs": 2,
                        "rerank_pairs_scored": 0,
                        "conformal_abstain_units": 1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        return {
            "accept_threshold": 0.05,
            "units": [{"unit_id": "u0001", "text": "Supported claim."}],
            "verifier_model_name": "nli/test-model",
            "pool_candidates": {"u0001": ["e1"]},
            "pre_rerank_candidates": {"u0001": [{"evidence_id": "e1", "score": None}]},
            "post_rerank_candidates": None,
            "verification_pairs": {
                "u0001": [
                    {
                        "evidence_id": "e1",
                        "entailment": 0.61,
                        "contradiction": 0.0,
                        "neutral": 0.39,
                    }
                ]
            },
            "used_evidence": {"u0001": ["e1"]},
            "verifier_scores": {
                "u0001": {
                    "entailment": 0.61,
                    "contradiction": 0.0,
                    "neutral": 0.39,
                    "label": "conformal_abstain",
                    "chosen_evidence_id": "e1",
                    "conformal_score": 0.61,
                    "conformal_gate": "abstain",
                }
            },
            "decisions": {"u0001": "abstain"},
            "coverage": {
                "u0001": {
                    "relevant_evidence_ids": ["e1"],
                    "used_evidence_ids": ["e1"],
                    "missing_evidence_ids": [],
                    "coverage_score": 1.0,
                }
            },
            "rewards": {
                "u0001": {
                    "total_reward": 0.0,
                    "support_score": 0.61,
                    "hallucination_penalty": 0.0,
                    "abstain_penalty": 1.0,
                    "coverage_score": 1.0,
                }
            },
            "conformal": {
                "threshold": 0.6,
                "meta": {"abstain_margin": 0.05},
                "score_min": 0.61,
                "score_max": 0.61,
                "abstain_margin": 0.05,
                "abstain_band": [0.55, 0.65],
                "band_hit_count": 1,
                "decision_counts": {"accept": 0, "reject": 0, "abstain": 1},
            },
            "verified_extract": [],
            "verified_text": "",
            "safe_answer_final_text": "",
            "safe_answer_summary": {
                "accepted_count": 0,
                "abstained_count": 1,
                "rejected_count": 0,
            },
            "safe_answer": {
                "final_text": "",
                "accepted_claims": [],
                "abstained_claims": [
                    {
                        "unit_id": "u0001",
                        "text": "Supported claim.",
                        "citations": ["e1"],
                        "decision": "abstain",
                    }
                ],
                "summary": {
                    "accepted_count": 0,
                    "abstained_count": 1,
                    "rejected_count": 0,
                },
            },
            "decision": {"refusal": True, "reason_code": "ALL_DROPPED", "summary_stats": {}},
            "stats": {
                "kept_units": 0,
                "dropped_units": 1,
                "coverage_avg_score": 1.0,
                "reward_total": 0.0,
                "reward_avg": 0.0,
                "reward_hallucination_rate": 0.0,
                "reward_abstention_rate": 1.0,
            },
        }

    monkeypatch.setattr(harness, "CrossEncoderReranker", FakeReranker)
    monkeypatch.setattr(harness, "run_pipeline", fake_run_pipeline)

    summary = harness.run_v2_eval(
        dataset_path=dataset_path,
        out_path=out_path,
        conformal_state_path=str(conformal_path),
        debug_dump_path=debug_path,
        accept_threshold=0.05,
    )

    debug_rows = [json.loads(line) for line in debug_path.read_text(encoding="utf-8").splitlines()]
    assert debug_rows
    debug_row = next(row for row in debug_rows if row["variant"] == "conformal_only")
    assert {
        "accept_threshold",
        "example_id",
        "variant",
        "verifier_model_name",
        "units",
        "gold_units",
        "pool_candidates",
        "pre_rerank_candidates",
        "post_rerank_candidates",
        "verification_pairs",
        "used_evidence",
        "verifier_scores",
        "decisions",
        "coverage",
        "rewards",
        "unit_risk_scores",
        "per_unit_pair_budget",
        "evaluated_pairs_count_per_unit",
        "planned_pairs_total",
        "evaluated_pairs_total",
        "pruned_pairs_total",
        "per_unit_pairs_before_budget",
        "per_unit_pairs_after_budget",
        "safe_answer_final_text",
        "safe_answer_summary",
        "safe_answer",
    }.issubset(debug_row.keys())
    assert debug_row["accept_threshold"] == 0.05
    assert debug_row["verifier_model_name"] == "nli/test-model"
    assert debug_row["safe_answer_summary"]["abstained_count"] == 1
    assert summary["variants"]["conformal_only"]["debug"]["accept_threshold"] == 0.05
    metadata = summary["variants"]["conformal_only"]["metrics_metadata"]
    assert metadata["conformal_threshold"] == 0.6
    assert metadata["conformal_band_hit_count"] == 1
    assert metadata["abstain_band_observed"] is True


def test_run_v2_eval_uses_relaxed_eval_policy_default(tmp_path: Path, monkeypatch) -> None:
    dataset_path = _write_jsonl(
        tmp_path / "dataset.jsonl",
        [
            {
                "id": "supported-example",
                "llm_summary_text": "Supported claim.",
                "evidence_json": [{"id": "e1", "text": "Supported claim.", "metadata": {}}],
            }
        ],
    )
    out_path = tmp_path / "summary.json"

    class FakeReranker:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

    def fake_run_pipeline(**kwargs):  # type: ignore[no-untyped-def]
        policy = kwargs["policy_config"]
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
        kept = 1 if getattr(policy, "threshold_entailment") <= 0.5 else 0
        return {
            "verified_extract": ([{"unit_id": "u0001", "text": "Supported claim."}] if kept else []),
            "verified_text": "Supported claim." if kept else "",
            "used_evidence": {"u0001": ["e1"]},
            "decision": {
                "refusal": kept == 0,
                "reason_code": "OK_FULL" if kept else "ALL_DROPPED",
                "summary_stats": {},
            },
            "stats": {"kept_units": kept, "dropped_units": 1 - kept},
        }

    monkeypatch.setattr(harness, "CrossEncoderReranker", FakeReranker)
    monkeypatch.setattr(harness, "run_pipeline", fake_run_pipeline)

    summary = harness.run_v2_eval(dataset_path=dataset_path, out_path=out_path)

    assert summary["variants"]["v1_baseline"]["metrics"]["kept_units"] == 1


def test_run_v2_eval_pilot_dataset_budget_variant_differs_from_baseline(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_path = Path("examples/v2/eval_dataset_pilot.jsonl")
    out_path = tmp_path / "summary.json"

    def fake_run_pipeline(**kwargs):  # type: ignore[no-untyped-def]
        summary_text = str(kwargs["llm_summary_text"])
        n_units = len([part for part in summary_text.split(".") if part.strip()])
        use_budget = kwargs.get("budget_policy") is not None
        budget_config = kwargs.get("budget_config")
        budget_cap = getattr(budget_config, "max_pairs_total", None) if budget_config is not None else None
        baseline_pairs = int(n_units * int(kwargs.get("topk_per_unit", 0)))
        n_pairs = baseline_pairs
        kept_units = n_units
        if use_budget and budget_cap is not None:
            n_pairs = min(int(budget_cap), baseline_pairs)
            kept_units = max(0, n_units - 1)
        trace_out = kwargs.get("trace_out")
        if isinstance(trace_out, str):
            Path(trace_out).write_text(
                json.dumps(
                    {
                        "total_seconds": 0.01,
                        "n_units": n_units,
                        "n_pairs": n_pairs,
                        "rerank_pairs_scored": 0,
                        "conformal_abstain_units": 0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        return {
            "verified_extract": [
                {"unit_id": f"u{i+1:04d}", "text": f"unit-{i+1}"}
                for i in range(kept_units)
            ],
            "verified_text": "\n".join(f"unit-{i+1}" for i in range(kept_units)),
            "decision": {"refusal": False, "reason_code": "OK_PARTIAL", "summary_stats": {}},
            "stats": {
                "kept_units": kept_units,
                "dropped_units": max(0, n_units - kept_units),
                "budget_active": bool(use_budget),
                "requested_budget_max_pairs": budget_cap,
                "effective_budget_max_pairs": n_pairs if use_budget else None,
                "effective_topk_per_unit": 1 if use_budget else kwargs.get("topk_per_unit"),
                "avg_pairs_per_unit": float(n_pairs) / float(n_units) if n_units > 0 else 0.0,
                "pairs_allocated_to_high_risk_units": n_pairs if use_budget else 0,
                "pairs_allocated_to_low_risk_units": 0,
                "planned_pairs_total": baseline_pairs,
                "evaluated_pairs_total": n_pairs,
                "pruned_pairs_total": max(0, baseline_pairs - n_pairs),
            },
        }

    monkeypatch.setattr(harness, "run_pipeline", fake_run_pipeline)

    summary = harness.run_v2_eval(
        dataset_path=dataset_path,
        out_path=out_path,
        budget_max_pairs=3,
    )

    baseline = summary["variants"]["v1_baseline"]["metrics"]
    budget_only = summary["variants"]["budget_only"]["metrics"]

    assert budget_only["kept_units"] < baseline["kept_units"]
    assert budget_only["budget_active"] is True
    assert budget_only["requested_budget_max_pairs"] == 3
    assert budget_only["verifier_calls_proxy"] <= budget_only["effective_budget_max_pairs"]
    assert budget_only["pruned_pairs_total"] > 0


def test_cli_v2_eval_command_calls_harness(tmp_path: Path, monkeypatch, capsys) -> None:
    dataset = _write_jsonl(
        tmp_path / "dataset.jsonl",
        [{"llm_summary_text": "A", "evidence_json": [{"id": "e1", "text": "A"}]}],
    )
    out_path = tmp_path / "out.json"

    seen: dict[str, object] = {}

    def fake_run_v2_eval(**kwargs):  # type: ignore[no-untyped-def]
        seen.update(kwargs)
        Path(kwargs["out_path"]).write_text(json.dumps({"ok": True}), encoding="utf-8")
        return {"ok": True}

    monkeypatch.setattr(cli, "run_v2_eval", fake_run_v2_eval)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "ega",
            "v2-eval",
            "--dataset",
            str(dataset),
            "--out",
            str(out_path),
            "--accept-threshold",
            "0.05",
        ],
    )

    exit_code = cli.main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert seen["accept_threshold"] == 0.05
    assert json.loads(captured.out.strip()) == {"ok": True}
