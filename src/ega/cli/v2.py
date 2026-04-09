"""Handlers for v2 and benchmark-related commands."""

from __future__ import annotations

import json
from pathlib import Path

from ega.benchmark import PolicyConfig, calibrate_policies, load_policy_config, run_benchmark
from ega.v2.calibrate import calibrate_jsonl_to_state, save_conformal_state_json
from ega.v2.eval_harness import run_v2_eval
from ega.v2.export_calibration_rows import CALIBRATION_SCORE_DEFINITION, export_calibration_rows
from ega.v2.poc_config import (
    DEFAULT_ACCEPT_THRESHOLD,
    DEFAULT_FINAL_CONFORMAL_STATE,
    DEFAULT_FINAL_REPORT,
    DEFAULT_FINAL_SOURCE_SUMMARY,
    DEFAULT_FINAL_SUMMARY,
)
from ega.v2.poc_release import build_final_poc_summary, write_poc_results_markdown
from ega.v2.threshold_sweep import run_threshold_sweep


def _resolve_path(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    return p.resolve()


def _resolve_existing_path(path_str: str) -> Path:
    resolved_path = _resolve_path(path_str)
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"File not found: {resolved_path}. Current working directory: {Path.cwd()}"
        )
    return resolved_path


def handle_conformal_calibrate(args: object) -> int:
    in_path = _resolve_existing_path(args.in_path)
    out_path = _resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state, n = calibrate_jsonl_to_state(
        in_path=in_path,
        epsilon=args.epsilon,
        mode=args.mode,
        min_calib=args.min_calib,
        abstain_margin=args.abstain_margin,
    )
    save_conformal_state_json(out_path, state)
    print(f"n={n} threshold={state.threshold:.6f} epsilon={float(args.epsilon):.6f}")
    return 0


def handle_export_calibration_rows(args: object) -> int:
    dataset_path = _resolve_existing_path(args.dataset)
    out_path = _resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = export_calibration_rows(
        dataset_path=str(dataset_path),
        out_path=str(out_path),
        nli_model_name=args.nli_model_name,
        topk_per_unit=args.topk_per_unit,
        max_pairs_total=args.max_pairs_total,
        nli_device=args.device,
        nli_dtype=args.dtype,
        accept_threshold=args.accept_threshold,
    )
    summary["score_definition"] = CALIBRATION_SCORE_DEFINITION
    print(json.dumps(summary, sort_keys=True))
    return 0


def handle_v2_eval(args: object) -> int:
    dataset_path = _resolve_existing_path(args.dataset)
    out_path = _resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    conformal_state = None
    if isinstance(args.conformal_state, str) and args.conformal_state.strip():
        conformal_state = str(_resolve_existing_path(args.conformal_state))
    emit_final_poc_summary = out_path.name == DEFAULT_FINAL_SUMMARY.name
    eval_out_path = out_path.with_name(DEFAULT_FINAL_SOURCE_SUMMARY.name) if emit_final_poc_summary else out_path
    summary = run_v2_eval(
        dataset_path=str(dataset_path),
        out_path=str(eval_out_path),
        conformal_state_path=conformal_state,
        reranker_model=args.reranker_model,
        rerank_topk=args.rerank_topk,
        latency_budget_ms=args.latency_budget_ms,
        budget_max_pairs=args.budget_max_pairs,
        topk_per_unit=args.topk_per_unit,
        max_pairs_total=args.max_pairs_total,
        nli_model_name=args.nli_model_name,
        nli_device=args.device,
        nli_dtype=args.dtype,
        accept_threshold=args.accept_threshold,
        debug_dump_path=args.debug_dump_path,
        render_safe_answer=args.render_safe_answer,
    )
    if emit_final_poc_summary:
        summary = build_final_poc_summary(
            source_summary_path=eval_out_path,
            out_path=out_path,
            dataset_path=dataset_path,
            conformal_state_path=(conformal_state if conformal_state is not None else DEFAULT_FINAL_CONFORMAL_STATE),
            accept_threshold=(DEFAULT_ACCEPT_THRESHOLD if args.accept_threshold is None else float(args.accept_threshold)),
            reranker_model=args.reranker_model,
        )
        write_poc_results_markdown(summary_path=out_path, out_path=DEFAULT_FINAL_REPORT)
    print(json.dumps(summary, sort_keys=True))
    return 0


def handle_threshold_sweep(args: object) -> int:
    dataset_path = _resolve_existing_path(args.dataset)
    out_path = _resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = run_threshold_sweep(
        dataset_path=str(dataset_path),
        out_path=str(out_path),
        nli_model_name=args.nli_model_name,
        topk_per_unit=args.topk_per_unit,
        max_pairs_total=args.max_pairs_total,
        nli_device=args.device,
        nli_dtype=args.dtype,
        accept_threshold=args.accept_threshold,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


def handle_benchmark(args: object) -> int:
    if args.calibrate and args.run_policy:
        raise ValueError("Cannot combine --calibrate with --run-policy.")

    if args.calibrate:
        calibration = calibrate_policies(
            data_path=args.data,
            model_name=args.model_name,
            out_path=args.out,
            topk=args.topk,
        )
        print(json.dumps(calibration, sort_keys=True))
        return 0

    policy = None
    if isinstance(args.run_policy, str) and args.run_policy.strip():
        policy = load_policy_config(_resolve_existing_path(args.run_policy))

    summary = run_benchmark(
        data_path=args.data,
        out_path=args.out,
        model_name=args.model_name,
        policy_config=policy
        or PolicyConfig(
            threshold_entailment=args.threshold_entailment,
            max_contradiction=args.max_contradiction,
            partial_allowed=args.partial_allowed,
        ),
    )
    print(json.dumps(summary, sort_keys=True))
    return 0
