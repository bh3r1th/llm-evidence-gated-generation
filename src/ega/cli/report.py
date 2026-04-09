"""Handlers for report-generation commands."""

from __future__ import annotations

import json
from pathlib import Path

from ega.v2.poc_release import build_final_poc_summary, write_poc_results_markdown


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


def handle_generate_poc_report(args: object) -> int:
    source_summary_path = _resolve_existing_path(args.source_summary)
    dataset_path = _resolve_existing_path(args.dataset)
    conformal_state_path = _resolve_existing_path(args.conformal_state)
    summary_out_path = _resolve_path(args.summary_out)
    report_out_path = _resolve_path(args.report_out)
    summary_out_path.parent.mkdir(parents=True, exist_ok=True)
    report_out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = build_final_poc_summary(
        source_summary_path=source_summary_path,
        out_path=summary_out_path,
        dataset_path=dataset_path,
        conformal_state_path=conformal_state_path,
        accept_threshold=args.accept_threshold,
        reranker_model=args.reranker_model,
        include_experimental=args.include_experimental,
    )
    write_poc_results_markdown(summary_path=summary_out_path, out_path=report_out_path)
    print(json.dumps(summary, sort_keys=True))
    return 0
