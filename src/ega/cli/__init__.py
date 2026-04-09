"""CLI entrypoint for EGA tooling."""

from __future__ import annotations

import argparse
import os
import sys

from ega import __version__
from ega.api import verify_answer
from ega.benchmark import run_benchmark
from ega.pipeline import run_pipeline
from ega.verifiers.nli_cross_encoder import NliCrossEncoderVerifier
from ega.v2.cross_encoder_reranker import CrossEncoderReranker
from ega.v2.eval_harness import DEFAULT_DEBUG_DUMP_PATH
from ega.v2.eval_harness import run_v2_eval
from ega.v2.export_calibration_rows import export_calibration_rows
from ega.v2.poc_config import (
    DEFAULT_ACCEPT_THRESHOLD,
    DEFAULT_FINAL_CONFORMAL_STATE,
    DEFAULT_FINAL_DATASET,
    DEFAULT_FINAL_REPORT,
    DEFAULT_FINAL_SOURCE_SUMMARY,
    DEFAULT_FINAL_SUMMARY,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_RERANK_TOPK,
)
from ega.v2.poc_release import build_final_poc_summary, write_poc_results_markdown
from ega.v2.threshold_sweep import run_threshold_sweep

_MIN_SUPPORTED_PYTHON = (3, 10)
_MAX_SUPPORTED_PYTHON = (3, 13)


def _python_version_supported(version_info: object) -> bool:
    if hasattr(version_info, "major") and hasattr(version_info, "minor"):
        major = int(version_info.major)  # type: ignore[attr-defined]
        minor = int(version_info.minor)  # type: ignore[attr-defined]
    else:
        major = int(version_info[0])  # type: ignore[index]
        minor = int(version_info[1])  # type: ignore[index]
    return _MIN_SUPPORTED_PYTHON <= (major, minor) < _MAX_SUPPORTED_PYTHON


def _runtime_version_info() -> object:
    return sys.version_info


def _enforce_supported_python_runtime() -> None:
    version_info = _runtime_version_info()
    if not _python_version_supported(version_info):
        if hasattr(version_info, "major") and hasattr(version_info, "minor"):
            major = int(version_info.major)  # type: ignore[attr-defined]
            minor = int(version_info.minor)  # type: ignore[attr-defined]
        else:
            major = int(version_info[0])  # type: ignore[index]
            minor = int(version_info[1])  # type: ignore[index]
        print(
            f"[EGA ERROR] Unsupported Python {major}.{minor}. Use 3.10-3.12.",
            file=sys.stderr,
        )
        raise SystemExit(1)


def _should_enforce_python_check() -> bool:
    if os.environ.get("EGA_ENFORCE_PYTHON_CHECK") == "1":
        return True
    return "PYTEST_CURRENT_TEST" not in os.environ


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the `ega` command."""
    parser = argparse.ArgumentParser(
        prog="ega",
        description="Evidence-Gated Answering (enforcement/decision layer)",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print package version and exit.",
    )

    subparsers = parser.add_subparsers(dest="command")
    run_parser = subparsers.add_parser(
        "run",
        help="Run policy enforcement for one answer/evidence pair.",
    )
    run_parser.add_argument("--answer-file", required=True, help="Path to plain text answer file.")
    run_parser.add_argument(
        "--evidence-file",
        required=True,
        help="Path to evidence JSON list containing {id,text,metadata} objects.",
    )
    run_parser.add_argument(
        "--unitizer",
        choices=("sentence", "bullets"),
        default="sentence",
        help="Unitizer mode for answer splitting.",
    )
    run_parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Entailment threshold for keeping a unit.",
    )
    run_parser.add_argument(
        "--partial-allowed",
        action="store_true",
        help="Allow partial answers when some units are dropped.",
    )
    run_parser.add_argument(
        "--scores-jsonl",
        default=None,
        help="Optional JSONL file with precomputed per-unit scores.",
    )
    run_parser.add_argument(
        "--emit-verified-only",
        dest="emit_verified_only",
        action="store_true",
        help="Emit only verified units in output payload (default).",
    )
    run_parser.add_argument(
        "--no-emit-verified-only",
        dest="emit_verified_only",
        action="store_false",
        help="Keep output unchanged for compatibility.",
    )
    run_parser.set_defaults(emit_verified_only=True)
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run verifier benchmark over JSONL data.",
    )
    benchmark_parser.add_argument("--data", required=True, help="Path to benchmark JSONL.")
    benchmark_parser.add_argument("--out", default=None, help="Optional output JSON path.")
    benchmark_parser.add_argument("--model-name", default=None, help="HF model id override.")
    benchmark_parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run deterministic threshold calibration grid search.",
    )
    benchmark_parser.add_argument(
    	"--topk",
	type=int,
	default=5,
	help="Number of top calibration configs to emit (only with --calibrate).",
    )
    benchmark_parser.add_argument(
    	"--run-policy",
        default=None,
        help="Path to a saved calibration/policy JSON file to use for benchmark run.",
    )

    benchmark_parser.add_argument(
        "--threshold-entailment",
        type=float,
        default=0.8,
        help="Entailment threshold for keeping a unit.",
    )
    benchmark_parser.add_argument(
        "--max-contradiction",
        type=float,
        default=0.2,
        help="Max contradiction score for keeping a unit.",
    )
    benchmark_parser.add_argument(
        "--partial-allowed",
        dest="partial_allowed",
        action="store_true",
        help="Allow partial answers when some units are dropped (default).",
    )
    benchmark_parser.add_argument(
        "--no-partial-allowed",
        dest="partial_allowed",
        action="store_false",
        help="Disallow partial answers.",
    )
    benchmark_parser.set_defaults(partial_allowed=True)
    conformal_parser = subparsers.add_parser(
        "conformal-calibrate",
        help="Fit conformal threshold from unit-level score/support JSONL; regenerate after verifier or threshold semantics change.",
    )
    conformal_parser.add_argument(
        "--in",
        dest="in_path",
        required=True,
        help="Path to calibration JSONL rows with {unit_id,score,supported}. Regenerate whenever verifier model, score definition, or thresholding semantics change.",
    )
    conformal_parser.add_argument(
        "--out",
        required=True,
        help="Output JSON path for calibrated conformal state.",
    )
    conformal_parser.add_argument(
        "--epsilon",
        required=True,
        type=float,
        help="Target unsupported risk bound in [0,1].",
    )
    conformal_parser.add_argument(
        "--mode",
        default="supported_rate",
        help="Conformal mode label stored in output metadata.",
    )
    conformal_parser.add_argument(
        "--min-calib",
        type=int,
        default=50,
        help="Minimum required calibration rows.",
    )
    conformal_parser.add_argument(
        "--abstain-margin",
        type=float,
        default=0.02,
        help="Abstain band width around the learned threshold.",
    )
    export_calibration_parser = subparsers.add_parser(
        "export-calibration-rows",
        help="Export gold-labeled unit rows from the current v2 verifier path for conformal calibration.",
    )
    export_calibration_parser.add_argument(
        "--dataset",
        required=True,
        help="Path to JSONL dataset with llm_summary_text, evidence_json, and gold_units.",
    )
    export_calibration_parser.add_argument(
        "--out",
        required=True,
        help="Output JSONL path.",
    )
    export_calibration_parser.add_argument(
        "--accept-threshold",
        type=float,
        default=None,
        help="Verifier acceptance threshold recorded with exported rows.",
    )
    export_calibration_parser.add_argument(
        "--nli-model-name",
        default=None,
        help="Optional NLI model id override for OSS verification.",
    )
    export_calibration_parser.add_argument(
        "--topk-per-unit",
        type=int,
        default=12,
        help="Base verifier top-k per unit.",
    )
    export_calibration_parser.add_argument(
        "--max-pairs-total",
        type=int,
        default=200,
        help="Base verifier max pair cap.",
    )
    export_calibration_parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Runtime device for OSS NLI verifier.",
    )
    export_calibration_parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="Runtime dtype for OSS NLI verifier.",
    )
    v2_eval_parser = subparsers.add_parser(
        "v2-eval",
        help="Run v2 variant evaluation harness over a JSONL dataset.",
    )
    v2_eval_parser.add_argument(
        "--dataset",
        required=True,
        help="Path to JSONL dataset with llm_summary_text and evidence_json.",
    )
    v2_eval_parser.add_argument(
        "--out",
        required=True,
        help="Output JSON summary path.",
    )
    v2_eval_parser.add_argument(
        "--conformal-state",
        default=None,
        help="Optional conformal state JSON path for conformal variants.",
    )
    v2_eval_parser.add_argument(
        "--reranker-model",
        default=DEFAULT_RERANKER_MODEL,
        help="HF model id for reranker variants.",
    )
    v2_eval_parser.add_argument(
        "--rerank-topk",
        type=int,
        default=DEFAULT_RERANK_TOPK,
        help="Per-unit reranker top-k.",
    )
    v2_eval_parser.add_argument(
        "--latency-budget-ms",
        type=int,
        default=None,
        help="Experimental latency budget for budget variants.",
    )
    v2_eval_parser.add_argument(
        "--budget-max-pairs",
        type=int,
        default=None,
        help="Experimental pair cap override for budget variants.",
    )
    v2_eval_parser.add_argument(
        "--debug-dump-path",
        default=None,
        help=(
            "Optional JSONL debug dump path. Omit for normal publish runs. "
            f"Example: {DEFAULT_DEBUG_DUMP_PATH}"
        ),
    )
    v2_eval_parser.add_argument(
        "--topk-per-unit",
        type=int,
        default=12,
        help="Base verifier top-k per unit.",
    )
    v2_eval_parser.add_argument(
        "--max-pairs-total",
        type=int,
        default=200,
        help="Base verifier max pair cap.",
    )
    v2_eval_parser.add_argument(
        "--nli-model-name",
        default=None,
        help="Optional NLI model id override for OSS verification.",
    )
    v2_eval_parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Runtime device for OSS NLI verifier.",
    )
    v2_eval_parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="Runtime dtype for OSS NLI verifier.",
    )
    v2_eval_parser.add_argument(
        "--accept-threshold",
        type=float,
        default=None,
        help="Verifier acceptance threshold override for unit retention.",
    )
    v2_eval_parser.add_argument(
        "--render-safe-answer",
        action="store_true",
        help="Include rendered safe-answer text in pipeline results and debug artifacts.",
    )
    threshold_sweep_parser = subparsers.add_parser(
        "threshold-sweep",
        help="Sweep baseline verifier acceptance thresholds over gold-labeled eval data.",
    )
    threshold_sweep_parser.add_argument(
        "--dataset",
        required=True,
        help="Path to JSONL dataset with llm_summary_text, evidence_json, and gold_units.",
    )
    threshold_sweep_parser.add_argument(
        "--out",
        required=True,
        help="Output JSON path.",
    )
    threshold_sweep_parser.add_argument(
        "--nli-model-name",
        default=None,
        help="Optional NLI model id override for OSS verification.",
    )
    threshold_sweep_parser.add_argument(
        "--topk-per-unit",
        type=int,
        default=12,
        help="Base verifier top-k per unit.",
    )
    threshold_sweep_parser.add_argument(
        "--max-pairs-total",
        type=int,
        default=200,
        help="Base verifier max pair cap.",
    )
    threshold_sweep_parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Runtime device for OSS NLI verifier.",
    )
    threshold_sweep_parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="Runtime dtype for OSS NLI verifier.",
    )
    threshold_sweep_parser.add_argument(
        "--accept-threshold",
        type=float,
        default=None,
        help="Verifier acceptance threshold override recorded with the sweep run.",
    )
    poc_report_parser = subparsers.add_parser(
        "generate-poc-report",
        help="Generate final publish-oriented POC JSON and Markdown artifacts from a v2 eval summary.",
    )
    poc_report_parser.add_argument(
        "--source-summary",
        default=str(DEFAULT_FINAL_SOURCE_SUMMARY),
        help="Input v2 eval summary JSON.",
    )
    poc_report_parser.add_argument(
        "--summary-out",
        default=str(DEFAULT_FINAL_SUMMARY),
        help="Output final POC summary JSON.",
    )
    poc_report_parser.add_argument(
        "--report-out",
        default=str(DEFAULT_FINAL_REPORT),
        help="Output publish Markdown report.",
    )
    poc_report_parser.add_argument(
        "--dataset",
        default=str(DEFAULT_FINAL_DATASET),
        help="Dataset path recorded in the final report metadata.",
    )
    poc_report_parser.add_argument(
        "--conformal-state",
        default=str(DEFAULT_FINAL_CONFORMAL_STATE),
        help="Conformal state path recorded in the final report metadata.",
    )
    poc_report_parser.add_argument(
        "--accept-threshold",
        type=float,
        default=DEFAULT_ACCEPT_THRESHOLD,
        help="Recommended publish accept threshold recorded in the final report metadata.",
    )
    poc_report_parser.add_argument(
        "--reranker-model",
        default=DEFAULT_RERANKER_MODEL,
        help="Recommended reranker model recorded in the final report metadata.",
    )
    poc_report_parser.add_argument(
        "--include-experimental",
        action="store_true",
        help="Include experimental budget variants in the final JSON. The Markdown headline table still stays recommended-only.",
    )

    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run file-driven provided-summary pipeline with optional polish validation.",
    )
    pipeline_parser.add_argument(
        "--llm-summary-file",
        required=True,
        help="Path to provided LLM summary text file.",
    )
    pipeline_parser.add_argument(
        "--evidence-json",
        required=True,
        help="Path to evidence JSON list containing {id,text,metadata?}.",
    )
    pipeline_scoring = pipeline_parser.add_mutually_exclusive_group(required=True)
    pipeline_scoring.add_argument(
        "--scores-jsonl",
        default=None,
        help="Path to precomputed per-unit scores JSONL.",
    )
    pipeline_scoring.add_argument(
        "--use-oss-nli",
        action="store_true",
        help="Use OSS Hugging Face NLI verifier (requires ega[nli]).",
    )
    pipeline_parser.add_argument(
        "--nli-model-name",
        default=None,
        help="Optional HF model id for OSS NLI mode.",
    )
    pipeline_parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Runtime device for OSS NLI verifier.",
    )
    pipeline_parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="Runtime dtype for OSS NLI verifier.",
    )
    pipeline_parser.add_argument(
        "--topk-per-unit",
        type=int,
        default=12,
        help="Stage-A evidence preselection top-k per unit for OSS NLI.",
    )
    pipeline_parser.add_argument(
        "--max-pairs-total",
        type=int,
        default=200,
        help="Hard cap on total unit-evidence pairs scored by OSS NLI.",
    )
    pipeline_parser.add_argument(
        "--use-reranker",
        action="store_true",
        help="Enable optional v2 cross-encoder reranker before verification.",
    )
    pipeline_parser.add_argument(
        "--reranker-model",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="HF model id for v2 cross-encoder reranker.",
    )
    pipeline_parser.add_argument(
        "--rerank-topk",
        type=int,
        default=6,
        help="Per-unit candidate count retained by reranker.",
    )
    pipeline_parser.add_argument(
        "--conformal-state",
        default=None,
        help="Path to conformal state JSON artifact for v2 per-unit risk gating.",
    )
    pipeline_parser.add_argument(
        "--use-budget",
        action="store_true",
        help="Enable experimental v2 greedy budget controller for topk/max-pairs selection.",
    )
    pipeline_parser.add_argument(
        "--latency-budget-ms",
        type=int,
        default=None,
        help="Optional experimental latency budget consumed by v2 budget policy.",
    )
    pipeline_parser.add_argument(
        "--budget-max-pairs",
        type=int,
        default=None,
        help="Optional experimental pair-cap override consumed by v2 budget policy.",
    )
    pipeline_parser.add_argument(
        "--max-evidence-per-request",
        type=int,
        default=None,
        help="Optional cap on evidence items considered per request in OSS NLI.",
    )
    pipeline_parser.add_argument(
        "--max-batch-tokens",
        type=int,
        default=None,
        help="Token-budget cap per NLI batch (defaults by device).",
    )
    pipeline_parser.add_argument(
        "--evidence-max-chars",
        type=int,
        default=800,
        help="Per-evidence char cap applied before BM25 and NLI scoring.",
    )
    pipeline_parser.add_argument(
        "--evidence-max-sentences",
        type=int,
        default=3,
        help="Per-evidence sentence cap applied before BM25 and NLI scoring.",
    )
    pipeline_parser.add_argument(
        "--unitizer",
        choices=("spacy_sentence", "sentence", "bullets"),
        default="sentence",
        help="Unitizer mode for summary splitting.",
    )
    pipeline_parser.add_argument(
        "--threshold-entailment",
        type=float,
        default=0.8,
        help="Entailment threshold for keeping a unit.",
    )
    pipeline_parser.add_argument(
        "--accept-threshold",
        type=float,
        default=None,
        help="Verifier acceptance threshold override for keeping a unit.",
    )
    pipeline_parser.add_argument(
        "--max-contradiction",
        type=float,
        default=0.2,
        help="Max contradiction score for keeping a unit.",
    )
    pipeline_parser.add_argument(
        "--partial-allowed",
        action="store_true",
        help="Allow partial answers when some units are dropped.",
    )
    pipeline_parser.add_argument(
        "--polished-json",
        default=None,
        help="Optional JSON file with {unit_id,edited_text} list or wrapper.",
    )
    pipeline_parser.add_argument(
        "--no-polish-validation",
        action="store_true",
        help="Skip deterministic polish validation checks.",
    )
    pipeline_parser.add_argument(
        "--trace-out",
        default=None,
        help="Path to a JSONL trace file for pipeline timing and stats",
    )
    pipeline_parser.add_argument(
        "--coverage",
        action="store_true",
        help="Enable v2 coverage analysis stage.",
    )
    pipeline_parser.add_argument(
        "--coverage-pool-topk",
        type=int,
        default=20,
        help="Relevant evidence pool top-k for v2 coverage analysis.",
    )
    pipeline_parser.add_argument(
        "--rewards",
        action="store_true",
        help="Enable v2 reward signal computation stage.",
    )
    pipeline_parser.add_argument(
        "--reward-w-support",
        type=float,
        default=1.0,
        help="Reward weight for support score.",
    )
    pipeline_parser.add_argument(
        "--reward-w-hallucination",
        type=float,
        default=2.0,
        help="Reward weight for hallucination penalty.",
    )
    pipeline_parser.add_argument(
        "--reward-w-abstain",
        type=float,
        default=0.5,
        help="Reward weight for abstain penalty.",
    )
    pipeline_parser.add_argument(
        "--reward-w-coverage",
        type=float,
        default=1.0,
        help="Reward weight for coverage score.",
    )
    pipeline_parser.add_argument(
        "--reward-clamp-min",
        type=float,
        default=-5.0,
        help="Minimum clamp for per-unit reward.",
    )
    pipeline_parser.add_argument(
        "--reward-clamp-max",
        type=float,
        default=5.0,
        help="Maximum clamp for per-unit reward.",
    )
    pipeline_parser.add_argument(
        "--render-safe-answer",
        action="store_true",
        help="Render a deterministic plain-text safe answer from accepted claims.",
    )
    pipeline_parser.add_argument(
        "--emit-training-jsonl",
        default=None,
        help="Optional JSONL output path for emitted training examples.",
    )
    pipeline_parser.add_argument(
        "--training-example-id",
        default=None,
        help="Optional stable id for emitted training example row.",
    )

    polish_parser = subparsers.add_parser(
        "polish-validate",
        help="Validate optional polished units against verified units.",
    )
    polish_parser.add_argument(
        "--verified-json",
        required=True,
        help="Path to serialized EnforcementResult JSON containing verified units.",
    )
    polish_parser.add_argument(
        "--polished-json",
        required=True,
        help="Path to JSON list/object containing {unit_id,edited_text} rows.",
    )
    shell_parser = subparsers.add_parser(
        "shell",
        help="Interactive EGA shell that keeps the verifier loaded.",
    )
    shell_parser.add_argument(
        "--model-name",
        default=None,
        help="HF model id override for the verifier.",
    )
    shell_parser.add_argument(
        "--unitizer",
        choices=("spacy_sentence", "sentence", "bullets"),
        default="sentence",
        help="Unitizer mode for summary splitting.",
    )
    shell_parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Runtime device for OSS NLI verifier.",
    )
    shell_parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="Runtime dtype for OSS NLI verifier.",
    )
    shell_parser.add_argument(
        "--topk-per-unit",
        type=int,
        default=12,
        help="Stage-A evidence preselection top-k per unit for OSS NLI.",
    )
    shell_parser.add_argument(
        "--max-pairs-total",
        type=int,
        default=200,
        help="Hard cap on total unit-evidence pairs scored by OSS NLI.",
    )
    shell_parser.add_argument(
        "--use-reranker",
        action="store_true",
        help="Enable optional v2 cross-encoder reranker before verification.",
    )
    shell_parser.add_argument(
        "--reranker-model",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="HF model id for v2 cross-encoder reranker.",
    )
    shell_parser.add_argument(
        "--rerank-topk",
        type=int,
        default=6,
        help="Per-unit candidate count retained by reranker.",
    )
    shell_parser.add_argument(
        "--conformal-state",
        default=None,
        help="Path to conformal state JSON artifact for v2 per-unit risk gating.",
    )
    shell_parser.add_argument(
        "--use-budget",
        action="store_true",
        help="Enable experimental v2 greedy budget controller for topk/max-pairs selection.",
    )
    shell_parser.add_argument(
        "--latency-budget-ms",
        type=int,
        default=None,
        help="Optional experimental latency budget consumed by v2 budget policy.",
    )
    shell_parser.add_argument(
        "--budget-max-pairs",
        type=int,
        default=None,
        help="Optional experimental pair-cap override consumed by v2 budget policy.",
    )
    shell_parser.add_argument(
        "--max-evidence-per-request",
        type=int,
        default=None,
        help="Optional cap on evidence items considered per request in OSS NLI.",
    )
    shell_parser.add_argument(
        "--max-batch-tokens",
        type=int,
        default=None,
        help="Token-budget cap per NLI batch (defaults by device).",
    )
    shell_parser.add_argument(
        "--evidence-max-chars",
        type=int,
        default=800,
        help="Per-evidence char cap applied before BM25 and NLI scoring.",
    )
    shell_parser.add_argument(
        "--evidence-max-sentences",
        type=int,
        default=3,
        help="Per-evidence sentence cap applied before BM25 and NLI scoring.",
    )
    shell_parser.add_argument(
        "--threshold-entailment",
        type=float,
        default=0.8,
        help="Default entailment threshold for policy.",
    )
    shell_parser.add_argument(
        "--max-contradiction",
        type=float,
        default=0.2,
        help="Default max contradiction for policy.",
    )
    shell_parser.add_argument(
        "--partial-allowed",
        action="store_true",
        default=True,
        help="Default partial_allowed flag for policy.",
    )
    shell_parser.add_argument(
        "--stdin-jsonl",
        action="store_true",
        help="Read one JSON request per line from stdin (non-interactive mode).",
    )
    shell_parser.add_argument(
        "--stdout-jsonl",
        action="store_true",
        help="Write one JSON response per line to stdout (non-interactive mode).",
    )
    shell_parser.add_argument(
        "--trace-out",
        default=None,
        help="Optional JSONL trace output path for per-request timings.",
    )

    return parser


def main() -> int:
    """Run the CLI."""
    if _should_enforce_python_check():
        _enforce_supported_python_runtime()

    from ega.cli import pipeline as pipeline_handlers
    from ega.cli import report as report_handlers
    from ega.cli import run as run_handlers
    from ega.cli import shell as shell_handlers
    from ega.cli import v2 as v2_handlers

    # Backward-compatible monkeypatch hooks used by tests and external callers.
    pipeline_handlers.verify_answer = verify_answer
    pipeline_handlers.CrossEncoderReranker = CrossEncoderReranker
    v2_handlers.run_v2_eval = run_v2_eval
    v2_handlers.export_calibration_rows = export_calibration_rows
    v2_handlers.run_threshold_sweep = run_threshold_sweep
    v2_handlers.build_final_poc_summary = build_final_poc_summary
    v2_handlers.write_poc_results_markdown = write_poc_results_markdown
    report_handlers.build_final_poc_summary = build_final_poc_summary
    report_handlers.write_poc_results_markdown = write_poc_results_markdown
    v2_handlers.run_benchmark = run_benchmark
    shell_handlers.NliCrossEncoderVerifier = NliCrossEncoderVerifier
    shell_handlers.run_pipeline = run_pipeline
    shell_handlers.CrossEncoderReranker = CrossEncoderReranker

    try:
        parser = build_parser()
        args = parser.parse_args()

        if args.version:
            print(__version__)
            return 0

        if args.command == "run":
            return run_handlers.handle_run(args)
        if args.command == "polish-validate":
            return pipeline_handlers.handle_polish_validate(args)
        if args.command == "pipeline":
            return pipeline_handlers.handle_pipeline(args)
        if args.command == "conformal-calibrate":
            return v2_handlers.handle_conformal_calibrate(args)
        if args.command == "export-calibration-rows":
            return v2_handlers.handle_export_calibration_rows(args)
        if args.command == "v2-eval":
            return v2_handlers.handle_v2_eval(args)
        if args.command == "threshold-sweep":
            return v2_handlers.handle_threshold_sweep(args)
        if args.command == "generate-poc-report":
            return report_handlers.handle_generate_poc_report(args)
        if args.command == "benchmark":
            return v2_handlers.handle_benchmark(args)
        if args.command == "shell":
            return shell_handlers.handle_shell(args, sys)

        parser.print_help()
        return 0
    except Exception as exc:  # pragma: no cover - exercised by CLI IO error tests.
        if os.environ.get("EGA_DEBUG") == "1":
            raise
        print(f"[EGA ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
