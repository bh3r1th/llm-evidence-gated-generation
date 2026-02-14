"""CLI entrypoint for EGA tooling."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from string import punctuation

from ega import __version__
from ega.benchmark import PolicyConfig, calibrate_policies, load_policy_config, run_benchmark
from ega.enforcer import Enforcer
from ega.pipeline import run_pipeline as run_pipeline
from ega.polish.gate import PolishGateConfig, apply_polish_gate
from ega.polish.types import PolishedUnit
from ega.providers.jsonl_scores import JsonlScoresProvider
from ega.serialization import from_json, to_json
from ega.text_clean import clean_text
from ega.types import EnforcementResult, EvidenceItem, EvidenceSet, Unit, VerificationScore
from ega.unitization import unitize_answer
from ega.verifiers.nli_cross_encoder import NliCrossEncoderVerifier

_MIN_SUPPORTED_PYTHON = (3, 10)
_MAX_SUPPORTED_PYTHON = (3, 13)


@dataclass(slots=True)
class OverlapVerifier:
    """Deterministic lexical-overlap verifier for CLI usage."""

    name: str = "lexical_overlap"

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        translator = str.maketrans("", "", punctuation + "`")
        return {
            token.lower().translate(translator)
            for token in text.split()
            if token.strip()
        }

    def verify(
        self,
        *,
        unit_text: str,
        unit_id: str,
        evidence: EvidenceSet,
    ) -> VerificationScore:
        unit_tokens = self._tokenize(unit_text)
        if not unit_tokens or not evidence.items:
            return VerificationScore(
                unit_id=unit_id,
                entailment=0.0,
                contradiction=0.0,
                neutral=1.0,
                label="neutral",
                raw={"verifier": self.name, "best_evidence_id": None},
            )

        best_overlap = 0.0
        best_evidence_id: str | None = None
        for item in evidence.items:
            evidence_tokens = self._tokenize(item.text)
            overlap = len(unit_tokens & evidence_tokens) / len(unit_tokens)
            if overlap > best_overlap:
                best_overlap = overlap
                best_evidence_id = item.id

        entailment = round(best_overlap, 6)
        contradiction = round(1.0 - entailment, 6)
        neutral = 0.0
        label = "entailment" if entailment >= 0.5 else "contradiction"
        return VerificationScore(
            unit_id=unit_id,
            entailment=entailment,
            contradiction=contradiction,
            neutral=neutral,
            label=label,
            raw={"verifier": self.name, "best_evidence_id": best_evidence_id},
        )


def _load_evidence(path: str) -> EvidenceSet:
    resolved_path = _resolve_path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"File not found: {resolved_path}. Current working directory: {Path.cwd()}"
        )
    try:
        with resolved_path.open(encoding="utf-8-sig") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in file {resolved_path}: {exc.msg}.") from exc

    if not isinstance(payload, list):
        raise ValueError("Evidence file must be a JSON list of {id,text,metadata} objects.")

    items: list[EvidenceItem] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Evidence item at index {idx} must be an object.")
        if "id" not in item or "text" not in item:
            raise ValueError(
                f"Evidence item at index {idx} must include 'id' and 'text' fields."
            )

        metadata = item.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"Evidence item at index {idx} has non-object metadata.")

        items.append(
            EvidenceItem(
                id=str(item["id"]),
                text=clean_text(str(item["text"])),
                metadata=metadata,
            )
        )

    return EvidenceSet(items=items)


def _load_answer(path: str) -> str:
    resolved_path = _resolve_path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"File not found: {resolved_path}. Current working directory: {Path.cwd()}"
        )
    with resolved_path.open(encoding="utf-8") as handle:
        return clean_text(handle.read())


def _load_polished_units(path: str) -> list[PolishedUnit]:
    resolved_path = _resolve_path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"File not found: {resolved_path}. Current working directory: {Path.cwd()}"
        )
    try:
        with resolved_path.open(encoding="utf-8-sig") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in file {resolved_path}: {exc.msg}.") from exc

    rows = payload
    if isinstance(payload, dict):
        if isinstance(payload.get("units"), list):
            rows = payload["units"]
        elif isinstance(payload.get("polished_units"), list):
            rows = payload["polished_units"]
        else:
            raise ValueError(
                "Polished JSON object must include a 'units' or 'polished_units' list."
            )
    if not isinstance(rows, list):
        raise ValueError("Polished JSON must be a list or wrapper object containing a list.")

    units: list[PolishedUnit] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Polished unit at index {idx} must be an object.")
        if "unit_id" not in row or "edited_text" not in row:
            raise ValueError(
                f"Polished unit at index {idx} must include 'unit_id' and 'edited_text'."
            )
        units.append(PolishedUnit(unit_id=str(row["unit_id"]), edited_text=str(row["edited_text"])))
    return units


def _load_enforcement_result(path: str) -> EnforcementResult:
    resolved_path = _resolve_path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"File not found: {resolved_path}. Current working directory: {Path.cwd()}"
        )
    payload = resolved_path.read_text(encoding="utf-8-sig")
    try:
        json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in file {resolved_path}: {exc.msg}.") from exc
    return from_json(payload, EnforcementResult)


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


def _strip_bom(s: str) -> str:
    return s.lstrip("\ufeff")


def _evidence_from_rows(rows: list[object]) -> EvidenceSet:
    items: list[EvidenceItem] = []
    for idx, item in enumerate(rows):
        if not isinstance(item, dict):
            raise ValueError(f"Evidence item at index {idx} must be an object.")
        if "id" not in item or "text" not in item:
            raise ValueError(
                f"Evidence item at index {idx} must include 'id' and 'text' fields."
            )
        metadata = item.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"Evidence item at index {idx} has non-object metadata.")
        items.append(
            EvidenceItem(
                id=str(item["id"]),
                text=clean_text(str(item["text"])),
                metadata=metadata,
            )
        )
    return EvidenceSet(items=items)


def _load_evidence_payload(payload: object) -> EvidenceSet:
    if isinstance(payload, str):
        return _load_evidence(payload)
    if isinstance(payload, list):
        return _evidence_from_rows(payload)
    raise ValueError("evidence_json must be a file path string or a list of evidence objects.")


def _ensure_trace_parent(trace_out: str | None) -> None:
    if not trace_out:
        return
    Path(trace_out).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def _append_shell_trace_row(trace_out: str | None, row: dict[str, object]) -> None:
    if not trace_out:
        return
    _ensure_trace_parent(trace_out)
    with Path(trace_out).expanduser().resolve().open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def _shell_trace_row(
    *,
    total_seconds: float = 0.0,
    verify_trace: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "total_seconds": total_seconds,
        "read_seconds": 0.0,
        "unitize_seconds": 0.0,
        "verify_seconds": 0.0,
        "enforce_seconds": 0.0,
        "polish_seconds": 0.0,
    }
    if isinstance(verify_trace, dict):
        payload.update(verify_trace)
        if "n_pairs" not in payload and "n_pairs_scored" in payload:
            payload["n_pairs"] = payload["n_pairs_scored"]
    return payload


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

    try:
        parser = build_parser()
        args = parser.parse_args()

        if args.version:
            print(__version__)
            return 0

        if args.command != "run":
            if args.command == "polish-validate":
                result = _load_enforcement_result(args.verified_json)
                original_units = [
                    Unit(
                        id=str(item["unit_id"]),
                        text=str(item["text"]),
                        metadata={},
                    )
                    for item in result.verified_units
                ]
                polished_units = _load_polished_units(args.polished_json)
                merged = apply_polish_gate(
                    result=result,
                    original_units=original_units,
                    polished_units=polished_units,
                    config=PolishGateConfig(),
                )
                print(to_json(merged))
                return 0

            if args.command == "pipeline":
                if not getattr(args, "llm_summary_file", None):
                    raise ValueError("Missing required argument: --llm-summary-file")
                if not getattr(args, "evidence_json", None):
                    raise ValueError("Missing required argument: --evidence-json")
                if not (bool(args.scores_jsonl) or bool(args.use_oss_nli)):
                    raise ValueError("Must provide either --scores-jsonl or --use-oss-nli")

                polished_payload = None
                if isinstance(args.polished_json, str) and args.polished_json.strip():
                    polished_path = _resolve_existing_path(args.polished_json)
                    try:
                        with polished_path.open(encoding="utf-8-sig") as handle:
                            polished_payload = json.load(handle)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Invalid JSON in file {polished_path}: {exc.msg}."
                        ) from exc

                scores_jsonl_path = None
                if isinstance(args.scores_jsonl, str) and args.scores_jsonl.strip():
                    scores_jsonl_path = str(_resolve_existing_path(args.scores_jsonl))

                llm_summary_text = _load_answer(args.llm_summary_file)
                evidence = _load_evidence(args.evidence_json)
                _ensure_trace_parent(args.trace_out)
                payload = run_pipeline(
                    llm_summary_text=llm_summary_text,
                    evidence=evidence,
                    unitizer_mode=args.unitizer,
                    policy_config=PolicyConfig(
                        threshold_entailment=args.threshold_entailment,
                        max_contradiction=args.max_contradiction,
                        partial_allowed=args.partial_allowed,
                    ),
                    scores_jsonl_path=scores_jsonl_path,
                    use_oss_nli=args.use_oss_nli,
                    nli_model_name=args.nli_model_name,
                    nli_device=args.device,
                    nli_dtype=args.dtype,
                    topk_per_unit=args.topk_per_unit,
                    max_pairs_total=args.max_pairs_total,
                    max_evidence_per_request=args.max_evidence_per_request,
                    max_batch_tokens=args.max_batch_tokens,
                    evidence_max_chars=args.evidence_max_chars,
                    evidence_max_sentences=args.evidence_max_sentences,
                    polished_json=polished_payload,
                    enable_polish_validation=not args.no_polish_validation,
                    trace_out=args.trace_out,
                )
                print(json.dumps(payload, sort_keys=True))
                return 0

            if args.command == "benchmark":
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
                    policy_dict = load_policy_config(_resolve_existing_path(args.run_policy))
                    policy = policy_dict  # already PolicyConfig

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

            if args.command == "shell":
                if bool(args.stdin_jsonl) != bool(args.stdout_jsonl):
                    raise ValueError("Use --stdin-jsonl and --stdout-jsonl together.")

                base_policy = PolicyConfig(
                    threshold_entailment=args.threshold_entailment,
                    max_contradiction=args.max_contradiction,
                    partial_allowed=args.partial_allowed,
                )
                verifier = NliCrossEncoderVerifier(model_name=args.model_name)

                if args.stdin_jsonl and args.stdout_jsonl:
                    for raw_line in sys.stdin:
                        line = raw_line.strip()
                        if not line:
                            continue
                        line_trace_out: str | None = args.trace_out
                        try:
                            row = json.loads(line)
                            if not isinstance(row, dict):
                                raise ValueError("Each stdin-jsonl line must be a JSON object.")
                            llm_summary = row.get("llm_summary", row.get("summary"))
                            if not isinstance(llm_summary, str) or not llm_summary.strip():
                                minimal = {
                                    "verified_extract": [],
                                    "verified_text": "",
                                    "decision": {
                                        "allowed_units": [],
                                        "dropped_units": [],
                                        "refusal": True,
                                        "reason_code": "invalid_input",
                                        "summary_stats": {"kept_units": 0, "dropped_units": 0},
                                    },
                                    "stats": {
                                        "kept_units": 0,
                                        "dropped_units": 0,
                                        "model_name": str(getattr(verifier, "model_name", "") or ""),
                                    },
                                    "polish_status": "skipped",
                                }
                                trace_out = str(row.get("trace_out", args.trace_out or "")).strip() or None
                                line_trace_out = trace_out
                                verify_trace: dict[str, object] = {}
                                get_trace = getattr(verifier, "get_last_verify_trace", None)
                                if callable(get_trace):
                                    payload = get_trace()
                                    if isinstance(payload, dict):
                                        verify_trace = dict(payload)
                                _append_shell_trace_row(trace_out, _shell_trace_row(verify_trace=verify_trace))
                                print(json.dumps(minimal, sort_keys=True))
                                continue

                            evidence_payload = row.get("evidence", row.get("evidence_json"))
                            if evidence_payload is None:
                                raise ValueError("evidence is required.")
                            evidence = _load_evidence_payload(evidence_payload)
                            overrides = row.get("overrides", {})
                            if not isinstance(overrides, dict):
                                overrides = {}
                            threshold = float(
                                overrides.get(
                                    "threshold_entailment",
                                    row.get(
                                        "threshold_entailment",
                                        base_policy.threshold_entailment,
                                    ),
                                )
                            )
                            max_contradiction = float(
                                overrides.get(
                                    "max_contradiction",
                                    row.get(
                                        "max_contradiction",
                                        base_policy.max_contradiction,
                                    ),
                                )
                            )
                            partial_allowed = bool(
                                overrides.get(
                                    "partial_allowed",
                                    row.get("partial_allowed", base_policy.partial_allowed),
                                )
                            )
                            unitizer_mode = str(
                                overrides.get(
                                    "unitizer_mode",
                                    row.get("unitizer_mode", "sentence"),
                                )
                            )
                            trace_out = str(
                                overrides.get(
                                    "trace_out",
                                    row.get("trace_out", args.trace_out or ""),
                                )
                            ).strip() or None
                            line_trace_out = trace_out
                            _ensure_trace_parent(trace_out)
                            policy = PolicyConfig(
                                threshold_entailment=threshold,
                                max_contradiction=max_contradiction,
                                partial_allowed=partial_allowed,
                            )
                            topk_per_unit = int(
                                overrides.get(
                                    "topk_per_unit",
                                    row.get("topk_per_unit", args.topk_per_unit),
                                )
                            )
                            max_pairs_total = overrides.get(
                                "max_pairs_total",
                                row.get("max_pairs_total", args.max_pairs_total),
                            )
                            max_evidence_per_request = overrides.get(
                                "max_evidence_per_request",
                                row.get("max_evidence_per_request", args.max_evidence_per_request),
                            )
                            max_batch_tokens = overrides.get(
                                "max_batch_tokens",
                                row.get("max_batch_tokens", args.max_batch_tokens),
                            )
                            evidence_max_chars = int(
                                overrides.get(
                                    "evidence_max_chars",
                                    row.get("evidence_max_chars", args.evidence_max_chars),
                                )
                            )
                            evidence_max_sentences = int(
                                overrides.get(
                                    "evidence_max_sentences",
                                    row.get("evidence_max_sentences", args.evidence_max_sentences),
                                )
                            )
                            nli_device = str(
                                overrides.get("device", row.get("device", args.device))
                            )
                            nli_dtype = str(
                                overrides.get("dtype", row.get("dtype", args.dtype))
                            )
                            t0 = time.perf_counter()
                            result = run_pipeline(
                                llm_summary_text=llm_summary,
                                evidence=evidence,
                                policy_config=policy,
                                unitizer_mode=unitizer_mode,
                                use_oss_nli=True,
                                verifier=verifier,
                                nli_model_name=args.model_name,
                                nli_device=nli_device,
                                nli_dtype=nli_dtype,
                                topk_per_unit=topk_per_unit,
                                max_pairs_total=max_pairs_total,
                                max_evidence_per_request=max_evidence_per_request,
                                max_batch_tokens=max_batch_tokens,
                                evidence_max_chars=evidence_max_chars,
                                evidence_max_sentences=evidence_max_sentences,
                                trace_out=trace_out,
                            )
                            _ = time.perf_counter() - t0
                            print(json.dumps(result, sort_keys=True))
                        except Exception as exc:
                            verify_trace: dict[str, object] = {}
                            get_trace = getattr(verifier, "get_last_verify_trace", None)
                            if callable(get_trace):
                                payload = get_trace()
                                if isinstance(payload, dict):
                                    verify_trace = dict(payload)
                            _append_shell_trace_row(line_trace_out, _shell_trace_row(verify_trace=verify_trace))
                            print(json.dumps({"error": str(exc)}))
                    return 0

                print("[EGA] Interactive shell starting. Loading verifier...")
                print("[EGA] Ready. Enter paths or 'exit' to quit.")
                while True:
                    summary_path = input("summary path> ").strip()
                    if summary_path.lower() in ("exit", "quit"):
                        break
                    if not summary_path:
                        print("[EGA] summary path required.")
                        continue
                    summary_file = Path(summary_path).expanduser()
                    if summary_file.is_dir():
                        print(f"[EGA ERROR] Expected file path, got directory: {summary_path}")
                        continue
                    if not summary_file.exists():
                        print(f"[EGA ERROR] File not found: {summary_path}")
                        continue

                    evidence_path = input("evidence path> ").strip()
                    if evidence_path.lower() in ("exit", "quit"):
                        break
                    if not evidence_path:
                        print("[EGA] evidence path required.")
                        continue
                    evidence_file = Path(evidence_path).expanduser()
                    if evidence_file.is_dir():
                        print(f"[EGA ERROR] Expected file path, got directory: {evidence_path}")
                        continue
                    if not evidence_file.exists():
                        print(f"[EGA ERROR] File not found: {evidence_path}")
                        continue

                    try:
                        t0 = time.perf_counter()
                        _ensure_trace_parent(args.trace_out)
                        result = run_pipeline(
                            llm_summary_text=_load_answer(summary_path),
                            evidence=_load_evidence(evidence_path),
                            policy_config=base_policy,
                            unitizer_mode=args.unitizer,
                            use_oss_nli=True,
                            verifier=verifier,
                            nli_model_name=args.model_name,
                            nli_device=args.device,
                            nli_dtype=args.dtype,
                            topk_per_unit=args.topk_per_unit,
                            max_pairs_total=args.max_pairs_total,
                            max_evidence_per_request=args.max_evidence_per_request,
                            max_batch_tokens=args.max_batch_tokens,
                            evidence_max_chars=args.evidence_max_chars,
                            evidence_max_sentences=args.evidence_max_sentences,
                            trace_out=args.trace_out,
                        )
                        t1 = time.perf_counter()
                        if isinstance(result.get("verified_text"), str):
                            result["verified_text"] = _strip_bom(result["verified_text"])
                        verified_extract = result.get("verified_extract")
                        if isinstance(verified_extract, list):
                            for row in verified_extract:
                                if isinstance(row, dict) and isinstance(row.get("text"), str):
                                    row["text"] = _strip_bom(row["text"])
                        stats = result.get("stats", {})
                        decision = result.get("decision", {})
                        kept = stats.get("kept_units")
                        dropped = stats.get("dropped_units")
                        refusal = decision.get("refusal")
                        print(
                            f"[EGA] request_seconds={t1 - t0:.6f} "
                            f"kept={kept} dropped={dropped} refusal={refusal}"
                        )
                        print(json.dumps(result, indent=2))
                    except Exception as e:
                        print(f"[EGA ERROR] {e}")

                return 0


        answer_text = _load_answer(args.answer_file)
        evidence = _load_evidence(args.evidence_file)
        unitizer_mode = "sentence" if args.unitizer == "sentence" else "markdown_bullet"
        candidate = unitize_answer(answer_text, mode=unitizer_mode)
        scores_provider = None
        if isinstance(args.scores_jsonl, str) and args.scores_jsonl.strip():
            scores_path = _resolve_existing_path(args.scores_jsonl)
            scores_provider = JsonlScoresProvider(path=str(scores_path))

        enforcer = Enforcer(
            verifier=OverlapVerifier(),
            scores_provider=scores_provider,
            config=PolicyConfig(
                threshold_entailment=args.threshold,
                partial_allowed=args.partial_allowed,
            ),
        )
        result = enforcer.enforce(candidate=candidate, evidence=evidence)
        if not args.emit_verified_only:
            result = EnforcementResult(
                final_text=result.final_text,
                kept_units=result.kept_units,
                dropped_units=result.dropped_units,
                refusal_message=result.refusal_message,
                decision=result.decision,
                scores=result.scores,
                verified_units=[],
                polished_units=result.polished_units,
                polish_status=result.polish_status,
                polish_fail_reasons=result.polish_fail_reasons,
                ega_schema_version=result.ega_schema_version,
            )
        print(to_json(result))
        return 0
    except Exception as exc:  # pragma: no cover - exercised by CLI IO error tests.
        if os.environ.get("EGA_DEBUG") == "1":
            raise
        print(f"[EGA ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
