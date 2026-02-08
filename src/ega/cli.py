"""CLI entrypoint for EGA tooling."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from string import punctuation

from ega import __version__
from ega.benchmark import calibrate_policies, run_benchmark
from ega.contract import PolicyConfig
from ega.enforcer import Enforcer
from ega.serialization import to_json
from ega.types import EvidenceItem, EvidenceSet, VerificationScore
from ega.unitization import unitize_answer


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
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)

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
                text=str(item["text"]),
                metadata=metadata,
            )
        )

    return EvidenceSet(items=items)


def _load_answer(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


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
    return parser


def main() -> int:
    """Run the CLI."""
    parser = build_parser()
    args = parser.parse_args()

    if args.version:
        print(__version__)
        return 0

    if args.command != "run":
        if args.command == "benchmark":
            if args.calibrate:
                calibration = calibrate_policies(
                    data_path=args.data,
                    model_name=args.model_name,
                    out_path=args.out,
                )
                print(json.dumps(calibration, sort_keys=True))
                return 0
            summary = run_benchmark(
                data_path=args.data,
                out_path=args.out,
                model_name=args.model_name,
                policy_config=PolicyConfig(
                    threshold_entailment=args.threshold_entailment,
                    max_contradiction=args.max_contradiction,
                    partial_allowed=args.partial_allowed,
                ),
            )
            print(json.dumps(summary, sort_keys=True))
            return 0
        parser.print_help()
        return 0

    answer_text = _load_answer(args.answer_file)
    evidence = _load_evidence(args.evidence_file)
    unitizer_mode = "sentence" if args.unitizer == "sentence" else "markdown_bullet"
    candidate = unitize_answer(answer_text, mode=unitizer_mode)

    enforcer = Enforcer(
        verifier=OverlapVerifier(),
        config=PolicyConfig(
            threshold_entailment=args.threshold,
            partial_allowed=args.partial_allowed,
        ),
    )
    result = enforcer.enforce(candidate=candidate, evidence=evidence)
    print(to_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
