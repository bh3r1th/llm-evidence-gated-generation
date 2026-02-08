from __future__ import annotations

import argparse
import json

from ega.benchmark import calibrate_policies, run_benchmark
from ega.contract import PolicyConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark verifier/enforcer over JSONL data.")
    parser.add_argument("--data", required=True, help="Path to benchmark JSONL.")
    parser.add_argument("--out", default=None, help="Optional output JSON path.")
    parser.add_argument("--model-name", default=None, help="HF model id override.")
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run deterministic threshold calibration grid search.",
    )
    parser.add_argument(
        "--threshold-entailment",
        type=float,
        default=0.8,
        help="Entailment threshold for keeping a unit.",
    )
    parser.add_argument(
        "--max-contradiction",
        type=float,
        default=0.2,
        help="Max contradiction score for keeping a unit.",
    )
    parser.add_argument(
        "--partial-allowed",
        dest="partial_allowed",
        action="store_true",
        help="Allow partial answers (default).",
    )
    parser.add_argument(
        "--no-partial-allowed",
        dest="partial_allowed",
        action="store_false",
        help="Disallow partial answers.",
    )
    parser.set_defaults(partial_allowed=True)
    args = parser.parse_args()

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


if __name__ == "__main__":
    raise SystemExit(main())
