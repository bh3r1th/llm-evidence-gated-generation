from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from uuid import uuid4


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * q))
    idx = max(0, min(idx, len(ordered) - 1))
    return float(ordered[idx])


def _build_request_lines(*, n: int, summary: str, evidence: list[dict[str, object]]) -> str:
    row = {"llm_summary": summary, "evidence": evidence, "unitizer_mode": "sentence"}
    return "".join(json.dumps(row) + "\n" for _ in range(n))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run smoke perf gate against resident shell mode.")
    parser.add_argument("--n", type=int, default=5, help="Number of requests (default: 5).")
    parser.add_argument(
        "--p95-threshold-seconds",
        type=float,
        default=5.0,
        help="Fail if p95 request_seconds exceeds this threshold (default: 5.0).",
    )
    parser.add_argument(
        "--summary-file",
        default="examples/pipeline_demo/llm_summary.txt",
        help="Summary text file for repeated requests.",
    )
    parser.add_argument(
        "--evidence-json",
        default="examples/pipeline_demo/evidence.json",
        help="Evidence JSON file containing a list of {id,text,metadata?}.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Optional model override passed to shell.",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary_file)
    evidence_path = Path(args.evidence_json)
    summary = summary_path.read_text(encoding="utf-8-sig")
    evidence_obj = json.loads(evidence_path.read_text(encoding="utf-8-sig"))
    if not isinstance(evidence_obj, list):
        print("evidence-json must contain a JSON list.", file=sys.stderr)
        return 2

    trace_path = Path("artifacts") / f"perf_gate_trace_{uuid4().hex}.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    stdin_payload = _build_request_lines(n=args.n, summary=summary, evidence=evidence_obj)

    cmd = [
        sys.executable,
        "-m",
        "ega.cli",
        "shell",
        "--stdin-jsonl",
        "--stdout-jsonl",
        "--trace-out",
        str(trace_path),
    ]
    if args.model_name:
        cmd.extend(["--model-name", args.model_name])

    proc = subprocess.run(cmd, input=stdin_payload, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
        return proc.returncode

    out_lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if len(out_lines) != args.n:
        print(
            f"unexpected shell output lines: got={len(out_lines)} expected={args.n}",
            file=sys.stderr,
        )
        return 3

    request_seconds: list[float] = []
    for raw in trace_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        if not isinstance(row, dict):
            continue
        if "request_seconds" in row:
            request_seconds.append(float(row["request_seconds"]))
        elif "total_seconds" in row:
            request_seconds.append(float(row["total_seconds"]))

    if len(request_seconds) != args.n:
        print(
            f"unexpected trace rows with request_seconds: got={len(request_seconds)} expected={args.n}",
            file=sys.stderr,
        )
        return 4

    p50 = _percentile(request_seconds, 0.50)
    p95 = _percentile(request_seconds, 0.95)
    print(f"p50_seconds={p50:.6f}")
    print(f"p95_seconds={p95:.6f}")
    print(f"threshold_seconds={args.p95_threshold_seconds:.6f}")

    if p95 > args.p95_threshold_seconds:
        print("perf gate failed: p95 exceeded threshold", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
