from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * q))
    idx = max(0, min(idx, len(ordered) - 1))
    return float(ordered[idx])


def _run_cmd(cmd: list[str], *, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Perf smoke for EGA pipeline and shell JSONL mode.")
    parser.add_argument("--llm-summary-file", required=True)
    parser.add_argument("--evidence-json", required=True)
    parser.add_argument("--scores-jsonl", default=None)
    parser.add_argument("--use-oss-nli", action="store_true")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--n", type=int, default=20, help="Number of warm requests.")
    parser.add_argument("--trace-out", default="artifacts/perf_smoke_trace.jsonl")
    args = parser.parse_args()

    if not args.scores_jsonl and not args.use_oss_nli:
        parser.error("Provide either --scores-jsonl or --use-oss-nli.")

    trace_path = Path(args.trace_out)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text("", encoding="utf-8")

    cold_cmd = [
        sys.executable,
        "-m",
        "ega.cli",
        "pipeline",
        "--llm-summary-file",
        args.llm_summary_file,
        "--evidence-json",
        args.evidence_json,
        "--trace-out",
        str(trace_path),
    ]
    if args.scores_jsonl:
        cold_cmd.extend(["--scores-jsonl", args.scores_jsonl])
    else:
        cold_cmd.append("--use-oss-nli")
        if args.model_name:
            cold_cmd.extend(["--nli-model-name", args.model_name])

    cold_proc = _run_cmd(cold_cmd)
    if cold_proc.returncode != 0:
        print(cold_proc.stderr, file=sys.stderr)
        return cold_proc.returncode
    trace_lines = [line for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    cold_trace = json.loads(trace_lines[-1]) if trace_lines else {}
    print("cold_trace:")
    print(json.dumps(cold_trace, indent=2, sort_keys=True))

    summary_text = Path(args.llm_summary_file).read_text(encoding="utf-8-sig")
    request = {"llm_summary": summary_text, "evidence_json": args.evidence_json}
    stdin_payload = "\n".join(json.dumps(request) for _ in range(args.n)) + "\n"
    warm_cmd = [
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
        warm_cmd.extend(["--model-name", args.model_name])

    warm_proc = _run_cmd(warm_cmd, input_text=stdin_payload)
    if warm_proc.returncode != 0:
        print(warm_proc.stderr, file=sys.stderr)
        return warm_proc.returncode

    out_lines = [line for line in warm_proc.stdout.splitlines() if line.strip()]
    if len(out_lines) != args.n:
        print(
            f"unexpected number of shell outputs: got {len(out_lines)}, expected {args.n}",
            file=sys.stderr,
        )
        return 1

    req_secs: list[float] = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, dict) and "request_seconds" in row:
            req_secs.append(float(row["request_seconds"]))

    print(f"warm_requests={len(req_secs)}")
    print(f"request_seconds_p50={_pct(req_secs, 0.50):.6f}")
    print(f"request_seconds_p95={_pct(req_secs, 0.95):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
