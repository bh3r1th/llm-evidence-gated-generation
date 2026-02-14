# ega (Evidence-Gated Answering)

EGA is a deterministic enforcement layer for evidence-gated summaries.
It splits summaries into units, verifies them against evidence, and emits keep/drop/refusal decisions.

## Quickstart

### Install
```bash
python -m pip install -e ".[dev,nli]"
```

### Basic run
```bash
ega run \
  --answer-file examples/pipeline_demo/llm_summary.txt \
  --evidence-file examples/pipeline_demo/evidence.json \
  --partial-allowed
```

### Pipeline (precomputed scores)
```bash
ega pipeline \
  --llm-summary-file examples/pipeline_demo/llm_summary.txt \
  --evidence-json examples/pipeline_demo/evidence.json \
  --scores-jsonl examples/pipeline_demo/scores.jsonl \
  --unitizer sentence \
  --partial-allowed
```

### Pipeline (OSS NLI)
```bash
ega pipeline \
  --llm-summary-file examples/pipeline_demo/llm_summary.txt \
  --evidence-json examples/pipeline_demo/evidence.json \
  --use-oss-nli \
  --unitizer sentence \
  --device auto \
  --dtype auto \
  --partial-allowed
```

### Shell (interactive)
```bash
ega shell --model-name MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
```

### Shell (JSONL)
```bash
ega shell --stdin-jsonl --stdout-jsonl --trace-out artifacts/shell_trace.jsonl
```

## Performance Baseline (v0.1-ega-poc)

### Environment

- Device: CPU
- dtype: float32
- Model: MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
- Units: 23
- Evidence items: 40

---

### Large Example (200 pair cap)

Cold run (pipeline CLI):
- total_seconds: 208.03
- verify_compute_seconds: 192.60
- n_pairs: 200

Warm run (shell, median of 6):
- total_seconds: 158.13
- verify_compute_seconds ≈ forward_seconds (compute dominated)

Speedup warm vs cold:
~24% reduction in end-to-end latency after model load.

---

### Pruned Configuration (topk=2, max_pairs=50)

- n_pairs: 46
- total_seconds: 95.48
- verify_compute_seconds: 81.87
- pairs_pruned_stage1: 874
- pairs_pruned_stage2: 0

Effect:
~54% latency reduction compared to 200-pair baseline.

---

### Observations

1. Forward pass dominates runtime (>90% of verify_seconds).
2. Model load cost is ~15s and disappears in warm runs.
3. Pair pruning materially reduces latency.
4. Trace schema captures compute, batching, token stats, and pruning metrics.

---

## Demos

- Pipeline demo data and script:
  - `examples/pipeline_demo/*`
  - `examples/run_pipeline_demo.sh`
- Larger synthetic perf fixtures:
  - `examples/large_summary.txt`
  - `examples/large_evidence.json`

## Tracing

Use `--trace-out` on `ega pipeline` or `ega shell` to append JSONL trace rows.

```bash
ega pipeline \
  --llm-summary-file examples/large_summary.txt \
  --evidence-json examples/large_evidence.json \
  --use-oss-nli \
  --trace-out artifacts/pipeline_trace.jsonl
```

Trace row schema is documented in `trace_schema.json`.

### Warm latency median (PowerShell)
```powershell
$rows = Get-Content artifacts\pipeline_trace.jsonl | ForEach-Object { $_ | ConvertFrom-Json }
$warm = $rows | Select-Object -ExpandProperty total_seconds | Sort-Object
$mid = [Math]::Floor($warm.Count / 2)
"median_total_seconds=$($warm[$mid])"
```

## Notes

- CPU vs GPU:
  - GPU can materially reduce verify latency for NLI workloads.
  - CPU timings vary by hardware and thread settings.
- Pruning knobs:
  - `--topk-per-unit`
  - `--max-pairs-total`
  - `--max-batch-tokens`
- `partial_allowed` behavior:
  - `true`: return supported subset when some units fail.
  - `false`: refuse when any unit fails policy.

## Development

```bash
make install-dev
make format
make lint
make test
```

Windows test wrapper:
```powershell
python scripts/pytest_wrapper.py -q
```

## Build

```bash
python -m pip install build
python -m build
```
