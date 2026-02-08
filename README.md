# ega (Evidence-Gated Answering)

`ega` is a Python package that provides an **enforcement and decision layer** for evidence-gated answering systems.

> Scope note: EGA is intentionally focused on policy enforcement, evidence checks, and final answer gating. It is **not** an evaluator suite or dashboard.

## Status

This repository currently contains project scaffolding and API placeholders.

## Installation

```bash
pip install -e .
pip install -e ".[nli]"
pip install -e ".[wandb]"
pip install -e ".[nli,wandb,dev]"
```

## Quickstart

```bash
ega --help
```

## Default verifier model

EGA's NLI cross-encoder verifier defaults to:

- `microsoft/deberta-v3-large-mnli`

Override it via constructor argument:

```python
from ega.verifiers.nli_cross_encoder import NliCrossEncoderVerifier

verifier = NliCrossEncoderVerifier(model_name="facebook/bart-large-mnli")
```

The verifier supports CPU execution (`device="cpu"` by default). Model/tokenizer files are
downloaded at runtime by Hugging Face transformers when not already present in cache.

## Running the benchmark

```bash
ega benchmark --data data/benchmark_example.jsonl
ega benchmark --data data/benchmark_example.jsonl --calibrate
```

The benchmark command prints aggregate JSON with metrics such as `keep_rate`,
`refusal_rate`, and `avg_entailment_kept`.

This benchmark output is operational telemetry only. It is **not** an evaluator or a
ground-truth quality metric.

## Optional integrations

### Weights & Biases sink

W&B support is optional and is not required for core runtime usage.

```bash
pip install -e .[wandb]
```

Then wire the adapter into `Enforcer` event emission:

```python
from ega.adapters import make_wandb_sink
from ega.enforcer import Enforcer

sink = make_wandb_sink(project="my-project", entity="my-team", tags=["ega"])
enforcer = Enforcer(verifier=my_verifier, event_sink=sink)
```

The sink lazily initializes a run on the first event, logs scalar metrics (for example
`kept_count`, `refusal`, and policy summary stats such as `mean_entailment`), and records
full event JSON in a lightweight W&B table.

## Development

```bash
make setup
make lint
make test
```

## License

Apache-2.0
