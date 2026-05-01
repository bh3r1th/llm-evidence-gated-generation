# EGA

EGA is not an eval tool. It is a runtime enforcement layer.

## What problem it solves

LLM outputs are candidates, not answers. EGA verifies each claim against source evidence at runtime before allowing output downstream. Eval tools score after the fact. EGA enforces before emit.

## Install

```bash
pip install ega
```

## Usage

```python
from ega import verify_answer
from ega.config import PipelineConfig, VerifierConfig
from ega.contract import PolicyConfig

config = PipelineConfig(
    policy=PolicyConfig(),
    verifier=VerifierConfig(use_oss_nli=True),
)

result = verify_answer(
    llm_output="The company was founded in 2001 and has 500 employees.",
    source_text="The company was founded in 2001. It currently employs 500 people.",
    config=config,
    return_pipeline_output=True,
)

print(result["payload_status"])
# ACCEPT or REJECT
```

## What you get back

Every response includes `payload_status`, per-unit audit records with `authority` and `decision`, and `tracking_id`. Distribution drift signal is available when calibration data is present. Use `summarize_result()` to extract operational signals for logging.

## Output modes

| Mode | Behavior |
|------|----------|
| `strict` | Full payload or rejection metadata — nothing partial |
| `adapter` | Validation envelope with accepted and rejected fields separated |

Set via `output_mode` in `PipelineConfig`.

## Current limitations

- Sentence-level segmentation, not semantic claim decomposition
- Calibration bootstrapped on pilot data, not production-grade
- Clean inputs may show `authority=conformal_oor` until recalibrated on production data
- Structured output BM25 routing untested on real pipelines
- Not benchmarked against RAGAS or TruLens

## CLI

```bash
ega run --answer-file answer.txt --evidence-file evidence.json
```

`answer.txt` — plain text LLM output. `evidence.json` — JSON array of `{id, text, metadata}` objects.

---

[Changelog](CHANGELOG.md) | [License](LICENSE) | [Issues](https://github.com/bh3r1th/llm-evidence-gated-generation/issues)
