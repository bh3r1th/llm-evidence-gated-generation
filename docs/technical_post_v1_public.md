# I built a runtime enforcement layer for LLM outputs

## The problem

Eval tools score LLM outputs after the fact. A hallucination can score 0.3 on faithfulness, get logged, and still reach downstream. Nothing in that loop stops it. I wanted enforcement at runtime — before the output is emitted, not after it's already been used.

## The core mechanic

Model outputs are treated as candidates, not answers. Each claim is segmented and verified against source evidence independently. The result is one of three outcomes: accept, selectively correct, or abstain. Correction is bounded — a fixed retry limit per unit. Not unbounded retries hoping the model eventually gets it right.

The authority on each decision is explicit: conformal (in-distribution), conformal_oor (out-of-calibration-range), or threshold fallback. Every response carries that provenance.

## What I built through four internal versions

This was not a linear success story.

**Stage 1 — whole-output verification.** Verified the full LLM output as a single unit against the full evidence block. The obvious failure: compound claims. If one half of a sentence was supported and the other was hallucinated, the whole thing passed. Not useful.

**Stage 2 — claim-level verification.** Moved to sentence-level segmentation. Ran NLI independently per unit. NLI alone was too brittle on noisy RAG chunks — high abstain rates on legitimate outputs, erratic behavior on truncated evidence. Added selective abstention rather than hard rejection.

**Stage 3 — conformal gating.** Replaced raw NLI thresholds with conformal prediction calibrated on pilot data. More principled accept/abstain/reject boundaries. The new problem: calibration fit on pilot data means clean production inputs frequently fall outside the calibration range and hit OOR-high. The system is honest about this — it labels the authority as `conformal_oor` — but it's not the same as being calibrated for production.

**Stage 4 — explicit authority and output contracts.** Added a formal authority decision function, structured output schemas, async pending contract with `tracking_id`, and bounded correction gating by failure type. That's what ships publicly now.

## A concrete failure the current release still has

Consider this claim:

> "The company was founded in 2001 and has 500 employees."

That's two factual claims fused into one sentence. EGA segments at sentence boundaries. The whole sentence becomes one verification unit. If the source confirms the founding year but not the employee count, EGA may accept the unit anyway — because the NLI score reflects partial support from the founding year.

This is not a subtle edge case. It is the primary limitation of sentence-level segmentation. Semantic claim decomposition — splitting "founded in 2001" and "has 500 employees" into separate units before verification — is deferred. The public release ships with this known gap.

## What ships in v1.0.0

- NLI-based claim verification via DeBERTa-v3
- Conformal gating with calibrated accept/abstain/reject thresholds
- Explicit authority on every decision: `conformal`, `conformal_oor`, or `threshold`
- Strict output mode: full payload or rejection metadata, nothing partial
- Adapter output mode: accepted and rejected fields separated in a validation envelope
- `tracking_id` on every response
- `summarize_result()` for extracting operational signals for logging

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

print(result["payload_status"])   # ACCEPT or REJECT
print(result["tracking_id"])      # present on every response
```

## What I still don't know

Three things I genuinely don't have answers to yet:

**Abstain rate on real RAG pipelines.** Pilot calibration data is not production data. I don't know whether the abstain rate holds, increases, or collapses when evidence quality varies the way it does in real retrieval pipelines.

**Whether strict and adapter modes match actual downstream needs.** Both modes were designed from first principles. I don't know if the distinction maps to how real systems want to consume partially-verified output.

**Where structured field verification breaks first.** Structured mode routes fields through BM25 before verification. The routing logic is untested on real pipelines. I expect it breaks at nested fields, at fields with low lexical overlap to evidence, or at schemas with high cardinality. I don't know which failure mode appears first.

## Direct ask

This is a first public release, not a finished product.

I'm looking for engineers building RAG pipelines who will plug this in and tell me specifically where it breaks. Not looking for stars or feedback on the concept — looking for failure reports. A description of the input, the output EGA produced, and what was wrong with it is more useful than any other form of feedback right now.

GitHub: [https://github.com/bh3r1th/llm-evidence-gated-generation](https://github.com/bh3r1th/llm-evidence-gated-generation)

Contact: Bharath Nunepalli — bn3020@protonmail.com
