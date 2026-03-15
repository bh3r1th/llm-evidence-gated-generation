# EGA v2 POC Results

- Dataset: `examples/v2/eval_dataset_pilot.jsonl`
- Examples: `10`
- Verifier model: `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`
- Reranker model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Accept threshold: `0.05`
- Conformal calibration: Conformal state regenerated from exported calibration rows under the current verifier semantics. (`runs/v2_compare/calibration/pilot_conformal_state.json`)

| Variant | Status | Kept | Dropped | Unsupported | Hallucination | Abstention | Gold Recall | Avg Reward | Verifier Calls | Verifier Cost | Reranker Cost | Cost | p50 s | p95 s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `v1_baseline` | recommended | 7 | 37 | 0.000 | 0.843 | 0.000 | 0.384 | -1.485 | 308 | 308 | 0 | 308 | 18.886 | 40.580 |
| `rerank_only` | recommended | 7 | 37 | 0.000 | 0.843 | 0.000 | 0.406 | -1.469 | 264 | 264 | 264 | 528 | 18.493 | 32.029 |
| `conformal_only` | recommended | 7 | 37 | 0.000 | 0.165 | 0.678 | 0.384 | -0.476 | 308 | 308 | 0 | 308 | 18.303 | 39.480 |
| `combined` | recommended | 7 | 37 | 0.000 | 0.165 | 0.678 | 0.406 | -0.459 | 264 | 264 | 264 | 528 | 18.357 | 31.441 |

## Takeaways

- The release POC is fixed to four headline variants: `v1_baseline`, `rerank_only`, `conformal_only`, and `combined`.
- `0.05` is the recommended operating threshold because it is the current safe point on the pilot artifact under the present verifier semantics.
- `rerank_only` reduces verifier calls modestly relative to `v1_baseline` while preserving the same verifier threshold.
- `conformal_only` changes answer behavior through abstention rather than verifier cost reduction.
- `combined` is the publishable composition of reranking and conformal abstention without budget claims.

## Safe-Answer Example

```text
The Eiffel Tower is in Paris. [e1]
It opened in 1889. [e3]
```

## Limitations

- The evaluation is still a small pilot dataset and should not be presented as a broad benchmark.
- The conformal threshold is specific to calibration rows exported under the current verifier semantics and should be regenerated if the verifier path changes.
- The reported cost fields are proxies based on scored verification and reranking pairs, not production billing measurements.
- Budget mode is experimental unless a binding budget run demonstrates real pair-count reduction with the expected tradeoff.

