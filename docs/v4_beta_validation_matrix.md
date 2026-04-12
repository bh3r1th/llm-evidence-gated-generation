# EGA v4 Beta Validation Matrix

Date: 2026-04-12

| scenario | expected behavior | actual behavior | status | issue / notes |
|---|---|---|---|---|
| legacy text | Legacy text flow remains backward-compatible; no structured/adapter-only leakage. | _TBD during beta runs._ | OPEN | Run baseline legacy fixtures before stable cut. |
| structured strict | Strict mode returns only valid accepted payloads; non-accept does not emit completed business payload. | _TBD during beta runs._ | OPEN | Blocker if any reject/repair emits completed payload. |
| structured adapter | Adapter mode emits accepted subset only; rejected units stay in rejection metadata. | _TBD during beta runs._ | OPEN | Blocker if rejected content appears in adapter payload. |
| malformed payload | Malformed structured payload is bounded (no crash) with deterministic safe reject/pending outcome. | _TBD during beta runs._ | OPEN | Capture stack traces and input samples for any crash. |
| empty payload | Empty structured payload is handled without crash and with consistent status/action semantics. | _TBD during beta runs._ | OPEN | Verify `{}` and `[]` both covered. |
| large payload | Large payload completes within agreed beta SLO and preserves classification/payload safety semantics. | _TBD during beta runs._ | OPEN | Record payload size, latency, and memory notes. |
| contradictory evidence | Contradictory evidence is classified safely (no false completion) and surfaced in rationale/metadata. | _TBD during beta runs._ | OPEN | Sample real contradictory cases; flag misclassification patterns. |
| repeated-run determinism | Same input/config yields identical status/action/classification and payload output across reruns. | _TBD during beta runs._ | OPEN | Minimum 3 repeated runs per sampled case. |
