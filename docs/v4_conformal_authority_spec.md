# Conformal Authority Decision Function — V4 Specification

**Audience:** Engineer implementing V4 who has not read the V3 codebase.
**Status:** Reference specification. No implementation details or code.

---

## 1. Authority Precedence

The decision function has two possible authorities: the **conformal gate** and the **accept_threshold gate**. Exactly one authority is authoritative for any given unit decision. When the conformal gate is authoritative, `accept_threshold` is not consulted at all, and vice versa.

**Conformal loaded, score inside calibration range.** Conformal is authoritative. The gate compares the raw entailment score against the calibrated threshold and the abstain band (a symmetric interval of width `band_width` centered on the threshold). If the score falls strictly above the upper band edge, the decision is *accept*. If the score falls within the band (inclusive on both edges), the decision is *abstain*. If the score falls strictly below the lower band edge, the decision is *reject*. The `accept_threshold` parameter has no effect on the outcome.

**Conformal loaded, score above calibration range maximum.** Conformal is authoritative. The decision is *accept*. A score above the maximum observed during calibration is further into the entailment region than any calibration example. The statistical guarantee that justifies the threshold still holds conservatively in this direction, and applying `accept_threshold` would introduce an inconsistency between the inferred and stated policy. See Section 3 for how the calibration range maximum is derived.

**Conformal loaded, score below calibration range minimum.** Conformal is authoritative. The decision is *reject*. A score below the calibration range minimum is further into the rejection region than any calibration example. No amount of threshold-tuning changes the interpretation: the score is out-of-distribution on the low side.

**Conformal loaded, calibration failed or state corrupt.** Conformal is not authoritative. The system must detect this condition before making a unit decision, not after. Concretely, if the `ConformalState` artifact is present but any required field (`threshold`, `band_width`, `score_mean`, `score_std`, `n_samples`) is missing, non-finite, or inconsistent (e.g., `n_samples < 50`), the conformal gate is treated as absent for the entire request. The decision falls back to `accept_threshold` for all units in that request. A single partially-valid state must not be used: the artifact is valid or invalid as a whole. The audit record must capture the failure reason and the fact that authority fell back.

**Conformal not loaded.** `accept_threshold` is authoritative. No conformal artifact was provided. The decision is *accept* when `entailment >= accept_threshold`, and *reject* otherwise. The conformal-related audit fields are set to null.

---

## 2. `accept_threshold` Fallback Behavior

`accept_threshold` is an explicit numeric threshold on the entailment score. It governs unit decisions only when conformal is not authoritative: either because no `ConformalState` was loaded, or because the state was detected as corrupt before evaluation began.

When `accept_threshold` is the authority, a unit is accepted if and only if its entailment score is greater than or equal to `accept_threshold`. A unit is rejected otherwise. There is no abstain outcome under `accept_threshold` authority. If `accept_threshold` is not explicitly supplied by the caller, it defaults to the value in the active `PolicyConfig` (`threshold_entailment`).

`accept_threshold` must not silently override a conformal decision. Specifically, if a valid `ConformalState` is loaded and the conformal gate reaches a decision, `accept_threshold` may not be applied in parallel or as a secondary check. Applying both would be inconsistent with the authority model and would produce non-reproducible outcomes if the two thresholds disagree.

When both authorities would produce a conflicting outcome for the same score — for example, conformal would reject a unit that `accept_threshold` would accept — the conflict is resolved by the authority model, not by majority vote or override. If conformal is authoritative, conformal wins. If conformal is absent or corrupt, `accept_threshold` wins. There is no condition under which both are applied to the same unit in the same request.

---

## 3. Out-of-Calibration-Range Policy

**How the calibration range is determined.** The OOR boundary is defined by `calibration_score_min` and `calibration_score_max` — the observed minimum and maximum entailment scores from the calibration row set, stored as required fields in `ConformalState`. A score is out-of-range if it is strictly less than `calibration_score_min` or strictly greater than `calibration_score_max`.

**Score above calibration range maximum.** The decision is *accept*. This score is higher than any entailment score observed in calibration. Because the conformal threshold is derived from the quantile of calibration scores, a score this high is unambiguously on the accept side of the distribution. Abstaining would introduce unnecessary rejections for the strongest-evidence units. Falling back to `accept_threshold` would undermine calibration-based authority for exactly the scores where evidence is clearest. Auto-accept is therefore the correct policy. This outcome is recorded with reason code `CONFORMAL_OOR_HIGH`.

**Score below calibration range minimum.** The decision is *reject*. This score is lower than any entailment score observed in calibration. The conformal threshold does not extend statistical guarantees to this region, but the directional interpretation is unambiguous: the score is further into the rejection region than the lowest calibration example. Auto-reject is the correct policy. Neither abstain nor fallback to `accept_threshold` is appropriate because `accept_threshold` would produce an inconsistent decision (accepting a unit whose score is out-of-distribution on the low side). This outcome is recorded with reason code `CONFORMAL_OOR_LOW`.

---

## 4. Required Audit Fields on Every Unit Decision

Every unit decision record must carry the following fields, regardless of which authority made the decision. Fields that are not applicable to the current authority must be explicitly set to null, not omitted.

**`authority`** — A string identifying which authority made the final decision. Must be one of `conformal`, `threshold`, or `fallback`. The value `conformal` indicates the conformal gate was loaded and valid and its decision is used. The value `threshold` indicates no conformal state was loaded and `accept_threshold` governs. The value `fallback` indicates a conformal state was present but detected as corrupt, so `accept_threshold` was used as a fallback.

**`raw_score`** — The raw entailment score as produced by the verifier, before any clipping or transformation. This is a float in the range `[0.0, 1.0]` for well-behaved verifiers, but must be recorded as-is even if the verifier emits a value outside that range.

**`conformal_decision`** — The conformal authority decision actually applied to the unit, including any out-of-range override. If the score was above `calibration_range` maximum and the OOR policy auto-accepted, `conformal_decision = "accept"`. If the score was below `calibration_range` minimum and the OOR policy auto-rejected, `conformal_decision = "reject"`. The field never reflects a pre-override intermediate state. Must be one of `accept`, `reject`, or `abstain` if authority is `conformal`, or null if authority is `threshold` or `fallback`. The `reason_code` field distinguishes in-range decisions from OOR decisions.

**`final_decision`** — The decision that was actually enforced for this unit. Must be one of `accept`, `reject`, or `abstain`.

**`reason_code`** — A string identifying the specific rule that produced the final decision. Required values are defined below. No other values are permitted.

| `reason_code` | Meaning |
|---|---|
| `CONFORMAL_ACCEPT` | Conformal gate, score above threshold + band, inside calibration range |
| `CONFORMAL_ABSTAIN` | Conformal gate, score within abstain band |
| `CONFORMAL_REJECT` | Conformal gate, score below threshold − band, inside calibration range |
| `CONFORMAL_OOR_HIGH` | Conformal gate, score above calibration range maximum, auto-accepted |
| `CONFORMAL_OOR_LOW` | Conformal gate, score below calibration range minimum, auto-rejected |
| `THRESHOLD_ACCEPT` | `accept_threshold` authority, score ≥ threshold |
| `THRESHOLD_REJECT` | `accept_threshold` authority, score < threshold |
| `FALLBACK_ACCEPT` | Fallback authority (corrupt state), score ≥ threshold |
| `FALLBACK_REJECT` | Fallback authority (corrupt state), score < threshold |

**`conformal_threshold`** — The `threshold` value from the `ConformalState` artifact at decision time. Null if authority is `threshold`.

**`conformal_band_width`** — The `band_width` value from the `ConformalState` artifact. Null if authority is `threshold`.

**`calibration_range`** — A two-element array `[min, max]` representing the calibration range as computed in Section 3. Null if authority is `threshold`.

**`fallback_reason`** — A human-readable string describing why conformal authority fell back. Null unless authority is `fallback`. Examples: `"n_samples below minimum"`, `"non-finite threshold"`, `"missing field: score_std"`.
