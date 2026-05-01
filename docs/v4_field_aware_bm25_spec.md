# Field-Aware BM25 Evidence Routing — V4 Specification

**Audience:** Engineer implementing the BM25 routing change in V4.
**Approach:** Option B — field name + JSONPath path terms + field value as BM25 query terms.
**No learned router. No evidence pre-annotation. No semantic routing.**

---

## 1. Query Construction

### Rule

For a structured field unit, the BM25 query string is constructed by concatenating three components in order, separated by single spaces:

1. **Path terms** — the human-readable tokens extracted from the JSONPath field path, excluding the root `$` symbol and path delimiters (`.`, `[`, `]`, `"`). Array index integers are excluded. Only alphabetic and alphanumeric key segments are included.
2. **Field name** — the `field_name` value from `unit.metadata`, as-is (no transformation at this stage; the existing BM25 tokenizer lowercases during tokenization).
3. **Type-expanded value terms** — the field value string, expanded according to the field type rules defined in Section 2.

The resulting query string is passed to the existing `_bm25_tokenize` function unchanged. No modifications to the tokenizer are required.

When a unit does not have `structured_mode: True` in its metadata, the query string is `unit.text` with no modification. This preserves the existing behavior for all unstructured modes (sentence, markdown bullet, spaCy sentence).

### Formal construction

Given:
- `field_path`: the full JSONPath string from `unit.metadata["path"]`
- `field_name`: the leaf key from `unit.metadata["field_name"]`
- `field_value`: the string value from `unit.text`
- `field_type`: the type tag from `unit.metadata.get("field_type")`, defaulting to `"string"` if absent

The query string is:

```
query = path_terms(field_path) + " " + field_name + " " + expand_value(field_value, field_type)
```

Where `path_terms` extracts all dot-separated and bracket-enclosed alphabetic segments from the path (see examples below), and `expand_value` produces a type-specific token string (see Section 2).

### Worked examples

---

**Example 1 — string field**

```
field_path:  $.product.description
field_name:  description
field_value: "Lightweight aluminum frame"
field_type:  string
```

Path terms: `product description` (segments from `$.product.description`, root `$` excluded, leaf key included)

Expanded value: `Lightweight aluminum frame` (no expansion for strings)

**Query string:** `product description description Lightweight aluminum frame`

The duplication of `description` (once from path terms, once from field name) is intentional and harmless. BM25 term frequency handles it without special treatment, and removing it would require a deduplication step with no measurable benefit.

---

**Example 2 — number field**

```
field_path:  $.product.price
field_name:  price
field_value: "499.99"
field_type:  number
```

Path terms: `product price`

Expanded value: `499.99 499` (full decimal string + integer floor string; see Section 2 for the expansion rule)

**Query string:** `product price price 499.99 499`

---

**Example 3 — date field**

```
field_path:  $.order.shipped_at
field_name:  shipped_at
field_value: "2024-01-15"
field_type:  date
```

Path terms: `order shipped_at`

Expanded value: `2024-01-15 2024 January 15` (ISO string + year + month name + day; see Section 2)

**Query string:** `order shipped_at shipped_at 2024-01-15 2024 January 15`

---

**Example 4 — nested array path**

```
field_path:  $.lineItems[0].unitCost
field_name:  unitCost
field_value: "89.50"
field_type:  number
```

Path terms from `$.lineItems[0].unitCost`: `lineItems unitCost` (the `[0]` array index integer is excluded; only alphabetic/alphanumeric key segments are kept)

Expanded value: `89.50 89`

**Query string:** `lineItems unitCost unitCost 89.50 89`

---

## 2. Type Metadata Behavior

Type metadata affects only the **value expansion step** of query construction. It does not affect BM25 index construction, evidence pre-filtering, topk limits, or the NLI scoring stage.

### `string` (default)

The field value string is used as-is. No expansion. If `field_type` is absent or set to any unrecognized value, `string` behavior applies.

**Expansion rule:** `expand_value(v, "string") = v`

### `number`

Evidence text frequently references numeric values using the integer portion only (e.g., "$499" rather than "$499.99"), or uses a rounded form. To improve recall, two tokens are produced:

1. The full decimal string representation (the `field_value` as stored)
2. The integer floor as a string: `str(int(float(field_value)))`, omitting the decimal part

If parsing `field_value` as a float fails, fall back to `string` behavior and emit only the raw value string.

Negative numbers retain their sign in both tokens (e.g., `-12.5` → `-12.5 -12`).

**What numeric expansion does NOT do in V4:**
- No range terms (e.g., no `499 500 501` neighborhood expansion)
- No unit-of-measure inference (no `$`, `USD`, `dollars` added automatically)
- No proximity scoring or boosting
- No learned numeric similarity

**Expansion rule:** `expand_value(v, "number") = v + " " + str(int(float(v)))`
(with fallback to raw `v` if float conversion fails)

### `date`

ISO 8601 date strings of the form `YYYY-MM-DD` are expanded to four tokens: the full ISO string, the four-digit year, the full English month name, and the zero-padded day as a string. Evidence may reference dates in any of these surface forms (e.g., "January 2024", "Q1 2024", "the 15th").

Month names are the standard English full names: January, February, March, April, May, June, July, August, September, October, November, December.

If the `field_value` does not match the `YYYY-MM-DD` pattern — for example, a partial date like `2024-Q1`, a timestamp like `2024-01-15T10:00:00Z`, or a free-form date like `Jan 15` — fall back to `string` behavior and emit only the raw value string. Timestamp parsing beyond the date portion is explicitly out of scope for V4.

**What date expansion does NOT do in V4:**
- No relative time terms (no "yesterday", "last month", "recently")
- No calendar quarter inference (no `Q1` from January–March dates)
- No temporal proximity expansion
- No learned date normalization

**Expansion rule for `YYYY-MM-DD`:**
`expand_value(v, "date") = v + " " + year + " " + month_name + " " + day`
(with fallback to raw `v` if the pattern does not match)

### What type metadata does NOT do in V4

Type metadata is a query construction hint only. It has no effect on:
- The BM25 index (evidence is indexed from its raw text in all cases)
- BM25 scoring weights (no type-specific IDF boosting)
- Candidate ranking or topk selection beyond what results naturally from BM25 scores
- Evidence filtering by type (evidence items are not tagged with field type)
- NLI pair scoring or aggregation
- The conformal gate or accept_threshold decisions

---

## 3. Integration Point with Existing BM25 Preselection

### Where field-aware query construction plugs in

The only change is in `_build_stage1_candidates` (`nli_cross_encoder.py`, line 354–357), specifically the line that produces `query_tokens` for each unit:

**Current behavior:**
```
query_tokens = self._bm25_tokenize(unit.text)
```

**V4 behavior:**
```
query_string = _build_field_query(unit)   # new function; see construction rule in Section 1
query_tokens = self._bm25_tokenize(query_string)
```

`_build_field_query` inspects `unit.metadata.get("structured_mode")`. If `True`, it constructs the field-aware query string per Section 1. If `False` or absent, it returns `unit.text` unchanged. The existing `_bm25_tokenize` (whitespace tokenize + lowercase) is called in both cases.

The `_fallback_scores` path (used when `rank_bm25` is unavailable, line 360) receives the same `query_string` rather than `unit.text`, applying identical logic.

### What does not change

- **`topk_per_unit`** — unchanged. The field-aware query changes which candidates score highest; it does not increase the number of candidates selected per unit.
- **`max_pairs_total`** — unchanged. The global pair budget applies after per-unit topk selection, as before.
- **Evidence truncation** — `_truncate_evidence_text`, `evidence_max_chars`, and `evidence_max_sentences` are unchanged.
- **NLI pair construction** — the pair passed to the cross-encoder is still `(unit.text, evidence_text)`, not `(query_string, evidence_text)`. The field-aware query only affects which evidence items are selected for scoring; the NLI model always sees the original unit text.
- **BM25 index construction** — evidence is indexed from its raw (truncated) text. No annotation or field tagging of evidence is required.
- **Score aggregation** — `max_entailment` aggregation over selected pairs is unchanged.
- **The `verify_unit` and `_verify_unit_with_id` paths** — these single-unit paths bypass `_build_stage1_candidates` and score all evidence directly. They are not affected by this change.

### How unstructured units are handled (no regression)

Units produced by `SentenceUnitizer`, `MarkdownBulletUnitizer`, or `SpaCySentenceUnitizer` carry `metadata` that does not contain `structured_mode: True`. For these units, `_build_field_query` returns `unit.text` without modification, and the downstream BM25 call is byte-for-byte identical to the existing behavior. No regression is possible on the unstructured path.

---

## 4. Failure and Fallback Behavior

### Candidate count after field-aware BM25 preselection

After `_build_stage1_candidates` runs with the field-aware query, two undercoverage situations can occur:

**Case A: Zero candidates for a unit.** This happens when no evidence item shares any token with the field-aware query string. The existing behavior in `verify_many` already handles zero pairs per unit: it emits a `VerificationScore` with `entailment=0.0`, `contradiction=0.0`, `neutral=1.0`, `label="neutral"`, and `chosen_evidence_id=None` (lines 718–735). No change to this handling is needed. The unit proceeds through the conformal gate and will be assigned failure class `MISSING_IN_SOURCE` by `_normalize_unit_decisions`.

**Case B: Fewer than `topk_per_unit` candidates.** BM25 returns however many evidence items scored above zero. If the query is specific enough that only two of twelve target candidates are retrieved, those two proceed to NLI scoring. This is expected and acceptable; the field-aware query improves precision at the potential cost of recall.

### Value-only fallback

When a field-aware query produces zero candidates for a structured unit, the implementation must retry with a value-only query (`unit.text` only) before accepting zero candidates. The retry uses the same `_bm25_tokenize` and the same BM25 index; no re-indexing occurs.

The retry is attempted exactly once per unit per request. If the value-only query also produces zero candidates, the unit is scored with zero pairs (Case A above). The fallback attempt must be recorded in the pipeline trace as `field_query_fallback: true` for the affected unit. `field_query_fallback` is trace metadata only and does not appear in any public response schema (`StrictAcceptedResponse`, `StrictRejectedResponse`, `AdapterEnvelope`, or `PendingResponse`). See the "Trace Metadata vs Response Schema" section in Doc 2 (V4 Output Schema Reference).

**The fallback is value-only, not abstain.** Zero candidates after both attempts maps to the existing zero-pairs handling, which produces `MISSING_IN_SOURCE`. The conformal or threshold gate still makes a decision on the resulting score. There is no new abstain outcome introduced by field-aware routing.

### No global fallback

If the BM25 index itself is unavailable (i.e., `rank_bm25` is not installed), the existing `_fallback_scores` word-overlap path is used with the field-aware query string, not with `unit.text`. The fallback scoring path and the primary BM25 path receive the same query. There is no secondary fallback that reverts all queries to `unit.text` globally; that would negate the entire feature for installations without `rank_bm25`.

---

## 5. Known Limitations

The following problems are explicitly not solved by this approach. They are documented here so V5 has a clear starting point.

### Evidence not tagged by field

The BM25 index is built from raw evidence text. Evidence items carry no field-type annotation and no knowledge of which structured fields they are relevant to. The query-side expansion (adding `price`, `product price`) increases the probability that field-relevant evidence scores higher, but there is no guarantee. Evidence that describes a price without using the word "price" will not be reliably ranked higher by this approach.

**V5 starting point:** Evidence-side annotation with field tags, or a learned bi-encoder that embeds field name and value jointly with evidence.

### Cross-field dependencies

Some claims can only be verified by combining information across multiple fields. For example, "the discounted price of 449.99 is 10% below the list price of 499.99" requires evidence that mentions both prices and their relationship. Field-aware BM25 routes each unit to evidence independently; it cannot construct a compound query across two fields.

**V5 starting point:** Multi-field unit representation that merges related fields into a single NLI query.

### Semantic routing

BM25 is a lexical retrieval method. A field value of "myocardial infarction" will not retrieve evidence that says "heart attack" unless those terms co-occur in the evidence text. The field-aware query adds `field_name` terms but does not close the lexical gap for synonym-heavy domains.

**V5 starting point:** Dense retrieval (bi-encoder) or hybrid BM25 + dense retrieval, with a learned router aware of field semantics.

### Numeric proximity and range

A price of 499.99 and evidence that says "between $450 and $550" share no BM25 tokens. The integer floor expansion adds `499` to the query, which improves coverage for exact matches but does not address range-expressed evidence. No numeric proximity expansion is implemented in V4.

**V5 starting point:** Numeric normalization of evidence at index time, or a dedicated numeric entailment head on the NLI model.

### Date granularity and partial dates

The V4 date expansion handles only `YYYY-MM-DD`. Timestamps, partial dates, fiscal dates, and relative dates are all passed through as string fields. Evidence that uses relative dates ("last quarter", "three months ago") will not be retrieved reliably.

**V5 starting point:** Temporal normalization layer that resolves relative and partial dates to absolute `YYYY-MM-DD` before indexing and query construction.
