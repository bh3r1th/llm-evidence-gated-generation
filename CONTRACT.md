# EGA Public Contract

## Schema Version
- Canonical schema field: `ega_schema_version`
- Current version: `"1"`
- Rule: any backward-incompatible change requires a version bump and, when possible, keeping parsers for older versions.

## Public Constants and Enums
- `EGA_SCHEMA_VERSION = "1"`
- `ReasonCode` values:
  - `OK_FULL`
  - `OK_PARTIAL`
  - `ALL_DROPPED`
  - `PARTIAL_NOT_ALLOWED`

## Public Dataclasses
- `PolicyConfig`
  - `threshold_entailment: float`
  - `max_contradiction: float`
  - `partial_allowed: bool`
- `Unit`
  - `id: str`
  - `text: str`
  - `metadata: dict[str, Any]`
  - `source_ids: Optional[list[str]]`
- `AnswerCandidate`
  - `raw_answer_text: str`
  - `units: list[Unit]`
- `EvidenceItem`
  - `id: str`
  - `text: str`
  - `metadata: dict[str, Any]`
- `EvidenceSet`
  - `items: list[EvidenceItem]`
- `VerificationScore`
  - `unit_id: str`
  - `entailment: float`
  - `contradiction: float`
  - `neutral: float`
  - `label: str`
  - `raw: dict[str, Any]`
- `GateDecision`
  - `allowed_units: list[str]`
  - `dropped_units: list[str]`
  - `refusal: bool`
  - `reason_code: str`
  - `summary_stats: dict[str, Any]`
- `EnforcementResult`
  - `final_text: Optional[str]`
  - `kept_units: list[str]`
  - `dropped_units: list[str]`
  - `refusal_message: Optional[str]`
  - `decision: GateDecision`
  - `scores: list[VerificationScore]`
  - `ega_schema_version: str`
