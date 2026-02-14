"""Benchmark helpers for verifier + enforcer evaluation on JSONL datasets."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from ega.contract import PolicyConfig
from ega.enforcer import Enforcer
from ega.policy import DefaultPolicy
from ega.types import EvidenceItem, EvidenceSet, VerificationScore
from ega.unitization import unitize_answer
from ega.verifiers.nli_cross_encoder import NliCrossEncoderVerifier

CALIBRATION_THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80]
CALIBRATION_MAX_CONTRADICTIONS = [0.20, 0.25, 0.30, 0.35]


class _NliVerifierAdapter:
    def __init__(self, verifier: NliCrossEncoderVerifier) -> None:
        self._verifier = verifier
        self.model_name = verifier.model_name

    def verify(self, *, unit_text: str, unit_id: str, evidence: EvidenceSet) -> VerificationScore:
        score = self._verifier.verify_unit(unit_text, evidence)
        return VerificationScore(
            unit_id=unit_id,
            entailment=score.entailment,
            contradiction=score.contradiction,
            neutral=score.neutral,
            label=score.label,
            raw=dict(score.raw),
        )

    def verify_many(
        self,
        candidate: Any,
        evidence: EvidenceSet,
    ) -> list[VerificationScore]:
        scores = self._verifier.verify_many(candidate, evidence)
        return [
            VerificationScore(
                unit_id=score.unit_id,
                entailment=score.entailment,
                contradiction=score.contradiction,
                neutral=score.neutral,
                label=score.label,
                raw=dict(score.raw),
            )
            for score in scores
        ]


def _iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at line {line_no}: {exc.msg}.") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Malformed JSONL at line {line_no}: expected object.")
            yield line_no, payload


def _build_evidence_set(raw_items: Any) -> EvidenceSet:
    if not isinstance(raw_items, list):
        raise ValueError("'evidence' must be a list of objects with id/text.")
    items: list[EvidenceItem] = []
    for idx, raw in enumerate(raw_items):
        if not isinstance(raw, dict):
            raise ValueError(f"Evidence item at index {idx} must be an object.")
        if "id" not in raw or "text" not in raw:
            raise ValueError(f"Evidence item at index {idx} must include 'id' and 'text'.")
        metadata = raw.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"Evidence item at index {idx} has non-object metadata.")
        items.append(
            EvidenceItem(
                id=str(raw["id"]),
                text=str(raw["text"]),
                metadata=dict(metadata),
            )
        )
    return EvidenceSet(items=items)


def _build_policy_config(base: PolicyConfig, override: Any) -> PolicyConfig:
    if override is None:
        return base
    if not isinstance(override, dict):
        raise ValueError("'policy' must be an object when provided.")
    payload = asdict(base)
    for key in ("threshold_entailment", "max_contradiction", "partial_allowed"):
        if key in override:
            payload[key] = override[key]
    return PolicyConfig(
        threshold_entailment=float(payload["threshold_entailment"]),
        max_contradiction=float(payload["max_contradiction"]),
        partial_allowed=bool(payload["partial_allowed"]),
    )


def _unitizer_mode(raw_mode: Any) -> str:
    mode = "sentence" if raw_mode is None else str(raw_mode).strip().lower()
    if mode == "sentence":
        return "sentence"
    if mode == "bullets":
        return "markdown_bullet"
    raise ValueError(f"Unsupported unitizer: {raw_mode!r}. Expected 'sentence' or 'bullets'.")


def run_benchmark(
    *,
    data_path: str | Path,
    out_path: str | Path | None = None,
    model_name: str | None = None,
    policy_config: PolicyConfig | None = None,
    verifier: Any | None = None,
    use_example_policy_overrides: bool = True,
) -> dict[str, Any]:
    base_policy = policy_config or PolicyConfig()
    data_file = Path(data_path)

    active_verifier = verifier
    if active_verifier is None:
        active_verifier = _NliVerifierAdapter(NliCrossEncoderVerifier(model_name=model_name))

    total_units = 0
    kept_units = 0
    dropped_units = 0
    refusal_count = 0
    kept_entailments: list[float] = []
    n_examples = 0

    for example in _iter_jsonl(data_file):
        line_no, row = example
        if "id" not in row:
            raise ValueError(f"Example at line {line_no} must include 'id'.")
        if "answer" not in row:
            raise ValueError(f"Example at line {line_no} must include 'answer'.")
        if "evidence" not in row:
            raise ValueError(f"Example at line {line_no} must include 'evidence'.")

        candidate = unitize_answer(
            str(row["answer"]),
            mode=_unitizer_mode(row.get("unitizer")),
        )
        evidence = _build_evidence_set(row["evidence"])
        config = (
            _build_policy_config(base_policy, row.get("policy"))
            if use_example_policy_overrides
            else base_policy
        )

        result = Enforcer(
            verifier=active_verifier,
            policy=DefaultPolicy(),
            config=config,
        ).enforce(candidate=candidate, evidence=evidence)

        n_examples += 1
        total_units += len(candidate.units)
        kept_units += len(result.kept_units)
        dropped_units += len(result.dropped_units)
        refusal_count += int(result.decision.refusal)

        kept_ids = set(result.kept_units)
        kept_entailments.extend(
            score.entailment for score in result.scores if score.unit_id in kept_ids
        )

    keep_rate = (kept_units / total_units) if total_units else 0.0
    refusal_rate = (refusal_count / n_examples) if n_examples else 0.0
    avg_entailment_kept = (
        sum(kept_entailments) / len(kept_entailments) if kept_entailments else None
    )
    output = {
        "n_examples": n_examples,
        "total_units": total_units,
        "kept_units": kept_units,
        "dropped_units": dropped_units,
        "keep_rate": keep_rate,
        "refusal_rate": refusal_rate,
        "avg_entailment_kept": avg_entailment_kept,
        "policy_config": asdict(base_policy),
        "model_name": str(getattr(active_verifier, "model_name", model_name or "")),
    }

    if out_path is not None:
        Path(out_path).write_text(json.dumps(output, sort_keys=True), encoding="utf-8")
    return output


def _calibration_sort_key(row: dict[str, Any]) -> tuple[int, float, float, float, float]:
    policy = row["policy_config"]
    refusal_rate = float(row["refusal_rate"])
    keep_rate = float(row["keep_rate"])
    threshold = float(policy["threshold_entailment"])
    max_contradiction = float(policy["max_contradiction"])
    meets_target = int(refusal_rate <= 0.20)
    return (meets_target, -refusal_rate, keep_rate, threshold, -max_contradiction)


def _iter_benchmark_examples(
    data_file: Path,
) -> Iterable[tuple[list[Any], EvidenceSet]]:
    for line_no, row in _iter_jsonl(data_file):
        if "id" not in row:
            raise ValueError(f"Example at line {line_no} must include 'id'.")
        if "answer" not in row:
            raise ValueError(f"Example at line {line_no} must include 'answer'.")
        if "evidence" not in row:
            raise ValueError(f"Example at line {line_no} must include 'evidence'.")

        candidate = unitize_answer(
            str(row["answer"]),
            mode=_unitizer_mode(row.get("unitizer")),
        )
        evidence = _build_evidence_set(row["evidence"])
        yield candidate.units, evidence


def _precompute_calibration_scores(
    *,
    data_file: Path,
    active_verifier: Any,
) -> list[dict[str, Any]]:
    cached_examples: list[dict[str, Any]] = []
    for units, evidence in _iter_benchmark_examples(data_file):
        scores = [
            active_verifier.verify(unit_text=unit.text, unit_id=unit.id, evidence=evidence)
            for unit in units
        ]
        cached_examples.append({"units": units, "scores": scores})
    return cached_examples

def calibrate_policies(
    *,
    data_path: str | Path,
    model_name: str | None = None,
    verifier: Any | None = None,
    out_path: str | Path | None = None,
    topk: int = 5,
) -> dict[str, Any]:
    data_file = Path(data_path)
    active_verifier = verifier
    if active_verifier is None:
        active_verifier = _NliVerifierAdapter(NliCrossEncoderVerifier(model_name=model_name))

    cached_examples = _precompute_calibration_scores(
        data_file=data_file,
        active_verifier=active_verifier,
    )
    policy = DefaultPolicy()

    rows: list[dict[str, Any]] = []
    for threshold in CALIBRATION_THRESHOLDS:
        for max_contradiction in CALIBRATION_MAX_CONTRADICTIONS:
            policy_config = PolicyConfig(
                threshold_entailment=threshold,
                max_contradiction=max_contradiction,
                partial_allowed=True,
            )

            n_examples = 0
            total_units = 0
            kept_units = 0
            refusal_count = 0
            for cached in cached_examples:
                units = cached["units"]
                scores = cached["scores"]
                decision = policy.decide(scores=scores, units=units, config=policy_config)
                n_examples += 1
                total_units += len(units)
                kept_units += len(decision.allowed_units)
                refusal_count += int(decision.refusal)

            keep_rate = (kept_units / total_units) if total_units else 0.0
            refusal_rate = (refusal_count / n_examples) if n_examples else 0.0
            rows.append(
                {
                    "policy_config": asdict(policy_config),
                    "keep_rate": keep_rate,
                    "refusal_rate": refusal_rate,
                }
            )

    sorted_rows = sorted(rows, key=_calibration_sort_key, reverse=True)
    output = {
        "best_policy_config": sorted_rows[0]["policy_config"] if sorted_rows else None,
        "top_configs": sorted_rows[:topk],
    }
    if out_path is not None:
        Path(out_path).write_text(json.dumps(output, sort_keys=True), encoding="utf-8")
    return output


def load_policy_config(path: str | Path) -> PolicyConfig:
    """
    Loads a saved policy file.
    Accepts:
      - calibration artifact containing 'best_policy_config'
      - wrapper with 'policy_config'
      - bare policy dict
    """
    p = Path(path)
    obj = json.loads(p.read_text(encoding="utf-8-sig"))

    if isinstance(obj, dict) and isinstance(obj.get("best_policy_config"), dict):
        obj = obj["best_policy_config"]
    elif isinstance(obj, dict) and isinstance(obj.get("policy_config"), dict):
        obj = obj["policy_config"]

    if not isinstance(obj, dict):
        raise ValueError("Invalid policy file format.")

    return PolicyConfig(
        threshold_entailment=float(obj["threshold_entailment"]),
        max_contradiction=float(obj.get("max_contradiction", 0.2)),
        partial_allowed=bool(obj.get("partial_allowed", True)),
    )
