"""Handlers for legacy `ega run` command."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from string import punctuation

from ega.benchmark import PolicyConfig
from ega.enforcer import Enforcer
from ega.providers.jsonl_scores import JsonlScoresProvider
from ega.serialization import to_json
from ega.text_clean import clean_text
from ega.types import EnforcementResult, EvidenceItem, EvidenceSet, VerificationScore
from ega.unitization import unitize_answer


@dataclass(slots=True)
class OverlapVerifier:
    """Deterministic lexical-overlap verifier for CLI usage."""

    name: str = "lexical_overlap"

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        translator = str.maketrans("", "", punctuation + "`")
        return {
            token.lower().translate(translator)
            for token in text.split()
            if token.strip()
        }

    def verify(
        self,
        *,
        unit_text: str,
        unit_id: str,
        evidence: EvidenceSet,
    ) -> VerificationScore:
        unit_tokens = self._tokenize(unit_text)
        if not unit_tokens or not evidence.items:
            return VerificationScore(
                unit_id=unit_id,
                entailment=0.0,
                contradiction=0.0,
                neutral=1.0,
                label="neutral",
                raw={"verifier": self.name, "best_evidence_id": None},
            )

        best_overlap = 0.0
        best_evidence_id: str | None = None
        for item in evidence.items:
            evidence_tokens = self._tokenize(item.text)
            overlap = len(unit_tokens & evidence_tokens) / len(unit_tokens)
            if overlap > best_overlap:
                best_overlap = overlap
                best_evidence_id = item.id

        entailment = round(best_overlap, 6)
        contradiction = round(1.0 - entailment, 6)
        neutral = 0.0
        label = "entailment" if entailment >= 0.5 else "contradiction"
        return VerificationScore(
            unit_id=unit_id,
            entailment=entailment,
            contradiction=contradiction,
            neutral=neutral,
            label=label,
            raw={"verifier": self.name, "best_evidence_id": best_evidence_id},
        )


def _resolve_path(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    return p.resolve()


def _resolve_existing_path(path_str: str) -> Path:
    resolved_path = _resolve_path(path_str)
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"File not found: {resolved_path}. Current working directory: {Path.cwd()}"
        )
    return resolved_path


def _load_evidence(path: str) -> EvidenceSet:
    resolved_path = _resolve_existing_path(path)
    try:
        with resolved_path.open(encoding="utf-8-sig") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in file {resolved_path}: {exc.msg}.") from exc

    if not isinstance(payload, list):
        raise ValueError("Evidence file must be a JSON list of {id,text,metadata} objects.")

    items: list[EvidenceItem] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Evidence item at index {idx} must be an object.")
        if "id" not in item or "text" not in item:
            raise ValueError(
                f"Evidence item at index {idx} must include 'id' and 'text' fields."
            )

        metadata = item.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"Evidence item at index {idx} has non-object metadata.")

        items.append(
            EvidenceItem(
                id=str(item["id"]),
                text=clean_text(str(item["text"])),
                metadata=metadata,
            )
        )

    return EvidenceSet(items=items)


def _load_answer(path: str) -> str:
    resolved_path = _resolve_existing_path(path)
    with resolved_path.open(encoding="utf-8") as handle:
        return clean_text(handle.read())


def handle_run(args: object) -> int:
    answer_text = _load_answer(args.answer_file)
    evidence = _load_evidence(args.evidence_file)
    unitizer_mode = "sentence" if args.unitizer == "sentence" else "markdown_bullet"
    candidate = unitize_answer(answer_text, mode=unitizer_mode)
    scores_provider = None
    if isinstance(args.scores_jsonl, str) and args.scores_jsonl.strip():
        scores_path = _resolve_existing_path(args.scores_jsonl)
        scores_provider = JsonlScoresProvider(path=str(scores_path))

    enforcer = Enforcer(
        verifier=OverlapVerifier(),
        scores_provider=scores_provider,
        config=PolicyConfig(
            threshold_entailment=args.threshold,
            partial_allowed=args.partial_allowed,
        ),
    )
    result = enforcer.enforce(candidate=candidate, evidence=evidence)
    if not args.emit_verified_only:
        result = EnforcementResult(
            final_text=result.final_text,
            kept_units=result.kept_units,
            dropped_units=result.dropped_units,
            refusal_message=result.refusal_message,
            decision=result.decision,
            scores=result.scores,
            verified_units=[],
            polished_units=result.polished_units,
            polish_status=result.polish_status,
            polish_fail_reasons=result.polish_fail_reasons,
            ega_schema_version=result.ega_schema_version,
        )
    print(to_json(result))
    return 0
