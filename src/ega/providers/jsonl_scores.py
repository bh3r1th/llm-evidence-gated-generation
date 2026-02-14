"""JSONL score provider for precomputed per-unit verifier outputs.

Each JSONL row is expected to include:
`{"query_id":"...", "unit_id":"u0001", "score":0.83, "label":"pass", "raw":{...}}`

Scalar mapping:
- If `score` is provided, it is copied to `raw["score"]`.
- `entailment = score`, `contradiction = 0.0`, `neutral = 1.0 - score`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ega.types import AnswerCandidate, EvidenceSet, VerificationScore


class JsonlScoresProvider:
    """Load per-unit verification scores from JSONL rows."""

    def __init__(self, *, path: str | Path, query_id: str | None = None) -> None:
        self._path = Path(path)
        self._query_id = query_id

    def load_scores(
        self,
        *,
        candidate: AnswerCandidate,
        evidence: EvidenceSet,
    ) -> list[VerificationScore]:
        _ = (candidate, evidence)
        rows = self._read_rows()
        return [self._to_score(row) for row in rows]

    def _read_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with self._path.open("r", encoding="utf-8-sig") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Malformed JSONL score at line {line_no}: {exc.msg}."
                    ) from exc
                if not isinstance(payload, dict):
                    raise ValueError(f"Malformed JSONL score at line {line_no}: expected object.")
                if self._query_id is not None and str(payload.get("query_id")) != self._query_id:
                    continue
                rows.append(payload)
        return rows

    def _to_score(self, payload: dict[str, Any]) -> VerificationScore:
        if "unit_id" not in payload:
            raise ValueError("JSONL score rows must include 'unit_id'.")

        unit_id = str(payload["unit_id"])
        label = str(payload.get("label", "unknown"))
        raw_payload = payload.get("raw", {})
        if raw_payload is None:
            raw_payload = {}
        if not isinstance(raw_payload, dict):
            raise ValueError(f"Score row for unit {unit_id!r} has non-object 'raw'.")
        raw = dict(raw_payload)

        if "score" in payload:
            score_value = float(payload["score"])
            raw["score"] = score_value
            raw["has_contradiction"] = False
            return VerificationScore(
                unit_id=unit_id,
                entailment=score_value,
                contradiction=0.0,
                neutral=max(0.0, 1.0 - score_value),
                label=label,
                raw=raw,
            )

        entailment_value = payload.get("entailment", payload.get("entail"))
        contradiction_value = payload.get("contradiction", payload.get("contrad"))
        neutral_value = payload.get("neutral")

        if entailment_value is None:
            raise ValueError(
                f"Score row for unit {unit_id!r} must include either 'score' or 'entailment'."
            )

        contradiction = 0.0 if contradiction_value is None else float(contradiction_value)
        neutral = 0.0 if neutral_value is None else float(neutral_value)
        raw.setdefault("has_contradiction", contradiction_value is not None)
        return VerificationScore(
            unit_id=unit_id,
            entailment=float(entailment_value),
            contradiction=contradiction,
            neutral=neutral,
            label=label,
            raw=raw,
        )
