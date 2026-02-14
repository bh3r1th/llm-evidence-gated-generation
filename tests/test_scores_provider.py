from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from ega.contract import PolicyConfig, ReasonCode
from ega.enforcer import Enforcer
from ega.providers.jsonl_scores import JsonlScoresProvider
from ega.types import EvidenceSet, VerificationScore
from ega.unitization import unitize_answer


class _FailingVerifier:
    def verify(self, *, unit_text: str, unit_id: str, evidence: EvidenceSet) -> VerificationScore:
        _ = (unit_text, unit_id, evidence)
        raise AssertionError("verifier path should not run")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _scores_file() -> Path:
    return Path("data") / f"scores_{uuid4().hex}.jsonl"


def test_jsonl_scores_provider_maps_scalar_and_preserves_raw() -> None:
    scores_path = _scores_file()
    try:
        _write_jsonl(
            scores_path,
            [
                {
                    "query_id": "q-1",
                    "unit_id": "u0001",
                    "score": 0.83,
                    "raw": {"source": "phoenix"},
                },
                {
                    "query_id": "q-1",
                    "unit_id": "u0002",
                    "entail": 0.52,
                    "contrad": 0.21,
                    "neutral": 0.27,
                    "label": "pass",
                    "raw": {"trace_id": "abc"},
                },
                {
                    "query_id": "q-2",
                    "unit_id": "u0001",
                    "score": 0.11,
                },
            ],
        )

        provider = JsonlScoresProvider(path=scores_path, query_id="q-1")
        candidate = unitize_answer("One. Two.", mode="sentence")
        loaded = provider.load_scores(candidate=candidate, evidence=EvidenceSet(items=[]))

        assert [score.unit_id for score in loaded] == ["u0001", "u0002"]
        assert loaded[0].entailment == 0.83
        assert loaded[0].contradiction == 0.0
        assert loaded[0].neutral == pytest.approx(0.17)
        assert loaded[0].label == "unknown"
        assert loaded[0].raw["score"] == 0.83
        assert loaded[0].raw["source"] == "phoenix"

        assert loaded[1].entailment == 0.52
        assert loaded[1].contradiction == 0.21
        assert loaded[1].neutral == 0.27
        assert loaded[1].label == "pass"
        assert loaded[1].raw["trace_id"] == "abc"
    finally:
        scores_path.unlink(missing_ok=True)


def test_enforcer_uses_scores_provider_without_verifier_for_partial_and_refusal() -> None:
    candidate = unitize_answer("Alpha is true. Beta is true. Gamma is true.", mode="sentence")
    scores_path = _scores_file()
    try:
        _write_jsonl(
            scores_path,
            [
                {"query_id": "ok", "unit_id": "u0001", "score": 0.93},
                {"query_id": "ok", "unit_id": "u0002", "score": 0.22},
                {"query_id": "ok", "unit_id": "u0003", "score": 0.91},
                {"query_id": "refuse", "unit_id": "u0001", "score": 0.20},
                {"query_id": "refuse", "unit_id": "u0002", "score": 0.10},
                {"query_id": "refuse", "unit_id": "u0003", "score": 0.05},
            ],
        )

        ok_result = Enforcer(
            scores_provider=JsonlScoresProvider(path=scores_path, query_id="ok"),
            verifier=_FailingVerifier(),
            config=PolicyConfig(
                threshold_entailment=0.8,
                max_contradiction=0.2,
                partial_allowed=True,
            ),
        ).enforce(candidate=candidate, evidence=EvidenceSet(items=[]))

        assert ok_result.decision.reason_code == ReasonCode.OK_PARTIAL.value
        assert ok_result.decision.refusal is False
        assert ok_result.kept_units == ["u0001", "u0003"]
        assert ok_result.dropped_units == ["u0002"]
        assert ok_result.final_text == "Alpha is true.\nGamma is true."

        refusal_result = Enforcer(
            scores_provider=JsonlScoresProvider(path=scores_path, query_id="refuse"),
            verifier=_FailingVerifier(),
            config=PolicyConfig(
                threshold_entailment=0.8,
                max_contradiction=0.2,
                partial_allowed=True,
            ),
        ).enforce(candidate=candidate, evidence=EvidenceSet(items=[]))

        assert refusal_result.decision.reason_code == ReasonCode.ALL_DROPPED.value
        assert refusal_result.decision.refusal is True
        assert refusal_result.final_text is None
    finally:
        scores_path.unlink(missing_ok=True)


def test_enforcer_scores_precedence_is_explicit_then_provider_then_verifier() -> None:
    candidate = unitize_answer("One. Two.", mode="sentence")
    scores_path = _scores_file()
    try:
        _write_jsonl(
            scores_path,
            [
                {"unit_id": "u0001", "score": 0.95},
                {"unit_id": "u0002", "score": 0.95},
            ],
        )
        explicit_scores = [
            VerificationScore(
                unit_id="u0001",
                entailment=0.1,
                contradiction=0.9,
                neutral=0.0,
                label="contradiction",
                raw={},
            ),
            VerificationScore(
                unit_id="u0002",
                entailment=0.1,
                contradiction=0.9,
                neutral=0.0,
                label="contradiction",
                raw={},
            ),
        ]

        result = Enforcer(
            scores_provider=JsonlScoresProvider(path=scores_path),
            verifier=_FailingVerifier(),
            config=PolicyConfig(
                threshold_entailment=0.8,
                max_contradiction=0.2,
                partial_allowed=True,
            ),
        ).enforce(candidate=candidate, evidence=EvidenceSet(items=[]), scores=explicit_scores)

        assert result.decision.reason_code == ReasonCode.ALL_DROPPED.value
        assert result.decision.refusal is True
    finally:
        scores_path.unlink(missing_ok=True)
