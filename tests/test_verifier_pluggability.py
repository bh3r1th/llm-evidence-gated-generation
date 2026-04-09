from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from ega.contract import PolicyConfig
from ega.pipeline import run_pipeline
from ega.types import EvidenceItem, EvidenceSet, Unit, VerificationScore
from ega.verifiers.adapter import LegacyVerifierAdapter


@dataclass
class _ScoreShape:
    entailment: float
    contradiction: float
    neutral: float
    label: str
    raw: dict[str, object]


class _DummyVerifier:
    model_name = "dummy-verifier"

    def verify(self, units: list[Unit], evidence: EvidenceSet) -> list[VerificationScore]:
        _ = evidence
        return [
            VerificationScore(
                unit_id=unit.id,
                entailment=0.95,
                contradiction=0.02,
                neutral=0.03,
                label="entailment",
                raw={"path": "dummy"},
            )
            for unit in units
        ]

    @staticmethod
    def get_last_verify_trace() -> dict[str, int]:
        return {"n_pairs_scored": 1}


class _VerifyManyOnly:
    model_name = "shape-many"

    def verify_many(self, candidate, evidence):  # type: ignore[no-untyped-def]
        _ = evidence
        return [
            _ScoreShape(0.8, 0.1, 0.1, "entailment", {"source": "many", "unit": unit.id})
            for unit in candidate.units
        ]


class _VerifyUnitOnly:
    model_name = "shape-unit"

    def verify_unit(self, unit_text, evidence):  # type: ignore[no-untyped-def]
        _ = (unit_text, evidence)
        return _ScoreShape(0.8, 0.1, 0.1, "entailment", {"source": "unit"})


def _policy() -> PolicyConfig:
    return PolicyConfig(threshold_entailment=0.5, max_contradiction=0.5, partial_allowed=True)


def _evidence() -> EvidenceSet:
    return EvidenceSet(items=[EvidenceItem(id="e1", text="Alpha fact.", metadata={})])


def test_pipeline_accepts_dummy_verifier_without_oss_nli_flag() -> None:
    output = run_pipeline(
        llm_summary_text="Alpha fact.",
        evidence=_evidence(),
        policy_config=_policy(),
        verifier=_DummyVerifier(),
        use_oss_nli=False,
    )

    assert output["decision"]["reason_code"] == "OK_FULL"
    assert output["trace"]["verifier_type"] == "custom_verifier"
    assert output["stats"]["model_name"] == "dummy-verifier"


def test_adapter_normalizes_verify_many_and_verify_unit_shapes() -> None:
    units = [Unit(id="u1", text="one", metadata={}), Unit(id="u2", text="two", metadata={})]
    evidence = _evidence()

    many_scores = LegacyVerifierAdapter(_VerifyManyOnly()).verify(units, evidence)
    unit_scores = LegacyVerifierAdapter(_VerifyUnitOnly()).verify(units, evidence)

    assert [score.unit_id for score in many_scores] == ["u1", "u2"]
    assert [score.unit_id for score in unit_scores] == ["u1", "u2"]
    assert all(isinstance(score, VerificationScore) for score in many_scores + unit_scores)
    assert all(score.label == "entailment" for score in many_scores + unit_scores)


def test_core_pipeline_module_does_not_import_concrete_nli_verifier() -> None:
    path = Path("src/ega/core/pipeline_core.py")
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    disallowed = []
    for node in ast.walk(module):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            imported = {alias.name for alias in node.names}
            if mod == "ega.verifiers.nli_cross_encoder" or (
                mod == "ega.verifiers" and "NliCrossEncoderVerifier" in imported
            ):
                disallowed.append((mod, sorted(imported)))

    assert disallowed == []
