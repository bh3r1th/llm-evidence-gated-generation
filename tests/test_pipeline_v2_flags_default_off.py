from __future__ import annotations

import json
from pathlib import Path

from ega.contract import PolicyConfig
from ega.pipeline import run_pipeline
from ega.types import EvidenceItem, EvidenceSet

_FIXTURES_DIR = Path("examples") / "pipeline_demo"


def _evidence() -> EvidenceSet:
    payload = json.loads((_FIXTURES_DIR / "evidence.json").read_text(encoding="utf-8"))
    return EvidenceSet(
        items=[
            EvidenceItem(
                id=str(item["id"]),
                text=str(item["text"]),
                metadata=dict(item.get("metadata", {})),
            )
            for item in payload
        ]
    )


def _summary() -> str:
    return (_FIXTURES_DIR / "llm_summary.txt").read_text(encoding="utf-8")


def test_pipeline_v2_flags_default_off_produces_identical_output() -> None:
    kwargs = {
        "llm_summary_text": _summary(),
        "evidence": _evidence(),
        "unitizer_mode": "sentence",
        "policy_config": PolicyConfig(
            threshold_entailment=0.8,
            max_contradiction=0.2,
            partial_allowed=True,
        ),
        "scores_jsonl_path": str(_FIXTURES_DIR / "scores.jsonl"),
    }
    baseline = run_pipeline(**kwargs)
    with_v2_nones = run_pipeline(
        **kwargs,
        reranker=None,
        rerank_topk=None,
        conformal_state_path=None,
        conformal_epsilon=None,
        budget_policy=None,
        budget_config=None,
    )

    assert with_v2_nones == baseline
