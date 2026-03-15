from ega.types import EvidenceItem, EvidenceSet, Unit
from ega.v2.coverage import CoverageConfig, EvidenceCoverageAnalyzer


def _unit(unit_id: str) -> Unit:
    return Unit(id=unit_id, text=f"text-{unit_id}", metadata={})


def _evidence(*ids: str) -> EvidenceSet:
    return EvidenceSet(items=[EvidenceItem(id=eid, text=f"ev-{eid}", metadata={}) for eid in ids])


def test_coverage_empty_pool_is_zero_and_stable() -> None:
    analyzer = EvidenceCoverageAnalyzer()
    units = [_unit("u1")]
    result = analyzer.analyze(
        units=units,
        evidence=_evidence("e1", "e2"),
        pool_candidates={"u1": []},
        used_evidence={"u1": ["e1"]},
        config=CoverageConfig(pool_topk=20),
    )

    u1 = result["u1"]
    assert u1.relevant_evidence_ids == []
    assert u1.used_evidence_ids == []
    assert u1.coverage_score == 0.0
    assert u1.missing_evidence_ids == []


def test_coverage_partial_uses_intersection_with_stable_used_order() -> None:
    analyzer = EvidenceCoverageAnalyzer()
    units = [_unit("u1")]
    result = analyzer.analyze(
        units=units,
        evidence=_evidence("e1", "e2", "e3"),
        pool_candidates={"u1": ["e1", "e2", "e3"]},
        used_evidence={"u1": ["e3", "e9", "e1", "e3"]},
        config=CoverageConfig(pool_topk=20),
    )

    u1 = result["u1"]
    assert u1.relevant_evidence_ids == ["e1", "e2", "e3"]
    assert u1.used_evidence_ids == ["e3", "e1"]
    assert u1.coverage_score == 2.0 / 3.0
    assert u1.missing_evidence_ids == ["e2"]


def test_coverage_full_when_all_relevant_are_used() -> None:
    analyzer = EvidenceCoverageAnalyzer()
    units = [_unit("u1")]
    result = analyzer.analyze(
        units=units,
        evidence=_evidence("e1", "e2", "e3"),
        pool_candidates={"u1": ["e1", "e2", "e3", "e4"]},
        used_evidence={"u1": ["e2", "e1", "e3", "e2"]},
        config=CoverageConfig(pool_topk=3),
    )

    u1 = result["u1"]
    assert u1.relevant_evidence_ids == ["e1", "e2", "e3"]
    assert u1.used_evidence_ids == ["e2", "e1", "e3"]
    assert u1.coverage_score == 1.0
    assert u1.missing_evidence_ids == []
