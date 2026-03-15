"""Deterministic safe-answer rendering for EGA v2 outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ega.types import Unit


@dataclass(frozen=True, slots=True)
class RenderedClaim:
    unit_id: str
    text: str
    citations: list[str]
    decision: str


@dataclass(frozen=True, slots=True)
class SafeAnswerRender:
    final_text: str
    accepted_claims: list[RenderedClaim]
    abstained_claims: list[RenderedClaim]
    summary: dict

    def to_dict(self) -> dict:
        return {
            "final_text": self.final_text,
            "accepted_claims": [asdict(claim) for claim in self.accepted_claims],
            "abstained_claims": [asdict(claim) for claim in self.abstained_claims],
            "summary": dict(self.summary),
        }


class SafeAnswerRenderer:
    def render(
        self,
        units: list[Unit],
        decisions: dict[str, str],
        used_evidence: dict[str, list[str]],
        citation_text_by_id: dict[str, str] | None = None,
    ) -> SafeAnswerRender:
        _ = citation_text_by_id
        accepted_claims: list[RenderedClaim] = []
        abstained_claims: list[RenderedClaim] = []
        rejected_count = 0

        for unit in units:
            decision = _normalize_decision(decisions.get(unit.id))
            citations = sorted({str(evidence_id) for evidence_id in used_evidence.get(unit.id, [])})
            claim = RenderedClaim(
                unit_id=str(unit.id),
                text=str(unit.text),
                citations=citations,
                decision=decision,
            )
            if decision == "accept":
                accepted_claims.append(claim)
            elif decision == "abstain":
                abstained_claims.append(claim)
            else:
                rejected_count += 1

        return SafeAnswerRender(
            final_text="\n".join(_render_claim_line(claim) for claim in accepted_claims),
            accepted_claims=accepted_claims,
            abstained_claims=abstained_claims,
            summary={
                "accepted_count": len(accepted_claims),
                "abstained_count": len(abstained_claims),
                "rejected_count": rejected_count,
            },
        )


def _normalize_decision(value: str | None) -> str:
    decision = (value or "reject").strip().lower()
    if decision == "keep":
        return "accept"
    if decision == "drop":
        return "reject"
    if decision in {"accept", "abstain", "reject"}:
        return decision
    return "reject"


def _render_claim_line(claim: RenderedClaim) -> str:
    if not claim.citations:
        return claim.text
    return f"{claim.text} [{', '.join(claim.citations)}]"
