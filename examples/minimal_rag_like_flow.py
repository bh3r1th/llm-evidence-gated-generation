"""Minimal deterministic RAG-like flow using EGA enforcement."""

from __future__ import annotations

from dataclasses import dataclass
from string import punctuation

from ega.enforcer import Enforcer
from ega.policy import PolicyConfig
from ega.serialization import to_json
from ega.types import AnswerCandidate, EvidenceItem, EvidenceSet, Unit, VerificationScore


@dataclass(slots=True)
class OverlapVerifier:
    """Simple deterministic verifier based on token overlap."""

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        translator = str.maketrans("", "", punctuation + "`")
        return {
            token.lower().translate(translator)
            for token in text.split()
            if token.strip()
        }

    def verify(self, *, unit_text: str, unit_id: str, evidence: EvidenceSet) -> VerificationScore:
        unit_tokens = self._tokenize(unit_text)
        best_overlap = 0.0
        for item in evidence.items:
            evidence_tokens = self._tokenize(item.text)
            overlap = len(unit_tokens & evidence_tokens) / len(unit_tokens)
            if overlap > best_overlap:
                best_overlap = overlap

        entailment = round(best_overlap, 6)
        contradiction = round(1.0 - entailment, 6)
        label = "entailment" if entailment >= 0.5 else "contradiction"
        return VerificationScore(
            unit_id=unit_id,
            entailment=entailment,
            contradiction=contradiction,
            neutral=0.0,
            label=label,
            raw={},
        )


def main() -> None:
    evidence = EvidenceSet(
        items=[
            EvidenceItem(id="doc-1", text="Paris is the capital of France.", metadata={}),
            EvidenceItem(id="doc-2", text="The Eiffel Tower is in Paris.", metadata={}),
        ]
    )
    answer = AnswerCandidate(
        raw_answer_text=(
            "Paris is the capital of France.\n"
            "France has 1000 official capitals."
        ),
        units=[
            Unit(id="u0001", text="Paris is the capital of France."),
            Unit(id="u0002", text="France has 1000 official capitals."),
        ],
    )

    verifier = OverlapVerifier()

    partial_result = Enforcer(
        verifier=verifier,
        config=PolicyConfig(threshold=0.7, partial_allowed=True),
    ).enforce(candidate=answer, evidence=evidence)
    print("Partial allowed:")
    print(to_json(partial_result))

    refusal_result = Enforcer(
        verifier=verifier,
        config=PolicyConfig(threshold=0.7, partial_allowed=False),
    ).enforce(candidate=answer, evidence=evidence)
    print("Partial not allowed:")
    print(to_json(refusal_result))


if __name__ == "__main__":
    main()
