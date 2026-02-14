"""Enforcement orchestration for deterministic policy gating."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from ega.contract import PolicyConfig
from ega.events import event_from_result
from ega.policy import DefaultPolicy, Policy
from ega.providers.base import ScoreProvider
from ega.types import AnswerCandidate, EnforcementResult, EvidenceSet, VerificationScore


class Verifier(Protocol):
    """Verifier interface used by the enforcer runtime."""

    def verify(self, *, unit_text: str, unit_id: str, evidence: EvidenceSet) -> VerificationScore:
        """Return a deterministic score for one answer unit."""


EventSink = Callable[[dict[str, Any]], None]


@dataclass(slots=True)
class Enforcer:
    """Deterministic enforcer that verifies units and applies a policy."""

    verifier: Verifier | None = None
    scores_provider: ScoreProvider | None = None
    policy: Policy = DefaultPolicy()
    config: PolicyConfig = PolicyConfig()
    refusal_message: str = "I can't provide a supported answer from the available evidence."
    event_sink: EventSink | None = None
    event_context: dict[str, Any] = field(default_factory=dict)

    def enforce(
        self,
        *,
        candidate: AnswerCandidate,
        evidence: EvidenceSet,
        scores: list[VerificationScore] | None = None,
    ) -> EnforcementResult:
        """Verify units, apply policy, and emit final gated output."""

        unit_text_by_id = {unit.id: unit.text for unit in candidate.units}
        active_scores = self._resolve_scores(candidate=candidate, evidence=evidence, scores=scores)
        decision = self.policy.decide(
            scores=active_scores,
            units=candidate.units,
            config=self.config,
        )

        if decision.refusal:
            refusal_message = (
                self.refusal_message
                if isinstance(self.refusal_message, str) and self.refusal_message.strip()
                else "I can't provide a supported answer from the available evidence."
            )
            result = EnforcementResult(
                final_text=None,
                kept_units=decision.allowed_units,
                dropped_units=decision.dropped_units,
                refusal_message=refusal_message,
                decision=decision,
                scores=active_scores,
                verified_units=[
                    {"unit_id": unit_id, "text": unit_text_by_id[unit_id]}
                    for unit_id in decision.allowed_units
                    if unit_id in unit_text_by_id
                ],
            )
            self._emit_event(result)
            return result

        final_text = "\n".join(
            unit_text_by_id[unit_id]
            for unit_id in decision.allowed_units
            if unit_id in unit_text_by_id
        )
        result = EnforcementResult(
            final_text=final_text,
            kept_units=decision.allowed_units,
            dropped_units=decision.dropped_units,
            refusal_message=None,
            decision=decision,
            scores=active_scores,
            verified_units=[
                {"unit_id": unit_id, "text": unit_text_by_id[unit_id]}
                for unit_id in decision.allowed_units
                if unit_id in unit_text_by_id
            ],
        )
        self._emit_event(result)
        return result

    def _resolve_scores(
        self,
        *,
        candidate: AnswerCandidate,
        evidence: EvidenceSet,
        scores: list[VerificationScore] | None,
    ) -> list[VerificationScore]:
        if scores is not None:
            return self._normalize_scores(candidate=candidate, raw_scores=scores)

        if self.scores_provider is not None:
            provider_scores = self.scores_provider.load_scores(
                candidate=candidate,
                evidence=evidence,
            )
            return self._normalize_scores(candidate=candidate, raw_scores=provider_scores)

        if self.verifier is None:
            raise ValueError(
                "Enforcer requires one of: explicit scores, scores_provider, or verifier."
            )

        verify_many = getattr(self.verifier, "verify_many", None)
        if callable(verify_many):
            verifier_scores = verify_many(candidate, evidence)
        else:
            verifier_scores = [
                self.verifier.verify(unit_text=unit.text, unit_id=unit.id, evidence=evidence)
                for unit in candidate.units
            ]
        return self._normalize_scores(candidate=candidate, raw_scores=verifier_scores)

    @staticmethod
    def _normalize_scores(
        *,
        candidate: AnswerCandidate,
        raw_scores: list[VerificationScore],
    ) -> list[VerificationScore]:
        score_by_unit: dict[str, VerificationScore] = {}
        candidate_ids = {unit.id for unit in candidate.units}
        for score in raw_scores:
            if score.unit_id not in candidate_ids:
                continue
            if score.unit_id in score_by_unit:
                raise ValueError(f"Duplicate score for unit_id {score.unit_id!r}.")
            score_by_unit[score.unit_id] = score

        normalized: list[VerificationScore] = []
        for unit in candidate.units:
            existing = score_by_unit.get(unit.id)
            if existing is not None:
                normalized.append(existing)
                continue
            normalized.append(
                VerificationScore(
                    unit_id=unit.id,
                    entailment=0.0,
                    contradiction=1.0,
                    neutral=0.0,
                    label="missing",
                    raw={"missing_score": True},
                )
            )
        return normalized

    def _emit_event(self, result: EnforcementResult) -> None:
        if self.event_sink is None:
            return

        context = dict(self.event_context)
        context.setdefault("policy_config", self.config)
        event = event_from_result(result, context)
        self.event_sink(asdict(event))
