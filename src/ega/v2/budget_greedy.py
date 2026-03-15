"""Greedy budget controller for EGA v2."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from ega.types import EvidenceSet, Unit
from ega.v2.budget import BudgetConfig, BudgetDecision, BudgetPolicy


@dataclass(frozen=True, slots=True)
class GreedyBudgetTrace:
    """Debug trace for last greedy budget allocation."""

    base_topk_per_unit: int
    base_pairs_total: int
    requested_budget_max_pairs: int | None
    requested_latency_budget_ms: int | None
    effective_budget_max_pairs: int
    chosen_topk_per_unit: int
    chosen_max_pairs_total: int
    allocated_steps: int
    risk_by_unit: dict[str, float] = field(default_factory=dict)
    per_unit_pair_budget: dict[str, int] = field(default_factory=dict)
    per_unit_added_k: dict[str, int] = field(default_factory=dict)


class GreedyBudgetPolicy(BudgetPolicy):
    """Allocate additional verification effort by greedy marginal benefit."""

    def __init__(self, *, default_risk: float = 0.5) -> None:
        self.default_risk = float(default_risk)
        self._last_trace = GreedyBudgetTrace(
            base_topk_per_unit=0,
            base_pairs_total=0,
            requested_budget_max_pairs=None,
            requested_latency_budget_ms=None,
            effective_budget_max_pairs=0,
            chosen_topk_per_unit=0,
            chosen_max_pairs_total=0,
            allocated_steps=0,
            risk_by_unit={},
            per_unit_pair_budget={},
            per_unit_added_k={},
        )

    def get_last_trace(self) -> GreedyBudgetTrace:
        return self._last_trace

    def choose(
        self,
        units: list[Unit],
        evidence: EvidenceSet,
        base_params: Mapping[str, Any],
        risk_features: Mapping[str, Any],
        budget: BudgetConfig,
    ) -> BudgetDecision:
        if "topk_per_unit" not in base_params:
            raise ValueError("base_params must include 'topk_per_unit'.")
        if "max_pairs_total" not in base_params:
            raise ValueError("base_params must include 'max_pairs_total'.")
        if "verifier_name" not in base_params:
            raise ValueError("base_params must include 'verifier_name'.")

        n_units = len(units)
        n_evidence = len(evidence.items)
        verifier_name = str(base_params["verifier_name"])
        base_topk = max(0, int(base_params["topk_per_unit"]))
        base_topk = min(base_topk, n_evidence)
        base_pairs = base_topk * n_units

        raw_pair_cap = int(base_params["max_pairs_total"])
        max_possible_pairs = max(0, n_units * n_evidence)
        pair_cap = max_possible_pairs if raw_pair_cap <= 0 else min(max_possible_pairs, raw_pair_cap)
        if budget.max_pairs_total is not None:
            pair_cap = min(pair_cap, max(0, int(budget.max_pairs_total)))
        cost = float(budget.cost_per_pair)
        if not math.isfinite(cost) or cost <= 0.0:
            raise ValueError("budget.cost_per_pair must be a finite positive value.")
        if budget.latency_budget_ms is not None:
            pair_cap = min(pair_cap, max(0, int(float(budget.latency_budget_ms) / cost)))

        if n_units == 0 or n_evidence == 0 or pair_cap <= 0:
            self._last_trace = GreedyBudgetTrace(
                base_topk_per_unit=base_topk,
                base_pairs_total=base_pairs,
                requested_budget_max_pairs=(
                    None if budget.max_pairs_total is None else int(budget.max_pairs_total)
                ),
                requested_latency_budget_ms=(
                    None if budget.latency_budget_ms is None else int(budget.latency_budget_ms)
                ),
                effective_budget_max_pairs=0,
                chosen_topk_per_unit=0,
                chosen_max_pairs_total=0,
                allocated_steps=0,
                risk_by_unit={},
                per_unit_pair_budget={},
                per_unit_added_k={},
            )
            return BudgetDecision(
                topk_per_unit=0,
                max_pairs_total=0,
                verifier_name=verifier_name,
                per_unit_pair_budget={},
            )

        if budget.max_pairs_total is None and budget.latency_budget_ms is None:
            self._last_trace = GreedyBudgetTrace(
                base_topk_per_unit=base_topk,
                base_pairs_total=base_pairs,
                requested_budget_max_pairs=None,
                requested_latency_budget_ms=None,
                effective_budget_max_pairs=min(base_pairs, pair_cap),
                chosen_topk_per_unit=base_topk,
                chosen_max_pairs_total=min(base_pairs, pair_cap),
                allocated_steps=0,
                risk_by_unit={unit.id: self.default_risk for unit in units},
                per_unit_pair_budget={unit.id: base_topk for unit in units},
                per_unit_added_k={unit.id: 0 for unit in units},
            )
            return BudgetDecision(
                topk_per_unit=base_topk,
                max_pairs_total=min(base_pairs, pair_cap),
                verifier_name=verifier_name,
                per_unit_pair_budget={unit.id: base_topk for unit in units},
            )

        pair_cap = min(pair_cap, max_possible_pairs)
        per_unit_pair_budget: dict[str, int] = {unit.id: 0 for unit in units}
        risk_by_unit = self._normalize_risk_features(units=units, risk_features=risk_features)

        allocated_steps = 0
        while allocated_steps < pair_cap:
            best_unit_id: str | None = None
            best_gain = -1.0
            for unit in units:
                unit_id = unit.id
                can_add = per_unit_pair_budget[unit_id] < n_evidence
                if not can_add:
                    continue
                gain = risk_by_unit[unit_id] * (1.0 / (1.0 + float(per_unit_pair_budget[unit_id])))
                if gain > best_gain or (
                    gain == best_gain and best_unit_id is not None and str(unit_id) < str(best_unit_id)
                ):
                    best_gain = gain
                    best_unit_id = unit_id

            if best_unit_id is None:
                break

            per_unit_pair_budget[best_unit_id] += 1
            allocated_steps += 1

        chosen_topk = max(per_unit_pair_budget.values()) if per_unit_pair_budget else 0
        chosen_pairs = int(sum(per_unit_pair_budget.values()))
        chosen_pairs = max(0, chosen_pairs)
        added_k = {
            unit.id: max(0, per_unit_pair_budget[unit.id] - min(base_topk, per_unit_pair_budget[unit.id]))
            for unit in units
        }

        self._last_trace = GreedyBudgetTrace(
            base_topk_per_unit=base_topk,
            base_pairs_total=base_pairs,
            requested_budget_max_pairs=(
                None if budget.max_pairs_total is None else int(budget.max_pairs_total)
            ),
            requested_latency_budget_ms=(
                None if budget.latency_budget_ms is None else int(budget.latency_budget_ms)
            ),
            effective_budget_max_pairs=pair_cap,
            chosen_topk_per_unit=chosen_topk,
            chosen_max_pairs_total=chosen_pairs,
            allocated_steps=allocated_steps,
            risk_by_unit=dict(risk_by_unit),
            per_unit_pair_budget=dict(per_unit_pair_budget),
            per_unit_added_k=dict(added_k),
        )
        return BudgetDecision(
            topk_per_unit=chosen_topk,
            max_pairs_total=chosen_pairs,
            verifier_name=verifier_name,
            per_unit_pair_budget=dict(per_unit_pair_budget),
        )

    def _normalize_risk_features(
        self,
        *,
        units: list[Unit],
        risk_features: Mapping[str, Any],
    ) -> dict[str, float]:
        normalized: dict[str, float] = {}
        for unit in units:
            raw = risk_features.get(unit.id, self.default_risk)
            try:
                value = float(raw)
            except (TypeError, ValueError):
                value = self.default_risk
            if not math.isfinite(value):
                value = self.default_risk
            normalized[unit.id] = max(0.0, min(1.0, value))
        return normalized
