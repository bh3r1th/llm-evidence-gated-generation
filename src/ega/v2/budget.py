"""Budget policy interfaces for EGA v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from ega.types import EvidenceSet, Unit


@dataclass(frozen=True, slots=True)
class BudgetConfig:
    """Runtime budget constraints for verification."""

    latency_budget_ms: int | None = None
    max_pairs_total: int | None = None
    max_verifier_calls: int | None = None
    cost_per_pair: float = 1.0


@dataclass(frozen=True, slots=True)
class BudgetDecision:
    """Selected runtime limits for a verification request."""

    topk_per_unit: int
    max_pairs_total: int
    verifier_name: str
    per_unit_pair_budget: dict[str, int] | None = None


class BudgetPolicy(Protocol):
    """Choose verifier parameters under a budget."""

    def choose(
        self,
        units: list[Unit],
        evidence: EvidenceSet,
        base_params: Mapping[str, Any],
        risk_features: Mapping[str, Any],
        budget: BudgetConfig,
    ) -> BudgetDecision:
        """Return a budgeted verification decision."""


class FixedBudgetPolicy:
    """Baseline policy that returns base parameters unchanged."""

    def choose(
        self,
        units: list[Unit],
        evidence: EvidenceSet,
        base_params: Mapping[str, Any],
        risk_features: Mapping[str, Any],
        budget: BudgetConfig,
    ) -> BudgetDecision:
        _ = (units, evidence, risk_features, budget)
        if "topk_per_unit" not in base_params:
            raise ValueError("base_params must include 'topk_per_unit'.")
        if "max_pairs_total" not in base_params:
            raise ValueError("base_params must include 'max_pairs_total'.")
        if "verifier_name" not in base_params:
            raise ValueError("base_params must include 'verifier_name'.")

        return BudgetDecision(
            topk_per_unit=int(base_params["topk_per_unit"]),
            max_pairs_total=int(base_params["max_pairs_total"]),
            verifier_name=str(base_params["verifier_name"]),
        )
