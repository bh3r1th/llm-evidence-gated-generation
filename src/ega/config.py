"""Structured configuration objects for the verification pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ega.contract import PolicyConfig
from ega.v2.budget import BudgetConfig, BudgetPolicy
from ega.v2.conformal import ConformalConfig
from ega.v2.reranker import EvidenceReranker


@dataclass(frozen=True, slots=True)
class VerifierConfig:
    """Verifier-specific settings for cross-encoder/NLI scoring."""

    model: str | None = None
    device: str = "auto"
    dtype: str = "auto"
    top_k: int = 12
    max_pairs: int | None = 200
    use_oss_nli: bool = False
    verifier: Any | None = None


@dataclass(frozen=True, slots=True)
class RerankerConfig:
    """Reranker toggles and optional runtime objects."""

    enabled: bool = False
    reranker: EvidenceReranker | None = None
    top_k: int | None = None


@dataclass(frozen=True, slots=True)
class OutputConfig:
    """Output and tracing flags."""

    render_safe_answer: bool = False
    trace_out: str | None = None
    enable_polish_validation: bool = True


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Structured config for :func:`ega.api.verify_answer`."""

    policy: PolicyConfig
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    conformal: ConformalConfig | None = None
    budget: BudgetConfig | None = None
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    scores_jsonl_path: str | None = None
    unitizer_mode: str = "sentence"
    accept_threshold: float | None = None
    conformal_state_path: str | None = None
    budget_policy: BudgetPolicy | None = None
    extras: dict[str, Any] = field(default_factory=dict)
