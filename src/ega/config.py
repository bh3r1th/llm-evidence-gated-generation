"""Structured configuration objects for the verification pipeline.

``PipelineConfig`` is the public configuration contract.
Other config dataclasses in this module support that contract but are not
package-level stable API guarantees.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ega.contract import PolicyConfig
from ega.v2.budget import BudgetConfig, BudgetPolicy
from ega.v2.conformal import ConformalConfig
from ega.v2.reranker import EvidenceReranker


@dataclass(frozen=True, slots=True)
class VerifierConfig:
    """Internal helper config for ``PipelineConfig.verifier`` (not stable API)."""

    model: str | None = None
    device: str = "auto"
    dtype: str = "auto"
    top_k: int = 12
    max_pairs: int | None = 200
    use_oss_nli: bool = False
    verifier: Any | None = None


@dataclass(frozen=True, slots=True)
class RerankerConfig:
    """Internal helper config for ``PipelineConfig.reranker`` (not stable API)."""

    enabled: bool = False
    reranker: EvidenceReranker | None = None
    top_k: int | None = None


@dataclass(frozen=True, slots=True)
class OutputConfig:
    """Internal helper config for ``PipelineConfig.output`` (not stable API)."""

    render_safe_answer: bool = False
    trace_out: str | None = None
    enable_polish_validation: bool = True
    downstream_compatibility_mode: str = "STRICT_PASSTHROUGH"


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Public config contract for :func:`ega.api.verify_answer`.

    Stable fields are:
    ``policy``, ``verifier``, ``conformal``, ``budget``, ``reranker``, ``output``,
    ``scores_jsonl_path``, ``unitizer_mode``, ``accept_threshold``,
    ``conformal_state_path``, ``budget_policy``, ``enable_correction``,
    ``max_retries``, and ``extras``.
    """

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
    enable_correction: bool = False
    max_retries: int = 1
    extras: dict[str, Any] = field(default_factory=dict)
