"""EGA v2 extension interfaces."""

from ega.v2.budget import BudgetConfig, BudgetDecision, BudgetPolicy, FixedBudgetPolicy
from ega.v2.budget_greedy import GreedyBudgetPolicy, GreedyBudgetTrace
from ega.v2.calibrate import (
    calibrate_jsonl_to_state,
    load_unit_calibration_jsonl,
    save_conformal_state_json,
)
from ega.v2.conformal import ConformalCalibrator, ConformalConfig, ConformalState
from ega.v2.coverage import CoverageConfig, CoverageResult, EvidenceCoverageAnalyzer
from ega.v2.cross_encoder_reranker import CrossEncoderReranker, RerankStats
from ega.v2.poc_config import (
    DEFAULT_ACCEPT_THRESHOLD,
    DEFAULT_FINAL_CONFORMAL_STATE,
    DEFAULT_FINAL_DATASET,
    DEFAULT_FINAL_REPORT,
    DEFAULT_FINAL_SOURCE_SUMMARY,
    DEFAULT_FINAL_SUMMARY,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_RERANK_TOPK,
    EXPERIMENTAL_VARIANTS,
    RECOMMENDED_VARIANTS,
)
from ega.v2.rewards import RewardComputer, RewardConfig, RewardSummary, UnitReward
from ega.v2.render import RenderedClaim, SafeAnswerRender, SafeAnswerRenderer
from ega.v2.risk import extract_unit_risks
from ega.v2.reranker import EvidenceReranker, NoopReranker

__all__ = [
    "EvidenceReranker",
    "NoopReranker",
    "BudgetConfig",
    "BudgetDecision",
    "BudgetPolicy",
    "FixedBudgetPolicy",
    "GreedyBudgetPolicy",
    "GreedyBudgetTrace",
    "load_unit_calibration_jsonl",
    "calibrate_jsonl_to_state",
    "save_conformal_state_json",
    "ConformalConfig",
    "ConformalState",
    "ConformalCalibrator",
    "CoverageConfig",
    "CoverageResult",
    "EvidenceCoverageAnalyzer",
    "RewardConfig",
    "UnitReward",
    "RewardSummary",
    "RewardComputer",
    "DEFAULT_ACCEPT_THRESHOLD",
    "DEFAULT_FINAL_CONFORMAL_STATE",
    "DEFAULT_FINAL_DATASET",
    "DEFAULT_FINAL_REPORT",
    "DEFAULT_FINAL_SOURCE_SUMMARY",
    "DEFAULT_FINAL_SUMMARY",
    "DEFAULT_RERANKER_MODEL",
    "DEFAULT_RERANK_TOPK",
    "RECOMMENDED_VARIANTS",
    "EXPERIMENTAL_VARIANTS",
    "RenderedClaim",
    "SafeAnswerRender",
    "SafeAnswerRenderer",
    "CrossEncoderReranker",
    "RerankStats",
    "extract_unit_risks",
]
