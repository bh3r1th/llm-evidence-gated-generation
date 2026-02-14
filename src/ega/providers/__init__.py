"""Score provider interfaces and implementations."""

from ega.providers.base import ScoreProvider
from ega.providers.jsonl_scores import JsonlScoresProvider

__all__ = ["JsonlScoresProvider", "ScoreProvider"]

