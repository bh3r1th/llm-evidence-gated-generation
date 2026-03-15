"""Cross-encoder evidence reranker for EGA v2."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ega.types import EvidenceSet, Unit
from ega.v2.reranker import EvidenceReranker


@dataclass(frozen=True, slots=True)
class RerankStats:
    """Lightweight timing and throughput stats for reranking."""

    n_pairs_scored: int
    seconds: float


class CrossEncoderReranker(EvidenceReranker):
    """Rerank candidate evidence ids per unit with a cross-encoder."""

    def __init__(
        self,
        *,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 32,
        max_pairs: int | None = None,
        cross_encoder: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.max_pairs = max_pairs
        self._model = cross_encoder
        self._last_stats = RerankStats(n_pairs_scored=0, seconds=0.0)

        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.max_pairs is not None and self.max_pairs <= 0:
            raise ValueError("max_pairs must be > 0 when provided")

        if self._model is None:
            self._model = self._load_cross_encoder(model_name=self.model_name)

    @staticmethod
    def _load_cross_encoder(*, model_name: str) -> Any:
        try:
            import torch
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "CrossEncoderReranker requires optional dependencies. "
                "Install with: pip install 'ega[rerank]'"
            ) from exc

        model = CrossEncoder(model_name)
        model._ega_torch = torch
        return model

    def get_last_stats(self) -> RerankStats:
        return self._last_stats

    def rerank_with_stats(
        self,
        units: list[Unit],
        evidence: EvidenceSet,
        candidates: dict[str, list[str]],
        topk: int,
    ) -> tuple[dict[str, list[str]], RerankStats]:
        t0 = time.perf_counter()
        if topk <= 0:
            empty = {unit.id: [] for unit in units}
            stats = RerankStats(n_pairs_scored=0, seconds=time.perf_counter() - t0)
            self._last_stats = stats
            return empty, stats

        evidence_text_by_id = {item.id: item.text for item in evidence.items}

        pair_rows: list[tuple[str, str, str, str]] = []
        for unit in units:
            for evidence_id in candidates.get(unit.id, []):
                evidence_text = evidence_text_by_id.get(evidence_id)
                if evidence_text is None:
                    continue
                pair_rows.append((unit.id, evidence_id, unit.text, evidence_text))

        if self.max_pairs is not None:
            pair_rows = pair_rows[: self.max_pairs]

        if not pair_rows:
            result = {unit.id: [] for unit in units}
            stats = RerankStats(n_pairs_scored=0, seconds=time.perf_counter() - t0)
            self._last_stats = stats
            return result, stats

        pairs = [(unit_text, evidence_text) for _, _, unit_text, evidence_text in pair_rows]
        scores = self._predict_scores(pairs)

        grouped: dict[str, list[tuple[str, float]]] = {unit.id: [] for unit in units}
        for idx, (unit_id, evidence_id, _unit_text, _evidence_text) in enumerate(pair_rows):
            grouped.setdefault(unit_id, []).append((evidence_id, scores[idx]))

        reranked: dict[str, list[str]] = {}
        for unit in units:
            scored = grouped.get(unit.id, [])
            scored_sorted = sorted(scored, key=lambda item: (-item[1], str(item[0])))
            reranked[unit.id] = [evidence_id for evidence_id, _ in scored_sorted[:topk]]

        stats = RerankStats(n_pairs_scored=len(pair_rows), seconds=time.perf_counter() - t0)
        self._last_stats = stats
        return reranked, stats

    def rerank(
        self,
        units: list[Unit],
        evidence: EvidenceSet,
        candidates: dict[str, list[str]],
        topk: int,
    ) -> dict[str, list[str]]:
        ranked, _stats = self.rerank_with_stats(
            units=units,
            evidence=evidence,
            candidates=candidates,
            topk=topk,
        )
        return ranked

    def _predict_scores(self, pairs: list[tuple[str, str]]) -> list[float]:
        torch_module = getattr(self._model, "_ega_torch", None)
        if torch_module is None:
            try:
                import torch as torch_runtime
            except ImportError:
                torch_runtime = None
            torch_module = torch_runtime

        if torch_module is not None:
            with torch_module.no_grad():
                raw = self._model.predict(pairs, batch_size=self.batch_size)
        else:
            raw = self._model.predict(pairs, batch_size=self.batch_size)

        if hasattr(raw, "tolist"):
            values = raw.tolist()
        else:
            values = list(raw)

        flattened: list[float] = []
        for value in values:
            if isinstance(value, (list, tuple)):
                flattened.append(float(value[0]) if value else 0.0)
            else:
                flattened.append(float(value))
        return flattened
