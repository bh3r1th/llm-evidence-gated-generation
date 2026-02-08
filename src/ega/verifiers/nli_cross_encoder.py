"""NLI cross-encoder verifier implementation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ega.types import AnswerCandidate, EvidenceSet, VerificationScore

PairPredictor = Callable[[list[tuple[str, str]]], list[dict[str, float]]]
DEFAULT_MODEL_NAME = "microsoft/deberta-v3-large-mnli"


@dataclass(slots=True)
class NliCrossEncoderVerifier:
    """Cross-encoder verifier backed by an MNLI-style sequence-pair classifier."""

    model_name: str | None = None
    max_length: int = 384
    batch_size: int = 16
    device: str = "cpu"
    aggregation_strategy: str = "max_entailment"
    pair_predictor: PairPredictor | None = None
    model: Any | None = None
    tokenizer: Any | None = None

    name: str = "nli_cross_encoder"

    def __post_init__(self) -> None:
        if self.model_name is None:
            self.model_name = DEFAULT_MODEL_NAME

        if self.aggregation_strategy != "max_entailment":
            msg = f"Unsupported aggregation strategy: {self.aggregation_strategy}"
            raise ValueError(msg)

        if self.pair_predictor is not None:
            return

        self._torch = self._import_torch()
        transformers = self._import_transformers()

        self._torch.manual_seed(0)

        if self.tokenizer is None:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
        if self.model is None:
            self.model = transformers.AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )

        self.model.to(self.device)
        self.model.eval()
        self._label_indices = self._resolve_label_indices(
            id2label=getattr(self.model.config, "id2label", {}),
            num_labels=getattr(self.model.config, "num_labels", 3),
        )

    @staticmethod
    def _import_torch() -> Any:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - exercised in runtime environments.
            msg = "NLI verifier requires optional dependency: pip install 'ega[nli]'"
            raise ImportError(msg) from exc
        return torch

    @staticmethod
    def _import_transformers() -> Any:
        try:
            import transformers
        except ImportError as exc:  # pragma: no cover - exercised in runtime environments.
            msg = "NLI verifier requires optional dependency: pip install 'ega[nli]'"
            raise ImportError(msg) from exc
        return transformers

    @staticmethod
    def _normalize_label(label: str) -> str:
        return "".join(ch for ch in label.lower() if ch.isalpha())

    @classmethod
    def _resolve_label_indices(
        cls,
        *,
        id2label: dict[Any, str] | None,
        num_labels: int,
    ) -> dict[str, int]:
        labels: dict[int, str] = {}
        for idx in range(num_labels):
            if id2label is None:
                labels[idx] = str(idx)
                continue
            raw_label = id2label.get(idx, id2label.get(str(idx), str(idx)))
            labels[idx] = str(raw_label)

        resolved: dict[str, int] = {}
        for idx, label in labels.items():
            normalized = cls._normalize_label(label)
            if "entail" in normalized:
                resolved["entailment"] = idx
            elif "contrad" in normalized:
                resolved["contradiction"] = idx
            elif "neutral" in normalized:
                resolved["neutral"] = idx

        if len(resolved) < 3 and num_labels == 3:
            resolved.setdefault("contradiction", 0)
            resolved.setdefault("neutral", 1)
            resolved.setdefault("entailment", 2)

        missing = {"entailment", "contradiction", "neutral"} - set(resolved)
        if missing:
            msg = f"Unable to resolve MNLI label indices, missing: {sorted(missing)}"
            raise ValueError(msg)

        return resolved

    def _predict_pair_probabilities(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        if self.pair_predictor is not None:
            return self.pair_predictor(pairs)

        probabilities: list[dict[str, float]] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            batch_units = [unit for unit, _ in batch]
            batch_evidence = [item for _, item in batch]
            encoded = self.tokenizer(
                batch_units,
                batch_evidence,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}
            with self._torch.no_grad():
                logits = self.model(**encoded).logits
                probs = self._torch.softmax(logits, dim=-1).cpu()

            for row in probs:
                probabilities.append(
                    {
                        "entailment": float(row[self._label_indices["entailment"]].item()),
                        "contradiction": float(row[self._label_indices["contradiction"]].item()),
                        "neutral": float(row[self._label_indices["neutral"]].item()),
                    }
                )

        return probabilities

    @staticmethod
    def _label_from_probs(probs: dict[str, float]) -> str:
        return max(("entailment", "contradiction", "neutral"), key=lambda key: probs[key])

    def _verify_unit_with_id(
        self,
        *,
        unit_id: str,
        unit_text: str,
        evidence: EvidenceSet,
    ) -> VerificationScore:
        if not evidence.items:
            empty = {"entailment": 0.0, "contradiction": 0.0, "neutral": 1.0}
            return VerificationScore(
                unit_id=unit_id,
                entailment=empty["entailment"],
                contradiction=empty["contradiction"],
                neutral=empty["neutral"],
                label="neutral",
                raw={
                    "chosen_evidence_id": None,
                    "per_item_probs": [],
                    "model_name": self.model_name,
                    "tokenizer_name": getattr(self.tokenizer, "name_or_path", self.model_name),
                },
            )

        pairs = [(unit_text, item.text) for item in evidence.items]
        per_item_probs = self._predict_pair_probabilities(pairs)
        best_idx = max(
            range(len(per_item_probs)),
            key=lambda idx: per_item_probs[idx]["entailment"],
        )
        best_probs = per_item_probs[best_idx]

        return VerificationScore(
            unit_id=unit_id,
            entailment=best_probs["entailment"],
            contradiction=best_probs["contradiction"],
            neutral=best_probs["neutral"],
            label=self._label_from_probs(best_probs),
            raw={
                "chosen_evidence_id": evidence.items[best_idx].id,
                "per_item_probs": [
                    {"evidence_id": item.id, **probs}
                    for item, probs in zip(evidence.items, per_item_probs, strict=True)
                ],
                "model_name": self.model_name,
                "tokenizer_name": getattr(self.tokenizer, "name_or_path", self.model_name),
            },
        )

    def verify_unit(self, unit_text: str, evidence: EvidenceSet) -> VerificationScore:
        """Verify one unit against evidence and return the best entailment score."""

        return self._verify_unit_with_id(unit_id="unit", unit_text=unit_text, evidence=evidence)

    def verify(self, candidate: AnswerCandidate, evidence: EvidenceSet) -> list[VerificationScore]:
        """Verify all candidate units against the provided evidence set."""

        return [
            self._verify_unit_with_id(unit_id=unit.id, unit_text=unit.text, evidence=evidence)
            for unit in candidate.units
        ]


NLICrossEncoderVerifier = NliCrossEncoderVerifier
