"""NLI cross-encoder verifier implementation."""

from __future__ import annotations

import contextlib
import time
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ega.interfaces import Verifier
from ega.types import AnswerCandidate, EvidenceSet, Unit, VerificationScore

PairPredictor = Callable[[list[tuple[str, str]]], list[dict[str, float]]]
DEFAULT_MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"


@dataclass(slots=True)
class NliCrossEncoderVerifier(Verifier):
    """Cross-encoder verifier backed by an MNLI-style sequence-pair classifier."""

    model_name: str | None = None
    max_length: int = 384
    batch_size: int = 16
    device: str = "auto"
    dtype: str = "auto"
    aggregation_strategy: str = "max_entailment"
    pair_predictor: PairPredictor | None = None
    model: Any | None = None
    tokenizer: Any | None = None
    verbose: bool = False
    topk_per_unit: int = 12
    max_pairs_total: int | None = 200
    max_evidence_per_request: int | None = None
    max_batch_tokens: int | None = None
    evidence_max_chars: int = 800
    evidence_max_sentences: int = 3

    name: str = "nli_cross_encoder"
    _torch: Any = field(init=False, repr=False)
    _label_indices: dict[str, int] = field(init=False, repr=False)
    _last_verify_trace: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _torch_dtype: Any = field(default=None, init=False, repr=False)
    amp_enabled: bool = field(default=False, init=False)
    dtype_overridden: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.model_name is None:
            self.model_name = DEFAULT_MODEL_NAME

        if self.aggregation_strategy != "max_entailment":
            msg = f"Unsupported aggregation strategy: {self.aggregation_strategy}"
            raise ValueError(msg)

        torch_probe = self._safe_import_torch()
        self.device, self.dtype, self._torch_dtype, self.amp_enabled, self.dtype_overridden = (
            self._resolve_runtime(
                requested_device=self.device,
                requested_dtype=self.dtype,
                torch_module=torch_probe,
            )
        )
        if self.max_batch_tokens is None:
            self.max_batch_tokens = 16000 if self.device == "cuda" else 4000

        if self.pair_predictor is not None:
            return

        self._torch = self._import_torch()
        _, _, self._torch_dtype, self.amp_enabled, self.dtype_overridden = self._resolve_runtime(
            requested_device=self.device,
            requested_dtype=self.dtype,
            torch_module=self._torch,
        )
        transformers = self._import_transformers()
        if not self.verbose:
            self._suppress_transformers_progress(transformers)

        self._torch.manual_seed(0)

        if self.tokenizer is None:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
        if self.model is None:
            self.model = transformers.AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )

        if self.device == "cuda" and self._torch_dtype is not None:
            self.model.to(device=self.device, dtype=self._torch_dtype)
        else:
            self.model.to(device=self.device)
        self.model.eval()
        self._label_indices = self._resolve_label_indices(
            id2label=getattr(self.model.config, "id2label", {}),
            num_labels=getattr(self.model.config, "num_labels", 3),
        )

    @staticmethod
    def _suppress_transformers_progress(transformers: Any) -> None:
        try:
            logging = transformers.utils.logging
            logging.set_verbosity_error()
            disable_progress = getattr(logging, "disable_progress_bar", None)
            if callable(disable_progress):
                disable_progress()
        except Exception:
            pass

        try:
            from huggingface_hub.utils import disable_progress_bars

            disable_progress_bars()
        except Exception:
            pass

    @staticmethod
    def _import_torch() -> Any:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - exercised in runtime environments.
            msg = "NLI verifier requires optional dependency: pip install 'ega[nli]'"
            raise ImportError(msg) from exc
        return torch

    @classmethod
    def _safe_import_torch(cls) -> Any | None:
        try:
            return cls._import_torch()
        except ImportError:
            return None

    @staticmethod
    def _resolve_runtime(
        *,
        requested_device: str,
        requested_dtype: str,
        torch_module: Any | None,
    ) -> tuple[str, str, Any | None, bool, bool]:
        normalized_device = (requested_device or "auto").lower()
        normalized_dtype = (requested_dtype or "auto").lower()
        if normalized_device not in {"auto", "cpu", "cuda"}:
            msg = f"Unsupported device: {requested_device}"
            raise ValueError(msg)
        if normalized_dtype not in {"auto", "float32", "float16", "bfloat16"}:
            msg = f"Unsupported dtype: {requested_dtype}"
            raise ValueError(msg)

        cuda_available = False
        if torch_module is not None:
            try:
                cuda_available = bool(torch_module.cuda.is_available())
            except Exception:
                cuda_available = False

        resolved_device = "cuda" if (normalized_device == "auto" and cuda_available) else normalized_device
        if normalized_device == "auto" and not cuda_available:
            resolved_device = "cpu"

        if normalized_dtype == "auto":
            resolved_dtype = "float16" if resolved_device == "cuda" else "float32"
        else:
            resolved_dtype = normalized_dtype

        dtype_overridden = False
        if resolved_device == "cpu" and resolved_dtype in {"float16", "bfloat16"}:
            resolved_dtype = "float32"
            dtype_overridden = True

        amp_enabled = resolved_device == "cuda" and resolved_dtype in {"float16", "bfloat16"}
        torch_dtype = getattr(torch_module, resolved_dtype, None) if torch_module is not None else None
        return resolved_device, resolved_dtype, torch_dtype, amp_enabled, dtype_overridden

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

    @classmethod
    def _raw_score_payload(
        cls,
        *,
        chosen_evidence_id: str | None,
        per_item_probs: list[dict[str, float | str]],
        label: str,
        tokenizer_name: str | None,
    ) -> dict[str, Any]:
        return {
            "chosen_evidence_id": chosen_evidence_id,
            "per_item_probs": per_item_probs,
            "model_name": DEFAULT_MODEL_NAME,
            "tokenizer_name": tokenizer_name,
            "has_contradiction": label == "contradiction",
        }

    @staticmethod
    def _bm25_tokenize(text: str) -> list[str]:
        return [token for token in text.lower().split() if token]

    @classmethod
    def _fallback_scores(cls, query: str, evidence_texts: list[str]) -> list[float]:
        query_tokens = set(cls._bm25_tokenize(query))
        if not query_tokens:
            return [0.0 for _ in evidence_texts]
        scores: list[float] = []
        for text in evidence_texts:
            evidence_tokens = set(cls._bm25_tokenize(text))
            if not evidence_tokens:
                scores.append(0.0)
                continue
            overlap = len(query_tokens & evidence_tokens) / max(1, len(query_tokens))
            scores.append(float(overlap))
        return scores

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        if not text.strip():
            return []
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [part.strip() for part in parts if part.strip()]

    def _truncate_evidence_text(self, text: str) -> str:
        out = text
        if self.evidence_max_sentences is not None and self.evidence_max_sentences > 0:
            sentences = self._split_sentences(out)
            if sentences:
                out = " ".join(sentences[: self.evidence_max_sentences])
        if self.evidence_max_chars is not None and self.evidence_max_chars > 0:
            out = out[: self.evidence_max_chars]
        return out

    def _preprocess_evidence_texts(self, evidence: EvidenceSet) -> tuple[list[str], dict[str, float]]:
        before_lengths = [len(item.text) for item in evidence.items]
        processed = [self._truncate_evidence_text(item.text) for item in evidence.items]
        after_lengths = [len(text) for text in processed]
        total_before = float(sum(before_lengths))
        total_after = float(sum(after_lengths))
        truncated_frac = 0.0
        if total_before > 0:
            truncated_frac = max(0.0, min(1.0, (total_before - total_after) / total_before))
        mean_before = (total_before / len(before_lengths)) if before_lengths else 0.0
        mean_after = (total_after / len(after_lengths)) if after_lengths else 0.0
        return processed, {
            "evidence_truncated_frac": truncated_frac,
            "evidence_chars_mean_before": mean_before,
            "evidence_chars_mean_after": mean_after,
        }

    def _build_stage1_candidates(
        self,
        *,
        candidate: AnswerCandidate,
        evidence_ids: list[str],
        evidence_texts: list[str],
        n_total_evidence: int,
    ) -> tuple[list[tuple[int, int, float]], dict[str, int], dict[str, dict[str, Any]]]:
        n_units = len(candidate.units)
        if n_units == 0 or n_total_evidence == 0:
            return [], {"pairs_pruned_stage1": 0, "pairs_pruned_stage2": 0}, {}

        tokenized = [self._bm25_tokenize(text) for text in evidence_texts]
        bm25 = None
        try:
            from rank_bm25 import BM25Okapi

            bm25 = BM25Okapi(tokenized)
        except Exception:
            bm25 = None

        topk = max(0, int(self.topk_per_unit))
        selected: list[tuple[int, int, float]] = []
        per_unit_trace: dict[str, dict[str, Any]] = {}
        for unit_idx, unit in enumerate(candidate.units):
            unit_query = unit.text
            used_field_query = False
            used_fallback = False
            field_meta = self._structured_field_metadata(unit)
            if field_meta is not None:
                field_path, field_name, field_type = field_meta
                unit_query = self._build_field_aware_query(
                    field_path=field_path,
                    field_name=field_name,
                    field_type=field_type,
                    field_value=unit.text,
                )
                used_field_query = True

            if bm25 is not None:
                query_tokens = self._bm25_tokenize(unit_query)
                raw_scores = bm25.get_scores(query_tokens)
                scores = [float(value) for value in raw_scores]
                if used_field_query and not any(score > 0.0 for score in scores):
                    fallback_query = unit.text
                    fallback_tokens = self._bm25_tokenize(fallback_query)
                    fallback_raw_scores = bm25.get_scores(fallback_tokens)
                    scores = [float(value) for value in fallback_raw_scores]
                    used_fallback = True
            else:
                scores = self._fallback_scores(unit_query, evidence_texts)
                if used_field_query and not any(score > 0.0 for score in scores):
                    scores = self._fallback_scores(unit.text, evidence_texts)
                    used_fallback = True

            ranked = sorted(
                ((idx, score) for idx, score in enumerate(scores)),
                key=lambda item: (-item[1], evidence_ids[item[0]]),
            )
            keep = ranked[:topk] if topk > 0 else []
            keep_by_index = sorted((evidence_idx for evidence_idx, _ in keep))
            selected.extend(
                (unit_idx, evidence_idx, scores[evidence_idx]) for evidence_idx in keep_by_index
            )
            if used_fallback:
                per_unit_trace[str(unit.id)] = {"field_query_fallback": True}

        original_pairs = n_units * n_total_evidence
        stage_a_pairs = len(selected)
        pairs_pruned_stage1 = max(0, original_pairs - stage_a_pairs)

        if self.max_pairs_total is not None and self.max_pairs_total > 0 and stage_a_pairs > self.max_pairs_total:
            selected = sorted(
                selected,
                key=lambda item: (-item[2], item[1], item[0]),
            )[: self.max_pairs_total]
            selected = sorted(selected, key=lambda item: (item[0], item[1]))
        stage_b_pairs = len(selected)
        pairs_pruned_stage2 = max(0, stage_a_pairs - stage_b_pairs)
        return selected, {
            "pairs_pruned_stage1": pairs_pruned_stage1,
            "pairs_pruned_stage2": pairs_pruned_stage2,
        }, per_unit_trace

    @staticmethod
    def _structured_field_metadata(unit: Unit) -> tuple[str, str, str] | None:
        metadata = unit.metadata if isinstance(unit.metadata, dict) else {}
        field_path = getattr(unit, "field_path", None) or metadata.get("field_path") or metadata.get("path")
        field_name = getattr(unit, "field_name", None) or metadata.get("field_name")
        field_type = getattr(unit, "field_type", None) or metadata.get("field_type")
        if not (isinstance(field_path, str) and field_path.strip()):
            return None
        if not (isinstance(field_name, str) and field_name.strip()):
            return None
        if not (isinstance(field_type, str) and field_type.strip()):
            return None
        return field_path.strip(), field_name.strip(), field_type.strip().lower()

    @staticmethod
    def _build_field_aware_query(
        *,
        field_path: str,
        field_name: str,
        field_type: str,
        field_value: str,
    ) -> str:
        value_str = str(field_value)
        if field_type == "number":
            value_token = value_str
        elif field_type == "date":
            value_token = value_str
        else:
            value_token = value_str
        return f"{field_name} {field_path} {value_token}".strip()

    @staticmethod
    def _pack_by_token_budget(
        *,
        ordered_pair_indices: list[int],
        estimated_lengths: list[int],
        max_batch_tokens: int,
    ) -> list[list[int]]:
        if not ordered_pair_indices:
            return []
        if max_batch_tokens <= 0:
            return [list(ordered_pair_indices)]

        batches: list[list[int]] = []
        current: list[int] = []
        current_tokens = 0
        for pair_idx in ordered_pair_indices:
            pair_tokens = max(1, int(estimated_lengths[pair_idx]))
            if current and current_tokens + pair_tokens > max_batch_tokens:
                batches.append(current)
                current = []
                current_tokens = 0
            current.append(pair_idx)
            current_tokens += pair_tokens
        if current:
            batches.append(current)
        return batches

    def _estimate_pair_lengths(self, pairs: list[tuple[str, str]]) -> list[int]:
        if not pairs:
            return []
        if self.pair_predictor is not None or self.tokenizer is None:
            return [min(self.max_length, len(a.split()) + len(b.split()) + 3) for a, b in pairs]
        units = [a for a, _ in pairs]
        evidences = [b for _, b in pairs]
        try:
            encoded = self.tokenizer(
                units,
                evidences,
                padding=False,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=True,
                return_length=True,
            )
            lengths = encoded.get("length")
            if lengths is None:
                raise ValueError("missing length")
            return [int(v) for v in lengths]
        except Exception:
            return [min(self.max_length, len(a.split()) + len(b.split()) + 3) for a, b in pairs]

    @staticmethod
    def _percentile(values: list[int], q: float) -> float:
        if not values:
            return 0.0
        if len(values) == 1:
            return float(values[0])
        sorted_values = sorted(values)
        idx = int(round((len(sorted_values) - 1) * q))
        idx = max(0, min(idx, len(sorted_values) - 1))
        return float(sorted_values[idx])


    def verify(
        self,
        units: list[Unit] | AnswerCandidate,
        evidence: EvidenceSet,
    ) -> list[VerificationScore]:
        if isinstance(units, AnswerCandidate):
            return self.verify_many(units, evidence)
        candidate = AnswerCandidate(raw_answer_text="\n".join(unit.text for unit in units), units=units)
        return self.verify_many(candidate, evidence)

    def get_last_verify_trace(self) -> dict[str, Any]:
        return dict(self._last_verify_trace)

    def _verify_unit_with_id(
        self,
        *,
        unit_id: str,
        unit_text: str,
        evidence: EvidenceSet,
    ) -> VerificationScore:
        if not evidence.items:
            empty = {"entailment": 0.0, "contradiction": 0.0, "neutral": 1.0}
            label = "neutral"
            return VerificationScore(
                unit_id=unit_id,
                entailment=empty["entailment"],
                contradiction=empty["contradiction"],
                neutral=empty["neutral"],
                label=label,
                raw=self._raw_score_payload(
                    chosen_evidence_id=None,
                    per_item_probs=[],
                    label=label,
                    tokenizer_name=getattr(self.tokenizer, "name_or_path", self.model_name),
                ),
            )

        pairs = [(unit_text, item.text) for item in evidence.items]
        per_item_probs = self._predict_pair_probabilities(pairs)
        best_idx = max(
            range(len(per_item_probs)),
            key=lambda idx: per_item_probs[idx]["entailment"],
        )
        best_probs = per_item_probs[best_idx]
        label = self._label_from_probs(best_probs)

        return VerificationScore(
            unit_id=unit_id,
            entailment=best_probs["entailment"],
            contradiction=best_probs["contradiction"],
            neutral=best_probs["neutral"],
            label=label,
            raw=self._raw_score_payload(
                chosen_evidence_id=evidence.items[best_idx].id,
                per_item_probs=[
                    {"evidence_id": item.id, **probs}
                    for item, probs in zip(evidence.items, per_item_probs, strict=True)
                ],
                label=label,
                tokenizer_name=getattr(self.tokenizer, "name_or_path", self.model_name),
            ),
            nli_score=best_probs["entailment"],
        )

    def verify_many(self, candidate: AnswerCandidate, evidence: EvidenceSet) -> list[VerificationScore]:
        """Verify all candidate units against evidence in a single batched model call."""
        trace: dict[str, Any] = {
            "preselect_seconds": 0.0,
            "tokenize_seconds": 0.0,
            "forward_seconds": 0.0,
            "post_seconds": 0.0,
            "num_batches": 0,
            "batch_size_mean": 0.0,
            "batch_size_max": 0,
            "seq_len_mean": 0.0,
            "seq_len_p50": 0.0,
            "seq_len_p95": 0.0,
            "tokens_total": 0,
            "device": None,
            "dtype": None,
            "amp_enabled": False,
            "compiled_enabled": False,
            "pairs_pruned_stage1": 0,
            "pairs_pruned_stage2": 0,
            "dtype_overridden": False,
            "n_pairs_scored": 0,
            "evidence_truncated_frac": 0.0,
            "evidence_chars_mean_before": 0.0,
            "evidence_chars_mean_after": 0.0,
        }
        trace["device"] = str(self.device)
        trace["dtype"] = str(self.dtype)
        trace["amp_enabled"] = bool(self.amp_enabled)
        trace["dtype_overridden"] = bool(self.dtype_overridden)
        if self.pair_predictor is None:
            trace["compiled_enabled"] = bool(
                getattr(self.model, "_orig_mod", None) is not None
                or getattr(self.model, "_compiled_call_impl", None) is not None
            )

        if not candidate.units:
            self._last_verify_trace = trace
            return []

        if not evidence.items:
            empty = {"entailment": 0.0, "contradiction": 0.0, "neutral": 1.0}
            scores = [
                VerificationScore(
                    unit_id=unit.id,
                    entailment=empty["entailment"],
                    contradiction=empty["contradiction"],
                    neutral=empty["neutral"],
                    label="neutral",
                    raw=self._raw_score_payload(
                        chosen_evidence_id=None,
                        per_item_probs=[],
                        label="neutral",
                        tokenizer_name=getattr(self.tokenizer, "name_or_path", self.model_name),
                    ),
                )
                for unit in candidate.units
            ]
            self._last_verify_trace = trace
            return sorted(scores, key=lambda s: s.unit_id)

        capped_evidence = evidence.items
        if self.max_evidence_per_request is not None and self.max_evidence_per_request > 0:
            capped_evidence = evidence.items[: self.max_evidence_per_request]
        processed_evidence_texts, trunc_stats = self._preprocess_evidence_texts(
            EvidenceSet(items=list(capped_evidence))
        )
        trace.update(trunc_stats)

        preselect_t0 = time.perf_counter()
        selected_pairs, prune_stats, per_unit_preselect = self._build_stage1_candidates(
            candidate=candidate,
            evidence_ids=[item.id for item in capped_evidence],
            evidence_texts=processed_evidence_texts,
            n_total_evidence=len(capped_evidence),
        )
        trace["preselect_seconds"] = time.perf_counter() - preselect_t0
        trace["pairs_pruned_stage1"] = int(prune_stats["pairs_pruned_stage1"])
        trace["pairs_pruned_stage2"] = int(prune_stats["pairs_pruned_stage2"])
        trace["n_pairs_scored"] = len(selected_pairs)
        if per_unit_preselect:
            trace["per_unit_preselect"] = per_unit_preselect

        pairs = [
            (candidate.units[unit_idx].text, processed_evidence_texts[evidence_idx])
            for unit_idx, evidence_idx, _ in selected_pairs
        ]
        if not pairs:
            empty = {"entailment": 0.0, "contradiction": 0.0, "neutral": 1.0}
            scores = [
                VerificationScore(
                    unit_id=unit.id,
                    entailment=empty["entailment"],
                    contradiction=empty["contradiction"],
                    neutral=empty["neutral"],
                    label="neutral",
                    raw=self._raw_score_payload(
                        chosen_evidence_id=None,
                        per_item_probs=[],
                        label="neutral",
                        tokenizer_name=getattr(self.tokenizer, "name_or_path", self.model_name),
                    ),
                )
                for unit in candidate.units
            ]
            self._last_verify_trace = trace
            return sorted(scores, key=lambda s: s.unit_id)

        estimated_lengths = self._estimate_pair_lengths(pairs)
        sorted_pair_indices = sorted(
            range(len(pairs)),
            key=lambda idx: (
                candidate.units[selected_pairs[idx][0]].id,
                capped_evidence[selected_pairs[idx][1]].id,
            ),
        )
        batches = self._pack_by_token_budget(
            ordered_pair_indices=sorted_pair_indices,
            estimated_lengths=estimated_lengths,
            max_batch_tokens=int(self.max_batch_tokens or 0),
        )
        batch_sizes = [len(batch) for batch in batches]
        trace["num_batches"] = len(batches)
        trace["batch_size_mean"] = (
            float(sum(batch_sizes)) / float(len(batch_sizes)) if batch_sizes else 0.0
        )
        trace["batch_size_max"] = max(batch_sizes) if batch_sizes else 0

        all_probs: list[dict[str, float]] = [
            {"entailment": 0.0, "contradiction": 0.0, "neutral": 1.0}
            for _ in pairs
        ]
        seq_lens_all: list[int] = []
        tokenize_total = 0.0
        forward_total = 0.0

        if self.pair_predictor is not None:
            forward_t0 = time.perf_counter()
            for batch in batches:
                batch_pairs = [pairs[idx] for idx in batch]
                batch_probs = self.pair_predictor(batch_pairs)
                for local_idx, pair_idx in enumerate(batch):
                    all_probs[pair_idx] = batch_probs[local_idx]
            forward_total = time.perf_counter() - forward_t0
            seq_lens_all = [int(estimated_lengths[idx]) for idx in range(len(estimated_lengths))]
        else:
            for batch in batches:
                batch_pairs = [pairs[idx] for idx in batch]
                batch_units = [unit_text for unit_text, _ in batch_pairs]
                batch_evidence = [evidence_text for _, evidence_text in batch_pairs]
                tokenize_t0 = time.perf_counter()
                encoded = self.tokenizer(
                    batch_units,
                    batch_evidence,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                tokenize_total += time.perf_counter() - tokenize_t0
                encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}
                attention_mask = encoded.get("attention_mask")
                if attention_mask is not None:
                    seq_lens_all.extend(int(x) for x in attention_mask.sum(dim=1).tolist())
                forward_t0 = time.perf_counter()
                with self._torch.inference_mode():
                    autocast_ctx = (
                        self._torch.autocast(device_type="cuda", dtype=self._torch_dtype)
                        if self.amp_enabled and self.device == "cuda" and self._torch_dtype is not None
                        else contextlib.nullcontext()
                    )
                    with autocast_ctx:
                        logits = self.model(**encoded).logits
                    probs = self._torch.softmax(logits, dim=-1).cpu()
                forward_total += time.perf_counter() - forward_t0
                for local_idx, pair_idx in enumerate(batch):
                    row = probs[local_idx]
                    all_probs[pair_idx] = {
                        "entailment": float(row[self._label_indices["entailment"]].item()),
                        "contradiction": float(row[self._label_indices["contradiction"]].item()),
                        "neutral": float(row[self._label_indices["neutral"]].item()),
                    }

        trace["tokenize_seconds"] = tokenize_total
        trace["forward_seconds"] = forward_total
        trace["tokens_total"] = int(sum(seq_lens_all))
        trace["seq_len_mean"] = (
            float(trace["tokens_total"]) / float(len(seq_lens_all)) if seq_lens_all else 0.0
        )
        trace["seq_len_p50"] = self._percentile(seq_lens_all, 0.50)
        trace["seq_len_p95"] = self._percentile(seq_lens_all, 0.95)

        post_t0 = time.perf_counter()
        per_unit_pairs: dict[int, list[tuple[int, dict[str, float]]]] = {}
        for idx, (unit_idx, evidence_idx, _score) in enumerate(selected_pairs):
            probs = all_probs[idx] if idx < len(all_probs) else {
                "entailment": 0.0,
                "contradiction": 0.0,
                "neutral": 1.0,
            }
            per_unit_pairs.setdefault(unit_idx, []).append((evidence_idx, probs))
        scores: list[VerificationScore] = []
        for unit_index, unit in enumerate(candidate.units):
            unit_pairs = per_unit_pairs.get(unit_index, [])
            if not unit_pairs:
                empty = {"entailment": 0.0, "contradiction": 0.0, "neutral": 1.0}
                scores.append(
                    VerificationScore(
                        unit_id=unit.id,
                        entailment=empty["entailment"],
                        contradiction=empty["contradiction"],
                        neutral=empty["neutral"],
                        label="neutral",
                        raw=self._raw_score_payload(
                            chosen_evidence_id=None,
                            per_item_probs=[],
                            label="neutral",
                            tokenizer_name=getattr(self.tokenizer, "name_or_path", self.model_name),
                        ),
                    )
                )
                continue

            best_evidence_idx, best_probs = max(
                unit_pairs,
                key=lambda row: row[1]["entailment"],
            )
            label = self._label_from_probs(best_probs)
            scores.append(
                VerificationScore(
                    unit_id=unit.id,
                    entailment=best_probs["entailment"],
                    contradiction=best_probs["contradiction"],
                    neutral=best_probs["neutral"],
                    label=label,
                    raw=self._raw_score_payload(
                        chosen_evidence_id=evidence.items[best_evidence_idx].id,
                        per_item_probs=[
                            {"evidence_id": evidence.items[evidence_idx].id, **probs}
                            for evidence_idx, probs in unit_pairs
                        ],
                        label=label,
                        tokenizer_name=getattr(self.tokenizer, "name_or_path", self.model_name),
                    ),
                    nli_score=best_probs["entailment"],
                )
            )
        trace["post_seconds"] = time.perf_counter() - post_t0
        self._last_verify_trace = trace
        return sorted(scores, key=lambda s: s.unit_id)

    def verify_unit(self, unit_text: str, evidence: EvidenceSet) -> VerificationScore:
        """Verify one unit against evidence and return the best entailment score."""
        candidate = AnswerCandidate(
            raw_answer_text=unit_text,
            units=[Unit(id="unit", text=unit_text, metadata={})],
        )
        return self.verify_many(candidate, evidence)[0]

NLICrossEncoderVerifier = NliCrossEncoderVerifier
