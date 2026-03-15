"""Helpers for packaging the final v2 pilot POC artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ega.v2.poc_config import (
    DEFAULT_ACCEPT_THRESHOLD,
    DEFAULT_FINAL_CONFORMAL_STATE,
    DEFAULT_FINAL_DATASET,
    DEFAULT_FINAL_REPORT,
    DEFAULT_FINAL_SOURCE_SUMMARY,
    DEFAULT_FINAL_SUMMARY,
    DEFAULT_MODEL_NAME,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_RERANK_TOPK,
    EXPERIMENTAL_VARIANTS,
    RECOMMENDED_VARIANTS,
)

_METRIC_KEYS = (
    "kept_units",
    "dropped_units",
    "unsupported_claim_rate",
    "hallucination_rate",
    "abstention_rate",
    "gold_coverage_recall",
    "avg_reward",
    "verifier_calls_proxy",
    "verifier_cost",
    "reranker_cost",
    "cost_proxy",
    "p50_total_seconds",
    "p95_total_seconds",
)


def build_final_poc_summary(
    *,
    source_summary_path: str | Path = DEFAULT_FINAL_SOURCE_SUMMARY,
    out_path: str | Path = DEFAULT_FINAL_SUMMARY,
    dataset_path: str | Path = DEFAULT_FINAL_DATASET,
    conformal_state_path: str | Path = DEFAULT_FINAL_CONFORMAL_STATE,
    accept_threshold: float = DEFAULT_ACCEPT_THRESHOLD,
    reranker_model: str = DEFAULT_RERANKER_MODEL,
    include_experimental: bool = False,
    budget_status: str = "experimental_not_recommended",
) -> dict[str, Any]:
    source = _load_json(source_summary_path)
    variants = source.get("variants", {})
    if not isinstance(variants, dict):
        raise ValueError("Source eval summary missing 'variants' object.")
    dataset_rows = _count_jsonl_rows(dataset_path)
    resolved_conformal_state_path = Path(conformal_state_path)
    recommended_variants = list(RECOMMENDED_VARIANTS)
    variant_entries = {
        variant_name: _build_variant_entry(
            variant_name=variant_name,
            source_variant=variants.get(variant_name),
            status="recommended",
            fallback_verifier_model_name=DEFAULT_MODEL_NAME,
            config=_variant_config(
                variant_name=variant_name,
                accept_threshold=accept_threshold,
                reranker_model=reranker_model,
                conformal_state_path=resolved_conformal_state_path,
            ),
        )
        for variant_name in RECOMMENDED_VARIANTS
    }
    if include_experimental:
        for variant_name in EXPERIMENTAL_VARIANTS:
            variant_entries[variant_name] = _build_variant_entry(
                variant_name=variant_name,
                source_variant=variants.get(variant_name),
                status=budget_status,
                fallback_verifier_model_name=DEFAULT_MODEL_NAME,
                config=_variant_config(
                    variant_name=variant_name,
                    accept_threshold=accept_threshold,
                    reranker_model=reranker_model,
                    conformal_state_path=resolved_conformal_state_path,
                ),
            )

    summary = {
        "dataset_path": str(Path(dataset_path)),
        "n_examples": dataset_rows,
        "source_eval_summary_path": str(Path(source_summary_path)),
        "verifier_model_name": _resolve_summary_verifier_model_name(
            variants=variants,
            fallback=DEFAULT_MODEL_NAME,
        ),
        "reranker_model_name": reranker_model,
        "accept_threshold": float(accept_threshold),
        "conformal_state_path": str(resolved_conformal_state_path),
        "conformal_note": (
            "Conformal state regenerated from exported calibration rows under the current "
            "verifier semantics."
        ),
        "budget_status": budget_status,
        "recommended_variants": recommended_variants,
        "experimental_variants": list(EXPERIMENTAL_VARIANTS) if include_experimental else [],
        "variants": variant_entries,
    }
    resolved_out = Path(out_path)
    resolved_out.parent.mkdir(parents=True, exist_ok=True)
    resolved_out.write_text(json.dumps(summary, sort_keys=True, indent=2), encoding="utf-8")
    return summary


def write_poc_results_markdown(
    *,
    summary_path: str | Path = DEFAULT_FINAL_SUMMARY,
    out_path: str | Path = DEFAULT_FINAL_REPORT,
) -> str:
    summary = _load_json(summary_path)
    variants = summary.get("variants", {})
    if not isinstance(variants, dict):
        raise ValueError("Final POC summary missing 'variants' object.")

    rows = [
        _table_row("v1_baseline", variants["v1_baseline"]),
        _table_row("rerank_only", variants["rerank_only"]),
        _table_row("conformal_only", variants["conformal_only"]),
        _table_row("combined", variants["combined"]),
    ]
    lines = [
        "# EGA v2 POC Results",
        "",
        f"- Dataset: `{summary['dataset_path']}`",
        f"- Examples: `{summary['n_examples']}`",
        f"- Verifier model: `{summary['verifier_model_name']}`",
        f"- Reranker model: `{summary['reranker_model_name']}`",
        f"- Accept threshold: `{summary['accept_threshold']}`",
        f"- Conformal calibration: {summary['conformal_note']} (`{summary['conformal_state_path']}`)",
        "",
        "| Variant | Status | Kept | Dropped | Unsupported | Hallucination | Abstention | Gold Recall | Avg Reward | Verifier Calls | Verifier Cost | Reranker Cost | Cost | p50 s | p95 s |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *rows,
        "",
        "## Takeaways",
        "",
        "- The release POC is fixed to four headline variants: `v1_baseline`, `rerank_only`, `conformal_only`, and `combined`.",
        "- `0.05` is the recommended operating threshold because it is the current safe point on the pilot artifact under the present verifier semantics.",
        "- `rerank_only` reduces verifier calls modestly relative to `v1_baseline` while preserving the same verifier threshold.",
        "- `conformal_only` changes answer behavior through abstention rather than verifier cost reduction.",
        "- `combined` is the publishable composition of reranking and conformal abstention without budget claims.",
    ]
    if "budget_only" in variants:
        lines.append(
            "- `budget_only` remains experimental and is excluded from the headline comparison table."
        )
    lines.extend(
        [
            "",
            "## Safe-Answer Example",
            "",
            "```text",
            "The Eiffel Tower is in Paris. [e1]",
            "It opened in 1889. [e3]",
            "```",
            "",
            "## Limitations",
            "",
            "- The evaluation is still a small pilot dataset and should not be presented as a broad benchmark.",
            "- The conformal threshold is specific to calibration rows exported under the current verifier semantics and should be regenerated if the verifier path changes.",
            "- The reported cost fields are proxies based on scored verification and reranking pairs, not production billing measurements.",
            "- Budget mode is experimental unless a binding budget run demonstrates real pair-count reduction with the expected tradeoff.",
            "",
        ]
    )
    markdown = "\n".join(lines)
    resolved_out = Path(out_path)
    resolved_out.parent.mkdir(parents=True, exist_ok=True)
    resolved_out.write_text(markdown + "\n", encoding="utf-8")
    return markdown


def _build_variant_entry(
    *,
    variant_name: str,
    source_variant: Any,
    status: str,
    fallback_verifier_model_name: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    source_dict = source_variant if isinstance(source_variant, dict) else {}
    source_metrics = source_dict.get("metrics", {}) if isinstance(source_dict, dict) else {}
    metrics = {
        key: (source_metrics.get(key) if isinstance(source_metrics, dict) else None)
        for key in _METRIC_KEYS
    }
    metrics_metadata = source_dict.get("metrics_metadata", {}) if isinstance(source_dict, dict) else {}
    debug = source_dict.get("debug", {}) if isinstance(source_dict, dict) else {}
    conformal_threshold = None
    if isinstance(metrics_metadata, dict):
        threshold_value = metrics_metadata.get("conformal_threshold")
        if isinstance(threshold_value, (int, float)):
            conformal_threshold = float(threshold_value)
    verifier_model_name = _resolve_variant_verifier_model_name(
        source_variant=source_dict,
        fallback=fallback_verifier_model_name,
    )
    config_with_model = dict(config)
    config_with_model["verifier_model_name"] = verifier_model_name
    debug_with_model = dict(debug) if isinstance(debug, dict) else {}
    debug_with_model["verifier_model_name"] = verifier_model_name
    entry = {
        "variant_name": variant_name,
        "status": status,
        "source_run_status": str(source_dict.get("status", "missing")),
        "source_errors": list(source_dict.get("errors", [])) if isinstance(source_dict.get("errors", []), list) else [],
        "config": config_with_model,
        "accept_threshold": config["accept_threshold"],
        "verifier_model_name": verifier_model_name,
        "reranker_enabled": config["reranker_enabled"],
        "reranker_model_name": config["reranker_model_name"],
        "conformal_threshold": conformal_threshold,
        "recommended": status == "recommended",
        "debug": debug_with_model,
    }
    for key, value in metrics.items():
        entry[key] = value
    if isinstance(metrics_metadata, dict) and metrics_metadata:
        entry["conformal_metadata"] = dict(metrics_metadata)
    return entry


def _table_row(name: str, entry: dict[str, Any]) -> str:
    status = str(entry.get("status", "unknown"))
    return (
        f"| `{name}` | {status} | "
        f"{_fmt(entry.get('kept_units'))} | {_fmt(entry.get('dropped_units'))} | "
        f"{_fmt(entry.get('unsupported_claim_rate'))} | {_fmt(entry.get('hallucination_rate'))} | "
        f"{_fmt(entry.get('abstention_rate'))} | {_fmt(entry.get('gold_coverage_recall'))} | "
        f"{_fmt(entry.get('avg_reward'))} | {_fmt(entry.get('verifier_calls_proxy'))} | "
        f"{_fmt(entry.get('verifier_cost'))} | {_fmt(entry.get('reranker_cost'))} | "
        f"{_fmt(entry.get('cost_proxy'))} | {_fmt(entry.get('p50_total_seconds'))} | "
        f"{_fmt(entry.get('p95_total_seconds'))} |"
    )


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}.")
    return payload


def _count_jsonl_rows(path: str | Path) -> int:
    total = 0
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                total += 1
    return total


def _variant_config(
    *,
    variant_name: str,
    accept_threshold: float,
    reranker_model: str,
    conformal_state_path: Path,
) -> dict[str, Any]:
    use_reranker = variant_name in {"rerank_only", "combined"}
    use_conformal = variant_name in {"conformal_only", "combined"}
    use_budget = variant_name == "budget_only"
    return {
        "accept_threshold": float(accept_threshold),
        "verifier_model_name": DEFAULT_MODEL_NAME,
        "reranker_enabled": use_reranker,
        "reranker_model_name": reranker_model if use_reranker else None,
        "rerank_topk": DEFAULT_RERANK_TOPK if use_reranker else None,
        "conformal_state_path": str(conformal_state_path) if use_conformal else None,
        "budget_enabled": use_budget,
    }


def _resolve_summary_verifier_model_name(*, variants: dict[str, Any], fallback: str) -> str:
    for variant_name in RECOMMENDED_VARIANTS:
        verifier_model_name = _resolve_variant_verifier_model_name(
            source_variant=variants.get(variant_name),
            fallback="",
        )
        if verifier_model_name:
            return verifier_model_name
    return fallback


def _resolve_variant_verifier_model_name(*, source_variant: Any, fallback: str) -> str:
    source_dict = source_variant if isinstance(source_variant, dict) else {}
    debug = source_dict.get("debug", {})
    if isinstance(debug, dict):
        model_name = debug.get("verifier_model_name")
        if isinstance(model_name, str) and model_name.strip():
            return model_name
    config = source_dict.get("config", {})
    if isinstance(config, dict):
        model_name = config.get("verifier_model_name")
        if isinstance(model_name, str) and model_name.strip():
            return model_name
    return fallback
