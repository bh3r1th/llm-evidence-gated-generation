"""Weights & Biases event sink adapter.

This module intentionally keeps W&B optional by importing it lazily when the
returned sink is first called.
"""

from __future__ import annotations

import json
from typing import Any

from ega.enforcer import EventSink


def make_wandb_sink(
    project: str,
    entity: str | None = None,
    tags: list[str] | None = None,
) -> EventSink:
    """Create an event sink that logs EGA events to Weights & Biases.

    The W&B run is initialized lazily on first event, which keeps the core
    runtime free from a hard dependency on ``wandb``.
    """

    run: Any | None = None
    wandb_module: Any | None = None

    def sink(event: dict[str, Any]) -> None:
        nonlocal run, wandb_module

        if run is None:
            try:
                import wandb as _wandb  # type: ignore[import-not-found]
            except ImportError as exc:  # pragma: no cover - environment-dependent
                raise ImportError(
                    "wandb is not installed. Install with `pip install ega[wandb]`."
                ) from exc

            wandb_module = _wandb
            run = wandb_module.init(project=project, entity=entity, tags=tags)

        log_payload: dict[str, Any] = {
            "ega/kept_count": event.get("kept_count"),
            "ega/unit_count": event.get("unit_count"),
            "ega/refusal": int(bool(event.get("refusal", False))),
        }

        summary_stats = event.get("summary_stats", {})
        if isinstance(summary_stats, dict):
            for key, value in summary_stats.items():
                if isinstance(value, (int, float, bool)):
                    log_payload[f"ega/{key}"] = value

        if wandb_module is None:  # pragma: no cover - defensive
            raise RuntimeError("wandb module was not initialized.")

        event_table = wandb_module.Table(columns=["timestamp", "run_id", "event_json"])
        event_table.add_data(
            str(event.get("timestamp", "")),
            str(event.get("run_id", "")),
            json.dumps(event, sort_keys=True, ensure_ascii=False),
        )
        log_payload["ega/event"] = event_table

        run.log(log_payload)

    return sink
