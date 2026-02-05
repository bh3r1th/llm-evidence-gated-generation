"""Tests for optional W&B event sink adapter."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from ega.adapters import make_wandb_sink


class _FakeTable:
    def __init__(self, *, columns: list[str]) -> None:
        self.columns = columns
        self.rows: list[tuple[str, str, str]] = []

    def add_data(self, timestamp: str, run_id: str, event_json: str) -> None:
        self.rows.append((timestamp, run_id, event_json))


class _FakeRun:
    def __init__(self) -> None:
        self.logged: list[dict[str, object]] = []

    def log(self, payload: dict[str, object]) -> None:
        self.logged.append(payload)


def test_wandb_sink_is_lazy_and_logs_scalars_and_event(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_run = _FakeRun()
    init_calls: list[dict[str, object]] = []

    def fake_init(*, project: str, entity: str | None, tags: list[str] | None) -> _FakeRun:
        init_calls.append({"project": project, "entity": entity, "tags": tags})
        return fake_run

    fake_module = SimpleNamespace(init=fake_init, Table=_FakeTable)
    monkeypatch.setitem(sys.modules, "wandb", fake_module)

    sink = make_wandb_sink(project="proj", entity="team", tags=["ega"])
    assert init_calls == []

    sink(
        {
            "timestamp": "2024-01-01T00:00:00+00:00",
            "run_id": "run-1",
            "kept_count": 2,
            "unit_count": 3,
            "refusal": False,
            "summary_stats": {"mean_entailment": 0.95, "reason": "ignored"},
        }
    )

    assert len(init_calls) == 1
    assert init_calls[0] == {"project": "proj", "entity": "team", "tags": ["ega"]}
    assert len(fake_run.logged) == 1
    payload = fake_run.logged[0]
    assert payload["ega/kept_count"] == 2
    assert payload["ega/unit_count"] == 3
    assert payload["ega/refusal"] == 0
    assert payload["ega/mean_entailment"] == 0.95
    assert "ega/reason" not in payload
    assert isinstance(payload["ega/event"], _FakeTable)


def test_wandb_sink_raises_clear_error_when_wandb_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "wandb", raising=False)

    sink = make_wandb_sink(project="proj")

    with pytest.raises(ImportError, match="pip install ega\\[wandb\\]"):
        sink({"summary_stats": {}})
