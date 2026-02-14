from __future__ import annotations

import types

import pytest

from ega import cli


def _version(major: int, minor: int) -> types.SimpleNamespace:
    return types.SimpleNamespace(major=major, minor=minor)


def test_cli_exits_on_unsupported_python(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setenv("EGA_ENFORCE_PYTHON_CHECK", "1")
    monkeypatch.setattr(cli, "_runtime_version_info", lambda: _version(3, 14))

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "[EGA ERROR] Unsupported Python 3.14. Use 3.10-3.12." in captured.err


def test_cli_allows_supported_python_when_check_enforced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EGA_ENFORCE_PYTHON_CHECK", "1")
    monkeypatch.setattr(cli, "_runtime_version_info", lambda: _version(3, 12))
    monkeypatch.setattr(cli.sys, "argv", ["ega", "--version"])

    exit_code = cli.main()

    assert exit_code == 0
