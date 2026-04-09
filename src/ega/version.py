"""Package version utilities.

The canonical version is defined in ``pyproject.toml``.
This module reads installed package metadata and falls back to
``pyproject.toml`` when running directly from source.
"""

from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _version_from_pyproject() -> str:
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    contents = pyproject_path.read_text(encoding="utf-8")
    match = re.search(r'(?m)^\s*version\s*=\s*"([^"]+)"\s*$', contents)
    if match is None:
        raise RuntimeError("Unable to read project.version from pyproject.toml.")
    return match.group(1)


def _resolve_version() -> str:
    try:
        return version("ega")
    except PackageNotFoundError:
        return _version_from_pyproject()


__version__ = _resolve_version()
