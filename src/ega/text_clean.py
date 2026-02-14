"""Shared text normalization helpers for runtime I/O."""

from __future__ import annotations


def clean_text(s: str) -> str:
    """Remove leading UTF-8 BOM and normalize Windows newlines."""
    return s.lstrip("\ufeff").replace("\r\n", "\n")
