from __future__ import annotations

import os
import sys
from pathlib import Path


def _configure_temp_dir() -> None:
    if os.name != "nt":
        return
    tmp_dir = Path(".tmp").resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TMP"] = str(tmp_dir)
    os.environ["TEMP"] = str(tmp_dir)


def main() -> int:
    _configure_temp_dir()
    try:
        import pytest
    except ImportError as exc:
        raise SystemExit("pytest is required. Install dev dependencies first.") from exc
    return int(pytest.main(sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
