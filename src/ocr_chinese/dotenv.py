from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: str | Path, *, override: bool = False) -> bool:
    """
    Minimal .env loader.

    - Supports KEY=VALUE lines (VALUE may be quoted with ' or ").
    - Ignores empty lines and comments starting with '#'.
    - By default does NOT override existing environment variables.
    """
    p = Path(path)
    if not p.exists() or not p.is_file():
        return False

    for raw in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        if override:
            os.environ[key] = value
        else:
            os.environ.setdefault(key, value)
    return True


def load_default_env(*, override: bool = False) -> str | None:
    """
    Load env vars from a file in the current working directory.

    Order:
    1) MASKPDF_ENV_FILE (explicit)
    2) .env
    3) venv (legacy / user naming)
    """
    explicit = os.getenv("MASKPDF_ENV_FILE")
    if explicit and load_env_file(explicit, override=override):
        return str(explicit)
    for name in (".env", "venv"):
        candidate = Path.cwd() / name
        if load_env_file(candidate, override=override):
            return str(candidate)
    return None
