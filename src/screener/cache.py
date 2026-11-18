"""Disk cache for external API responses.

Both SEC and Yahoo rate-limit, and iterating on the screen means re-running it
many times against the same data. Cached responses keep development fast and
keep us well inside fair-access limits.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache"


def _path_for(key: str) -> Path:
    digest = hashlib.sha256(key.encode()).hexdigest()[:32]
    return CACHE_DIR / f"{digest}.json"


def get(key: str, ttl_seconds: float) -> Any | None:
    """Return the cached value for `key`, or None if absent or stale."""
    path = _path_for(key)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if time.time() - payload.get("stored_at", 0) > ttl_seconds:
        return None
    return payload.get("value")


def put(key: str, value: Any) -> None:
    """Store `value` under `key`. Cache failures are never fatal."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        _path_for(key).write_text(json.dumps({"stored_at": time.time(), "value": value}))
    except (OSError, TypeError):
        pass


def clear() -> int:
    """Delete all cached entries. Returns the number of files removed."""
    if not CACHE_DIR.exists():
        return 0
    files = list(CACHE_DIR.glob("*.json"))
    for f in files:
        f.unlink()
    return len(files)
