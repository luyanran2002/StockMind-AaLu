"""Small shared helpers."""

from __future__ import annotations

from datetime import datetime


def format_local_time(iso: str | None) -> str:
    """Convert an ISO timestamp to a human-readable local time (to the second)."""
    if not iso:
        return "unknown"
    try:
        local = datetime.fromisoformat(iso).astimezone()
    except (ValueError, TypeError):
        return str(iso)
    return f"{local.strftime('%Y-%m-%d %H:%M:%S')} ({local.strftime('%z')})"

