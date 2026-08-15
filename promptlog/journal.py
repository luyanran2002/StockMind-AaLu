"""Core storage and rendering for the prompt journal."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parent
JSONL_PATH = BASE / "prompts.jsonl"
MARKDOWN_PATH = BASE / "prompts.md"


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _display_ts(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return iso


def load_entries() -> list[dict[str, Any]]:
    """Read all entries from the JSONL source of truth."""
    if not JSONL_PATH.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in JSONL_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def render_markdown(entry: dict[str, Any]) -> str:
    """Render a single entry as a Markdown section."""
    lines = [f"## {_display_ts(entry['timestamp'])}", ""]
    tags = entry.get("tags") or []
    if tags:
        lines.append("Tags: " + ", ".join(f"`{tag}`" for tag in tags))
        lines.append("")
    prompt = entry["prompt"].replace("\n", "\n> ")
    lines.append(f"> {prompt}")
    if entry.get("note"):
        lines.append("")
        lines.append(f"Note: {entry['note']}")
    return "\n".join(lines)


def add_prompt(
    prompt: str,
    tags: list[str] | None = None,
    note: str | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append a prompt to the journal and return the stored entry."""
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("prompt must not be empty")

    entry: dict[str, Any] = {
        "id": uuid.uuid4().hex[:8],
        "timestamp": _now_iso(),
        "prompt": prompt,
        "tags": tags or [],
        "note": note,
        "meta": meta or {},
    }

    with JSONL_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    with MARKDOWN_PATH.open("a", encoding="utf-8") as fh:
        fh.write(render_markdown(entry) + "\n\n")

    return entry


def list_prompts(limit: int = 20) -> list[dict[str, Any]]:
    """Return the most recent ``limit`` entries (newest last)."""
    return load_entries()[-limit:]


def export_markdown() -> int:
    """Rebuild ``prompts.md`` from ``prompts.jsonl`` and return the entry count."""
    entries = load_entries()
    content = "\n\n".join(render_markdown(entry) for entry in entries)
    if content:
        content += "\n"
    MARKDOWN_PATH.write_text(content, encoding="utf-8")
    return len(entries)

