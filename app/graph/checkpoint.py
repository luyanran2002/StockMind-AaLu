"""Checkpoint saver construction.

``memory`` uses LangGraph's :class:`InMemorySaver` (no extra dependency, works
in-process). ``sqlite`` uses the optional ``langgraph-checkpoint-sqlite``
package for durable, cross-process persistence.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.memory import InMemorySaver

# Pydantic models stored in checkpoint state must be whitelisted for msgpack.
ALLOWED_MSGPACK_MODULES = [("app.schemas.report", "StockResearchReport")]


def _serializer() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_MSGPACK_MODULES)


@asynccontextmanager
async def _sqlite_checkpointer(db_path: str | None):
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    async with aiosqlite.connect(db_path or ":memory:") as conn:
        yield AsyncSqliteSaver(conn, serde=_serializer())


def build_checkpointer(kind: str = "memory", db_path: str | None = None) -> Any:
    """Return a checkpointer for the given backend.

    * ``"memory"`` -> an ``InMemorySaver`` instance (ready to use).
    * ``"sqlite"``  -> an *async context manager* from ``AsyncSqliteSaver``;
      enter it inside an ``async with`` block before compiling/invoking.
    """
    kind = (kind or "memory").strip().lower()
    if kind == "memory":
        return InMemorySaver(serde=_serializer())
    if kind == "sqlite":
        return _sqlite_checkpointer(db_path)
    raise ValueError(f"Unsupported checkpointer kind: {kind!r} (expected 'memory' or 'sqlite')")
