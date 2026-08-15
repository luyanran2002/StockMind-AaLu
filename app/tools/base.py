"""Shared tool primitives: provenance models and serialization helpers.

Every tool observation carries ``source``, ``timestamp`` and ``data_period`` so
the final report can distinguish observed data from interpretation, and so
simulated data is never silently presented as real market data.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class DataPoint(BaseModel):
    """A single grounded value with its provenance."""

    value: Any = Field(..., description="The observed or computed value")
    source: str = Field(..., description="Where the value came from")
    timestamp: datetime = Field(..., description="When the value was retrieved")
    data_period: str | None = Field(default=None, description="The period the value covers")


def to_json(obj: Any) -> str:
    """Serialize a Pydantic model or plain structure to compact JSON for the LLM."""
    if isinstance(obj, BaseModel):
        return obj.model_dump_json()
    return json.dumps(obj, default=str)


def tool_call_signature(name: str, args: dict[str, Any]) -> str:
    """Canonical, order-insensitive signature used for duplicate detection."""
    canonical = json.dumps(args, sort_keys=True, default=str)
    return f"{name}:{canonical}"


def metric_json(
    metric: str,
    value: Any,
    inputs: dict[str, Any],
    source: str,
    note: str | None = None,
) -> str:
    """Serialise a deterministic computation result with its provenance."""
    return json.dumps(
        {
            "metric": metric,
            "value": value,
            "inputs": inputs,
            "source": f"computed from {source} data",
            "timestamp": utc_now().isoformat(),
            "note": note,
        }
    )
