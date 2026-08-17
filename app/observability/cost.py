"""LLM cost estimation from token usage.

Prices are per 1M tokens and are rough, configurable estimates — they change
over time and should not be treated as billing-grade.
"""

from __future__ import annotations

from typing import Any

MODEL_PRICING_USD_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "claude-3-5-haiku-latest": (0.80, 4.00),
    "claude-3-5-sonnet-latest": (3.00, 15.00),
    "claude-3-opus-latest": (15.00, 75.00),
}


def estimate_cost_usd(model: str | None, input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost for the given token counts."""
    if not model:
        return 0.0
    pricing = MODEL_PRICING_USD_PER_1M.get(model)
    if not pricing:
        return 0.0
    input_price, output_price = pricing
    return (input_tokens / 1_000_000 * input_price) + (
        output_tokens / 1_000_000 * output_price
    )


def token_counts(usage: dict[str, Any] | None) -> tuple[int, int]:
    """Extract (input_tokens, output_tokens) from LangChain usage metadata."""
    if not usage:
        return 0, 0
    input_tokens = (
        usage.get("input_tokens")
        or usage.get("prompt_tokens")
        or 0
    )
    output_tokens = (
        usage.get("output_tokens")
        or usage.get("completion_tokens")
        or 0
    )
    return int(input_tokens or 0), int(output_tokens or 0)

