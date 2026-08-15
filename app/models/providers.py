"""LLM-provider abstraction.

Business logic depends only on ``langchain_core.language_models.BaseChatModel``.
OpenAI and Anthropic are selected here from configuration, keeping provider
specifics out of the agent core.
"""

from __future__ import annotations

import os
from typing import Any

from langchain_core.language_models import BaseChatModel

SUPPORTED_PROVIDERS = ("openai", "anthropic")


def get_chat_model(
    provider: str | None = None,
    model: str | None = None,
    *,
    temperature: float = 0.0,
    **kwargs: Any,
) -> BaseChatModel:
    """Return a chat model for the configured (or requested) provider.

    API keys are read from the environment by the underlying client; they are
    never embedded in code.
    """
    name = (provider or os.getenv("STOCKMIND_LLM_PROVIDER") or "openai").strip().lower()
    model = model or os.getenv("STOCKMIND_LLM_MODEL")

    if name == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model or "gpt-4o-mini", temperature=temperature, **kwargs)

    if name == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model or "claude-3-5-haiku-latest", temperature=temperature, **kwargs)

    raise ValueError(f"Unsupported LLM provider: {name!r}. Supported: {SUPPORTED_PROVIDERS}")

