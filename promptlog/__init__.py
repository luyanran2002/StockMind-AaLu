"""Prompt journal — a standalone framework for recording user prompts.

Usage (from the repository root)::

    python -m promptlog add "你的 prompt 内容" --tag finance
    python -m promptlog list --limit 10
    python -m promptlog export
"""

from promptlog.journal import add_prompt, export_markdown, list_prompts

__all__ = ["add_prompt", "list_prompts", "export_markdown"]
__version__ = "0.1.0"

