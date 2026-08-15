"""Agent package.

Exports are lazy to avoid a circular import between ``app.agents.react_agent``
and ``app.graph.workflow`` during package initialisation.
"""

__all__ = ["StockMindAgent"]


def __getattr__(name: str):
    if name == "StockMindAgent":
        from app.agents.react_agent import StockMindAgent

        return StockMindAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
