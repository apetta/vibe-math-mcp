"""Type definitions for the Maths MCP server."""

from typing import Annotated
from pydantic import Field

# Reusable type alias for optional context parameter
# This allows LLMs to pass context through tools to maintain reasoning state
ContextParam = Annotated[
    str | None,
    Field(
        description="Optional context for LLM to track reasoning and maintain state across tool calls"
    ),
]

__all__ = ["ContextParam"]
