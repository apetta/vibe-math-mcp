"""Vibe Math - High-performance mathematical operations using Polars and scientific Python."""

import json
from typing import Annotated, Any

from fastmcp import FastMCP
from pydantic import Field
from fastmcp.tools import Tool
from fastmcp.tools.tool_transform import forward
from mcp.types import TextContent


class CustomMCP(FastMCP):
    """Custom FastMCP subclass that automatically handles context parameter for all tools.

    This custom subclass intercepts tool registration using FastMCP's Tool Transformation API.
    Every tool is automatically wrapped to accept an optional 'context' parameter that flows
    through to the response without requiring manual handling in tool implementations.

    Architecture:
    - Overrides add_tool() to transform tools at registration time
    - Uses Tool.from_tool() with transform_fn for context injection
    - Leverages FastMCP's built-in transformation system (no hacks)
    - Works with Pydantic validation (transformation happens AFTER tool creation)

    Benefits:
    - Zero boilerplate in tool functions
    - Impossible to forget context handling
    - Single source of truth for context injection
    - Automatic for all existing and future tools
    - Type-safe and production-grade
    """

    def add_tool(self, tool: Tool) -> Tool:
        """Override add_tool to automatically inject context handling via Tool transformation.

        Uses FastMCP's official Tool.from_tool() API to wrap each tool with context support.
        """

        # Define the transform function that adds context parameter
        async def context_transform(
            context: Annotated[
                str | None,
                Field(
                    description=(
                        "Optional annotation to label this calculation "
                        "(e.g., 'Bond A PV', 'Q2 revenue'). "
                        "Appears in results for easy identification."
                    )
                )
            ] = None,
            **kwargs: Any
        ) -> str:
            """Transform function that injects context into tool results.

            Args:
                context: Optional context string from LLM (added by transformation)
                **kwargs: All original tool arguments (passed through)

            Returns:
                Original tool result with context injected if provided
            """
            # Call the original tool using FastMCP's forward() function
            tool_result = await forward(**kwargs)

            # Extract text from ToolResult content
            # All tools return JSON strings as TextContent
            if (
                tool_result.content
                and len(tool_result.content) > 0
                and isinstance(tool_result.content[0], TextContent)
            ):
                result_str = tool_result.content[0].text
            else:
                # This should never happen as all tools return TextContent
                # If it does, something is fundamentally wrong with the tool
                raise ValueError(
                    f"Expected TextContent from tool, got {type(tool_result.content[0]) if tool_result.content else 'no content'}"
                )

            # Inject context into JSON result if provided
            if context is not None:
                try:
                    result_data = json.loads(result_str)
                    result_data['context'] = context
                    return json.dumps(result_data, indent=2, default=str)
                except (json.JSONDecodeError, TypeError):
                    # Tool returned non-JSON (unexpected) - return original result
                    # Context parameter will not be included in this edge case
                    pass

            return result_str

        # Transform the tool to add context handling
        transformed_tool = Tool.from_tool(
            tool=tool,
            transform_fn=context_transform,
            # Preserve all original metadata
            name=tool.name,
            description=tool.description,
        )

        # Register the transformed tool with parent class
        return super().add_tool(transformed_tool)


# Create CustomMCP server instance
mcp = CustomMCP("vibe-math")

# Import and register all tools (must be after mcp instance creation for decorators)
from .tools import array, basic, batch, calculus, financial, linalg, statistics  # noqa: E402

# Explicitly declare as part of module interface (tools registered via decorators)
__all__ = ["mcp", "basic", "array", "batch", "statistics", "financial", "linalg", "calculus"]


def main():
    """Entry point for uvx."""
    mcp.run()


if __name__ == "__main__":
    main()
