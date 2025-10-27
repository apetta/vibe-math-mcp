"""Vibe Math - High-performance mathematical operations using Polars and scientific Python."""

import json
from typing import Annotated, Any, Dict, Literal

from fastmcp import FastMCP
from pydantic import Field
from fastmcp.tools import Tool
from fastmcp.tools.tool_transform import forward
from mcp.types import TextContent


# ============================================================================
# Output Transformation Helpers
# ============================================================================

def extract_primary_value(data: Dict[str, Any]) -> Any:
    """Extract the primary value from a tool response.

    Intelligently detects the main value field based on response structure:
    - Basic/Financial tools: data["result"]
    - Array tools: data["values"]
    - Stats tools: entire object (multiple values)
    """
    if "result" in data:
        return data["result"]
    elif "values" in data:
        return data["values"]
    else:
        # Stats tools or complex responses - return entire object
        return data


def transform_single_response(data: Dict[str, Any], mode: str) -> Dict[str, Any]:
    """Transform single tool response based on output mode.

    Args:
        data: Original tool response (already JSON-parsed)
        mode: Output mode (full, compact, minimal, value)

    Returns:
        Transformed response dictionary
    """
    if mode == "full":
        return data

    if mode == "compact":
        # Remove None/null values, preserve structure
        return {k: v for k, v in data.items() if v is not None}

    if mode == "minimal":
        # Keep only primary field + context if present
        if "result" in data:
            minimal = {"result": data["result"]}
        elif "values" in data:
            minimal = {"values": data["values"]}
        else:
            # Stats tools - already minimal (multiple result fields)
            minimal = data

        # Preserve context if present
        if "context" in data:
            minimal["context"] = data["context"]
        return minimal

    if mode == "value":
        # Normalize to {value: X} structure
        extracted = extract_primary_value(data)
        result = {"value": extracted}

        # Preserve context if present
        if "context" in data:
            result["context"] = data["context"]
        return result

    # Fallback (should never reach here)
    return data


def transform_batch_response(data: Dict[str, Any], mode: str) -> Dict[str, Any]:
    """Transform batch execution response based on output mode.

    Args:
        data: Batch response with 'results' and 'summary' keys
        mode: Output mode (full, compact, minimal, value)

    Returns:
        Transformed batch response
    """
    results = data.get("results", [])
    summary = data.get("summary", {})

    if mode == "value":
        # Flat {id: value} mapping + minimal summary
        value_map = {}
        for r in results:
            if r.get("status") == "success" and r.get("result"):
                op_id = r["id"]
                value_map[op_id] = extract_primary_value(r["result"])

        return {
            **value_map,
            "summary": {
                "succeeded": summary.get("succeeded", 0),
                "failed": summary.get("failed", 0),
                "time_ms": summary.get("total_execution_time_ms", 0),
            }
        }

    if mode == "minimal":
        # Simplified operation objects
        minimal_results = []
        for r in results:
            minimal_op = {
                "id": r["id"],
                "status": r["status"],
                "wave": r.get("wave", 0),
            }

            if r.get("status") == "success" and r.get("result"):
                minimal_op["value"] = extract_primary_value(r["result"])
            elif r.get("error"):
                minimal_op["error"] = r["error"].get("message", "Unknown error")

            minimal_results.append(minimal_op)

        return {
            "results": minimal_results,
            "summary": summary
        }

    if mode == "compact":
        # Remove nulls from each result
        compact_results = [
            {k: v for k, v in r.items() if v is not None}
            for r in results
        ]
        return {
            "results": compact_results,
            "summary": summary
        }

    # full mode - return as-is
    return data


class CustomMCP(FastMCP):
    """Custom FastMCP subclass with automatic context injection and output control.

    This custom subclass intercepts tool registration using FastMCP's Tool Transformation API.
    Every tool is automatically wrapped to accept optional parameters that enhance LLM usability:

    1. **context**: Label calculations for identification (e.g., "Bond A PV", "Q2 revenue")
    2. **output_mode**: Control response verbosity and structure

    Output Modes:
        - "full" (default): Complete response with all metadata - backward compatible
        - "compact": Remove null fields, minimize whitespace (~20-30% smaller)
        - "minimal": Primary value(s) only, no metadata (~60-70% smaller)
        - "value": Normalized {value: X} structure (~70-80% smaller)

    Architecture:
        - Overrides add_tool() to transform tools at registration time
        - Uses Tool.from_tool() with transform_fn for parameter injection
        - Leverages FastMCP's built-in transformation system (no hacks)
        - Works with Pydantic validation (transformation happens AFTER tool creation)
        - Special handling for batch_execute vs single tools

    Benefits:
        - Zero boilerplate in tool functions
        - Automatic for all existing and future tools
        - Type-safe and production-grade
        - Massive token savings for LLM consumers
    """

    def add_tool(self, tool: Tool) -> Tool:
        """Override add_tool to inject context and output_mode parameters.

        Uses FastMCP's official Tool.from_tool() API to wrap each tool with
        automatic context injection and intelligent output control.
        """

        # Detect if this is the batch_execute tool for special handling
        is_batch_tool = (tool.name == "batch_execute")

        # Define the unified transform function
        async def unified_transform(
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
            output_mode: Annotated[
                Literal["full", "compact", "minimal", "value"],
                Field(
                    description=(
                        "Control response verbosity:\n"
                        "- 'full' (default): Complete response with all metadata\n"
                        "- 'compact': Remove null fields, minimize whitespace (~20-30% smaller)\n"
                        "- 'minimal': Primary value(s) only, no metadata (~60-70% smaller)\n"
                        "- 'value': Normalized {value: X} structure (~70-80% smaller)"
                    )
                )
            ] = "full",
            **kwargs: Any
        ) -> str:
            """Transform function for context injection and output control.

            Args:
                context: Optional context string from LLM
                output_mode: Output verbosity control
                **kwargs: All original tool arguments (passed through)

            Returns:
                Transformed tool result as JSON string
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
                raise ValueError(
                    f"Expected TextContent from tool, got "
                    f"{type(tool_result.content[0]) if tool_result.content else 'no content'}"
                )

            # Parse JSON result
            try:
                result_data = json.loads(result_str)
            except (json.JSONDecodeError, TypeError):
                # Tool returned non-JSON (unexpected) - return original
                return result_str

            # Inject context if provided (before transformation)
            if context is not None:
                result_data['context'] = context

            # Apply output transformation based on tool type
            if is_batch_tool:
                result_data = transform_batch_response(result_data, output_mode)
            else:
                result_data = transform_single_response(result_data, output_mode)

            # Serialize based on mode
            if output_mode == "compact":
                # No indentation for compact mode
                return json.dumps(result_data, separators=(',', ':'), default=str)
            else:
                # Pretty-print for all other modes
                return json.dumps(result_data, indent=2, default=str)

        # Transform the tool to add context and output_mode handling
        transformed_tool = Tool.from_tool(
            tool=tool,
            transform_fn=unified_transform,
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
