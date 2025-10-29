"""Vibe Math - High-performance mathematical operations using Polars and scientific Python."""

import json
from typing import Annotated, Any, Dict, Literal

from fastmcp import FastMCP
from pydantic import Field
from fastmcp.tools import Tool
from fastmcp.tools.tool_transform import forward
from mcp.types import TextContent

# Version is defined here to avoid circular import with __init__.py
__version__ = "2.0.0"


# ============================================================================
# Output Transformation Helpers
# ============================================================================


def is_sequential_chain(results: list) -> bool:
    """Detect if operations form pure sequential chain (no branching)."""
    if len(results) <= 1:
        return True

    dependents = {}
    for r in results:
        for dep in r.get("dependencies", []):
            if dep not in dependents:
                dependents[dep] = []
            dependents[dep].append(r["id"])

    all_ids = {r["id"] for r in results}
    roots = [r["id"] for r in results if not r.get("dependencies")]
    terminals = [op_id for op_id in all_ids if op_id not in dependents]

    if len(roots) != 1 or len(terminals) != 1:
        return False

    for op_id in all_ids:
        if op_id != terminals[0]:
            if op_id not in dependents or len(dependents[op_id]) != 1:
                return False

    return True


def find_terminal_operation(results: list) -> str | None:
    """Find terminal operation (one with no dependents)."""
    if not results:
        return None

    has_dependents = set()
    for r in results:
        has_dependents.update(r.get("dependencies", []))

    terminals = [r["id"] for r in results if r["id"] not in has_dependents]
    return terminals[0] if len(terminals) == 1 else None


def transform_single_response(data: Dict[str, Any], mode: str) -> Dict[str, Any]:
    """Transform single tool response based on output mode.

    Args:
        data: Original tool response (already JSON-parsed)
        mode: Output mode (full, compact, minimal, value, final)

    Returns:
        Transformed response dictionary
    """
    if mode == "final":
        mode = "value"

    if mode == "full":
        return data

    if mode == "compact":
        # Remove None/null values, preserve structure
        return {k: v for k, v in data.items() if v is not None}

    if mode == "minimal":
        # Keep only result + context if present
        minimal = {"result": data["result"]}

        # Preserve context if present
        if "context" in data:
            minimal["context"] = data["context"]
        return minimal

    if mode == "value":
        # Normalize to {value: X} structure
        result = {"value": data["result"]}

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
        mode: Output mode (full, compact, minimal, value, final)

    Returns:
        Transformed batch response
    """
    results = data.get("results", [])
    summary = data.get("summary", {})
    batch_context = data.get("context")

    if mode == "final":
        failed_count = summary.get("failed", 0)

        # Check for failures first - if any failures exist, use minimal mode
        # This ensures error visibility even in sequential chains
        if failed_count > 0:
            return transform_batch_response(data, "minimal")

        # No failures - check if sequential chain for terminal-only output
        if is_sequential_chain(results):
            terminal_id = find_terminal_operation(results)
            if terminal_id:
                terminal = next((r for r in results if r["id"] == terminal_id), None)

                if terminal and terminal.get("status") == "success":
                    result = {
                        "result": terminal["result"]["result"],
                        "summary": {
                            "succeeded": summary.get("succeeded", 0),
                            "failed": summary.get("failed", 0),
                            "time_ms": summary.get("total_execution_time_ms", 0),
                        },
                    }
                    if batch_context is not None:
                        result["context"] = batch_context
                    return result

        # Non-sequential with no failures - fall back to value mode
        return transform_batch_response(data, "value")

    if mode == "value":
        value_map = {}
        errors = {}

        for r in results:
            if r.get("status") == "success" and r.get("result"):
                op_id = r["id"]
                value_map[op_id] = r["result"]["result"]
            elif r.get("status") == "error":
                # Extract error message (could be string or dict)
                error_info = r.get("error")
                if isinstance(error_info, dict):
                    errors[r["id"]] = error_info.get("message", str(error_info))
                else:
                    errors[r["id"]] = str(error_info)

        result = {
            **value_map,
            "summary": {
                "succeeded": summary.get("succeeded", 0),
                "failed": summary.get("failed", 0),
                "time_ms": summary.get("total_execution_time_ms", 0),
            },
        }

        # Add errors if any operations failed
        if errors:
            result["errors"] = errors

        if batch_context is not None:
            result["context"] = batch_context
        return result

    if mode == "minimal":
        minimal_results = []
        for r in results:
            minimal_op = {
                "id": r["id"],
                "status": r["status"],
                "wave": r.get("wave", 0),
            }

            if r.get("status") == "success" and r.get("result"):
                minimal_op["value"] = r["result"]["result"]
                if "context" in r["result"] and r["result"]["context"] is not None:
                    minimal_op["context"] = r["result"]["context"]
            elif r.get("error"):
                minimal_op["error"] = r["error"].get("message", "Unknown error")

            minimal_results.append(minimal_op)

        result = {"results": minimal_results, "summary": summary}
        if batch_context is not None:
            result["context"] = batch_context
        return result

    if mode == "compact":
        compact_results = [{k: v for k, v in r.items() if v is not None} for r in results]
        result = {"results": compact_results, "summary": summary}
        if batch_context is not None:
            result["context"] = batch_context
        return result

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
        is_batch_tool = tool.name == "batch_execute"

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
                ),
            ] = None,
            output_mode: Annotated[
                Literal["full", "compact", "minimal", "value", "final"],
                Field(
                    description=(
                        "Control response verbosity:\n"
                        "- 'full' (default): Complete response with all metadata\n"
                        "- 'compact': Remove null fields, minimize whitespace (~20-30% smaller)\n"
                        "- 'minimal': Primary value(s) only, no metadata (~60-70% smaller)\n"
                        "- 'value': Normalized {value: X} structure (~70-80% smaller)\n"
                        "- 'final': For sequential chains, return only terminal result (~95% smaller)"
                    )
                ),
            ] = "full",
            **kwargs: Any,
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
                result_data["context"] = context

            # Apply output transformation based on tool type
            if is_batch_tool:
                result_data = transform_batch_response(result_data, output_mode)
            else:
                result_data = transform_single_response(result_data, output_mode)

            # Serialize based on mode
            if output_mode == "compact":
                # No indentation for compact mode
                return json.dumps(result_data, separators=(",", ":"), default=str)
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
mcp = CustomMCP(
    "vibe-math-mcp",
    version=__version__,
    instructions="""Use this server for ANY calculation, formula evaluation, or quantitative analysis. Delegate to production-grade tools (Polars, NumPy, SciPy, SymPy) for precision, never manually compute or approximate.

**Comprehensive coverage (21 tools):**
• Basic math (expressions, percentages, rounding, unit conversion)
• Arrays (operations, statistics, aggregations, transformations)
• Statistics (descriptive analysis, pivot tables, correlations)
• Financial (TVM/PV/FV/IRR/NPV, compound interest, perpetuities, growing annuities)
• Linear algebra (matrices, systems of equations, decompositions)
• Calculus (derivatives, integrals, limits, series expansions)

**Key capabilities:**
• Exact symbolic computation (not approximations)
• Multi-step workflows with dependency chaining (batch_execute)
• Token-efficient output modes (up to 95% reduction)
• Context tracking for complex calculations
• Professional numerical libraries (battle-tested, validated)

**Use when:**
• ANY quantitative calculation is needed
• Precision required (no rounding errors or mental math)
• Multi-step workflows (financial models, data transformations, statistical pipelines)
• Matrix operations, calculus, or financial formulas

**Default behavior:** Reach for these tools for quantitative work instead of manual calculation or approximation.""",
)

# Import and register all tools (must be after mcp instance creation for decorators)
from .tools import array, basic, batch, calculus, financial, linalg, statistics  # noqa: E402

# Explicitly declare as part of module interface (tools registered via decorators)
__all__ = ["mcp", "basic", "array", "batch", "statistics", "financial", "linalg", "calculus"]


def main():
    """Entry point for uvx."""
    mcp.run()


if __name__ == "__main__":
    main()
