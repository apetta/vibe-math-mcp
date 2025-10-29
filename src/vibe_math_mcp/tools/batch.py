"""Batch execution tool with auto-discovered tool registry and intelligent orchestration."""

import json
from typing import Annotated, List, Literal
from pydantic import Field
from mcp.types import ToolAnnotations

from ..server import mcp
from ..core.batch_models import BatchOperation, BatchResponse
from ..core.batch_executor import BatchExecutor


# Single source of truth for tool organization
TOOL_CATEGORIES = {
    "Basic": ["calculate", "percentage", "round", "convert_units"],
    "Arrays": ["array_operations", "array_statistics", "array_aggregate", "array_transform"],
    "Statistics": ["statistics", "pivot_table", "correlation"],
    "Financial": ["financial_calcs", "compound_interest", "perpetuity"],
    "Linear Algebra": ["matrix_operations", "solve_linear_system", "matrix_decomposition"],
    "Calculus": ["derivative", "integral", "limits_series"],
}


async def _build_tool_registry_async():
    """Build registry of wrapped tools from MCP server.

    Uses CustomMCP-transformed tools which allows the same transformation layer that individual tool calls use.

    Returns:
        Dictionary mapping tool_name -> Tool instance (with wrapper support)
    """
    # Get all tool names from TOOL_CATEGORIES
    tool_names = [tool for tools in TOOL_CATEGORIES.values() for tool in tools]

    # Build registry from wrapped tools in MCP server
    registry = {}
    for name in tool_names:
        tool = await mcp._tool_manager.get_tool(name)
        registry[name] = tool

    # Validate all expected tools were found
    expected_tools = set(tool_names)
    actual_tools = set(registry.keys())
    assert expected_tools == actual_tools, (
        f"Registry mismatch! Missing: {expected_tools - actual_tools}, "
        f"Extra: {actual_tools - expected_tools}"
    )

    return registry


def _generate_tool_reference() -> str:
    """Dynamically generate compact list of batchable tool IDs from TOOL_CATEGORIES."""
    total = sum(len(tools) for tools in TOOL_CATEGORIES.values())
    lines = [f"Available tools ({total}):"]
    for category, tools in TOOL_CATEGORIES.items():
        lines.append(f"• {category}: {', '.join(tools)}")
    return "\n".join(lines)


@mcp.tool(
    name="batch_execute",
    description=f"""Execute multiple math operations in a single request with automatic dependency chaining.

**USE THIS TOOL when you need 2+ calculations where outputs feed into inputs** (bond pricing, statistical workflows, multi-step formulas). Don't make sequential individual tool calls.

Benefits: 90-95% token reduction, single API call, highly flexible workflows

## Quick Start

{_generate_tool_reference()}

**Result referencing:**

Pass `$op_id.result` directly in any parameter:
- `$op_id.result` - Use output from prior operation
- `$op_id.result[0]` - Array indexing
- `$op_id.metadata.field` - Nested fields

Example: `"payment": "$coupon.result"` or `"variables": {{"x": "$op1.result"}}`

**Example - Bond valuation:**
```json
{{
  "operations": [
    {{"id": "coupon", "tool": "calculate",
     "context": "Calculate annual coupon payment",
     "arguments": {{"expression": "principal * 0.04", "variables": {{"principal": 8306623.86}}}}}},
    {{"id": "fv", "tool": "financial_calcs",
     "context": "Future value of coupon payments",
     "arguments": {{"calculation": "fv", "rate": 0.04, "periods": 10,
                   "payment": "$coupon.result", "present_value": 0}}}},
    {{"id": "total", "tool": "calculate",
     "context": "Total bond maturity value",
     "arguments": {{"expression": "fv + principal",
                   "variables": {{"fv": "$fv.result", "principal": 8306623.86}}}}}}
  ],
  "execution_mode": "auto",
  "output_mode": "minimal",
  "context": "Bond A 10-year valuation"
}}
```

## When to Use
✅ Multi-step calculations (financial models, statistics, transformations)
✅ Data pipelines where step N needs output from step N-1
✅ Any workflow requiring 2+ operations from the tools above

❌ Single standalone calculation
❌ Need to inspect/validate intermediate results before proceeding

## Execution Modes
- `auto` (recommended): DAG-based optimization, parallel where possible
- `sequential`: Strict order
- `parallel`: All concurrent (only if truly independent)

## Output Modes
- `full`: Complete metadata (default)
- `compact`: Remove nulls/whitespace
- `minimal`: Basic operation objects with values
- `value`: Flat {{id: value}} map (~90% smaller) - **use this for most cases**
- `final`: Sequential chains only, returns terminal result (~95% smaller)

## Structure
Each operation:
- `tool`: Tool name (required)
- `arguments`: Tool parameters (required)
- `id`: Unique identifier (auto-generated if omitted)
- `context`: Optional label for this operation

Batch-level `context` parameter labels entire workflow across all output modes.

Response includes: per-operation status, result/error, execution_time_ms, dependency wave, summary stats.
""",
    annotations=ToolAnnotations(
        title="Batch Execute",
        readOnlyHint=True,
    ),
)
async def batch_execute(
    operations: Annotated[
        List[BatchOperation],
        Field(
            description=(
                "List of operations to execute. Each operation MUST include: "
                "tool (name), arguments (dict). Optional: id (UUID/string), context, label, "
                "timeout_ms (int)"
            ),
            min_length=1,
            max_length=100,
        ),
    ],
    execution_mode: Annotated[
        Literal["sequential", "parallel", "auto"],
        Field(
            description="Execution strategy: sequential (order), parallel (concurrent), auto (DAG-based)"
        ),
    ] = "auto",
    max_concurrent: Annotated[
        int,
        Field(
            description="Maximum concurrent operations (applies to parallel/auto modes)",
            ge=1,
            le=20,
        ),
    ] = 5,
    stop_on_error: Annotated[
        bool,
        Field(
            description=(
                "Whether to stop execution on first error. "
                "If False, independent operations continue even if others fail."
            )
        ),
    ] = False,
) -> str:
    """Execute batch of mathematical operations with dependency management.

    This tool orchestrates multiple tool calls in a single request, automatically
    detecting dependencies and executing operations in optimal parallel waves.

    Each operation is tracked by its unique ID, providing crystal-clear mapping
    between inputs and outputs for easy LLM consumption and debugging.

    Returns:
        JSON string with results array and execution summary
    """
    try:
        # Build tool registry from wrapped tools (supports context/output_mode)
        tool_registry = await _build_tool_registry_async()

        # Validate tool names
        for op in operations:
            if op.tool not in tool_registry:
                available = ", ".join(sorted(tool_registry.keys()))
                raise ValueError(
                    f"Unknown tool '{op.tool}' in operation '{op.id}'. Available tools: {available}"
                )

        # Create executor
        executor = BatchExecutor(
            operations=operations,
            tool_registry=tool_registry,
            mode=execution_mode,
            max_concurrent=max_concurrent,
            stop_on_error=stop_on_error,
        )

        # Execute batch
        response: BatchResponse = await executor.execute()

        # Convert to JSON
        # Note: CustomMCP will inject batch-level context at top level
        return json.dumps(
            {
                "results": [result.model_dump() for result in response.results],
                "summary": response.summary.model_dump(),
            },
            indent=2,
            default=str,
        )

    except Exception as e:
        # Return structured error response
        return json.dumps(
            {
                "error": {
                    "type": type(e).__name__,
                    "message": str(e),
                    "tool": "batch_execute",
                },
                "results": [],  # No partial results on batch-level error
            },
            indent=2,
        )
