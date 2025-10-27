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


def _build_tool_registry():
    """Auto-discover all available tools by importing tool modules.

    This is inspired by CustomMCP's automatic tool registration pattern.
    Instead of manually maintaining a registry, we dynamically import
    all tool modules and extract their exported functions.

    FastMCP wraps decorated functions in FunctionTool objects, so we extract
    the underlying async callable via the .fn attribute.

    Returns:
        Dictionary mapping tool_name -> async function
    """
    # Import all tool modules
    from . import basic, array, statistics as stats_module, financial, linalg, calculus

    # Build registry by extracting underlying callables from FunctionTool wrappers
    # Tool names match the @mcp.tool(name="...") decorators
    registry = {
        # Basic calculations (4 tools)
        "calculate": basic.calculate.fn,
        "percentage": basic.percentage.fn,
        "round": basic.round_values.fn,
        "convert_units": basic.convert_units.fn,
        # Array operations (4 tools)
        "array_operations": array.array_operations.fn,
        "array_statistics": array.array_statistics.fn,
        "array_aggregate": array.array_aggregate.fn,
        "array_transform": array.array_transform.fn,
        # Statistics (3 tools)
        "statistics": stats_module.statistics.fn,
        "pivot_table": stats_module.pivot_table.fn,
        "correlation": stats_module.correlation.fn,
        # Financial mathematics (3 tools)
        "financial_calcs": financial.financial_calcs.fn,
        "compound_interest": financial.compound_interest.fn,
        "perpetuity": financial.perpetuity.fn,
        # Linear algebra (3 tools)
        "matrix_operations": linalg.matrix_operations.fn,
        "solve_linear_system": linalg.solve_linear_system.fn,
        "matrix_decomposition": linalg.matrix_decomposition.fn,
        # Calculus (3 tools)
        "derivative": calculus.derivative.fn,
        "integral": calculus.integral.fn,
        "limits_series": calculus.limits_series.fn,
    }

    # Validate registry matches TOOL_CATEGORIES (DRY enforcement)
    expected_tools = {tool for tools in TOOL_CATEGORIES.values() for tool in tools}
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
    description=f"""Execute multiple operations in a single request with dependency management and parallel execution.

{_generate_tool_reference()}

## Structure
Each operation requires:
- `id`: Unique identifier (auto-generated UUID if omitted)
- `tool`: Tool name from list above
- `arguments`: Tool-specific parameters
- `depends_on`: (optional) Array of operation IDs to wait for

## Result Referencing
Reference prior results in arguments:
- `$op_id.result` - Main result value
- `$op_id.metadata.field` - Nested field access
- `$op_id.values[0]` - Array indexing

## Execution Modes
- `sequential`: Execute in specified order
- `parallel`: All operations run concurrently
- `auto`: DAG-based optimization (recommended)

## Output Control (via output_mode parameter)
- `full`: Complete operation details (default)
- `compact`: Remove nulls, minimize whitespace
- `minimal`: Simplified operation objects with values
- `value`: Flat {{id: value}} mapping (~90% token reduction)

## Example
```json
{{
  "operations": [
    {{"id": "calc1", "tool": "calculate", "arguments": {{"expression": "10 + 5"}}}},
    {{
      "id": "calc2",
      "tool": "calculate",
      "arguments": {{"expression": "x * 2", "variables": {{"x": "$calc1.result"}}}},
      "depends_on": ["calc1"]
    }}
  ],
  "execution_mode": "auto",
  "output_mode": "value"
}}
```

**Note:** With `value` mode, results are returned as a flat map (e.g., `{{"calc1": 15, "calc2": 30, "summary": ...}}`), making client-side extraction trivial.

Response includes: `id`, `status` (success/error/timeout), `result`/`error`, `execution_time_ms`, `wave`, `dependencies`.
Per-operation `context` field flows through to results. Summary shows total/succeeded/failed counts and wave depth.
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
                "depends_on (list of IDs), result_mapping (dict), timeout_ms (int)"
            ),
            min_length=1,
            max_length=100,
        ),
    ],
    execution_mode: Annotated[
        Literal["sequential", "parallel", "auto"],
        Field(description="Execution strategy: sequential (order), parallel (concurrent), auto (DAG-based)"),
    ] = "auto",
    max_concurrent: Annotated[
        int, Field(description="Maximum concurrent operations (applies to parallel/auto modes)", ge=1, le=20)
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
        # Build tool registry (DRY: auto-discovered from imports)
        tool_registry = _build_tool_registry()

        # Validate tool names
        for op in operations:
            if op.tool not in tool_registry:
                available = ', '.join(sorted(tool_registry.keys()))
                raise ValueError(
                    f"Unknown tool '{op.tool}' in operation '{op.id}'. "
                    f"Available tools: {available}"
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
