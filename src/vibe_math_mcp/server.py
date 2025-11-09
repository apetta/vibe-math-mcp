"""Vibe Math - High-performance mathematical operations using Polars and scientific Python."""

import json
from typing import Annotated, Any, Dict, Literal

from fastmcp import FastMCP
from pydantic import Field
from fastmcp.tools import Tool
from fastmcp.tools.tool_transform import forward
from mcp.types import TextContent

# Version is defined here to avoid circular import with __init__.py
__version__ = "2.0.2"


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
                    description="Output format: full (default), compact, minimal, value, or final. See batch_execute tool for details."
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


# ============================================================================
# MCP Prompts - Common Mathematical Workflows
# ============================================================================


@mcp.prompt()
def financial_calculation(
    calculation_type: Annotated[
        Literal["bond_pricing", "loan_payment", "investment_analysis"],
        Field(description="Type of financial calculation to perform"),
    ] = "bond_pricing",
) -> str:
    """Financial calculation workflows: bond pricing, loans, NPV/IRR."""
    workflows = {
        "bond_pricing": """Bond Pricing (PV of coupon bond)

1. calculate("principal * rate", {principal: 1000, rate: 0.04}) → 40 (annual coupon)
2. financial_calcs(calculation="pv", rate=0.05, periods=10, payment=40, future_value=1000) → bond price

Batch example:
{"operations": [
  {"id": "coup", "tool": "calculate", "arguments": {"expression": "p*r", "variables": {"p": 1000, "r": 0.04}}},
  {"id": "pv", "tool": "financial_calcs", "arguments": {"calculation": "pv", "rate": 0.05, "periods": 10, "payment": "$coup.result", "future_value": 1000}}
], "output_mode": "value"}""",
        "loan_payment": """Loan Payment (monthly mortgage)

1. calculate("apr / 12", {apr: 0.045}) → 0.00375 (monthly rate)
2. calculate("years * 12", {years: 30}) → 360 (periods)
3. financial_calcs(calculation="pmt", rate=0.00375, periods=360, present_value=-200000, future_value=0) → monthly payment""",
        "investment_analysis": """NPV/IRR (project evaluation)

NPV: financial_calcs(calculation="npv", rate=0.10, cash_flows=[-100000, 30000, 40000, 50000])
IRR: financial_calcs(calculation="irr", cash_flows=[-100000, 30000, 40000, 50000])

Choose higher NPV or IRR > required return.

Batch compare:
{"operations": [
  {"id": "a_npv", "tool": "financial_calcs", "arguments": {"calculation": "npv", "rate": 0.10, "cash_flows": [-100000, 30000, 40000, 50000]}},
  {"id": "a_irr", "tool": "financial_calcs", "arguments": {"calculation": "irr", "cash_flows": [-100000, 30000, 40000, 50000]}},
  {"id": "b_npv", "tool": "financial_calcs", "arguments": {"calculation": "npv", "rate": 0.10, "cash_flows": [-80000, 25000, 35000, 35000]}},
  {"id": "b_irr", "tool": "financial_calcs", "arguments": {"calculation": "irr", "cash_flows": [-80000, 25000, 35000, 35000]}}
], "execution_mode": "parallel", "output_mode": "value"}""",
    }
    return workflows[calculation_type]


@mcp.prompt()
def statistical_analysis(
    analysis_type: Annotated[
        Literal["descriptive", "correlation", "outliers"],
        Field(description="Type of statistical analysis to perform"),
    ] = "descriptive",
) -> str:
    """Statistical analysis: descriptive stats, correlation, outlier detection."""
    workflows = {
        "descriptive": """Descriptive Statistics

statistics(data=[12,15,18,20,22,25,28,30,100], analyses=["describe","quartiles","outliers"])
→ count, mean, std, min, max, median, Q1, Q2, Q3, IQR, outlier detection

Multi-dimensional: array_statistics(data=[[1,2,3],[4,5,6]], operations=["mean","std"], axis=0) → column stats""",
        "correlation": """Correlation Analysis

Pearson (linear): correlation(data={"height": [170,175,168], "weight": [65,78,62]}, method="pearson", output_format="matrix")
→ correlation matrix: 1.0=perfect positive, -1.0=perfect negative, 0.0=no relationship

Spearman (monotonic, robust to outliers): same but method="spearman" """,
        "outliers": """Outlier Detection (IQR method)

statistics(data=[12,15,18,20,22,25,28,30,100,105], analyses=["outliers"])
→ outlier_values, outlier_count, bounds (Q1-1.5×IQR to Q3+1.5×IQR)

Compare with/without: Use batch_execute to run analyses on original + cleaned data

Multi-dimensional: array_transform(data, transform="standardize") then |Z| > 3 = outliers""",
    }
    return workflows[analysis_type]


@mcp.prompt()
def matrix_problem(
    problem_type: Annotated[
        Literal["solve_system", "decomposition", "operations"],
        Field(description="Type of matrix problem to solve"),
    ] = "solve_system",
) -> str:
    """Linear algebra: solve systems, decompositions, matrix operations."""
    workflows = {
        "solve_system": """Solve Ax = b

Square (exact): solve_linear_system(coefficients=[[2,3],[1,1]], constants=[8,3], method="direct") → [x=1, y=2]
Overdetermined (best-fit): solve_linear_system(coefficients=[[1,2],[3,4],[5,6]], constants=[5,6,7], method="least_squares")""",
        "decomposition": """Matrix Decompositions

eigen (PCA, stability): matrix_decomposition(matrix=[[4,2],[1,3]], decomposition="eigen") → eigenvalues, eigenvectors
svd (dimensionality reduction): decomposition="svd" → U, singular_values, Vt
qr (least squares): decomposition="qr" → Q (orthogonal), R (upper triangular)
cholesky (positive definite): decomposition="cholesky" → L where A=LL^T
lu (solving, determinant): decomposition="lu" → P, L, U where PA=LU""",
        "operations": """Matrix Operations

multiply: matrix_operations(operation="multiply", matrix1=[[1,2],[3,4]], matrix2=[[5,6],[7,8]])
inverse: operation="inverse" → A⁻¹ (square matrices only)
transpose: operation="transpose" → Aᵀ
determinant: operation="determinant" → scalar
trace: operation="trace" → sum of diagonal

Element-wise: Use array_operations(operation="add/subtract/multiply/divide/power", ...)""",
    }
    return workflows[problem_type]


@mcp.prompt()
def batch_workflow(
    workflow_type: Annotated[
        Literal["financial_model", "data_pipeline", "transformation"],
        Field(description="Type of multi-step workflow to demonstrate"),
    ] = "financial_model",
) -> str:
    """Multi-step batch_execute workflows with dependency chaining."""
    workflows = {
        "financial_model": """Bond Maturity Value (dependency chaining + parallel execution)

{"operations": [
  {"id": "c", "tool": "calculate", "arguments": {"expression": "p*r", "variables": {"p": 1000000, "r": 0.04}}},
  {"id": "cfv", "tool": "financial_calcs", "arguments": {"calculation": "fv", "rate": 0.04, "periods": 10, "payment": "$c.result", "present_value": 0}},
  {"id": "pfv", "tool": "compound_interest", "arguments": {"principal": 1000000, "rate": 0.04, "time": 10, "frequency": "annual"}},
  {"id": "tot", "tool": "calculate", "arguments": {"expression": "cfv+pfv", "variables": {"cfv": "$cfv.result", "pfv": "$pfv.result"}}}
], "execution_mode": "auto", "output_mode": "value"}

Wave 1: c | Wave 2: cfv, pfv (parallel) | Wave 3: tot""",
        "data_pipeline": """Data Pipeline (transform → analyze → outliers)

{"operations": [
  {"id": "norm", "tool": "array_transform", "arguments": {"data": [[10,20,30],[40,50,60],[70,80,90]], "transform": "minmax_scale", "axis": 0}},
  {"id": "stats", "tool": "array_statistics", "arguments": {"data": "$norm.result", "operations": ["mean","std"], "axis": 0}},
  {"id": "sums", "tool": "array_statistics", "arguments": {"data": "$norm.result", "operations": ["sum"], "axis": 1}},
  {"id": "out", "tool": "statistics", "arguments": {"data": "$sums.result.sum", "analyses": ["outliers"]}}
], "execution_mode": "auto", "output_mode": "minimal"}

Wave 1: norm | Wave 2: stats, sums (parallel) | Wave 3: out""",
        "transformation": """Calculus Chain (derive → evaluate → integrate → round)

{"operations": [
  {"id": "d", "tool": "derivative", "arguments": {"expression": "x^3+2*x^2+x", "variable": "x", "order": 1}},
  {"id": "ev", "tool": "calculate", "arguments": {"expression": "3*x^2+4*x+1", "variables": {"x": 2}}},
  {"id": "i", "tool": "integral", "arguments": {"expression": "x^3+2*x^2+x", "variable": "x", "lower_bound": 0, "upper_bound": 2, "method": "symbolic"}},
  {"id": "r", "tool": "round", "arguments": {"values": ["$ev.result","$i.result"], "method": "round", "decimals": 2}}
], "execution_mode": "auto", "output_mode": "final"}

output_mode options: "full" (all metadata), "value" (flat map, 70-80% reduction), "final" (terminal only, 95% reduction)""",
    }
    return workflows[workflow_type]


# ============================================================================
# MCP Resources - Server Documentation
# ============================================================================


@mcp.resource("tools://available")
def available_tools() -> str:
    """List all 21 available mathematical tools with descriptions.

    Returns structured documentation of all tools organised by category.
    """
    return """Available Mathematical Tools (21)

BASIC (4)
calculate: SymPy expressions | calculate(expression="x^2+2*x+1", variables={"x": 3}) → 16
percentage: of/increase/decrease/change | percentage(operation="of", value=200, percentage=15) → 30
round: nearest/floor/ceil/trunc | round(values=3.14159, method="round", decimals=2) → 3.14
convert_units: degrees ↔ radians | convert_units(value=180, from_unit="degrees", to_unit="radians") → π

ARRAY (4)
array_operations: element-wise add/subtract/multiply/divide/power | array_operations(operation="multiply", array1=[[1,2],[3,4]], array2=2) → [[2,4],[6,8]]
array_statistics: mean/median/std/min/max/sum (axis: 0=col, 1=row, None=all) | array_statistics(data=[[1,2,3],[4,5,6]], operations=["mean"], axis=0) → [2.5,3.5,4.5]
array_aggregate: sumproduct/weighted_average/dot_product | array_aggregate(operation="sumproduct", array1=[1,2,3], array2=[4,5,6]) → 32
array_transform: normalize/standardize/minmax_scale/log_transform | array_transform(data=[[3,4]], transform="normalize") → [[0.6,0.8]]

STATISTICS (3)
statistics: describe/quartiles/outliers (IQR-based) | statistics(data=[1,2,3,4,5,100], analyses=["describe","outliers"])
pivot_table: reshape tabular data (sum/mean/count/min/max) | pivot_table(data=[...], index="region", columns="product", values="sales")
correlation: pearson/spearman (matrix/pairs format) | correlation(data={"x":[1,2,3], "y":[2,4,6]}, method="pearson")

FINANCIAL (3)
financial_calcs: PV/FV/PMT/rate/IRR/NPV (TVM) | financial_calcs(calculation="pv", rate=0.05, periods=10, payment=30, future_value=1000)
compound_interest: discrete/continuous compounding | compound_interest(principal=1000, rate=0.05, time=10, frequency="monthly")
perpetuity: level/growing, ordinary/due | perpetuity(payment=1000, rate=0.05) → 20000

LINEAR ALGEBRA (3)
matrix_operations: multiply/inverse/transpose/determinant/trace | matrix_operations(operation="multiply", matrix1=[[1,2],[3,4]], matrix2=[[5,6],[7,8]])
solve_linear_system: Ax=b (direct/least_squares) | solve_linear_system(coefficients=[[2,3],[1,1]], constants=[8,3]) → [1,2]
matrix_decomposition: eigen/svd/qr/cholesky/lu | matrix_decomposition(matrix=[[4,2],[1,3]], decomposition="eigen")

CALCULUS (3)
derivative: symbolic/numerical, partial derivatives | derivative(expression="x^3+2*x^2", variable="x", order=1) → "3*x^2+4*x"
integral: symbolic/numerical, definite/indefinite | integral(expression="x^2", variable="x", lower_bound=0, upper_bound=1) → 0.333
limits_series: limits (point/infinity/one-sided), Taylor/Maclaurin | limits_series(expression="sin(x)/x", variable="x", point=0, operation="limit") → 1

BATCH (1)
batch_execute: multi-step workflows with dependency chaining ($op_id.result) | DAG-based auto parallelization | execution_mode: auto/sequential/parallel | 90-95% token reduction | See docs://batch-execution

Global params (all tools): context="label", output_mode="full/compact/minimal/value/final" (70-95% token reduction)
"""


@mcp.resource("docs://batch-execution")
def batch_execution_guide() -> str:
    """Comprehensive guide to using batch_execute for multi-step workflows.

    Covers dependency chaining, execution modes, output modes, and best practices.
    """
    return """Batch Execution Guide

Use for: Multi-step workflows with dependencies (step N needs step N-1), parallel scenarios, data pipelines
Skip for: Single operations, workflows needing intermediate validation

DEPENDENCY CHAINING
Reference: $op_id.result (primary), $op_id.result[0] (array index), $op_id.metadata.field (nested)

{"operations": [
  {"id": "s1", "tool": "calculate", "arguments": {"expression": "100*0.05"}},
  {"id": "s2", "tool": "calculate", "arguments": {"expression": "x+10", "variables": {"x": "$s1.result"}}}
]}

EXECUTION MODES
auto (recommended): DAG-based, parallel within wave, sequential across waves → max performance
sequential: strict order (first to last) → use when order matters beyond dependencies
parallel: all concurrent → use only for truly independent operations (fails if dependencies exist)

OUTPUT MODES (See docs://output-modes for details)
full: all metadata (~4-5KB for 4 ops) → debugging
compact: remove nulls (~20-30% reduction) → production with moderate logging
minimal: operation objects only (~60-70% reduction) → focus on results
value: flat {id: value} map (~70-80% reduction) → production minimalist
final: terminal result only (~95% reduction) → sequential chains, max efficiency

ERROR HANDLING
stop_on_error=false (default): independent ops continue even if others fail
stop_on_error=true: stop at first error → use for strict sequential dependencies
Errors include: failed op ID, message, partial results, summary

EXAMPLE: Bond Maturity Value
{"operations": [
  {"id": "coup", "tool": "calculate", "arguments": {"expression": "p*r", "variables": {"p": 1000000, "r": 0.04}}},
  {"id": "cfv", "tool": "financial_calcs", "arguments": {"calculation": "fv", "rate": 0.04, "periods": 10, "payment": "$coup.result", "present_value": 0}},
  {"id": "pfv", "tool": "compound_interest", "arguments": {"principal": 1000000, "rate": 0.04, "time": 10, "frequency": "annual"}},
  {"id": "tot", "tool": "calculate", "arguments": {"expression": "cfv+pfv", "variables": {"cfv": "$cfv.result", "pfv": "$pfv.result"}}}
], "execution_mode": "auto", "output_mode": "value"}

Execution: W1=[coup] → W2=[cfv,pfv parallel] → W3=[tot]
Response: {"coup": 40000, "cfv": 480244.28, "pfv": 1480244.28, "tot": 1960488.56, "summary": {...}}
Token savings: value vs full = 75% (4.2KB → 1.1KB)

PATTERNS
Parallel compare: [scenario_a, scenario_b, scenario_c] | execution_mode="parallel"
Sequential pipeline: load → transform → analyse → report | output_mode="final"
Diamond: root → [branch_a, branch_b parallel] → merge | execution_mode="auto"

LIMITS: max 100 ops, timeout 100ms-300s per op, no circular deps (DAG only), no nested batch_execute

See batch_workflow prompt for financial/data/calculus examples
"""


@mcp.resource("docs://output-modes")
def output_modes_guide() -> str:
    """Guide to output modes for controlling response size and structure.

    Explains the 5 output modes and when to use each for optimal token efficiency.
    """
    return """Output Modes Guide

All 21 tools support output_mode parameter for controlling response verbosity (70-95% token reduction possible)

5 MODES

1. full (default, 100% baseline)
   Use: development, debugging, first-time usage
   Returns: complete metadata (result, tool, timestamp, execution_time_ms, input echo, context)
   Single: {"result": 42, "metadata": {...}, "context": "...", "output_mode": "full"}
   Batch: full operation objects with dependencies, waves, execution times

2. compact (20-30% reduction)
   Use: production with moderate logging
   Returns: result + non-null metadata, no indentation (single-line JSON)
   Single: {"result":42,"context":"..."}
   Batch: compact operation array

3. minimal (60-70% reduction)
   Use: production focus on results, operation-level tracking needed
   Returns: result/value, context (if provided), status/wave (batch only), summary (batch only)
   Single: {"result": 42, "context": "..."}
   Batch: {"results": [{"id": "s1", "status": "success", "value": 42, "wave": 0}, ...], "summary": {...}}
   Removes: input echo, timestamps, detailed metadata, dependencies

4. value (70-80% reduction) ← RECOMMENDED for production batch workflows
   Use: production minimalist, flat key-value mapping
   Returns: flat structure, direct op_id → value mapping
   Single: {"value": 42, "context": "..."}
   Batch: {"step1": 42, "step2": 84, "step3": 126, "summary": {...}, "context": "..."}
   Errors: {"good_op": 42, "summary": {...}, "errors": {"bad_op": "Division by zero"}}

5. final (95% reduction, most aggressive)
   Use: sequential chains (A→B→C→D), only need final answer, max token efficiency
   Returns: terminal result only
   Batch: {"result": 168, "summary": {...}, "context": "..."}
   Requirements: pure sequential chain (no branching), all ops successful, batch_execute only
   Fallback: non-sequential → value mode, any failure → minimal mode, non-batch → value mode

COMPARISON (4-op bond workflow)
full: ~4200 bytes | compact: ~3000 bytes | minimal: ~1400 bytes (67% ↓) | value: ~800 bytes (81% ↓) | final: ~200 bytes (95% ↓)

Single tool: full ~850B → value ~25B (97% reduction)
Batch (10 ops): full ~10KB → value ~1.5KB (85% ↓) → final ~200B (98% ↓)

RECOMMENDATION: Use value for production batch workflows, minimal for single tool calls in production, full for development only. Context preserved in all modes. Errors always included (safety first).
"""


# Import and register all tools (must be after mcp instance creation for decorators)
from .tools import array, basic, batch, calculus, financial, linalg, statistics  # noqa: E402

# Explicitly declare as part of module interface (tools registered via decorators)
__all__ = ["mcp", "basic", "array", "batch", "statistics", "financial", "linalg", "calculus"]


def main():
    """Entry point for uvx."""
    mcp.run()


if __name__ == "__main__":
    main()
