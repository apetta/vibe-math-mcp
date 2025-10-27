# Vibe Math

A high-performance Model Context Protocol (MCP) server for mathematical operations, leveraging Polars for optimal calculation speed and comprehensive mathematical capabilities from basic arithmetic to advanced calculus and linear algebra.

## Features

**21 Mathematical Tools** across 6 domains + batch orchestration:

- **Basic Calculations** (4 tools): Expression evaluation, percentages, rounding, unit conversion
- **Array Operations** (4 tools): Element-wise operations, statistics, aggregations, transformations
- **Statistics** (3 tools): Descriptive statistics, pivot tables, correlations
- **Financial Mathematics** (3 tools): Time value of money, compound interest, perpetuity
- **Linear Algebra** (3 tools): Matrix operations, system solving, decompositions
- **Calculus** (3 tools): Derivatives, integrals, limits & series
- **Batch Execution** (1 tool): Multi-tool orchestration with DAG-based parallelization & UUID tracking

=� **Performance**: Leverages Polars for 30x faster data operations; batch execution offers 3-10x speedup for independent operations

<� **Type-Safe**: Full Pydantic validation with clear, actionable error messages

**Tested**: 72 tests with 43% code coverage

## Installation

### Using uvx (Recommended)

```bash
uvx vibe-math
```

### Using uv

```bash
# Clone the repository
git clone <repository-url>
cd vibe-math

# Install with uv
uv sync

# Run the server
uv run vibe-math
```

## Setup with Claude

### Claude Desktop

Open **Settings > Developer > Edit Config** and add:

**For published package:**

```json
{
  "mcpServers": {
    "math": {
      "command": "uvx",
      "args": ["--from", "/absolute/path/to/vibe-math", "vibe-math"]
    }
  }
}
```

**For local development:**

```json
{
  "mcpServers": {
    "math": {
      "command": "uv",
      "args": [
        "--directory",
        "/Users/akshaypetta/Desktop/Dev/vibe-math",
        "run",
        "vibe-math"
      ]
    }
  }
}
```

Restart Claude Desktop. Look for the 🔌 icon to confirm it's working.

**Config file locations:**

- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

### Claude Code

**Quick setup (CLI):**

Published package:

```bash
claude mcp add --transport stdio math -- uvx vibe-math
```

Local development:

```bash
claude mcp add --transport stdio math -- uvx --from /absolute/path/to/vibe-math vibe-math
```

**Team setup** (create `.mcp.json` in project root):

```json
{
  "mcpServers": {
    "math": {
      "command": "uvx",
      "args": ["--from", "/absolute/path/to/vibe-math", "vibe-math"]
    }
  }
}
```

**Verify:** Run `claude mcp list` or use `/mcp` in Claude Code.

**Test it:**

- "Calculate 15% of 250" → uses `percentage`
- "Find determinant of [[1,2],[3,4]]" → uses `matrix_operations`
- "Integrate x^2 from 0 to 1" → uses `integral`

## Quick Start

### Tool Examples

#### Basic Calculations

```json
// Evaluate mathematical expressions
{
  "tool": "calculate",
  "expression": "x^2 + 2*x + 1",
  "variables": {"x": 3}
}
// � 16.0

// Calculate percentages
{
  "tool": "percentage",
  "operation": "increase",
  "value": 100,
  "percentage": 20
}
// � 120.0
```

#### Array Operations

```json
// Array statistics
{
  "tool": "array_statistics",
  "data": [[1,2,3],[4,5,6]],
  "operations": ["mean", "std"],
  "axis": 0
}

// Weighted average
{
  "tool": "array_aggregate",
  "operation": "weighted_average",
  "array1": [10, 20, 30],
  "weights": [1, 2, 3]
}
// � 23.33...
```

#### Financial Mathematics

```json
// Compound interest
{
  "tool": "compound_interest",
  "principal": 1000,
  "rate": 0.05,
  "time": 10,
  "frequency": "monthly"
}

// Present value
{
  "tool": "financial_calcs",
  "calculation": "pv",
  "rate": 0.05,
  "periods": 10,
  "payment": -100
}
```

#### Linear Algebra

```json
// Solve linear system Ax = b
{
  "tool": "solve_linear_system",
  "coefficients": [[2, 3], [1, 1]],
  "constants": [8, 3]
}
// � x=1, y=2

// Matrix decomposition
{
  "tool": "matrix_decomposition",
  "matrix": [[4,2],[1,3]],
  "decomposition": "svd"
}
```

#### Calculus

```json
// Derivative
{
  "tool": "derivative",
  "expression": "x^3 + 2*x^2",
  "variable": "x",
  "order": 2
}
// � "6*x + 4"

// Definite integral
{
  "tool": "integral",
  "expression": "x^2",
  "variable": "x",
  "lower_bound": 0,
  "upper_bound": 1
}
// � 0.333...
```

### Batch Execution

The `batch_execute` tool enables orchestrating multiple operations in a single request with intelligent dependency management and parallel execution.

**Key Features:**
- **UUID Tracking**: Each operation has a unique ID for clear input/output mapping
- **Dependency Management**: Automatic DAG-based execution with parallel waves
- **Result Referencing**: JSONPath-like syntax to reference prior operation results
- **Three Execution Modes**: Sequential, parallel, or auto (recommended)
- **Performance**: 3-10x faster for independent operations, 40-60% token savings

#### Simple Batch (Parallel Execution)

```json
{
  "tool": "batch_execute",
  "operations": [
    {
      "id": "calc1",
      "tool": "calculate",
      "arguments": {"expression": "2 + 2"}
    },
    {
      "id": "pct1",
      "tool": "percentage",
      "arguments": {"operation": "of", "value": 100, "percentage": 15}
    }
  ],
  "execution_mode": "auto"
}
```

Response includes 1:1 UUID mapping:
```json
{
  "results": [
    {"id": "calc1", "status": "success", "result": {"result": 4}, "wave": 0},
    {"id": "pct1", "status": "success", "result": {"result": 15}, "wave": 0}
  ],
  "summary": {
    "total_operations": 2,
    "succeeded": 2,
    "num_waves": 1
  }
}
```

#### Dependent Operations (DAG Execution)

Chain operations using `$operation_id.result` syntax:

```json
{
  "tool": "batch_execute",
  "operations": [
    {
      "id": "bond_a",
      "tool": "financial_calcs",
      "context": "Corporate Bond A",
      "arguments": {
        "calculation": "pv",
        "rate": 0.05,
        "periods": 10,
        "future_value": 1000
      }
    },
    {
      "id": "bond_b",
      "tool": "financial_calcs",
      "context": "Corporate Bond B",
      "arguments": {
        "calculation": "pv",
        "rate": 0.06,
        "periods": 10,
        "future_value": 1000
      }
    },
    {
      "id": "portfolio",
      "tool": "calculate",
      "context": "Total Portfolio Value",
      "arguments": {
        "expression": "a + b",
        "variables": {
          "a": "$bond_a.result",
          "b": "$bond_b.result"
        }
      },
      "depends_on": ["bond_a", "bond_b"]
    }
  ]
}
```

**Execution**: `bond_a` and `bond_b` run in parallel (wave 0), then `portfolio` runs (wave 1).

#### Result Referencing Syntax

**Universal Accessor (Recommended):**
- `$op_id.value` - Smart accessor that works with any tool

**Legacy Accessors:**
- `$op_id.result` - For tools with "result" field
- `$op_id.values` - For tools with "values" field
- `$op_id.metadata.field` - Nested metadata access
- `$op_id.values[0]` - Array indexing
- `$op_id` - Entire result object

**Why use `.value`?**
Different tools return values in different fields (`result` vs `values`). The universal `.value` accessor automatically detects and extracts the primary value from any tool, eliminating guesswork when chaining operations.

#### Execution Modes

| Mode         | Behaviour                                           | Best For                    |
| ------------ | --------------------------------------------------- | --------------------------- |
| `sequential` | Execute in exact order specified                    | Order-dependent pipelines   |
| `parallel`   | All operations run concurrently (ignores deps)      | Independent calculations    |
| `auto`       | Automatic DAG detection & wave-based parallelization | Mixed dependencies (default) |

---

## Output Control

All tools automatically support output control for maximum flexibility and token efficiency.

### Output Modes

Control response verbosity using the `output_mode` parameter (available on **every tool**):

| Mode       | Description                                    | Token Savings | Use Case                               |
| ---------- | ---------------------------------------------- | ------------- | -------------------------------------- |
| `full`     | Complete response with all metadata (default)  | 0% (baseline) | Debugging, full context needed         |
| `compact`  | Remove null fields, minimize whitespace        | ~20-30%       | Moderate reduction, preserve structure |
| `minimal`  | Primary value(s) only, strip metadata          | ~60-70%       | Fast extraction, minimal context       |
| `value`    | Normalized `{value: X}` structure              | ~70-80%       | Consistent chaining, maximum simplicity|
| `final`    | For sequential chains, return only terminal result | ~95%      | Simple calculations, predictable extraction|

**Example - Single Tool:**

```json
// Full mode (default)
{"result": 105.0, "expression": "100 * 1.05", "variables": null}

// Compact mode
{"result":105.0,"expression":"100 * 1.05"}

// Minimal mode
{"result": 105.0}

// Value mode
{"value": 105.0}
```

**Example - Batch Execution:**

```json
// Full mode - Complete operation details
{
  "results": [
    {"id": "step1", "tool": "calculate", "status": "success", "result": {...}, "wave": 0, ...},
    {"id": "step2", "tool": "percentage", "status": "success", "result": {...}, "wave": 1, ...}
  ],
  "summary": {...}
}

// Value mode - Flat mapping (~90% smaller!)
{
  "step1": 105.0,
  "step2": 115.5,
  "summary": {"succeeded": 2, "failed": 0, "time_ms": 0.85}
}
```

**Client-Side Filtering:**

With `value` mode, batch responses are flat maps that are trivially filterable:

```javascript
// If you only need step2:
const response = await batch_execute({operations: [...], output_mode: "value"});
const finalValue = response.step2;  // Simple property access
```

**Final Mode for Sequential Chains:**

For simple calculations that form a chain (step1 → step2 → step3), use `final` mode:

```json
{
  "operations": [
    {"id": "step1", "tool": "calculate", "arguments": {"expression": "1000 * 1.05"}},
    {"id": "step2", "tool": "calculate", "arguments": {"expression": "$step1.value * 1.10"}, "depends_on": ["step1"]},
    {"id": "step3", "tool": "round", "arguments": {"values": "$step2.value", "decimals": 2}, "depends_on": ["step2"]}
  ],
  "output_mode": "final"
}

// Returns: {"result": 1155.00, "summary": {"succeeded": 3, "failed": 0}}
// Extract: response.result ✨ (predictable!)
```

**Automatic fallback:** If operations have branching/parallelism, automatically falls back to `value` mode.

**Token Savings:**
- Full batch (20 ops): ~2000 tokens
- Value mode: ~200 tokens (90% reduction)
- Final mode (sequential): ~25 tokens (95% reduction)

---

## Result Structure Reference

Different tool categories return different response structures. Here's your quick reference guide:

| Tool Category         | Primary Field | Example Access             | Minimal Output           |
| --------------------- | ------------- | -------------------------- | ------------------------ |
| **Basic**             | `result`      | `response["result"]`       | `{"result": 105.0}`      |
| - calculate           |               |                            |                          |
| - percentage          |               |                            |                          |
| - round               |               |                            |                          |
| - convert_units       |               |                            |                          |
| **Arrays**            | `values`      | `response["values"]`       | `{"values": [[1,2],[3,4]]}` |
| - array_operations    |               |                            |                          |
| - array_statistics    |               |                            |                          |
| - array_aggregate     |               |                            |                          |
| - array_transform     |               |                            |                          |
| **Statistics**        | Multiple keys | `response["describe"]["mean"]` | Full object (already minimal) |
| - statistics          | `describe`, `quartiles`, `outliers` | | |
| - pivot_table         | Pivot structure | | |
| - correlation         | Correlation matrix | | |
| **Financial**         | `result`      | `response["result"]`       | `{"result": 1628.89}`    |
| - financial_calcs     |               |                            |                          |
| - compound_interest   |               |                            |                          |
| - perpetuity          |               |                            |                          |
| **Linear Algebra**    | `result` or `values` | Tool-specific       | Tool-specific            |
| - matrix_operations   | `values` for matrices | `response["values"]` | |
| - solve_linear_system | `result` (solution vector) | `response["result"]` | |
| - matrix_decomposition | Multiple keys | `response["eigenvalues"]` | |
| **Calculus**          | `result` or symbolic | Tool-specific       | Tool-specific            |
| - derivative          | `result` (symbolic expression) | | |
| - integral            | `result` (value or symbolic) | | |
| - limits_series       | `result` | | |

**With `output_mode="value"`:** All tools normalize to `{"value": X}` for consistent chaining!

---

## Common Use Patterns

### Pattern 1: Just Need the Final Answer

```json
{
  "operations": [
    {"id": "step1", "tool": "calculate", "arguments": {"expression": "100 * 1.05"}},
    {"id": "step2", "tool": "percentage", "arguments": {"operation": "increase", "value": "$step1.result", "percentage": 10}},
    {"id": "final", "tool": "calculate", "arguments": {"expression": "x - 25", "variables": {"x": "$step2.result"}}}
  ],
  "output_mode": "value"
}

// Response: ~50 tokens instead of ~500 (90% reduction)
{
  "step1": 105.0,
  "step2": 115.5,
  "final": 90.5,
  "summary": {"succeeded": 3, "failed": 0}
}

// Extract what you need client-side:
const finalAnswer = response.final;  // 90.5
```

### Pattern 2: Chain with Consistent Structure

```json
// Use value mode for consistent {value: X} across all tools
{
  "tool": "calculate",
  "expression": "2 + 2",
  "output_mode": "value"
}
// Returns: {"value": 4}

{
  "tool": "array_operations",
  "operation": "add",
  "array1": [[1,2]],
  "array2": [[3,4]],
  "output_mode": "value"
}
// Returns: {"value": [[4,6]]}

// Now you can reference "$op.value" consistently!
```

### Pattern 3: Debugging with Full Context

```json
// Use full mode when you need to understand what happened
{
  "operations": [...],
  "output_mode": "full"  // See all metadata, execution times, waves, etc.
}
```

### Pattern 4: Production Efficiency

```json
// Use value mode in production for 70-90% token savings
{
  "operations": [...],
  "output_mode": "value",  // Minimize tokens
  "context": "Q4 Financial Analysis"  // But keep identification
}
```

---

## Complete Tool Reference

**Note:** All tool parameters include detailed descriptions with concrete examples directly in the MCP interface. Each parameter shows expected format, use cases, and sample values to make usage obvious without referring to external documentation.

### Basic Calculations

| Tool            | Description                                                  |
| --------------- | ------------------------------------------------------------ |
| `calculate`     | Evaluate mathematical expressions with variable substitution |
| `percentage`    | Percentage calculations (of, increase, decrease, change)     |
| `round`         | Advanced rounding (round, floor, ceil, trunc)                |
| `convert_units` | Unit conversions (degrees � radians)                         |

### Array Operations

| Tool               | Description                                                      |
| ------------------ | ---------------------------------------------------------------- |
| `array_operations` | Element-wise operations (add, subtract, multiply, divide, power) |
| `array_statistics` | Statistical measures (mean, median, std, min, max, sum)          |
| `array_aggregate`  | Aggregations (sumproduct, weighted average, dot product)         |
| `array_transform`  | Transformations (normalise, standardise, scale, log)             |

### Statistics

| Tool          | Description                                            |
| ------------- | ------------------------------------------------------ |
| `statistics`  | Comprehensive analysis (describe, quartiles, outliers) |
| `pivot_table` | Create pivot tables with aggregation                   |
| `correlation` | Correlation matrices (Pearson, Spearman)               |

### Financial Mathematics

| Tool                | Description                                 |
| ------------------- | ------------------------------------------- |
| `financial_calcs`   | Time value of money (PV, FV, PMT, IRR, NPV) |
| `compound_interest` | Compound interest with various frequencies  |

### Linear Algebra

| Tool                   | Description                                                          |
| ---------------------- | -------------------------------------------------------------------- |
| `matrix_operations`    | Matrix operations (multiply, inverse, transpose, determinant, trace) |
| `solve_linear_system`  | Solve Ax = b systems                                                 |
| `matrix_decomposition` | Decompositions (eigen, SVD, QR, Cholesky, LU)                        |

### Calculus

| Tool            | Description                            |
| --------------- | -------------------------------------- |
| `derivative`    | Symbolic and numerical differentiation |
| `integral`      | Symbolic and numerical integration     |
| `limits_series` | Limits and series expansions           |

## Development

### Running Tests

```bash
# Install dev dependencies
uv sync --dev

# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=vibe_math --cov-report=html
```

### Project Structure

```
vibe-math/
   src/vibe_math/
      server.py           # FastMCP server
      core/               # Core utilities
         validators.py   # Pydantic models
         formatters.py   # Response formatting
         converters.py   # Data conversions
      tools/              # Tool modules
          basic.py
          array.py
          statistics.py
          financial.py
          linalg.py
          calculus.py
   tests/                  # Comprehensive tests
```

## Technology Stack

- **FastMCP**: MCP server framework
- **Polars**: High-performance dataframe operations (30x faster than pandas)
- **NumPy**: Numerical computing with BLAS optimisation
- **SciPy**: Advanced scientific algorithms
- **SymPy**: Symbolic mathematics
- **Pydantic**: Type-safe input validation

## Performance

The server is optimised for speed:

- Polars for data operations (>10x faster than pandas)
- NumPy BLAS for linear algebra
- Lazy evaluation where possible
- Streaming support for large datasets

## License

MIT

## Contributing

Contributions welcome! Please ensure:

1. Tests pass (`uv run pytest`)
2. Code is formatted (`uv run ruff format`)
3. Type hints are included
4. Clear, actionable error messages

## Support

For issues and questions, please open an issue on GitHub.
