# Vibe Math

[![PyPI version](https://badge.fury.io/py/vibe-math-mcp.svg)](https://badge.fury.io/py/vibe-math-mcp)
[![Python Version](https://img.shields.io/pypi/pyversions/vibe-math-mcp.svg)](https://pypi.org/project/vibe-math-mcp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Smithery](https://smithery.ai/badge/@apetta/vibe-math-mcp)](https://smithery.ai/server/@apetta/vibe-math-mcp)
[![Test Coverage](https://img.shields.io/badge/coverage-87%25-brightgreen)](https://github.com/apetta/vibe-math)
[![Tests](https://img.shields.io/badge/tests-245%20passing-brightgreen)](https://github.com/apetta/vibe-math)

A high-performance Model Context Protocol (MCP) server for math-ing whilst vibing with LLMs. Leveraging Polars for optimal calculation speed and comprehensive mathematical capabilities from basic arithmetic to advanced calculus and linear algebra.

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

**Tested**: 245 tests with 87% code coverage

## Installation

## Setup with Claude

### Claude Desktop

Open **Settings > Developer > Edit Config** and add:

**For published package:**

```json
{
  "mcpServers": {
    "Math": {
      "command": "uvx",
      "args": ["vibe-math-mcp"]
    }
  }
}
```

**For local development:**

```json
{
  "mcpServers": {
    "Math": {
      "command": "uv",
      "args": ["--directory", "/absolute/path/to/vibe-math", "run", "vibe-math"]
    }
  }
}
```

### Claude Code

**Quick setup (CLI):**

Published package:

```bash
claude mcp add --transport stdio math -- uvx vibe-math-mcp
```

Local development:

```bash
claude mcp add --transport stdio math -- uvx --from /absolute/path/to/vibe-math vibe-math
```

**Team setup** (create `.mcp.json` in project root for shared use with Claude Code & IDEs):

```json
{
  "mcpServers": {
    "math": {
      "command": "uvx",
      "args": ["vibe-math-mcp"]
    }
  }
}
```

**Verify:** Run `claude mcp list` or use `/mcp` in Claude Code.

**Test it:**

- "Calculate 15% of 250" → uses `percentage`
- "Find determinant of [[1,2],[3,4]]" → uses `matrix_operations`
- "Integrate x^2 from 0 to 1" → uses `integral`

## Output Control

All tools automatically support output control for maximum flexibility and token efficiency. The LLM can specify the desired verbosity.

Control response verbosity using the `output_mode` parameter (available on **every tool**):

| Mode      | Description                                        | Token Savings | Use Case                                    |
| --------- | -------------------------------------------------- | ------------- | ------------------------------------------- |
| `full`    | Complete response with all metadata (default)      | 0% (baseline) | Debugging, full context needed              |
| `compact` | Remove null fields, minimize whitespace            | ~20-30%       | Moderate reduction, preserve structure      |
| `minimal` | Primary value(s) only, strip metadata              | ~60-70%       | Fast extraction, minimal context            |
| `value`   | Normalized `{value: X}` structure                  | ~70-80%       | Consistent chaining, maximum simplicity     |
| `final`   | For sequential chains, return only terminal result | ~95%          | Simple calculations, predictable extraction |

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

---

## Development

### Running Tests

```bash
# Install dev dependencies
uv sync --dev

# Run all tests
uv run poe test

```

## License

MIT

## Contributing

Contributions welcome via PRs! Please ensure:

1. Tests pass
2. Code is formatted
3. Type hints are included
4. Clear, actionable error messages are provided

## Support

For issues and questions, please open an issue on GitHub.
