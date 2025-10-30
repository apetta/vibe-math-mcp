# Vibe Math MCP

[![smithery badge](https://smithery.ai/badge/@apetta/vibe-math-mcp)](https://smithery.ai/server/@apetta/vibe-math-mcp)
[![PyPI version](https://badge.fury.io/py/vibe-math-mcp.svg)](https://badge.fury.io/py/vibe-math-mcp)
[![Python Version](https://img.shields.io/pypi/pyversions/vibe-math-mcp.svg)](https://pypi.org/project/vibe-math-mcp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Smithery](https://smithery.ai/badge/@apetta/vibe-math-mcp)](https://smithery.ai/server/@apetta/vibe-math-mcp)
[![Test Coverage](https://img.shields.io/badge/coverage-87%25-brightgreen)](https://github.com/apetta/vibe-math)
[![Tests](https://img.shields.io/badge/tests-245%20passing-brightgreen)](https://github.com/apetta/vibe-math)

A high-performance Model Context Protocol (MCP) server for math-ing whilst vibing with LLMs. Built with Polars, Pandas, NumPy, SciPy, and SymPy for optimal calculation speed and comprehensive mathematical capabilities from basic arithmetic to advanced calculus and linear algebra.

## Features

**21 Mathematical Tools** across 6 domains + batch orchestration:

- **Basic Calculations** (4 tools): Expression evaluation, percentages, rounding, unit conversion
- **Array Operations** (4 tools): Element-wise operations, statistics, aggregations, transformations
- **Statistics** (3 tools): Descriptive statistics, pivot tables, correlations
- **Financial Mathematics** (3 tools): Time value of money, compound interest, perpetuity
- **Linear Algebra** (3 tools): Matrix operations, system solving, decompositions
- **Calculus** (3 tools): Derivatives, integrals, limits & series
- **Batch Execution** (1 tool): Multi-tool orchestration for complex workflows

## Installation

### IDEs

[![Install in VS Code](https://img.shields.io/badge/VS_Code-Install_Math-0098FF?style=flat-square&logo=visualstudiocode&logoColor=ffffff)](vscode:mcp/install?%7B%22name%22%3A%22Math%22%2C%22type%22%3A%22stdio%22%2C%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22vibe-math-mcp%22%5D%7D)

[![Install MCP Server](https://cursor.com/deeplink/mcp-install-dark.svg)](https://cursor.com/en-US/install-mcp?name=Math&config=eyJjb21tYW5kIjoidXZ4IHZpYmUtbWF0aC1tY3AifQ%3D%3D)

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
      "args": [
        "--directory",
        "/absolute/path/to/vibe-math-mcp",
        "run",
        "vibe-math-mcp"
      ]
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
claude mcp add --transport stdio math -- uvx --from /absolute/path/to/vibe-math-mcp vibe-math-mcp
```

**Team setup** (create `mcp.json` in project root for shared use with Claude Code and/or IDEs)

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

**Verify:** Run `claude mcp list` or use `/mcp` or view available servers in IDEs.

### Try it

- "Calculate 15% of 250" → uses `percentage`
- "Find determinant of [[1,2],[3,4]]" → uses `matrix_operations`
- "Integrate x^2 from 0 to 1" → uses `integral`
- "If I invest $1000 at 5% annual interest compounded monthly for 10 years, what will be the future value?" → uses `compound_interest`
- If I was paid the square root of $69m in 10 years, what's the present value at 7% discount rate? → uses `batch_execute (calculate -> financial_calcs)`

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

## Batch Execution

For multi-step workflows, `batch_execute` chains multiple calculations in a single request—**achieving 90-95% token reduction**. Reference prior outputs using `$operation_id.result` syntax, and the engine automatically handles dependency resolution and parallel execution for speed.

**Perfect for:** Bond pricing, financial models, statistical pipelines, complex transformations

## Complete Tool Reference

**Note:** All tool parameters include detailed descriptions with concrete examples directly in the MCP interface. Each parameter shows expected format, use cases, and sample values to make usage obvious without referring to external documentation.

### Basic Calculations

| Tool            | Description                                                  |
| --------------- | ------------------------------------------------------------ |
| `calculate`     | Evaluate mathematical expressions with variable substitution |
| `percentage`    | Percentage calculations (of, increase, decrease, change)     |
| `round`         | Advanced rounding (round, floor, ceil, trunc)                |
| `convert_units` | Unit conversions (degrees – radians)                         |

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
# Install dependencies
uv sync

# Run all tests
uv run poe test
```

### Local Development Modes

**STDIO Mode** (default - for Claude Desktop, IDEs):

```bash
uv run vibe-math-mcp
```

**HTTP Mode** (for container testing):

```bash
uv run python -m vibe_math_mcp.http_server
```

## License

MIT License. See `LICENSE` file for details.

## Contributing

Contributions welcome via PRs! Please ensure:

1. Tests pass, and new ones are added if applicable
2. Code is linted & formatted
3. Type hints are included
4. Clear, actionable error messages are provided

## Support

For issues and questions, please open an issue on GitHub.