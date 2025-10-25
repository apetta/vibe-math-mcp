# Math MCP Server - Comprehensive Implementation Plan

## Executive Summary

A high-performance Model Context Protocol (MCP) server for mathematical operations, leveraging Polars for optimal calculation speed and comprehensive mathematical capabilities from basic arithmetic to advanced calculus and linear algebra. The server is designed to be intuitive, avoiding decision paralysis through carefully balanced tool selection whilst maintaining powerful functionality.

---

## 1. Project Overview

### 1.1 Purpose
Create a Python MCP server (`math_mcp`) that provides mathematical computation tools optimised for LLM interaction, with emphasis on:
- **Performance**: Utilising Polars (30x faster than pandas) for array/dataframe operations
- **Usability**: Intuitive tool design preventing decision paralysis
- **Comprehensiveness**: Coverage from basic calculations to advanced mathematics
- **Reliability**: Thorough testing with pytest across diverse scenarios

### 1.2 Core Principles
Following MCP best practices:
- **Agent-centric design**: Tools enable complete workflows, not just API wrappers
- **Context optimisation**: Return high-signal information, avoid data dumps
- **Actionable errors**: Error messages guide towards correct usage
- **Natural task subdivision**: Tool names reflect how humans think about mathematical tasks

### 1.3 Technical Foundation
- **Primary Library**: Polars for data operations (prioritised)
- **Fallback Library**: Pandas for operations where Polars lacks coverage
- **Scientific Computing**: NumPy, SciPy for numerical operations
- **Symbolic Mathematics**: SymPy for calculus and algebraic manipulation
- **Framework**: FastMCP (MCP Python SDK)
- **Testing**: pytest with comprehensive scenario coverage

---

## 2. Architecture Design

### 2.1 Server Structure
```
math_mcp/
├── server.py              # Main server file with FastMCP initialisation
├── tools/
│   ├── __init__.py
│   ├── basic.py          # Basic arithmetic operations
│   ├── array.py          # Array calculations
│   ├── statistics.py     # Statistical operations
│   ├── financial.py      # Financial mathematics
│   ├── linalg.py         # Linear algebra operations
│   └── calculus.py       # Calculus operations
├── core/
│   ├── __init__.py
│   ├── validators.py     # Input validation helpers
│   ├── formatters.py     # Output formatting utilities
│   └── converters.py     # Data conversion (Polars ↔ Pandas)
├── tests/
│   ├── __init__.py
│   ├── test_basic.py
│   ├── test_array.py
│   ├── test_statistics.py
│   ├── test_financial.py
│   ├── test_linalg.py
│   └── test_calculus.py
├── pyproject.toml        # Project configuration
├── README.md
├── PLAN.md
└── .gitignore
```

### 2.2 Data Flow Architecture
```
Input (JSON/Arrays) → Validation → Polars Processing → Results → Formatting → Output
                                        ↓
                                  Pandas Fallback
                                  (if needed)
```

### 2.3 Module Organisation
- **Modular tools**: Each mathematical domain in separate file
- **Shared utilities**: Common functionality extracted to core modules
- **Zero duplication**: DRY principle strictly enforced
- **Type safety**: Full type hints throughout codebase

---

## 3. Tool Categories & Specific Tools

### Tool Selection Philosophy
**Balance Point**: 15-20 tools total
- Enough coverage for comprehensive maths
- Few enough to avoid decision paralysis
- Each tool serves distinct, clear purpose
- Tools work together for complex workflows

### 3.1 Basic Calculations (4 tools)

#### 3.1.1 `math_calculate`
**Purpose**: Evaluate mathematical expressions
**Inputs**:
- `expression` (str): Mathematical expression (e.g., "2 + 3 * 4", "sin(pi/2)", "sqrt(16)")
- `variables` (Optional[dict]): Variable substitutions (e.g., {"x": 5, "y": 10})
**Outputs**: Calculation result with step-by-step evaluation (optional)
**Backend**: SymPy for parsing and evaluation
**Example**: `math_calculate("x^2 + 2*x + 1", {"x": 3})` → 16

#### 3.1.2 `math_percentage`
**Purpose**: Percentage calculations (increase, decrease, of total)
**Inputs**:
- `operation` (Literal["of", "increase", "decrease", "change"]): Type of percentage calc
- `value` (float): Base value
- `percentage` (float): Percentage amount
**Outputs**: Result and explanation
**Backend**: Pure Python calculations
**Example**: `math_percentage("increase", 100, 20)` → 120 (increased by 20%)

#### 3.1.3 `math_round`
**Purpose**: Advanced rounding operations
**Inputs**:
- `values` (Union[float, List[float]]): Value(s) to round
- `method` (Literal["round", "floor", "ceil", "trunc"]): Rounding method
- `decimals` (int): Number of decimal places
**Outputs**: Rounded value(s)
**Backend**: NumPy for array operations
**Example**: `math_round([3.14159, 2.71828], "round", 2)` → [3.14, 2.72]

#### 3.1.4 `math_convert_units`
**Purpose**: Convert between mathematical units (radians/degrees, etc.)
**Inputs**:
- `value` (float): Value to convert
- `from_unit` (str): Source unit
- `to_unit` (str): Target unit
**Outputs**: Converted value
**Backend**: NumPy
**Example**: `math_convert_units(180, "degrees", "radians")` → 3.14159...

### 3.2 Array Calculations (4 tools)

#### 3.2.1 `math_array_operations`
**Purpose**: Element-wise operations on arrays/matrices
**Inputs**:
- `operation` (Literal["add", "subtract", "multiply", "divide", "power"]): Operation type
- `array1` (List[List[float]]): First array
- `array2` (Union[List[List[float]], float]): Second array or scalar
**Outputs**: Result array
**Backend**: Polars for 2D operations, NumPy for compatibility
**Example**: `math_array_operations("multiply", [[1,2],[3,4]], 2)` → [[2,4],[6,8]]

#### 3.2.2 `math_array_statistics`
**Purpose**: Statistical measures on arrays (mean, median, std, etc.)
**Inputs**:
- `data` (List[List[float]]): Data array
- `operations` (List[str]): Statistics to compute ["mean", "median", "std", "min", "max", "sum"]
- `axis` (Optional[int]): Axis for computation (0=columns, 1=rows, None=all)
**Outputs**: Statistical results
**Backend**: Polars for performance
**Example**: `math_array_statistics([[1,2,3],[4,5,6]], ["mean", "std"], axis=0)` → {mean: [2.5, 3.5, 4.5], std: [...]}

#### 3.2.3 `math_array_aggregate`
**Purpose**: Aggregation operations (sumproduct, weighted average, etc.)
**Inputs**:
- `operation` (Literal["sumproduct", "weighted_average", "correlation"]): Aggregation type
- `array1` (List[float]): First array
- `array2` (Optional[List[float]]): Second array (for operations requiring two arrays)
- `weights` (Optional[List[float]]): Weights for weighted operations
**Outputs**: Aggregated result
**Backend**: Polars
**Example**: `math_array_aggregate("sumproduct", [1,2,3], [4,5,6])` → 32

#### 3.2.4 `math_array_transform`
**Purpose**: Array transformations (normalise, standardise, scale)
**Inputs**:
- `data` (List[List[float]]): Input data
- `transform` (Literal["normalize", "standardize", "minmax_scale", "log_transform"]): Transform type
- `axis` (Optional[int]): Axis for transformation
**Outputs**: Transformed array
**Backend**: Polars + NumPy
**Example**: `math_array_transform([[1,2],[3,4]], "normalize")` → normalised values

### 3.3 Statistics & Data Analysis (3 tools)

#### 3.3.1 `math_statistics`
**Purpose**: Comprehensive statistical analysis
**Inputs**:
- `data` (List[float]): Data series
- `analyses` (List[str]): Analyses to perform ["describe", "quartiles", "outliers", "distribution"]
**Outputs**: Statistical report
**Backend**: Polars
**Example**: `math_statistics([1,2,3,4,5,100], ["describe", "outliers"])` → full statistical summary

#### 3.3.2 `math_pivot_table`
**Purpose**: Create pivot tables from tabular data
**Inputs**:
- `data` (List[dict]): Tabular data
- `index` (Union[str, List[str]]): Row grouping column(s)
- `columns` (Optional[str]): Column grouping
- `values` (str): Column to aggregate
- `aggfunc` (Literal["sum", "mean", "count", "min", "max"]): Aggregation function
**Outputs**: Pivot table as nested dict/array
**Backend**: Polars
**Example**: Sales data → pivot by region and product

#### 3.3.3 `math_correlation`
**Purpose**: Correlation and covariance analysis
**Inputs**:
- `data` (Dict[str, List[float]]): Multiple series as dictionary
- `method` (Literal["pearson", "spearman"]): Correlation method
- `output_format` (Literal["matrix", "pairs"]): Output format
**Outputs**: Correlation matrix or pairwise correlations
**Backend**: Polars
**Example**: Analyse correlations between multiple variables

### 3.4 Financial Mathematics (2 tools)

#### 3.4.1 `math_financial_calcs`
**Purpose**: Common financial calculations
**Inputs**:
- `calculation` (Literal["pv", "fv", "pmt", "irr", "npv"]): Calculation type
- `rate` (float): Interest rate
- `periods` (int): Number of periods
- `payment` (Optional[float]): Payment amount
- `present_value` (Optional[float]): Present value
- `future_value` (Optional[float]): Future value
- `cash_flows` (Optional[List[float]]): Cash flow series (for IRR/NPV)
**Outputs**: Calculated financial result
**Backend**: NumPy financial functions
**Example**: `math_financial_calcs("fv", 0.05, 10, payment=-100)` → future value

#### 3.4.2 `math_compound_interest`
**Purpose**: Compound interest calculations with various frequencies
**Inputs**:
- `principal` (float): Initial amount
- `rate` (float): Annual interest rate
- `time` (float): Time period
- `frequency` (Literal["annual", "semi-annual", "quarterly", "monthly", "daily", "continuous"]): Compounding frequency
**Outputs**: Final amount and breakdown
**Backend**: NumPy
**Example**: Calculate compound interest with monthly compounding

### 3.5 Linear Algebra (3 tools)

#### 3.5.1 `math_matrix_operations`
**Purpose**: Core matrix operations
**Inputs**:
- `operation` (Literal["multiply", "inverse", "transpose", "determinant", "trace"]): Operation type
- `matrix1` (List[List[float]]): First matrix
- `matrix2` (Optional[List[List[float]]]): Second matrix (for operations requiring two)
**Outputs**: Result matrix or scalar
**Backend**: NumPy (BLAS/LAPACK optimised)
**Example**: `math_matrix_operations("multiply", [[1,2],[3,4]], [[5,6],[7,8]])` → matrix product

#### 3.5.2 `math_solve_linear_system`
**Purpose**: Solve systems of linear equations
**Inputs**:
- `coefficients` (List[List[float]]): Coefficient matrix (A in Ax = b)
- `constants` (List[float]): Constants vector (b in Ax = b)
- `method` (Optional[Literal["direct", "least_squares"]]): Solution method
**Outputs**: Solution vector
**Backend**: SciPy (scipy.linalg.solve)
**Example**: Solve system of equations

#### 3.5.3 `math_matrix_decomposition`
**Purpose**: Matrix decompositions (eigenvalues, SVD, QR, etc.)
**Inputs**:
- `matrix` (List[List[float]]): Input matrix
- `decomposition` (Literal["eigen", "svd", "qr", "cholesky", "lu"]): Decomposition type
**Outputs**: Decomposition results (eigenvalues/vectors, factors, etc.)
**Backend**: SciPy (scipy.linalg)
**Example**: `math_matrix_decomposition([[4,2],[1,3]], "eigen")` → eigenvalues and eigenvectors

### 3.6 Calculus (3 tools)

#### 3.6.1 `math_derivative`
**Purpose**: Symbolic and numerical differentiation
**Inputs**:
- `expression` (str): Mathematical expression
- `variable` (str): Variable to differentiate with respect to
- `order` (int): Derivative order (default: 1)
- `point` (Optional[float]): Point for numerical evaluation
**Outputs**: Symbolic derivative and/or numerical value
**Backend**: SymPy
**Example**: `math_derivative("x^3 + 2*x^2", "x", order=2)` → "6*x + 4"

#### 3.6.2 `math_integral`
**Purpose**: Symbolic and numerical integration
**Inputs**:
- `expression` (str): Mathematical expression
- `variable` (str): Integration variable
- `lower_bound` (Optional[float]): Lower bound for definite integral
- `upper_bound` (Optional[float]): Upper bound for definite integral
- `method` (Literal["symbolic", "numerical"]): Integration method
**Outputs**: Integral result (symbolic or numerical)
**Backend**: SymPy (symbolic), SciPy (numerical)
**Example**: `math_integral("x^2", "x", 0, 1)` → 1/3

#### 3.6.3 `math_limits_series`
**Purpose**: Limits and series expansions
**Inputs**:
- `expression` (str): Mathematical expression
- `variable` (str): Variable
- `point` (Union[float, str]): Point for limit ("oo" for infinity)
- `operation` (Literal["limit", "series"]): Operation type
- `order` (Optional[int]): Order for series expansion
**Outputs**: Limit value or series expansion
**Backend**: SymPy
**Example**: `math_limits_series("sin(x)/x", "x", 0, "limit")` → 1

---

## 4. Technical Implementation Details

### 4.1 Input Validation Strategy
```python
# Pydantic models for each tool
class CalculateInput(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )
    
    expression: str = Field(
        ...,
        description="Mathematical expression to evaluate (e.g., '2+2', 'sin(pi/2)')",
        min_length=1,
        max_length=1000
    )
    variables: Optional[Dict[str, float]] = Field(
        default=None,
        description="Variable substitutions as dict (e.g., {'x': 5, 'y': 10})"
    )
```

### 4.2 Response Format Standards
All tools support two formats:
- **JSON** (machine-readable, complete data)
- **Markdown** (human-readable, formatted)

```python
class ResponseFormat(str, Enum):
    MARKDOWN = "markdown"
    JSON = "json"
```

### 4.3 Error Handling Framework
```python
# Custom error types
class MathCalculationError(Exception):
    """Raised when mathematical calculation fails"""
    pass

class InvalidInputError(Exception):
    """Raised when input validation fails"""
    pass

# Error message guidelines:
# 1. Explain what went wrong
# 2. Suggest specific next steps
# 3. Provide examples if helpful
```

### 4.4 Performance Optimisations

#### Polars Usage Priority
```python
# Use Polars for:
- Tabular data operations
- Array statistics
- Aggregations
- Pivot tables
- Large dataset operations

# Use NumPy for:
- Matrix operations (better BLAS integration)
- Scientific functions
- When Polars doesn't support operation

# Use Pandas for:
- Operations unavailable in Polars
- Compatibility requirements
```

#### Lazy Evaluation
```python
# For large datasets, use Polars LazyFrames
df_lazy = pl.scan_csv("data.csv")
result = df_lazy.filter(...).group_by(...).collect()
```

### 4.5 Data Conversion Utilities
```python
# core/converters.py
def polars_to_pandas(df: pl.DataFrame) -> pd.DataFrame:
    """Convert Polars to Pandas when necessary"""
    
def pandas_to_polars(df: pd.DataFrame) -> pl.DataFrame:
    """Convert Pandas to Polars for performance"""
    
def list_to_polars(data: List[List[float]]) -> pl.DataFrame:
    """Convert nested list to Polars DataFrame"""
```

---

## 5. Testing Strategy

### 5.1 Test Categories

#### Unit Tests
- Each tool function tested in isolation
- Edge cases (empty arrays, division by zero, etc.)
- Invalid inputs
- Type validation

#### Integration Tests
- Tools working together in workflows
- Data format conversions
- Error propagation

#### Performance Tests
- Benchmark Polars vs Pandas operations
- Large dataset handling
- Memory efficiency

### 5.2 Test Scenarios

#### Basic Calculations
```python
def test_calculate_simple():
    """Test basic arithmetic"""
    assert calculate("2 + 2") == 4

def test_calculate_with_variables():
    """Test variable substitution"""
    assert calculate("x^2 + y", {"x": 3, "y": 1}) == 10

def test_calculate_invalid_expression():
    """Test error handling"""
    with pytest.raises(InvalidInputError):
        calculate("2 +* 3")
```

#### Array Operations
```python
def test_array_operations_multiply():
    """Test array multiplication"""
    result = array_operations("multiply", [[1,2],[3,4]], 2)
    expected = [[2,4],[6,8]]
    assert result == expected

def test_array_operations_incompatible_shapes():
    """Test shape mismatch error"""
    with pytest.raises(ValueError):
        array_operations("multiply", [[1,2]], [[1,2,3]])
```

#### Linear Algebra
```python
def test_solve_linear_system_2x2():
    """Test 2x2 system solution"""
    A = [[1, 2], [3, 4]]
    b = [5, 6]
    x = solve_linear_system(A, b)
    # Verify Ax = b
    assert np.allclose(np.dot(A, x), b)

def test_matrix_decomposition_eigen():
    """Test eigenvalue decomposition"""
    matrix = [[4, 2], [1, 3]]
    result = matrix_decomposition(matrix, "eigen")
    # Verify eigenvalues exist and are correct
```

#### Calculus
```python
def test_derivative_polynomial():
    """Test polynomial differentiation"""
    result = derivative("x^3 + 2*x^2 + x", "x")
    expected = "3*x^2 + 4*x + 1"
    assert simplify(result) == simplify(expected)

def test_integral_definite():
    """Test definite integration"""
    result = integral("x^2", "x", 0, 1)
    assert abs(result - 1/3) < 1e-10
```

### 5.3 Test Coverage Goals
- **Target**: 95%+ code coverage
- All error paths tested
- All edge cases covered
- Performance benchmarks recorded

---

## 6. Development Roadmap

### Phase 1: Foundation (Week 1)
1. Set up project structure
2. Configure FastMCP server
3. Implement core utilities (validators, formatters, converters)
4. Basic calculations tools (4 tools)
5. Unit tests for Phase 1

### Phase 2: Data Operations (Week 2)
1. Array calculations tools (4 tools)
2. Statistics tools (3 tools)
3. Integration tests for data pipeline
4. Performance benchmarking (Polars vs Pandas)

### Phase 3: Advanced Mathematics (Week 3)
1. Financial mathematics tools (2 tools)
2. Linear algebra tools (3 tools)
3. Calculus tools (3 tools)
4. Comprehensive integration tests

### Phase 4: Polish & Documentation (Week 4)
1. Error message refinement
2. Performance optimisation
3. Complete documentation
4. Full test suite execution
5. Evaluation scenarios creation

---

## 7. Server Configuration

### 7.1 Server Initialisation
```python
# server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("math_mcp")

# Module-level constants
CHARACTER_LIMIT = 25000
DEFAULT_PRECISION = 10
```

### 7.2 Tool Registration Pattern
```python
@mcp.tool(
    name="math_calculate",
    annotations={
        "title": "Mathematical Expression Calculator",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def calculate(params: CalculateInput) -> str:
    """
    Evaluate mathematical expressions with optional variable substitution.
    
    Supports arithmetic, trigonometric, logarithmic, and algebraic functions.
    
    Args:
        params (CalculateInput): Validated input containing:
            - expression (str): Mathematical expression
            - variables (Optional[Dict]): Variable substitutions
    
    Returns:
        str: JSON or Markdown formatted result
    """
    # Implementation
```

### 7.3 Dependencies
```toml
# pyproject.toml
[project]
name = "math-mcp-server"
version = "0.1.0"
dependencies = [
    "mcp>=1.0.0",
    "polars>=0.20.0",
    "pandas>=2.2.0",
    "numpy>=1.26.0",
    "scipy>=1.12.0",
    "sympy>=1.12",
    "pydantic>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=4.1.0",
    "pytest-asyncio>=0.23.0",
    "black>=24.0.0",
    "ruff>=0.3.0",
]
```

---

## 8. Documentation Requirements

### 8.1 README.md Contents
- Project overview and features
- Installation instructions (uvx compatible)
- Quick start examples
- Tool reference (brief)
- Contributing guidelines
- License

### 8.2 Tool Documentation
Each tool requires:
- Clear description of purpose
- Input parameter specifications with examples
- Output format explanation
- Usage examples (simple and complex)
- Common error scenarios

### 8.3 Code Documentation
- Comprehensive docstrings (Google style)
- Type hints throughout
- Inline comments for complex logic
- Architecture decision records (ADRs) for major choices

---

## 9. Quality Assurance Checklist

### Strategic Design
- [ ] Tools enable complete mathematical workflows
- [ ] Tool names reflect natural task subdivisions
- [ ] Response formats optimised for agent context
- [ ] Human-readable identifiers used appropriately
- [ ] Error messages guide towards correct usage

### Implementation Quality
- [ ] Most important tools implemented
- [ ] All tools have descriptive names and documentation
- [ ] Return types consistent across similar operations
- [ ] Error handling for all operations
- [ ] Server name follows format: `math_mcp`
- [ ] All operations use async/await where appropriate
- [ ] Common functionality extracted into reusable functions
- [ ] Error messages clear, actionable, and educational
- [ ] Outputs properly validated and formatted

### Tool Configuration
- [ ] All tools implement name and annotations
- [ ] Annotations correctly set (readOnlyHint, etc.)
- [ ] All tools use Pydantic BaseModel for input validation
- [ ] All Pydantic Fields have explicit types and descriptions
- [ ] All tools have comprehensive docstrings
- [ ] Docstrings include complete schema structure

### Code Quality
- [ ] Proper imports including Pydantic imports
- [ ] Type hints used throughout
- [ ] Constants defined at module level in UPPER_CASE
- [ ] Zero code duplication (DRY principle)
- [ ] Composable, reusable helper functions

### Testing
- [ ] Server runs successfully
- [ ] All imports resolve correctly
- [ ] Sample tool calls work as expected
- [ ] Error scenarios handled gracefully
- [ ] 95%+ code coverage achieved
- [ ] Performance benchmarks pass

---

## 10. Success Metrics

### Performance Targets
- Polars operations >10x faster than pandas equivalents
- Response time <100ms for basic operations
- Response time <1s for complex matrix operations
- Memory efficient for datasets up to 1GB

### Quality Targets
- Zero critical bugs in production
- 95%+ test coverage
- 100% of tools fully documented
- Clear error messages for all failure modes

### Usability Targets
- Users can accomplish common tasks with 1-3 tool calls
- Tool names intuitively suggest their purpose
- Error messages lead to successful retries

---

## 11. Future Enhancements (Post-MVP)

### Potential Additions
1. **Optimisation tools**: Linear programming, constraint solving
2. **Graph theory**: Shortest path, network analysis
3. **Numerical methods**: ODE solvers, root finding
4. **Signal processing**: FFT, filters
5. **Machine learning utilities**: Regression, clustering basics
6. **Geometric calculations**: Areas, volumes, transformations
7. **Probability distributions**: PDF, CDF, sampling
8. **Time series**: Trend analysis, forecasting basics

### Scaling Considerations
- Streaming support for very large datasets (Polars lazy frames)
- Caching for repeated calculations
- Parallel processing for independent operations
- Resource management for long-running computations

---

## 12. Risk Mitigation

### Technical Risks
1. **Polars limitations**: Mitigation: Pandas fallback implemented
2. **Symbolic computation timeout**: Mitigation: Set timeouts, provide numerical alternatives
3. **Memory exhaustion**: Mitigation: Input size limits, streaming for large data
4. **Numerical instability**: Mitigation: Use SciPy's robust algorithms, document limitations

### Usability Risks
1. **Too many tools**: Mitigation: 15-20 tool limit, clear categorisation
2. **Confusing tool names**: Mitigation: User testing, clear documentation
3. **Unexpected errors**: Mitigation: Comprehensive testing, actionable error messages

---

## Appendix A: Example Workflows

### Workflow 1: Statistical Analysis Pipeline
```
1. math_array_statistics → Get descriptive statistics
2. math_correlation → Identify relationships
3. math_pivot_table → Summarise by categories
```

### Workflow 2: Financial Planning
```
1. math_compound_interest → Calculate growth
2. math_financial_calcs → Compute present value
3. math_array_aggregate → Sum investment returns
```

### Workflow 3: Linear System Analysis
```
1. math_matrix_operations → Build coefficient matrix
2. math_solve_linear_system → Find solution
3. math_matrix_decomposition → Verify solution stability
```

### Workflow 4: Calculus Problem Solving
```
1. math_derivative → Find critical points
2. math_integral → Calculate area under curve
3. math_limits_series → Analyse behaviour at boundaries
```

---

## Appendix B: Performance Benchmarks (Expected)

| Operation | Polars | Pandas | Speedup |
|-----------|--------|--------|---------|
| 1M row aggregation | 50ms | 800ms | 16x |
| 100k row filtering | 10ms | 150ms | 15x |
| Pivot table (medium) | 30ms | 400ms | 13x |
| Group by statistics | 40ms | 600ms | 15x |

---

## Appendix C: Tool Decision Matrix

| Tool Name | Frequency | Complexity | Priority | Status |
|-----------|-----------|------------|----------|--------|
| math_calculate | High | Low | P0 | Phase 1 |
| math_array_statistics | High | Medium | P0 | Phase 2 |
| math_matrix_operations | Medium | High | P1 | Phase 3 |
| math_derivative | Medium | High | P1 | Phase 3 |
| math_pivot_table | Medium | Medium | P1 | Phase 2 |
| math_integral | Low | High | P2 | Phase 3 |
| ... | ... | ... | ... | ... |

---

## Conclusion

This comprehensive plan provides a roadmap for building a high-quality, performant math MCP server that balances functionality with usability. By prioritising Polars for performance, maintaining strict code quality standards, and thoroughly testing across diverse scenarios, we'll deliver a robust mathematical computation tool that integrates seamlessly with LLM workflows.

The server design follows MCP best practices, ensuring tools are intuitive, error messages are actionable, and response formats are optimised for agent context. With 15-18 carefully selected tools covering basic calculations through advanced mathematics, users can accomplish complex workflows without decision paralysis.

**Next Steps**: Begin Phase 1 implementation, starting with project structure setup and basic calculation tools.
