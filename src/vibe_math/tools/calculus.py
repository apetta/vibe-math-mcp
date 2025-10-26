"""Calculus tools using SymPy for symbolic computation."""

from typing import Annotated, Literal, Union
from pydantic import Field
from mcp.types import ToolAnnotations
from sympy import sympify, diff, integrate, limit, series, Symbol, oo, N, lambdify
import scipy.integrate as integrate_numeric

from ..server import mcp
from ..core import format_json, ContextParam


@mcp.tool(
    name="derivative",
    description="""Compute symbolic and numerical derivatives with support for higher orders and partial derivatives.

Examples:

FIRST DERIVATIVE:
    expression="x^3 + 2*x^2", variable="x", order=1
    Result: derivative="3*x^2 + 4*x"

SECOND DERIVATIVE (acceleration/concavity):
    expression="x^3", variable="x", order=2
    Result: derivative="6*x"

EVALUATE AT POINT:
    expression="sin(x)", variable="x", order=1, point=0
    Result: derivative="cos(x)", value_at_point=1.0

PRODUCT RULE:
    expression="sin(x)*cos(x)", variable="x", order=1
    Result: derivative="cos(x)^2 - sin(x)^2"

PARTIAL DERIVATIVE:
    expression="x^2*y", variable="y", order=1
    Result: derivative="x^2" (treating x as constant)""",
    annotations=ToolAnnotations(
        title="Derivative Calculator",
        readOnlyHint=True,
        idempotentHint=True,
    ),
)
async def derivative(
    expression: Annotated[
        str,
        Field(
            description="Mathematical expression to differentiate (e.g., 'x^3 + 2*x^2', 'sin(x)')",
            min_length=1,
        ),
    ],
    variable: Annotated[
        str,
        Field(
            description="Variable to differentiate with respect to (e.g., 'x', 't')", min_length=1
        ),
    ],
    order: Annotated[
        int, Field(description="Derivative order (1=first derivative, 2=second, etc.)", ge=1)
    ] = 1,
    point: Annotated[
        float | None, Field(description="Optional point for numerical evaluation of the derivative")
    ] = None,
    context: ContextParam = None,
) -> str:
    """Compute symbolic derivatives using SymPy. Supports higher orders and partial derivatives. Optional numerical evaluation at a point."""
    try:
        expr = sympify(expression)
        var = Symbol(variable)

        # Compute derivative
        derivative_expr = diff(expr, var, order)
        derivative_str = str(derivative_expr)

        result_data = {
            "expression": expression,
            "variable": variable,
            "order": order,
            "derivative": derivative_str,
        }

        # Evaluate at point if provided
        if point is not None:
            value = float(N(derivative_expr.subs(var, point)))
            result_data["value_at_point"] = value
            result_data["point"] = point

        # Add context if provided
        if context is not None:
            result_data["context"] = context

        return format_json(result_data)

    except Exception as e:
        raise ValueError(
            f"Derivative calculation failed: {str(e)}. Example: expression='x^2', variable='x'"
        )


@mcp.tool(
    name="integral",
    description="""Compute symbolic and numerical integrals (definite and indefinite).

Examples:

INDEFINITE INTEGRAL (antiderivative):
    expression="x^2", variable="x"
    Result: "x^3/3"

DEFINITE INTEGRAL (area):
    expression="x^2", variable="x", lower_bound=0, upper_bound=1
    Result: 0.333

TRIGONOMETRIC:
    expression="sin(x)", variable="x", lower_bound=0, upper_bound=3.14159
    Result: 2.0 (area under one period)

NUMERICAL METHOD (non-elementary):
    expression="exp(-x^2)", variable="x", lower_bound=0, upper_bound=1, method="numerical"
    Result: 0.746824 (Gaussian integral approximation)

SYMBOLIC ANTIDERIVATIVE:
    expression="1/x", variable="x"
    Result: "log(x)" """,
    annotations=ToolAnnotations(
        title="Integral Calculator",
        readOnlyHint=True,
        idempotentHint=True,
    ),
)
async def integral(
    expression: Annotated[
        str,
        Field(
            description="Mathematical expression to integrate (e.g., 'x^2', 'sin(x)')", min_length=1
        ),
    ],
    variable: Annotated[
        str, Field(description="Integration variable (e.g., 'x', 't')", min_length=1)
    ],
    lower_bound: Annotated[
        float | None, Field(description="Lower bound for definite integral (omit for indefinite)")
    ] = None,
    upper_bound: Annotated[
        float | None, Field(description="Upper bound for definite integral (omit for indefinite)")
    ] = None,
    method: Annotated[
        Literal["symbolic", "numerical"],
        Field(
            description="Integration method: symbolic=exact/analytical, numerical=approximate (requires bounds)"
        ),
    ] = "symbolic",
    context: ContextParam = None,
) -> str:
    """Compute integrals using SymPy (symbolic/exact) or SciPy (numerical/approximate). Supports indefinite (antiderivatives) and definite (area) integrals."""
    try:
        expr = sympify(expression)
        var = Symbol(variable)

        is_definite = lower_bound is not None and upper_bound is not None

        if method == "symbolic":
            if is_definite:
                result = integrate(expr, (var, lower_bound, upper_bound))
                result_data = {
                    "expression": expression,
                    "variable": variable,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "result": float(N(result)),
                    "symbolic_result": str(result),
                    "type": "definite",
                }
            else:
                result = integrate(expr, var)
                result_data = {
                    "expression": expression,
                    "variable": variable,
                    "result": str(result),
                    "type": "indefinite",
                }

        elif method == "numerical":
            if not is_definite:
                raise ValueError("Numerical integration requires lower_bound and upper_bound")

            # Convert SymPy expression to numeric function
            func = lambdify(var, expr, "numpy")

            # Use SciPy's quad for numerical integration
            result, error = integrate_numeric.quad(func, lower_bound, upper_bound)

            result_data = {
                "expression": expression,
                "variable": variable,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "result": float(result),
                "error_estimate": float(error),
                "method": "numerical",
                "type": "definite",
            }

        else:
            raise ValueError(f"Unknown method: {method}")

        # Add context if provided
        if context is not None:
            result_data["context"] = context

        return format_json(result_data)

    except Exception as e:
        raise ValueError(
            f"Integration failed: {str(e)}. "
            f"Example: expression='x^2', variable='x', lower_bound=0, upper_bound=1"
        )


@mcp.tool(
    name="limits_series",
    description="""Compute limits and series expansions using SymPy.

Examples:

CLASSIC LIMIT:
    expression="sin(x)/x", variable="x", point=0, operation="limit"
    Result: limit=1

LIMIT AT INFINITY:
    expression="1/x", variable="x", point="oo", operation="limit"
    Result: limit=0

ONE-SIDED LIMIT:
    expression="1/x", variable="x", point=0, operation="limit", direction="+"
    Result: limit=+∞ (approaching from right)

REMOVABLE DISCONTINUITY:
    expression="(x^2-1)/(x-1)", variable="x", point=1, operation="limit"
    Result: limit=2

MACLAURIN SERIES (at 0):
    expression="exp(x)", variable="x", point=0, operation="series", order=4
    Result: "1 + x + x^2/2 + x^3/6 + O(x^4)"

TAYLOR SERIES (at point):
    expression="sin(x)", variable="x", point=3.14159, operation="series", order=4
    Result: expansion around π""",
    annotations=ToolAnnotations(
        title="Limits and Series",
        readOnlyHint=True,
        idempotentHint=True,
    ),
)
async def limits_series(
    expression: Annotated[
        str,
        Field(
            description="Mathematical expression to analyse (e.g., 'sin(x)/x', 'exp(x)')",
            min_length=1,
        ),
    ],
    variable: Annotated[
        str, Field(description="Variable for limit/expansion (e.g., 'x', 't')", min_length=1)
    ],
    point: Annotated[
        Union[float, str],
        Field(
            description="Point for limit/expansion (number, 'oo' for infinity, '-oo' for -infinity)"
        ),
    ],
    operation: Annotated[
        Literal["limit", "series"],
        Field(description="Operation: limit=compute limit, series=Taylor/Maclaurin expansion"),
    ] = "limit",
    order: Annotated[int, Field(description="Series expansion order (number of terms)", ge=1)] = 6,
    direction: Annotated[
        Literal["+", "-", "+-"],
        Field(description="Limit direction: +=from right, -=from left, +-=both sides"),
    ] = "+-",
    context: ContextParam = None,
) -> str:
    """Compute limits (lim[x→a]f(x)) and Taylor/Maclaurin series expansions using SymPy. Handles infinity, one-sided limits, removable discontinuities."""
    try:
        expr = sympify(expression)
        var = Symbol(variable)

        # Handle infinity
        if point == "oo" or point == "inf":
            point_sym = oo
        elif point == "-oo" or point == "-inf":
            point_sym = -oo
        else:
            point_sym = float(point)

        if operation == "limit":
            if direction == "+-":
                # Two-sided limit
                result = limit(expr, var, point_sym)
            else:
                # One-sided limit
                result = limit(expr, var, point_sym, direction)

            result_data = {
                "expression": expression,
                "variable": variable,
                "point": str(point),
                "direction": direction,
                "limit": str(result),
                "numeric_value": float(N(result)) if result.is_number else None,
            }

        elif operation == "series":
            # Series expansion
            series_expr = series(expr, var, point_sym, order)  # type: ignore[arg-type]
            series_str = str(series_expr)

            # Remove O() term for cleaner output
            series_no_o = series_expr.removeO()

            result_data = {
                "expression": expression,
                "variable": variable,
                "point": str(point),
                "order": order,
                "series": series_str,
                "series_without_O": str(series_no_o),
            }

        else:
            raise ValueError(f"Unknown operation: {operation}")

        # Add context if provided
        if context is not None:
            result_data["context"] = context

        return format_json(result_data)

    except Exception as e:
        raise ValueError(
            f"Limit/series calculation failed: {str(e)}. "
            f"Example: expression='sin(x)/x', variable='x', point=0, operation='limit'"
        )
