"""Calculus tools using SymPy for symbolic computation."""

from typing import Literal, Optional, Union
from mcp.types import ToolAnnotations
from sympy import sympify, diff, integrate, limit, series, Symbol, oo, N, lambdify
import scipy.integrate as integrate_numeric

from ..server import mcp
from ..core import format_json


@mcp.tool(
    name="math_derivative",
    description="Compute symbolic and numerical derivatives with support for higher orders and partial derivatives.",
    annotations=ToolAnnotations(
        title="Derivative Calculator",
        readOnlyHint=True,
        idempotentHint=True,
    ),
)
async def derivative(
    expression: str, variable: str, order: int = 1, point: Optional[float] = None
) -> str:
    """
    Calculate derivatives using SymPy.

    Examples:
        - expression="x^3 + 2*x^2", variable="x", order=1 → "3*x^2 + 4*x"
        - expression="x^3 + 2*x^2", variable="x", order=2 → "6*x + 4"
        - expression="sin(x)", variable="x", order=1, point=0 → derivative expression and value at x=0

    Args:
        expression: Mathematical expression
        variable: Variable to differentiate with respect to
        order: Derivative order (default: 1)
        point: Optional point for numerical evaluation

    Returns:
        JSON with symbolic derivative and optional numerical value
    """
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

        return format_json(result_data)

    except Exception as e:
        raise ValueError(
            f"Derivative calculation failed: {str(e)}. Example: expression='x^2', variable='x'"
        )


@mcp.tool(
    name="math_integral",
    description="Compute symbolic and numerical integrals (definite and indefinite).",
    annotations=ToolAnnotations(
        title="Integral Calculator",
        readOnlyHint=True,
        idempotentHint=True,
    ),
)
async def integral(
    expression: str,
    variable: str,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
    method: Literal["symbolic", "numerical"] = "symbolic",
) -> str:
    """
    Calculate integrals using SymPy (symbolic) or SciPy (numerical).

    Examples:
        - expression="x^2", variable="x" → "x^3/3" (indefinite)
        - expression="x^2", variable="x", lower_bound=0, upper_bound=1 → 0.333... (definite)
        - expression="sin(x)", variable="x", lower_bound=0, upper_bound=pi → 2.0

    Args:
        expression: Mathematical expression
        variable: Integration variable
        lower_bound: Lower bound for definite integral
        upper_bound: Upper bound for definite integral
        method: Integration method (symbolic or numerical)

    Returns:
        JSON with integral result
    """
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

        return format_json(result_data)

    except Exception as e:
        raise ValueError(
            f"Integration failed: {str(e)}. "
            f"Example: expression='x^2', variable='x', lower_bound=0, upper_bound=1"
        )


@mcp.tool(
    name="math_limits_series",
    description="Compute limits and series expansions using SymPy.",
    annotations=ToolAnnotations(
        title="Limits and Series",
        readOnlyHint=True,
        idempotentHint=True,
    ),
)
async def limits_series(
    expression: str,
    variable: str,
    point: Union[float, str],
    operation: Literal["limit", "series"] = "limit",
    order: int = 6,
    direction: Literal["+", "-", "+-"] = "+-",
) -> str:
    """
    Calculate limits or series expansions.

    Examples:
        - expression="sin(x)/x", variable="x", point=0, operation="limit" → 1
        - expression="1/x", variable="x", point=0, operation="limit", direction="+" → ∞
        - expression="exp(x)", variable="x", point=0, operation="series", order=4 → 1 + x + x^2/2 + x^3/6 + O(x^4)

    Args:
        expression: Mathematical expression
        variable: Variable
        point: Point for limit/expansion ("oo" for infinity)
        operation: Operation type (limit or series)
        order: Order for series expansion (default: 6)
        direction: Direction for limit ("+", "-", or "+-" for both sides)

    Returns:
        JSON with limit or series result
    """
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

        return format_json(result_data)

    except Exception as e:
        raise ValueError(
            f"Limit/series calculation failed: {str(e)}. "
            f"Example: expression='sin(x)/x', variable='x', point=0, operation='limit'"
        )
