"""Basic mathematical calculation tools."""

import math
from typing import Dict, Literal, Optional, Union, List
from sympy import sympify, simplify, N
from mcp.types import ToolAnnotations
import numpy as np

from ..server import mcp
from ..core import format_result


@mcp.tool(
    name="math_calculate",
    description="Evaluate mathematical expressions with optional variable substitution. Supports arithmetic, trigonometric, logarithmic, and algebraic functions.",
    annotations=ToolAnnotations(
        title="Mathematical Expression Calculator",
        readOnlyHint=True,
        idempotentHint=True,
    ),
)
async def calculate(expression: str, variables: Optional[Dict[str, float]] = None) -> str:
    """
    Evaluate mathematical expressions using SymPy.

    Examples:
        - "2 + 2" → 4
        - "sin(pi/2)" → 1.0
        - "x^2 + 2*x + 1" with {"x": 3} → 16

    Args:
        expression: Mathematical expression (e.g., "2+2", "sin(pi/2)", "x^2+1")
        variables: Variable substitutions as dict (e.g., {"x": 5, "y": 10})

    Returns:
        JSON with result and expression details
    """
    try:
        expr = sympify(expression)

        if variables:
            result = float(N(expr.subs(variables)))
        else:
            result = float(N(simplify(expr)))

        return format_result(result, {"expression": expression, "variables": variables})
    except Exception as e:
        raise ValueError(
            f"Failed to evaluate expression '{expression}'. "
            f"Error: {str(e)}. "
            f"Example: '2*x + 3' with variables={{'x': 5}}"
        )


@mcp.tool(
    name="math_percentage",
    description="Perform percentage calculations: percentage of a value, increase, decrease, or percentage change.",
    annotations=ToolAnnotations(
        title="Percentage Calculator",
        readOnlyHint=True,
        idempotentHint=True,
    ),
)
async def percentage(
    operation: Literal["of", "increase", "decrease", "change"], value: float, percentage: float
) -> str:
    """
    Calculate percentages with different operations.

    Examples:
        - operation="of", value=200, percentage=15 → 30 (15% of 200)
        - operation="increase", value=100, percentage=20 → 120 (100 increased by 20%)
        - operation="decrease", value=100, percentage=20 → 80 (100 decreased by 20%)
        - operation="change", value=80, percentage=100 → 25 (percentage change from 80 to 100)

    Args:
        operation: Type of percentage calculation
        value: Base value
        percentage: Percentage amount

    Returns:
        JSON with result and calculation details
    """
    try:
        if operation == "of":
            result = (percentage / 100) * value
            explanation = f"{percentage}% of {value}"
        elif operation == "increase":
            result = value * (1 + percentage / 100)
            explanation = f"{value} increased by {percentage}%"
        elif operation == "decrease":
            result = value * (1 - percentage / 100)
            explanation = f"{value} decreased by {percentage}%"
        elif operation == "change":
            # percentage is actually the new value in this case
            if value == 0:
                raise ValueError("Cannot calculate percentage change from zero")
            result = ((percentage - value) / value) * 100
            explanation = f"Percentage change from {value} to {percentage}"
        else:
            raise ValueError(f"Unknown operation: {operation}")

        return format_result(
            result,
            {
                "operation": operation,
                "value": value,
                "percentage": percentage,
                "explanation": explanation,
            },
        )
    except Exception as e:
        raise ValueError(f"Percentage calculation failed: {str(e)}")


@mcp.tool(
    name="math_round",
    description="Advanced rounding operations: round, floor, ceil, or truncate values with specified decimal places.",
    annotations=ToolAnnotations(
        title="Advanced Rounding",
        readOnlyHint=True,
        idempotentHint=True,
    ),
)
async def round_values(
    values: Union[float, List[float]],
    method: Literal["round", "floor", "ceil", "trunc"] = "round",
    decimals: int = 0,
) -> str:
    """
    Perform various rounding operations.

    Examples:
        - values=3.14159, method="round", decimals=2 → 3.14
        - values=3.14159, method="floor", decimals=2 → 3.14
        - values=3.14159, method="ceil", decimals=2 → 3.15
        - values=[3.14159, 2.71828], method="round", decimals=2 → [3.14, 2.72]

    Args:
        values: Single value or list of values to round
        method: Rounding method (round, floor, ceil, trunc)
        decimals: Number of decimal places

    Returns:
        JSON with rounded value(s)
    """
    try:
        is_single = isinstance(values, (int, float))
        vals = [values] if is_single else values

        arr = np.array(vals, dtype=float)

        if method == "round":
            result = np.round(arr, decimals)
        elif method == "floor":
            result = np.floor(arr * 10**decimals) / 10**decimals
        elif method == "ceil":
            result = np.ceil(arr * 10**decimals) / 10**decimals
        elif method == "trunc":
            result = np.trunc(arr * 10**decimals) / 10**decimals
        else:
            raise ValueError(f"Unknown method: {method}")

        final_result = float(result[0]) if is_single else result.tolist()

        return format_result(final_result, {"method": method, "decimals": decimals})
    except Exception as e:
        raise ValueError(f"Rounding operation failed: {str(e)}")


@mcp.tool(
    name="math_convert_units",
    description="Convert between mathematical units (degrees/radians, etc.).",
    annotations=ToolAnnotations(
        title="Unit Converter",
        readOnlyHint=True,
        idempotentHint=True,
    ),
)
async def convert_units(
    value: float, from_unit: Literal["degrees", "radians"], to_unit: Literal["degrees", "radians"]
) -> str:
    """
    Convert between angle units.

    Examples:
        - value=180, from_unit="degrees", to_unit="radians" → 3.14159...
        - value=3.14159, from_unit="radians", to_unit="degrees" → 180

    Args:
        value: Value to convert
        from_unit: Source unit
        to_unit: Target unit

    Returns:
        JSON with converted value
    """
    try:
        if from_unit == to_unit:
            result = value
        elif from_unit == "degrees" and to_unit == "radians":
            result = math.radians(value)
        elif from_unit == "radians" and to_unit == "degrees":
            result = math.degrees(value)
        else:
            raise ValueError(f"Unsupported conversion: {from_unit} to {to_unit}")

        return format_result(
            result, {"from_unit": from_unit, "to_unit": to_unit, "original_value": value}
        )
    except Exception as e:
        raise ValueError(f"Unit conversion failed: {str(e)}")
