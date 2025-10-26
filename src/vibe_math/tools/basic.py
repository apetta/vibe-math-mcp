"""Basic mathematical calculation tools."""

import math
from typing import Annotated, Dict, Literal, Optional, Union, List
from pydantic import Field
from sympy import sympify, simplify, N
from mcp.types import ToolAnnotations
import numpy as np

from ..server import mcp
from ..core import format_result


@mcp.tool(
    name="calculate",
    description="""Evaluate mathematical expressions using SymPy.

Supports:
    - Arithmetic: +, -, *, /, ^
    - Trigonometry: sin, cos, tan, asin, acos, atan
    - Logarithms: log, ln, exp
    - Constants: pi, e
    - Functions: sqrt, abs

Examples:

SIMPLE ARITHMETIC:
    expression="2 + 2"
    Result: 4

TRIGONOMETRY:
    expression="sin(pi/2)"
    Result: 1.0

WITH VARIABLES:
    expression="x^2 + 2*x + 1", variables={"x": 3}
    Result: 16

MULTIPLE VARIABLES:
    expression="x^2 + y^2", variables={"x": 3, "y": 4}
    Result: 25""",
    annotations=ToolAnnotations(
        title="Mathematical Expression Calculator",
        readOnlyHint=True,
        idempotentHint=True,
    ),
)
async def calculate(
    expression: Annotated[str, Field(description="Mathematical expression (e.g., '2+2', 'sin(pi/2)', 'x^2+1')", min_length=1)],
    variables: Annotated[Dict[str, float] | None, Field(description="Variable substitutions (e.g., {'x': 5, 'y': 10})")] = None,
) -> str:
    """Evaluate mathematical expressions."""
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
    name="percentage",
    description="""Perform percentage calculations: of, increase, decrease, or change.

Examples:

PERCENTAGE OF: 15% of 200
    operation="of", value=200, percentage=15
    Result: 30

INCREASE: 100 increased by 20%
    operation="increase", value=100, percentage=20
    Result: 120

DECREASE: 100 decreased by 20%
    operation="decrease", value=100, percentage=20
    Result: 80

PERCENTAGE CHANGE: from 80 to 100
    operation="change", value=80, percentage=100
    Result: 25 (25% increase)""",
    annotations=ToolAnnotations(
        title="Percentage Calculator",
        readOnlyHint=True,
        idempotentHint=True,
    ),
)
async def percentage(
    operation: Annotated[Literal["of", "increase", "decrease", "change"], Field(description="Type of calculation")],
    value: Annotated[float, Field(description="Base value")],
    percentage: Annotated[float, Field(description="Percentage amount (or new value for 'change')")],
) -> str:
    """Perform percentage calculations."""
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
    name="round",
    description="""Advanced rounding operations with multiple methods.

Methods:
    - round: Round to nearest (3.145 → 3.15 at 2dp)
    - floor: Always round down (3.149 → 3.14)
    - ceil: Always round up (3.141 → 3.15)
    - trunc: Truncate towards zero (-3.7 → -3, 3.7 → 3)

Examples:

ROUND TO NEAREST:
    values=3.14159, method="round", decimals=2
    Result: 3.14

FLOOR (DOWN):
    values=3.14159, method="floor", decimals=2
    Result: 3.14

CEIL (UP):
    values=3.14159, method="ceil", decimals=2
    Result: 3.15

MULTIPLE VALUES:
    values=[3.14159, 2.71828], method="round", decimals=2
    Result: [3.14, 2.72]""",
    annotations=ToolAnnotations(
        title="Advanced Rounding",
        readOnlyHint=True,
        idempotentHint=True,
    ),
)
async def round_values(
    values: Annotated[Union[float, List[float]], Field(description="Single value or list (e.g., 3.14159 or [3.14, 2.71])")],
    method: Annotated[Literal["round", "floor", "ceil", "trunc"], Field(description="Rounding method")] = "round",
    decimals: Annotated[int, Field(description="Number of decimal places", ge=0)] = 0,
) -> str:
    """Advanced rounding operations."""
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
    name="convert_units",
    description="""Convert between angle units: degrees ↔ radians.

Examples:

DEGREES TO RADIANS:
    value=180, from_unit="degrees", to_unit="radians"
    Result: 3.14159... (π)

RADIANS TO DEGREES:
    value=3.14159, from_unit="radians", to_unit="degrees"
    Result: 180

RIGHT ANGLE:
    value=90, from_unit="degrees", to_unit="radians"
    Result: 1.5708... (π/2)""",
    annotations=ToolAnnotations(
        title="Unit Converter",
        readOnlyHint=True,
        idempotentHint=True,
    ),
)
async def convert_units(
    value: Annotated[float, Field(description="Value to convert (e.g., 180, 3.14159)")],
    from_unit: Annotated[Literal["degrees", "radians"], Field(description="Source unit")],
    to_unit: Annotated[Literal["degrees", "radians"], Field(description="Target unit")],
) -> str:
    """Convert between angle units."""
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
