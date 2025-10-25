"""Input validation using Pydantic models for type safety and clear error messages."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator


class CalculateInput(BaseModel):
    """Input model for mathematical expression evaluation."""

    model_config = ConfigDict(str_strip_whitespace=True)

    expression: str = Field(
        ...,
        description="Mathematical expression to evaluate (e.g., '2+2', 'sin(pi/2)', 'x^2+1')",
        min_length=1,
        max_length=2000,
    )
    variables: Optional[Dict[str, float]] = Field(
        default=None, description="Variable substitutions as dict (e.g., {'x': 5, 'y': 10})"
    )


class ArrayInput(BaseModel):
    """Input model for array operations."""

    data: List[List[float]] = Field(..., description="2D array/matrix as nested list")

    @field_validator("data")
    @classmethod
    def validate_array(cls, v: List[List[float]]) -> List[List[float]]:
        if not v or not v[0]:
            raise ValueError("Array cannot be empty")
        row_length = len(v[0])
        if not all(len(row) == row_length for row in v):
            raise ValueError("All rows must have the same length")
        return v


def validate_matrix_square(matrix: List[List[float]]) -> None:
    """Validate that a matrix is square (n×n)."""
    if len(matrix) != len(matrix[0]):
        raise ValueError(
            f"Matrix must be square for this operation. Got shape: {len(matrix)}×{len(matrix[0])}"
        )


def validate_arrays_compatible(arr1: List[List[float]], arr2: List[List[float]]) -> None:
    """Validate that two arrays have compatible shapes for operations."""
    if len(arr1) != len(arr2) or len(arr1[0]) != len(arr2[0]):
        raise ValueError(
            f"Arrays must have same shape. "
            f"Got {len(arr1)}×{len(arr1[0])} and {len(arr2)}×{len(arr2[0])}"
        )
