"""Tests for core validators module."""

import pytest
from pydantic import ValidationError

from vibe_math_mcp.core.validators import (
    CalculateInput,
    ArrayInput,
    validate_matrix_square,
    validate_arrays_compatible,
)


def test_calculate_input_valid():
    """Test valid CalculateInput."""
    input_data = CalculateInput(expression="2 + 2", variables={"x": 5})
    assert input_data.expression == "2 + 2"
    assert input_data.variables == {"x": 5}


def test_calculate_input_no_variables():
    """Test CalculateInput without variables."""
    input_data = CalculateInput(expression="sin(pi/2)")
    assert input_data.expression == "sin(pi/2)"
    assert input_data.variables is None


def test_calculate_input_whitespace_stripping():
    """Test that whitespace is stripped from expression."""
    input_data = CalculateInput(expression="  2 + 2  ")
    assert input_data.expression == "2 + 2"


def test_calculate_input_empty_expression():
    """Test error when expression is empty."""
    with pytest.raises(ValidationError) as exc_info:
        CalculateInput(expression="")
    assert "expression" in str(exc_info.value).lower()


def test_calculate_input_too_long():
    """Test error when expression exceeds maximum length."""
    # Max length is 2000 characters
    long_expression = "x + " * 1000  # > 2000 chars
    with pytest.raises(ValidationError) as exc_info:
        CalculateInput(expression=long_expression)
    assert "expression" in str(exc_info.value).lower()


def test_array_input_valid():
    """Test valid ArrayInput."""
    input_data = ArrayInput(data=[[1.0, 2.0], [3.0, 4.0]])
    assert len(input_data.data) == 2
    assert len(input_data.data[0]) == 2


def test_array_input_empty_array():
    """Test error when array is empty."""
    with pytest.raises(ValidationError) as exc_info:
        ArrayInput(data=[])
    assert "cannot be empty" in str(exc_info.value).lower()


def test_array_input_empty_rows():
    """Test error when array has empty rows."""
    with pytest.raises(ValidationError) as exc_info:
        ArrayInput(data=[[]])
    assert "cannot be empty" in str(exc_info.value).lower()


def test_array_input_irregular_rows():
    """Test error when array has irregular row lengths."""
    with pytest.raises(ValidationError) as exc_info:
        ArrayInput(data=[[1.0, 2.0], [3.0, 4.0, 5.0]])
    assert "same length" in str(exc_info.value).lower()


def test_validate_matrix_square_valid():
    """Test validation of square matrix."""
    matrix = [[1.0, 2.0], [3.0, 4.0]]
    # Should not raise any error
    validate_matrix_square(matrix)


def test_validate_matrix_square_non_square():
    """Test error when matrix is not square."""
    matrix = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    with pytest.raises(ValueError) as exc_info:
        validate_matrix_square(matrix)
    assert "square" in str(exc_info.value).lower()


def test_validate_arrays_compatible_valid():
    """Test validation of compatible arrays."""
    arr1 = [[1.0, 2.0], [3.0, 4.0]]
    arr2 = [[5.0, 6.0], [7.0, 8.0]]
    # Should not raise any error
    validate_arrays_compatible(arr1, arr2)


def test_validate_arrays_compatible_mismatch():
    """Test error when arrays have incompatible shapes."""
    arr1 = [[1.0, 2.0], [3.0, 4.0]]
    arr2 = [[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]
    with pytest.raises(ValueError) as exc_info:
        validate_arrays_compatible(arr1, arr2)
    assert "same shape" in str(exc_info.value).lower()


def test_validate_arrays_compatible_different_rows():
    """Test error when arrays have different number of rows."""
    arr1 = [[1.0, 2.0], [3.0, 4.0]]
    arr2 = [[5.0, 6.0]]
    with pytest.raises(ValueError) as exc_info:
        validate_arrays_compatible(arr1, arr2)
    assert "same shape" in str(exc_info.value).lower()
