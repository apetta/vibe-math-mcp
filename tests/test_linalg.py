"""Tests for linear algebra tools."""

import json
import pytest
import numpy as np


@pytest.mark.asyncio
async def test_matrix_multiply(mcp_client, sample_array_2x2):
    """Test matrix multiplication."""
    result = await mcp_client.call_tool(
        "math_matrix_operations",
        {"operation": "multiply", "matrix1": sample_array_2x2, "matrix2": sample_array_2x2}
    )
    data = json.loads(result.content[0].text)
    # [[1,2],[3,4]] * [[1,2],[3,4]] = [[7,10],[15,22]]
    assert data["values"] == [[7.0, 10.0], [15.0, 22.0]]


@pytest.mark.asyncio
async def test_matrix_transpose(mcp_client, sample_array_2x2):
    """Test matrix transpose."""
    result = await mcp_client.call_tool(
        "math_matrix_operations",
        {"operation": "transpose", "matrix1": sample_array_2x2}
    )
    data = json.loads(result.content[0].text)
    assert data["values"] == [[1.0, 3.0], [2.0, 4.0]]


@pytest.mark.asyncio
async def test_matrix_determinant(mcp_client, sample_array_2x2):
    """Test matrix determinant."""
    result = await mcp_client.call_tool(
        "math_matrix_operations",
        {"operation": "determinant", "matrix1": sample_array_2x2}
    )
    data = json.loads(result.content[0].text)
    # det([[1,2],[3,4]]) = 1*4 - 2*3 = -2
    assert abs(data["result"] - (-2.0)) < 1e-10


@pytest.mark.asyncio
async def test_matrix_trace(mcp_client, sample_array_2x2):
    """Test matrix trace."""
    result = await mcp_client.call_tool(
        "math_matrix_operations",
        {"operation": "trace", "matrix1": sample_array_2x2}
    )
    data = json.loads(result.content[0].text)
    # trace([[1,2],[3,4]]) = 1 + 4 = 5
    assert data["result"] == 5.0


@pytest.mark.asyncio
async def test_solve_linear_system(mcp_client):
    """Test solving linear system."""
    # 2x + 3y = 8
    # x + y = 3
    coefficients = [[2, 3], [1, 1]]
    constants = [8, 3]
    result = await mcp_client.call_tool(
        "math_solve_linear_system",
        {"coefficients": coefficients, "constants": constants, "method": "direct"}
    )
    data = json.loads(result.content[0].text)
    # Solution: x=1, y=2
    assert abs(data["result"][0] - 1.0) < 1e-10
    assert abs(data["result"][1] - 2.0) < 1e-10


@pytest.mark.asyncio
async def test_matrix_decomposition_svd(mcp_client, sample_array_2x2):
    """Test SVD decomposition."""
    result = await mcp_client.call_tool(
        "math_matrix_decomposition",
        {"matrix": sample_array_2x2, "decomposition": "svd"}
    )
    data = json.loads(result.content[0].text)
    assert "U" in data
    assert "singular_values" in data
    assert "Vt" in data


@pytest.mark.asyncio
async def test_matrix_decomposition_qr(mcp_client, sample_array_2x2):
    """Test QR decomposition."""
    result = await mcp_client.call_tool(
        "math_matrix_decomposition",
        {"matrix": sample_array_2x2, "decomposition": "qr"}
    )
    data = json.loads(result.content[0].text)
    assert "Q" in data
    assert "R" in data
