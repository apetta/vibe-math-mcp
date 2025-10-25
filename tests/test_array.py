"""Tests for array calculation tools."""

import json
import pytest


@pytest.mark.asyncio
async def test_array_operations_multiply_scalar(mcp_client, sample_array_2x2):
    """Test scalar multiplication."""
    result = await mcp_client.call_tool(
        "math_array_operations", {"operation": "multiply", "array1": sample_array_2x2, "array2": 2}
    )
    data = json.loads(result.content[0].text)
    assert data["values"] == [[2.0, 4.0], [6.0, 8.0]]


@pytest.mark.asyncio
async def test_array_operations_add(mcp_client, sample_array_2x2):
    """Test array addition."""
    result = await mcp_client.call_tool(
        "math_array_operations",
        {"operation": "add", "array1": sample_array_2x2, "array2": sample_array_2x2},
    )
    data = json.loads(result.content[0].text)
    assert data["values"] == [[2.0, 4.0], [6.0, 8.0]]


@pytest.mark.asyncio
async def test_array_statistics_mean(mcp_client, sample_array_2x2):
    """Test array mean calculation."""
    result = await mcp_client.call_tool(
        "math_array_statistics", {"data": sample_array_2x2, "operations": ["mean"], "axis": None}
    )
    data = json.loads(result.content[0].text)
    assert data["result"]["mean"] == 2.5


@pytest.mark.asyncio
async def test_array_statistics_multiple(mcp_client, sample_array_2x2):
    """Test multiple statistics."""
    result = await mcp_client.call_tool(
        "math_array_statistics",
        {"data": sample_array_2x2, "operations": ["mean", "min", "max"], "axis": None},
    )
    data = json.loads(result.content[0].text)
    assert data["result"]["mean"] == 2.5
    assert data["result"]["min"] == 1.0
    assert data["result"]["max"] == 4.0


@pytest.mark.asyncio
async def test_array_aggregate_sumproduct(mcp_client):
    """Test sumproduct operation."""
    result = await mcp_client.call_tool(
        "math_array_aggregate",
        {"operation": "sumproduct", "array1": [1, 2, 3], "array2": [4, 5, 6]},
    )
    data = json.loads(result.content[0].text)
    assert data["result"] == 32.0  # 1*4 + 2*5 + 3*6


@pytest.mark.asyncio
async def test_array_aggregate_weighted_average(mcp_client):
    """Test weighted average."""
    result = await mcp_client.call_tool(
        "math_array_aggregate",
        {"operation": "weighted_average", "array1": [10, 20, 30], "weights": [1, 2, 3]},
    )
    data = json.loads(result.content[0].text)
    expected = (10 * 1 + 20 * 2 + 30 * 3) / (1 + 2 + 3)  # 23.333...
    assert abs(data["result"] - expected) < 1e-10


@pytest.mark.asyncio
async def test_array_transform_normalize(mcp_client, sample_array_2x2):
    """Test normalization."""
    result = await mcp_client.call_tool(
        "math_array_transform", {"data": sample_array_2x2, "transform": "normalize", "axis": None}
    )
    data = json.loads(result.content[0].text)
    # Result should be normalized (check that it's a valid array)
    assert len(data["values"]) == 2
    assert len(data["values"][0]) == 2


@pytest.mark.asyncio
async def test_array_transform_standardize(mcp_client, sample_array_2x2):
    """Test standardization (z-score)."""
    result = await mcp_client.call_tool(
        "math_array_transform", {"data": sample_array_2x2, "transform": "standardize", "axis": None}
    )
    data = json.loads(result.content[0].text)
    # Check structure
    assert len(data["values"]) == 2
    assert len(data["values"][0]) == 2
