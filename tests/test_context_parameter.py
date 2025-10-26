"""Tests for context parameter pass-through across all tool modules."""

import json
import pytest


@pytest.mark.asyncio
async def test_basic_context_included(mcp_client):
    """Test that context is included in response when provided - basic.py."""
    result = await mcp_client.call_tool(
        "calculate",
        {"expression": "2 + 2", "context": "test context for calculation"},
    )
    data = json.loads(result.content[0].text)
    assert "context" in data
    assert data["context"] == "test context for calculation"


@pytest.mark.asyncio
async def test_basic_context_excluded(mcp_client):
    """Test that context key is NOT in response when omitted - basic.py."""
    result = await mcp_client.call_tool("calculate", {"expression": "2 + 2"})
    data = json.loads(result.content[0].text)
    assert "context" not in data


@pytest.mark.asyncio
async def test_array_context_included(mcp_client):
    """Test that context is included in response when provided - array.py."""
    result = await mcp_client.call_tool(
        "array_operations",
        {
            "operation": "multiply",
            "array1": [[1, 2], [3, 4]],
            "array2": 2,
            "context": "scaling matrix by 2",
        },
    )
    data = json.loads(result.content[0].text)
    assert "context" in data
    assert data["context"] == "scaling matrix by 2"


@pytest.mark.asyncio
async def test_array_context_excluded(mcp_client):
    """Test that context key is NOT in response when omitted - array.py."""
    result = await mcp_client.call_tool(
        "array_operations",
        {"operation": "multiply", "array1": [[1, 2], [3, 4]], "array2": 2},
    )
    data = json.loads(result.content[0].text)
    assert "context" not in data


@pytest.mark.asyncio
async def test_statistics_context_included(mcp_client):
    """Test that context is included in response when provided - statistics.py."""
    result = await mcp_client.call_tool(
        "statistics",
        {
            "data": [1, 2, 3, 4, 5],
            "analyses": ["describe"],
            "context": "analyzing sample data",
        },
    )
    data = json.loads(result.content[0].text)
    assert "context" in data
    assert data["context"] == "analyzing sample data"


@pytest.mark.asyncio
async def test_statistics_context_excluded(mcp_client):
    """Test that context key is NOT in response when omitted - statistics.py."""
    result = await mcp_client.call_tool(
        "statistics", {"data": [1, 2, 3, 4, 5], "analyses": ["describe"]}
    )
    data = json.loads(result.content[0].text)
    assert "context" not in data


@pytest.mark.asyncio
async def test_financial_context_included(mcp_client):
    """Test that context is included in response when provided - financial.py."""
    result = await mcp_client.call_tool(
        "financial_calcs",
        {
            "calculation": "pv",
            "rate": 0.05,
            "periods": 10,
            "future_value": 1000,
            "context": "calculating bond present value",
        },
    )
    data = json.loads(result.content[0].text)
    assert "context" in data
    assert data["context"] == "calculating bond present value"


@pytest.mark.asyncio
async def test_financial_context_excluded(mcp_client):
    """Test that context key is NOT in response when omitted - financial.py."""
    result = await mcp_client.call_tool(
        "financial_calcs",
        {"calculation": "pv", "rate": 0.05, "periods": 10, "future_value": 1000},
    )
    data = json.loads(result.content[0].text)
    assert "context" not in data


@pytest.mark.asyncio
async def test_linalg_context_included(mcp_client):
    """Test that context is included in response when provided - linalg.py."""
    result = await mcp_client.call_tool(
        "matrix_operations",
        {
            "operation": "determinant",
            "matrix1": [[1, 2], [3, 4]],
            "context": "checking matrix invertibility",
        },
    )
    data = json.loads(result.content[0].text)
    assert "context" in data
    assert data["context"] == "checking matrix invertibility"


@pytest.mark.asyncio
async def test_linalg_context_excluded(mcp_client):
    """Test that context key is NOT in response when omitted - linalg.py."""
    result = await mcp_client.call_tool(
        "matrix_operations", {"operation": "determinant", "matrix1": [[1, 2], [3, 4]]}
    )
    data = json.loads(result.content[0].text)
    assert "context" not in data


@pytest.mark.asyncio
async def test_calculus_context_included(mcp_client):
    """Test that context is included in response when provided - calculus.py."""
    result = await mcp_client.call_tool(
        "derivative",
        {
            "expression": "x^2",
            "variable": "x",
            "order": 1,
            "context": "finding velocity from position",
        },
    )
    data = json.loads(result.content[0].text)
    assert "context" in data
    assert data["context"] == "finding velocity from position"


@pytest.mark.asyncio
async def test_calculus_context_excluded(mcp_client):
    """Test that context key is NOT in response when omitted - calculus.py."""
    result = await mcp_client.call_tool(
        "derivative", {"expression": "x^2", "variable": "x", "order": 1}
    )
    data = json.loads(result.content[0].text)
    assert "context" not in data
