"""Tests for calculus tools."""

import json
import pytest


@pytest.mark.asyncio
async def test_derivative_basic(mcp_client):
    """Test basic derivative."""
    result = await mcp_client.call_tool(
        "derivative", {"expression": "x^2", "variable": "x", "order": 1}
    )
    data = json.loads(result.content[0].text)
    assert "result" in data
    assert "2*x" in data["result"] or "2x" in data["result"]


@pytest.mark.asyncio
async def test_derivative_second_order(mcp_client):
    """Test second order derivative."""
    result = await mcp_client.call_tool(
        "derivative", {"expression": "x^3", "variable": "x", "order": 2}
    )
    data = json.loads(result.content[0].text)
    assert "result" in data
    assert "6*x" in data["result"] or "6x" in data["result"]


@pytest.mark.asyncio
async def test_derivative_at_point(mcp_client):
    """Test derivative evaluation at a point."""
    result = await mcp_client.call_tool(
        "derivative", {"expression": "x^2", "variable": "x", "order": 1, "point": 3}
    )
    data = json.loads(result.content[0].text)
    # d/dx(x^2) = 2x, at x=3 → 6
    assert data["value_at_point"] == 6.0


@pytest.mark.asyncio
async def test_integral_indefinite(mcp_client):
    """Test indefinite integral."""
    result = await mcp_client.call_tool(
        "integral", {"expression": "x^2", "variable": "x", "method": "symbolic"}
    )
    data = json.loads(result.content[0].text)
    assert "result" in data
    assert "x**3/3" in data["result"] or "x^3/3" in data["result"]


@pytest.mark.asyncio
async def test_integral_definite(mcp_client):
    """Test definite integral."""
    result = await mcp_client.call_tool(
        "integral",
        {
            "expression": "x^2",
            "variable": "x",
            "lower_bound": 0,
            "upper_bound": 1,
            "method": "symbolic",
        },
    )
    data = json.loads(result.content[0].text)
    assert "result" in data
    # ∫₀¹ x² dx = 1/3
    assert abs(data["result"] - (1 / 3)) < 1e-10


@pytest.mark.asyncio
async def test_integral_numerical(mcp_client):
    """Test numerical integration."""
    result = await mcp_client.call_tool(
        "integral",
        {
            "expression": "sin(x)",
            "variable": "x",
            "lower_bound": 0,
            "upper_bound": 3.14159,
            "method": "numerical",
        },
    )
    data = json.loads(result.content[0].text)
    assert "result" in data
    # ∫₀^π sin(x) dx ≈ 2
    assert abs(data["result"] - 2.0) < 0.01


@pytest.mark.asyncio
async def test_limit_basic(mcp_client):
    """Test basic limit."""
    result = await mcp_client.call_tool(
        "limits_series",
        {"expression": "sin(x)/x", "variable": "x", "point": 0, "operation": "limit"},
    )
    data = json.loads(result.content[0].text)
    # lim(x→0) sin(x)/x = 1
    assert data["numeric_value"] == 1.0


@pytest.mark.asyncio
async def test_limit_infinity(mcp_client):
    """Test limit at infinity."""
    result = await mcp_client.call_tool(
        "limits_series",
        {"expression": "1/x", "variable": "x", "point": "oo", "operation": "limit"},
    )
    data = json.loads(result.content[0].text)
    # lim(x→∞) 1/x = 0
    assert data["numeric_value"] == 0.0


@pytest.mark.asyncio
async def test_series_expansion(mcp_client):
    """Test series expansion."""
    result = await mcp_client.call_tool(
        "limits_series",
        {"expression": "exp(x)", "variable": "x", "point": 0, "operation": "series", "order": 4},
    )
    data = json.loads(result.content[0].text)
    # Taylor series of e^x around 0 - result contains the series string
    assert "result" in data
    # Series should contain basic expansion terms
    assert "x" in data["result"]
