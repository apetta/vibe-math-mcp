"""Tests for basic calculation tools."""

import json
import pytest


@pytest.mark.asyncio
async def test_calculate_simple(mcp_client):
    """Test basic arithmetic calculation."""
    result = await mcp_client.call_tool("math_calculate", {"expression": "2 + 2"})
    data = json.loads(result.content[0].text)
    assert data["result"] == 4.0


@pytest.mark.asyncio
async def test_calculate_with_variables(mcp_client):
    """Test calculation with variable substitution."""
    result = await mcp_client.call_tool(
        "math_calculate",
        {"expression": "x^2 + 2*x + 1", "variables": {"x": 3}}
    )
    data = json.loads(result.content[0].text)
    assert data["result"] == 16.0


@pytest.mark.asyncio
async def test_calculate_trigonometric(mcp_client):
    """Test trigonometric functions."""
    result = await mcp_client.call_tool("math_calculate", {"expression": "sin(pi/2)"})
    data = json.loads(result.content[0].text)
    assert abs(data["result"] - 1.0) < 1e-10


@pytest.mark.asyncio
async def test_percentage_of(mcp_client):
    """Test percentage 'of' operation."""
    result = await mcp_client.call_tool(
        "math_percentage",
        {"operation": "of", "value": 200, "percentage": 15}
    )
    data = json.loads(result.content[0].text)
    assert data["result"] == 30.0


@pytest.mark.asyncio
async def test_percentage_increase(mcp_client):
    """Test percentage increase operation."""
    result = await mcp_client.call_tool(
        "math_percentage",
        {"operation": "increase", "value": 100, "percentage": 20}
    )
    data = json.loads(result.content[0].text)
    assert data["result"] == 120.0


@pytest.mark.asyncio
async def test_percentage_decrease(mcp_client):
    """Test percentage decrease operation."""
    result = await mcp_client.call_tool(
        "math_percentage",
        {"operation": "decrease", "value": 100, "percentage": 20}
    )
    data = json.loads(result.content[0].text)
    assert data["result"] == 80.0


@pytest.mark.asyncio
async def test_round_basic(mcp_client):
    """Test basic rounding."""
    result = await mcp_client.call_tool(
        "math_round",
        {"values": 3.14159, "method": "round", "decimals": 2}
    )
    data = json.loads(result.content[0].text)
    assert data["result"] == 3.14


@pytest.mark.asyncio
async def test_round_list(mcp_client):
    """Test rounding a list of values."""
    result = await mcp_client.call_tool(
        "math_round",
        {"values": [3.14159, 2.71828], "method": "round", "decimals": 2}
    )
    data = json.loads(result.content[0].text)
    assert data["result"] == [3.14, 2.72]


@pytest.mark.asyncio
async def test_convert_degrees_to_radians(mcp_client):
    """Test degrees to radians conversion."""
    result = await mcp_client.call_tool(
        "math_convert_units",
        {"value": 180, "from_unit": "degrees", "to_unit": "radians"}
    )
    data = json.loads(result.content[0].text)
    assert abs(data["result"] - 3.14159265) < 1e-6


@pytest.mark.asyncio
async def test_convert_radians_to_degrees(mcp_client):
    """Test radians to degrees conversion."""
    result = await mcp_client.call_tool(
        "math_convert_units",
        {"value": 3.14159265, "from_unit": "radians", "to_unit": "degrees"}
    )
    data = json.loads(result.content[0].text)
    assert abs(data["result"] - 180.0) < 1e-6
