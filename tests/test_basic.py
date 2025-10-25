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
        "math_calculate", {"expression": "x^2 + 2*x + 1", "variables": {"x": 3}}
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
        "math_percentage", {"operation": "of", "value": 200, "percentage": 15}
    )
    data = json.loads(result.content[0].text)
    assert data["result"] == 30.0


@pytest.mark.asyncio
async def test_percentage_increase(mcp_client):
    """Test percentage increase operation."""
    result = await mcp_client.call_tool(
        "math_percentage", {"operation": "increase", "value": 100, "percentage": 20}
    )
    data = json.loads(result.content[0].text)
    assert data["result"] == 120.0


@pytest.mark.asyncio
async def test_percentage_decrease(mcp_client):
    """Test percentage decrease operation."""
    result = await mcp_client.call_tool(
        "math_percentage", {"operation": "decrease", "value": 100, "percentage": 20}
    )
    data = json.loads(result.content[0].text)
    assert data["result"] == 80.0


@pytest.mark.asyncio
async def test_round_basic(mcp_client):
    """Test basic rounding."""
    result = await mcp_client.call_tool(
        "math_round", {"values": 3.14159, "method": "round", "decimals": 2}
    )
    data = json.loads(result.content[0].text)
    assert data["result"] == 3.14


@pytest.mark.asyncio
async def test_round_list(mcp_client):
    """Test rounding a list of values."""
    result = await mcp_client.call_tool(
        "math_round", {"values": [3.14159, 2.71828], "method": "round", "decimals": 2}
    )
    data = json.loads(result.content[0].text)
    assert data["result"] == [3.14, 2.72]


@pytest.mark.asyncio
async def test_convert_degrees_to_radians(mcp_client):
    """Test degrees to radians conversion."""
    result = await mcp_client.call_tool(
        "math_convert_units", {"value": 180, "from_unit": "degrees", "to_unit": "radians"}
    )
    data = json.loads(result.content[0].text)
    assert abs(data["result"] - 3.14159265) < 1e-6


@pytest.mark.asyncio
async def test_convert_radians_to_degrees(mcp_client):
    """Test radians to degrees conversion."""
    result = await mcp_client.call_tool(
        "math_convert_units", {"value": 3.14159265, "from_unit": "radians", "to_unit": "degrees"}
    )
    data = json.loads(result.content[0].text)
    assert abs(data["result"] - 180.0) < 1e-6


@pytest.mark.asyncio
async def test_calculate_invalid_expression(mcp_client):
    """Test error when expression is invalid."""
    with pytest.raises(Exception):
        await mcp_client.call_tool("math_calculate", {"expression": "2 +* 3"})


@pytest.mark.asyncio
async def test_calculate_undefined_variable(mcp_client):
    """Test error when variable is undefined."""
    with pytest.raises(Exception):
        await mcp_client.call_tool("math_calculate", {"expression": "x + y", "variables": {"x": 5}})


@pytest.mark.asyncio
async def test_calculate_complex_expression(mcp_client):
    """Test complex mathematical expression."""
    result = await mcp_client.call_tool(
        "math_calculate", {"expression": "sqrt(16) + log(exp(1)) + cos(0)"}
    )
    data = json.loads(result.content[0].text)
    # sqrt(16)=4, log(e)=1, cos(0)=1, total=6
    assert abs(data["result"] - 6.0) < 1e-10


@pytest.mark.asyncio
async def test_calculate_division_operation(mcp_client):
    """Test division in expression."""
    result = await mcp_client.call_tool("math_calculate", {"expression": "10 / 2"})
    data = json.loads(result.content[0].text)
    assert data["result"] == 5.0


@pytest.mark.asyncio
async def test_percentage_change(mcp_client):
    """Test percentage change operation."""
    result = await mcp_client.call_tool(
        "math_percentage", {"operation": "change", "value": 100, "percentage": 50}
    )
    data = json.loads(result.content[0].text)
    # Change from 100 to 50 is -50%
    assert data["result"] == -50.0


@pytest.mark.asyncio
async def test_percentage_negative_values(mcp_client):
    """Test percentage operations with negative values."""
    result = await mcp_client.call_tool(
        "math_percentage", {"operation": "of", "value": -200, "percentage": 25}
    )
    data = json.loads(result.content[0].text)
    # 25% of -200 = -50
    assert data["result"] == -50.0


@pytest.mark.asyncio
async def test_percentage_zero_value(mcp_client):
    """Test percentage of zero."""
    result = await mcp_client.call_tool(
        "math_percentage", {"operation": "of", "value": 0, "percentage": 50}
    )
    data = json.loads(result.content[0].text)
    assert data["result"] == 0.0


@pytest.mark.asyncio
async def test_round_floor(mcp_client):
    """Test floor rounding."""
    result = await mcp_client.call_tool(
        "math_round", {"values": 3.7, "method": "floor", "decimals": 0}
    )
    data = json.loads(result.content[0].text)
    assert data["result"] == 3.0


@pytest.mark.asyncio
async def test_round_ceil(mcp_client):
    """Test ceiling rounding."""
    result = await mcp_client.call_tool(
        "math_round", {"values": 3.1, "method": "ceil", "decimals": 0}
    )
    data = json.loads(result.content[0].text)
    assert data["result"] == 4.0


@pytest.mark.asyncio
async def test_round_trunc(mcp_client):
    """Test truncation rounding."""
    result = await mcp_client.call_tool(
        "math_round", {"values": -3.7, "method": "trunc", "decimals": 0}
    )
    data = json.loads(result.content[0].text)
    # Truncate towards zero: -3.7 -> -3
    assert data["result"] == -3.0


@pytest.mark.asyncio
async def test_round_with_decimals(mcp_client):
    """Test rounding with decimal places."""
    result = await mcp_client.call_tool(
        "math_round", {"values": [3.14159, 2.71828, 1.41421], "method": "round", "decimals": 3}
    )
    data = json.loads(result.content[0].text)
    assert data["result"] == [3.142, 2.718, 1.414]


@pytest.mark.asyncio
async def test_convert_units_zero(mcp_client):
    """Test unit conversion with zero value."""
    result = await mcp_client.call_tool(
        "math_convert_units", {"value": 0, "from_unit": "degrees", "to_unit": "radians"}
    )
    data = json.loads(result.content[0].text)
    assert data["result"] == 0.0
