"""Tests for statistical analysis tools."""

import json
import pytest


@pytest.mark.asyncio
async def test_statistics_describe(mcp_client, sample_data_list):
    """Test descriptive statistics."""
    result = await mcp_client.call_tool(
        "math_statistics",
        {"data": sample_data_list, "analyses": ["describe"]}
    )
    data = json.loads(result.content[0].text)
    assert data["describe"]["count"] == 10
    assert data["describe"]["mean"] == 5.5
    assert data["describe"]["min"] == 1.0
    assert data["describe"]["max"] == 10.0


@pytest.mark.asyncio
async def test_statistics_quartiles(mcp_client, sample_data_list):
    """Test quartile calculation."""
    result = await mcp_client.call_tool(
        "math_statistics",
        {"data": sample_data_list, "analyses": ["quartiles"]}
    )
    data = json.loads(result.content[0].text)
    assert "Q1" in data["quartiles"]
    assert "Q2" in data["quartiles"]
    assert "Q3" in data["quartiles"]
    assert "IQR" in data["quartiles"]


@pytest.mark.asyncio
async def test_statistics_outliers(mcp_client):
    """Test outlier detection."""
    data_with_outliers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]
    result = await mcp_client.call_tool(
        "math_statistics",
        {"data": data_with_outliers, "analyses": ["outliers"]}
    )
    data = json.loads(result.content[0].text)
    assert len(data["outliers"]["outlier_values"]) > 0


@pytest.mark.asyncio
async def test_pivot_table(mcp_client):
    """Test pivot table creation."""
    data = [
        {"region": "North", "product": "A", "sales": 100},
        {"region": "North", "product": "B", "sales": 150},
        {"region": "South", "product": "A", "sales": 200},
        {"region": "South", "product": "B", "sales": 250},
    ]
    result = await mcp_client.call_tool(
        "math_pivot_table",
        {"data": data, "index": "region", "columns": "product", "values": "sales", "aggfunc": "sum"}
    )
    result_data = json.loads(result.content[0].text)
    assert "result" in result_data


@pytest.mark.asyncio
async def test_correlation_pearson(mcp_client):
    """Test Pearson correlation."""
    data = {
        "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        "y": [2.0, 4.0, 6.0, 8.0, 10.0],
        "z": [1.0, 1.0, 1.0, 1.0, 1.0]
    }
    result = await mcp_client.call_tool(
        "math_correlation",
        {"data": data, "method": "pearson", "output_format": "matrix"}
    )
    result_data = json.loads(result.content[0].text)
    # x and y should be perfectly correlated
    assert result_data["result"]["x"]["y"] == 1.0 or abs(result_data["result"]["x"]["y"] - 1.0) < 1e-10
