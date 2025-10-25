"""Tests for statistical analysis tools."""

import json
import pytest


@pytest.mark.asyncio
async def test_statistics_describe(mcp_client, sample_data_list):
    """Test descriptive statistics."""
    result = await mcp_client.call_tool(
        "math_statistics", {"data": sample_data_list, "analyses": ["describe"]}
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
        "math_statistics", {"data": sample_data_list, "analyses": ["quartiles"]}
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
        "math_statistics", {"data": data_with_outliers, "analyses": ["outliers"]}
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
        {
            "data": data,
            "index": "region",
            "columns": "product",
            "values": "sales",
            "aggfunc": "sum",
        },
    )
    result_data = json.loads(result.content[0].text)
    assert "result" in result_data


@pytest.mark.asyncio
async def test_correlation_pearson(mcp_client):
    """Test Pearson correlation."""
    data = {
        "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        "y": [2.0, 4.0, 6.0, 8.0, 10.0],
        "z": [1.0, 1.0, 1.0, 1.0, 1.0],
    }
    result = await mcp_client.call_tool(
        "math_correlation", {"data": data, "method": "pearson", "output_format": "matrix"}
    )
    result_data = json.loads(result.content[0].text)
    # x and y should be perfectly correlated
    assert (
        result_data["result"]["x"]["y"] == 1.0 or abs(result_data["result"]["x"]["y"] - 1.0) < 1e-10
    )


@pytest.mark.asyncio
async def test_correlation_spearman(mcp_client):
    """Test Spearman rank correlation."""
    data = {
        "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        "y": [1.0, 4.0, 9.0, 16.0, 25.0],  # Non-linear but monotonic
    }
    result = await mcp_client.call_tool(
        "math_correlation", {"data": data, "method": "spearman", "output_format": "matrix"}
    )
    result_data = json.loads(result.content[0].text)
    # Spearman correlation should be perfect for monotonic relationship
    assert abs(result_data["result"]["x"]["y"] - 1.0) < 1e-10


@pytest.mark.asyncio
async def test_correlation_pairs_format(mcp_client):
    """Test correlation with pairs output format."""
    data = {
        "a": [1.0, 2.0, 3.0],
        "b": [2.0, 4.0, 6.0],
        "c": [3.0, 6.0, 9.0],
    }
    result = await mcp_client.call_tool(
        "math_correlation", {"data": data, "method": "pearson", "output_format": "pairs"}
    )
    result_data = json.loads(result.content[0].text)
    # Should return pairwise correlations
    assert isinstance(result_data["result"], list)
    # Should have 3 pairs: (a,b), (a,c), (b,c)
    assert len(result_data["result"]) == 3
    assert all("var1" in pair and "var2" in pair for pair in result_data["result"])


@pytest.mark.asyncio
async def test_correlation_unequal_lengths(mcp_client):
    """Test error when variables have unequal lengths."""
    data = {
        "x": [1.0, 2.0, 3.0],
        "y": [1.0, 2.0],  # Different length
    }
    with pytest.raises(Exception) as exc_info:
        await mcp_client.call_tool("math_correlation", {"data": data, "method": "pearson"})
    # Error should mention height/shape mismatch or same number of observations
    error_msg = str(exc_info.value).lower()
    assert ("same number" in error_msg or "height" in error_msg or "shape" in error_msg)


@pytest.mark.asyncio
async def test_pivot_table_mean_aggfunc(mcp_client):
    """Test pivot table with mean aggregation."""
    data = [
        {"region": "North", "product": "A", "sales": 100},
        {"region": "North", "product": "A", "sales": 150},
        {"region": "South", "product": "A", "sales": 200},
    ]
    result = await mcp_client.call_tool(
        "math_pivot_table",
        {
            "data": data,
            "index": "region",
            "columns": "product",
            "values": "sales",
            "aggfunc": "mean",
        },
    )
    result_data = json.loads(result.content[0].text)
    assert "result" in result_data


@pytest.mark.asyncio
async def test_pivot_table_count_aggfunc(mcp_client):
    """Test pivot table with count aggregation."""
    data = [
        {"region": "North", "product": "A", "sales": 100},
        {"region": "North", "product": "A", "sales": 150},
        {"region": "North", "product": "B", "sales": 200},
    ]
    result = await mcp_client.call_tool(
        "math_pivot_table",
        {
            "data": data,
            "index": "region",
            "columns": "product",
            "values": "sales",
            "aggfunc": "count",
        },
    )
    result_data = json.loads(result.content[0].text)
    assert "result" in result_data


@pytest.mark.asyncio
async def test_pivot_table_missing_column(mcp_client):
    """Test error when pivot table references missing column."""
    data = [
        {"region": "North", "product": "A", "sales": 100},
    ]
    with pytest.raises(Exception) as exc_info:
        await mcp_client.call_tool(
            "math_pivot_table",
            {
                "data": data,
                "index": "region",
                "columns": "category",  # Doesn't exist
                "values": "sales",
                "aggfunc": "sum",
            },
        )
    # Should mention the missing column in error
    assert "column" in str(exc_info.value).lower() or "category" in str(exc_info.value)


@pytest.mark.asyncio
async def test_statistics_combined_analyses(mcp_client):
    """Test multiple analyses at once."""
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0]
    result = await mcp_client.call_tool(
        "math_statistics", {"data": data, "analyses": ["describe", "quartiles", "outliers"]}
    )
    result_data = json.loads(result.content[0].text)
    # All three analyses should be present
    assert "describe" in result_data
    assert "quartiles" in result_data
    assert "outliers" in result_data
    # Outliers should detect the 100
    assert 100.0 in result_data["outliers"]["outlier_values"]
