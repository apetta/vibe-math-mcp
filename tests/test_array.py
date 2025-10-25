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


@pytest.mark.asyncio
async def test_array_operations_subtract_scalar(mcp_client, sample_array_2x2):
    """Test scalar subtraction."""
    result = await mcp_client.call_tool(
        "math_array_operations", {"operation": "subtract", "array1": sample_array_2x2, "array2": 1}
    )
    data = json.loads(result.content[0].text)
    assert data["values"] == [[0.0, 1.0], [2.0, 3.0]]


@pytest.mark.asyncio
async def test_array_operations_subtract_arrays(mcp_client, sample_array_2x2):
    """Test array subtraction."""
    array2 = [[1.0, 1.0], [1.0, 1.0]]
    result = await mcp_client.call_tool(
        "math_array_operations", {"operation": "subtract", "array1": sample_array_2x2, "array2": array2}
    )
    data = json.loads(result.content[0].text)
    assert data["values"] == [[0.0, 1.0], [2.0, 3.0]]


@pytest.mark.asyncio
async def test_array_operations_divide_scalar(mcp_client, sample_array_2x2):
    """Test scalar division."""
    result = await mcp_client.call_tool(
        "math_array_operations", {"operation": "divide", "array1": sample_array_2x2, "array2": 2}
    )
    data = json.loads(result.content[0].text)
    assert data["values"] == [[0.5, 1.0], [1.5, 2.0]]


@pytest.mark.asyncio
async def test_array_operations_divide_by_zero(mcp_client, sample_array_2x2):
    """Test division by zero error."""
    with pytest.raises(Exception) as exc_info:
        await mcp_client.call_tool(
            "math_array_operations", {"operation": "divide", "array1": sample_array_2x2, "array2": 0}
        )
    assert "Division by zero" in str(exc_info.value)


@pytest.mark.asyncio
async def test_array_operations_divide_arrays(mcp_client):
    """Test array division."""
    array1 = [[10.0, 20.0], [30.0, 40.0]]
    array2 = [[2.0, 4.0], [5.0, 8.0]]
    result = await mcp_client.call_tool(
        "math_array_operations", {"operation": "divide", "array1": array1, "array2": array2}
    )
    data = json.loads(result.content[0].text)
    assert data["values"] == [[5.0, 5.0], [6.0, 5.0]]


@pytest.mark.asyncio
async def test_array_operations_power_scalar(mcp_client, sample_array_2x2):
    """Test power operation with scalar exponent."""
    result = await mcp_client.call_tool(
        "math_array_operations", {"operation": "power", "array1": sample_array_2x2, "array2": 2}
    )
    data = json.loads(result.content[0].text)
    assert data["values"] == [[1.0, 4.0], [9.0, 16.0]]


@pytest.mark.asyncio
async def test_array_operations_power_arrays(mcp_client):
    """Test element-wise power operation between arrays."""
    array1 = [[2.0, 3.0], [4.0, 5.0]]
    array2 = [[2.0, 2.0], [2.0, 2.0]]
    result = await mcp_client.call_tool(
        "math_array_operations", {"operation": "power", "array1": array1, "array2": array2}
    )
    data = json.loads(result.content[0].text)
    assert data["values"] == [[4.0, 9.0], [16.0, 25.0]]


@pytest.mark.asyncio
async def test_array_statistics_axis_0_mean(mcp_client):
    """Test column-wise (axis=0) mean."""
    data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    result = await mcp_client.call_tool(
        "math_array_statistics", {"data": data, "operations": ["mean"], "axis": 0}
    )
    result_data = json.loads(result.content[0].text)
    # Column means: [2.5, 3.5, 4.5]
    expected = [2.5, 3.5, 4.5]
    assert len(result_data["result"]["mean"]) == 3
    for i, val in enumerate(expected):
        assert abs(result_data["result"]["mean"][i] - val) < 1e-10


@pytest.mark.asyncio
async def test_array_statistics_axis_0_all_operations(mcp_client):
    """Test all statistics with axis=0 (column-wise)."""
    data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    result = await mcp_client.call_tool(
        "math_array_statistics",
        {"data": data, "operations": ["mean", "median", "std", "min", "max", "sum"], "axis": 0},
    )
    result_data = json.loads(result.content[0].text)
    # Verify all operations returned lists for column-wise results
    assert isinstance(result_data["result"]["mean"], list)
    assert isinstance(result_data["result"]["median"], list)
    assert isinstance(result_data["result"]["std"], list)
    assert len(result_data["result"]["min"]) == 3
    assert result_data["result"]["min"] == [1.0, 2.0, 3.0]
    assert result_data["result"]["max"] == [7.0, 8.0, 9.0]


@pytest.mark.asyncio
async def test_array_statistics_axis_1_mean(mcp_client):
    """Test row-wise (axis=1) mean."""
    data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    result = await mcp_client.call_tool(
        "math_array_statistics", {"data": data, "operations": ["mean"], "axis": 1}
    )
    result_data = json.loads(result.content[0].text)
    # Row means: [2.0, 5.0]
    expected = [2.0, 5.0]
    assert len(result_data["result"]["mean"]) == 2
    for i, val in enumerate(expected):
        assert abs(result_data["result"]["mean"][i] - val) < 1e-10


@pytest.mark.asyncio
async def test_array_statistics_axis_1_all_operations(mcp_client):
    """Test all statistics with axis=1 (row-wise)."""
    data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    result = await mcp_client.call_tool(
        "math_array_statistics",
        {"data": data, "operations": ["mean", "median", "std", "min", "max", "sum"], "axis": 1},
    )
    result_data = json.loads(result.content[0].text)
    # Verify all operations returned lists for row-wise results
    assert isinstance(result_data["result"]["mean"], list)
    assert len(result_data["result"]["mean"]) == 3
    assert result_data["result"]["min"] == [1.0, 4.0, 7.0]
    assert result_data["result"]["max"] == [3.0, 6.0, 9.0]
    assert result_data["result"]["sum"] == [6.0, 15.0, 24.0]


@pytest.mark.asyncio
async def test_array_statistics_median(mcp_client):
    """Test median calculation."""
    data = [[1.0, 2.0, 3.0], [4.0, 5.0, 100.0]]
    result = await mcp_client.call_tool(
        "math_array_statistics", {"data": data, "operations": ["median"], "axis": None}
    )
    result_data = json.loads(result.content[0].text)
    # Median of [1, 2, 3, 4, 5, 100] = 3.5
    assert result_data["result"]["median"] == 3.5


@pytest.mark.asyncio
async def test_array_statistics_std(mcp_client):
    """Test standard deviation calculation."""
    data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    result = await mcp_client.call_tool(
        "math_array_statistics", {"data": data, "operations": ["std"], "axis": None}
    )
    result_data = json.loads(result.content[0].text)
    # Verify std is positive and reasonable
    assert result_data["result"]["std"] > 0
    assert result_data["result"]["std"] < 10


@pytest.mark.asyncio
async def test_array_statistics_sum(mcp_client):
    """Test sum calculation."""
    data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    result = await mcp_client.call_tool(
        "math_array_statistics", {"data": data, "operations": ["sum"], "axis": None}
    )
    result_data = json.loads(result.content[0].text)
    # Sum of 1+2+3+4+5+6 = 21
    assert result_data["result"]["sum"] == 21.0


@pytest.mark.asyncio
async def test_array_aggregate_dot_product(mcp_client):
    """Test dot product operation."""
    result = await mcp_client.call_tool(
        "math_array_aggregate", {"operation": "dot_product", "array1": [1, 2, 3], "array2": [4, 5, 6]}
    )
    data = json.loads(result.content[0].text)
    # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    assert data["result"] == 32.0


@pytest.mark.asyncio
async def test_array_aggregate_missing_array2(mcp_client):
    """Test error when array2 is missing for sumproduct."""
    with pytest.raises(Exception) as exc_info:
        await mcp_client.call_tool(
            "math_array_aggregate", {"operation": "sumproduct", "array1": [1, 2, 3]}
        )
    assert "requires array2" in str(exc_info.value)


@pytest.mark.asyncio
async def test_array_aggregate_missing_weights(mcp_client):
    """Test error when weights are missing for weighted_average."""
    with pytest.raises(Exception) as exc_info:
        await mcp_client.call_tool(
            "math_array_aggregate", {"operation": "weighted_average", "array1": [1, 2, 3]}
        )
    assert "requires weights" in str(exc_info.value)


@pytest.mark.asyncio
async def test_array_aggregate_length_mismatch(mcp_client):
    """Test error when arrays have different lengths."""
    with pytest.raises(Exception) as exc_info:
        await mcp_client.call_tool(
            "math_array_aggregate",
            {"operation": "sumproduct", "array1": [1, 2, 3], "array2": [4, 5]},
        )
    assert "same length" in str(exc_info.value)


@pytest.mark.asyncio
async def test_array_aggregate_weights_length_mismatch(mcp_client):
    """Test error when weights length doesn't match array length."""
    with pytest.raises(Exception) as exc_info:
        await mcp_client.call_tool(
            "math_array_aggregate",
            {"operation": "weighted_average", "array1": [1, 2, 3], "weights": [1, 2]},
        )
    assert "same length" in str(exc_info.value)


@pytest.mark.asyncio
async def test_array_transform_minmax_scale_none(mcp_client):
    """Test min-max scaling with axis=None."""
    data = [[1.0, 2.0], [3.0, 4.0]]
    result = await mcp_client.call_tool(
        "math_array_transform", {"data": data, "transform": "minmax_scale", "axis": None}
    )
    result_data = json.loads(result.content[0].text)
    # Min=1, Max=4, range=3
    # Scaled values should be in [0, 1]
    flat_values = [val for row in result_data["values"] for val in row]
    assert min(flat_values) == 0.0
    assert max(flat_values) == 1.0


@pytest.mark.asyncio
async def test_array_transform_minmax_scale_axis_0(mcp_client):
    """Test min-max scaling column-wise (axis=0)."""
    data = [[1.0, 10.0], [5.0, 20.0]]
    result = await mcp_client.call_tool(
        "math_array_transform", {"data": data, "transform": "minmax_scale", "axis": 0}
    )
    result_data = json.loads(result.content[0].text)
    # Column 1: min=1, max=5, Column 2: min=10, max=20
    # First column: [0, 1], Second column: [0, 1]
    assert len(result_data["values"]) == 2


@pytest.mark.asyncio
async def test_array_transform_minmax_scale_axis_1(mcp_client):
    """Test min-max scaling row-wise (axis=1)."""
    data = [[1.0, 5.0], [10.0, 20.0]]
    result = await mcp_client.call_tool(
        "math_array_transform", {"data": data, "transform": "minmax_scale", "axis": 1}
    )
    result_data = json.loads(result.content[0].text)
    # Each row should be scaled independently
    assert len(result_data["values"]) == 2
    assert len(result_data["values"][0]) == 2


@pytest.mark.asyncio
async def test_array_transform_log_transform(mcp_client):
    """Test logarithmic transformation."""
    data = [[1.0, 2.0], [3.0, 4.0]]
    result = await mcp_client.call_tool(
        "math_array_transform", {"data": data, "transform": "log_transform", "axis": None}
    )
    result_data = json.loads(result.content[0].text)
    # Result should contain positive values (log1p of positive numbers)
    assert len(result_data["values"]) == 2
    assert len(result_data["values"][0]) == 2
    # All values should be positive (log1p of positive inputs)
    for row in result_data["values"]:
        for val in row:
            assert val > 0


@pytest.mark.asyncio
async def test_array_transform_normalize_axis_0(mcp_client):
    """Test L2 normalization column-wise (axis=0)."""
    data = [[3.0, 4.0], [4.0, 3.0]]
    result = await mcp_client.call_tool(
        "math_array_transform", {"data": data, "transform": "normalize", "axis": 0}
    )
    result_data = json.loads(result.content[0].text)
    # Each column should have unit norm
    assert len(result_data["values"]) == 2


@pytest.mark.asyncio
async def test_array_transform_standardize_axis_1(mcp_client):
    """Test standardization row-wise (axis=1)."""
    data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    result = await mcp_client.call_tool(
        "math_array_transform", {"data": data, "transform": "standardize", "axis": 1}
    )
    result_data = json.loads(result.content[0].text)
    # Each row should be standardized independently
    assert len(result_data["values"]) == 2


@pytest.mark.asyncio
async def test_array_transform_zero_range_edge_case(mcp_client):
    """Test minmax_scale with zero range (all values same)."""
    data = [[5.0, 5.0], [5.0, 5.0]]
    result = await mcp_client.call_tool(
        "math_array_transform", {"data": data, "transform": "minmax_scale", "axis": None}
    )
    result_data = json.loads(result.content[0].text)
    # When all values are the same, range is 0, should handle gracefully
    assert len(result_data["values"]) == 2
