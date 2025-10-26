"""Tests for linear algebra tools."""

import json
import pytest


@pytest.mark.asyncio
async def test_matrix_multiply(mcp_client, sample_array_2x2):
    """Test matrix multiplication."""
    result = await mcp_client.call_tool(
        "matrix_operations",
        {"operation": "multiply", "matrix1": sample_array_2x2, "matrix2": sample_array_2x2},
    )
    data = json.loads(result.content[0].text)
    # [[1,2],[3,4]] * [[1,2],[3,4]] = [[7,10],[15,22]]
    assert data["values"] == [[7.0, 10.0], [15.0, 22.0]]


@pytest.mark.asyncio
async def test_matrix_transpose(mcp_client, sample_array_2x2):
    """Test matrix transpose."""
    result = await mcp_client.call_tool(
        "matrix_operations", {"operation": "transpose", "matrix1": sample_array_2x2}
    )
    data = json.loads(result.content[0].text)
    assert data["values"] == [[1.0, 3.0], [2.0, 4.0]]


@pytest.mark.asyncio
async def test_matrix_determinant(mcp_client, sample_array_2x2):
    """Test matrix determinant."""
    result = await mcp_client.call_tool(
        "matrix_operations", {"operation": "determinant", "matrix1": sample_array_2x2}
    )
    data = json.loads(result.content[0].text)
    # det([[1,2],[3,4]]) = 1*4 - 2*3 = -2
    assert abs(data["result"] - (-2.0)) < 1e-10


@pytest.mark.asyncio
async def test_matrix_trace(mcp_client, sample_array_2x2):
    """Test matrix trace."""
    result = await mcp_client.call_tool(
        "matrix_operations", {"operation": "trace", "matrix1": sample_array_2x2}
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
        "solve_linear_system",
        {"coefficients": coefficients, "constants": constants, "method": "direct"},
    )
    data = json.loads(result.content[0].text)
    # Solution: x=1, y=2
    assert abs(data["result"][0] - 1.0) < 1e-10
    assert abs(data["result"][1] - 2.0) < 1e-10


@pytest.mark.asyncio
async def test_matrix_decomposition_svd(mcp_client, sample_array_2x2):
    """Test SVD decomposition."""
    result = await mcp_client.call_tool(
        "matrix_decomposition", {"matrix": sample_array_2x2, "decomposition": "svd"}
    )
    data = json.loads(result.content[0].text)
    assert "U" in data
    assert "singular_values" in data
    assert "Vt" in data


@pytest.mark.asyncio
async def test_matrix_decomposition_qr(mcp_client, sample_array_2x2):
    """Test QR decomposition."""
    result = await mcp_client.call_tool(
        "matrix_decomposition", {"matrix": sample_array_2x2, "decomposition": "qr"}
    )
    data = json.loads(result.content[0].text)
    assert "Q" in data
    assert "R" in data


@pytest.mark.asyncio
async def test_matrix_inverse_2x2(mcp_client):
    """Test 2x2 matrix inversion."""
    matrix = [[4.0, 7.0], [2.0, 6.0]]
    result = await mcp_client.call_tool(
        "matrix_operations", {"operation": "inverse", "matrix1": matrix}
    )
    data = json.loads(result.content[0].text)
    # Verify inverse exists and has correct shape
    assert len(data["values"]) == 2
    assert len(data["values"][0]) == 2
    # For [[4,7],[2,6]], det = 24-14 = 10, inverse = [[0.6,-0.7],[-0.2,0.4]]
    assert abs(data["values"][0][0] - 0.6) < 1e-10


@pytest.mark.asyncio
async def test_matrix_inverse_3x3(mcp_client):
    """Test 3x3 matrix inversion."""
    matrix = [[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]]
    result = await mcp_client.call_tool(
        "matrix_operations", {"operation": "inverse", "matrix1": matrix}
    )
    data = json.loads(result.content[0].text)
    # Verify inverse has correct shape
    assert len(data["values"]) == 3
    assert len(data["values"][0]) == 3


@pytest.mark.asyncio
async def test_matrix_inverse_singular(mcp_client):
    """Test error when trying to invert singular matrix."""
    # Singular matrix (rows are linearly dependent)
    matrix = [[1.0, 2.0], [2.0, 4.0]]
    with pytest.raises(Exception) as exc_info:
        await mcp_client.call_tool(
            "matrix_operations", {"operation": "inverse", "matrix1": matrix}
        )
    assert "singular" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_matrix_inverse_non_square(mcp_client):
    """Test error when trying to invert non-square matrix."""
    matrix = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    with pytest.raises(Exception) as exc_info:
        await mcp_client.call_tool(
            "matrix_operations", {"operation": "inverse", "matrix1": matrix}
        )
    assert "square" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_matrix_multiply_incompatible_shapes(mcp_client):
    """Test error when multiplying incompatible matrices."""
    matrix1 = [[1.0, 2.0], [3.0, 4.0]]  # 2x2
    matrix2 = [[1.0, 2.0, 3.0]]  # 1x3
    with pytest.raises(Exception) as exc_info:
        await mcp_client.call_tool(
            "matrix_operations",
            {"operation": "multiply", "matrix1": matrix1, "matrix2": matrix2},
        )
    assert "Incompatible" in str(exc_info.value) or "shape" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_matrix_multiply_missing_matrix2(mcp_client, sample_array_2x2):
    """Test error when matrix2 is missing for multiplication."""
    with pytest.raises(Exception) as exc_info:
        await mcp_client.call_tool(
            "matrix_operations", {"operation": "multiply", "matrix1": sample_array_2x2}
        )
    assert "requires matrix2" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_matrix_determinant_non_square(mcp_client):
    """Test error when computing determinant of non-square matrix."""
    matrix = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    with pytest.raises(Exception) as exc_info:
        await mcp_client.call_tool(
            "matrix_operations", {"operation": "determinant", "matrix1": matrix}
        )
    assert "square" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_matrix_trace_non_square(mcp_client):
    """Test error when computing trace of non-square matrix."""
    matrix = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    with pytest.raises(Exception) as exc_info:
        await mcp_client.call_tool(
            "matrix_operations", {"operation": "trace", "matrix1": matrix}
        )
    assert "square" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_solve_linear_system_dimension_mismatch(mcp_client):
    """Test error when coefficient matrix rows != constants vector length."""
    coefficients = [[1.0, 2.0], [3.0, 4.0]]
    constants = [5.0, 6.0, 7.0]  # Too many constants
    with pytest.raises(Exception) as exc_info:
        await mcp_client.call_tool(
            "solve_linear_system",
            {"coefficients": coefficients, "constants": constants, "method": "direct"},
        )
    assert "dimension" in str(exc_info.value).lower() or "incompatible" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_solve_linear_system_non_square_direct(mcp_client):
    """Test error when using direct method on non-square system."""
    coefficients = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    constants = [7.0, 8.0, 9.0]
    with pytest.raises(Exception) as exc_info:
        await mcp_client.call_tool(
            "solve_linear_system",
            {"coefficients": coefficients, "constants": constants, "method": "direct"},
        )
    assert "square" in str(exc_info.value).lower() or "least_squares" in str(exc_info.value)


@pytest.mark.asyncio
async def test_solve_linear_system_singular(mcp_client):
    """Test error when system is singular."""
    # Singular system (rows are linearly dependent)
    coefficients = [[1.0, 2.0], [2.0, 4.0]]
    constants = [3.0, 6.0]
    with pytest.raises(Exception) as exc_info:
        await mcp_client.call_tool(
            "solve_linear_system",
            {"coefficients": coefficients, "constants": constants, "method": "direct"},
        )
    assert "singular" in str(exc_info.value).lower() or "poorly conditioned" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_matrix_decomposition_eigen_2x2(mcp_client):
    """Test eigenvalue decomposition of 2x2 matrix."""
    matrix = [[4.0, 2.0], [1.0, 3.0]]
    result = await mcp_client.call_tool(
        "matrix_decomposition", {"matrix": matrix, "decomposition": "eigen"}
    )
    data = json.loads(result.content[0].text)
    assert "eigenvalues" in data
    assert "eigenvectors" in data
    assert len(data["eigenvalues"]) == 2


@pytest.mark.asyncio
async def test_matrix_decomposition_eigen_3x3(mcp_client):
    """Test eigenvalue decomposition of 3x3 matrix."""
    matrix = [[1.0, 2.0, 3.0], [0.0, 4.0, 5.0], [0.0, 0.0, 6.0]]
    result = await mcp_client.call_tool(
        "matrix_decomposition", {"matrix": matrix, "decomposition": "eigen"}
    )
    data = json.loads(result.content[0].text)
    assert len(data["eigenvalues"]) == 3
    assert len(data["eigenvectors"]) == 3


@pytest.mark.asyncio
async def test_matrix_decomposition_eigen_non_square(mcp_client):
    """Test error when computing eigenvalues of non-square matrix."""
    matrix = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    with pytest.raises(Exception) as exc_info:
        await mcp_client.call_tool(
            "matrix_decomposition", {"matrix": matrix, "decomposition": "eigen"}
        )
    assert "square" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_matrix_decomposition_cholesky_positive_definite(mcp_client):
    """Test Cholesky decomposition of positive definite matrix."""
    # Symmetric positive definite matrix
    matrix = [[4.0, 2.0], [2.0, 3.0]]
    result = await mcp_client.call_tool(
        "matrix_decomposition", {"matrix": matrix, "decomposition": "cholesky"}
    )
    data = json.loads(result.content[0].text)
    assert "L" in data
    assert len(data["L"]) == 2


@pytest.mark.asyncio
async def test_matrix_decomposition_cholesky_non_symmetric(mcp_client):
    """Test error when Cholesky decomposition is applied to non-symmetric matrix."""
    matrix = [[4.0, 1.0], [2.0, 3.0]]
    with pytest.raises(Exception) as exc_info:
        await mcp_client.call_tool(
            "matrix_decomposition", {"matrix": matrix, "decomposition": "cholesky"}
        )
    assert "symmetric" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_matrix_decomposition_cholesky_not_positive_definite(mcp_client):
    """Test error when matrix is symmetric but not positive definite."""
    # Symmetric but not positive definite (has negative eigenvalue)
    matrix = [[1.0, 2.0], [2.0, 1.0]]
    with pytest.raises(Exception) as exc_info:
        await mcp_client.call_tool(
            "matrix_decomposition", {"matrix": matrix, "decomposition": "cholesky"}
        )
    assert "positive definite" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_matrix_decomposition_lu_2x2(mcp_client):
    """Test LU decomposition of 2x2 matrix."""
    matrix = [[3.0, 1.0], [6.0, 4.0]]
    result = await mcp_client.call_tool(
        "matrix_decomposition", {"matrix": matrix, "decomposition": "lu"}
    )
    data = json.loads(result.content[0].text)
    assert "P" in data
    assert "L" in data
    assert "U" in data
    assert len(data["L"]) == 2


@pytest.mark.asyncio
async def test_matrix_decomposition_lu_3x3(mcp_client):
    """Test LU decomposition of 3x3 matrix."""
    matrix = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]]
    result = await mcp_client.call_tool(
        "matrix_decomposition", {"matrix": matrix, "decomposition": "lu"}
    )
    data = json.loads(result.content[0].text)
    assert len(data["P"]) == 3
    assert len(data["L"]) == 3
    assert len(data["U"]) == 3
