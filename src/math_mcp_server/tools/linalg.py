"""Linear algebra tools using NumPy and SciPy."""

from typing import List, Literal, Union, cast
from mcp.types import ToolAnnotations
import json
import numpy as np
from numpy.typing import NDArray
import scipy.linalg as la

from ..server import mcp
from ..core import format_result, format_array_result, list_to_numpy, numpy_to_list


@mcp.tool(
    name="math_matrix_operations",
    description="Core matrix operations: multiply, inverse, transpose, determinant, trace.",
    annotations=ToolAnnotations(
        title="Matrix Operations",
        readOnlyHint=True,
        idempotentHint=True,
    ),
)
async def matrix_operations(
    operation: Literal["multiply", "inverse", "transpose", "determinant", "trace"],
    matrix1: List[List[float]],
    matrix2: Union[str, List[List[float]], None] = None,
) -> str:
    """
    Perform matrix operations using NumPy's BLAS-optimised routines.

    Examples:
        - operation="multiply", matrix1=[[1,2],[3,4]], matrix2=[[5,6],[7,8]] → matrix product
        - operation="inverse", matrix1=[[1,2],[3,4]] → inverse matrix
        - operation="transpose", matrix1=[[1,2],[3,4]] → transposed matrix
        - operation="determinant", matrix1=[[1,2],[3,4]] → -2.0
        - operation="trace", matrix1=[[1,2],[3,4]] → 5.0

    Args:
        operation: Matrix operation type
        matrix1: First matrix (2D list)
        matrix2: Second matrix (required for multiply)

    Returns:
        JSON with result (matrix or scalar)
    """
    try:
        # Parse stringified JSON from XML serialization
        if isinstance(matrix2, str):
            matrix2 = cast(List[List[float]], json.loads(matrix2))

        mat1 = list_to_numpy(matrix1)

        if operation == "multiply":
            if matrix2 is None:
                raise ValueError("Matrix multiplication requires matrix2")
            mat2 = list_to_numpy(matrix2)
            if mat1.shape[1] != mat2.shape[0]:
                raise ValueError(
                    f"Incompatible shapes for multiplication: {mat1.shape} and {mat2.shape}. "
                    f"First matrix columns must equal second matrix rows."
                )
            result = np.dot(mat1, mat2)
            return format_array_result(numpy_to_list(result), {"operation": operation})

        elif operation == "inverse":
            if mat1.shape[0] != mat1.shape[1]:
                raise ValueError(f"Matrix must be square for inversion. Got shape: {mat1.shape}")
            try:
                result = la.inv(mat1)
                return format_array_result(numpy_to_list(result), {"operation": operation})
            except np.linalg.LinAlgError:
                raise ValueError("Matrix is singular and cannot be inverted")

        elif operation == "transpose":
            result = mat1.T
            return format_array_result(numpy_to_list(result), {"operation": operation})

        elif operation == "determinant":
            if mat1.shape[0] != mat1.shape[1]:
                raise ValueError(f"Matrix must be square for determinant. Got shape: {mat1.shape}")
            result = float(la.det(mat1))
            return format_result(
                result, {"operation": operation, "shape": f"{mat1.shape[0]}×{mat1.shape[1]}"}
            )

        elif operation == "trace":
            if mat1.shape[0] != mat1.shape[1]:
                raise ValueError(f"Matrix must be square for trace. Got shape: {mat1.shape}")
            result = float(np.trace(mat1))
            return format_result(
                result, {"operation": operation, "shape": f"{mat1.shape[0]}×{mat1.shape[1]}"}
            )

        else:
            raise ValueError(f"Unknown operation: {operation}")

    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Matrix operation failed: {str(e)}")


@mcp.tool(
    name="math_solve_linear_system",
    description="Solve systems of linear equations (Ax = b) using SciPy's optimised solver.",
    annotations=ToolAnnotations(
        title="Linear System Solver",
        readOnlyHint=True,
        idempotentHint=True,
    ),
)
async def solve_linear_system(
    coefficients: List[List[float]],
    constants: List[float],
    method: Literal["direct", "least_squares"] = "direct",
) -> str:
    """
    Solve linear systems using SciPy (preferred over matrix inversion).

    Examples:
        - coefficients=[[1,2],[3,4]], constants=[5,6] → solution vector x
        - coefficients=[[1,2],[3,4],[5,6]], constants=[7,8,9], method="least_squares" → overdetermined system

    Args:
        coefficients: Coefficient matrix A (m×n)
        constants: Constants vector b (m×1)
        method: Solution method (direct for square systems, least_squares for overdetermined)

    Returns:
        JSON with solution vector
    """
    try:
        A = list_to_numpy(coefficients)
        b = np.array(constants, dtype=float)

        if A.shape[0] != len(b):
            raise ValueError(
                f"Incompatible dimensions: coefficient matrix has {A.shape[0]} rows "
                f"but constants vector has {len(b)} elements"
            )

        if method == "direct":
            if A.shape[0] != A.shape[1]:
                raise ValueError(
                    f"Direct method requires square matrix. Got {A.shape}. "
                    f"Use method='least_squares' for overdetermined systems."
                )
            try:
                x = la.solve(A, b)
            except np.linalg.LinAlgError:
                raise ValueError("System is singular or poorly conditioned")

        elif method == "least_squares":
            x, residuals, rank, _s = la.lstsq(A, b)  # type: ignore[misc]
            metadata = {
                "method": method,
                "rank": int(rank),
                "residuals": residuals.tolist() if len(residuals) > 0 else None,
            }
            return format_result(x.tolist(), metadata)

        else:
            raise ValueError(f"Unknown method: {method}")

        return format_result(x.tolist(), {"method": method})

    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Linear system solution failed: {str(e)}")


@mcp.tool(
    name="math_matrix_decomposition",
    description="Matrix decompositions: eigenvalues/vectors, SVD, QR, Cholesky, LU.",
    annotations=ToolAnnotations(
        title="Matrix Decomposition",
        readOnlyHint=True,
        idempotentHint=True,
    ),
)
async def matrix_decomposition(
    matrix: List[List[float]], decomposition: Literal["eigen", "svd", "qr", "cholesky", "lu"]
) -> str:
    """
    Perform matrix decompositions using SciPy.

    Examples:
        - decomposition="eigen", matrix=[[4,2],[1,3]] → eigenvalues and eigenvectors
        - decomposition="svd", matrix=[[1,2],[3,4]] → U, Σ, V^T
        - decomposition="qr", matrix=[[1,2],[3,4]] → Q, R matrices
        - decomposition="cholesky", matrix=[[4,2],[2,3]] → L (lower triangular)
        - decomposition="lu", matrix=[[1,2],[3,4]] → P, L, U matrices

    Args:
        matrix: Input matrix (2D list)
        decomposition: Type of decomposition

    Returns:
        JSON with decomposition results
    """
    try:
        mat = list_to_numpy(matrix)

        if decomposition == "eigen":
            if mat.shape[0] != mat.shape[1]:
                raise ValueError(
                    f"Eigenvalue decomposition requires square matrix. Got shape: {mat.shape}"
                )

            eigenvalues: NDArray[np.complexfloating]
            eigenvectors: NDArray[np.complexfloating]
            eigenvalues, eigenvectors = la.eig(mat)  # type: ignore[misc]

            return format_json(
                {
                    "eigenvalues": eigenvalues.tolist(),
                    "eigenvectors": eigenvectors.tolist(),
                    "decomposition": decomposition,
                }
            )

        elif decomposition == "svd":
            U, s, Vt = la.svd(mat)

            return format_json(
                {
                    "U": U.tolist(),
                    "singular_values": s.tolist(),
                    "Vt": Vt.tolist(),
                    "decomposition": decomposition,
                }
            )

        elif decomposition == "qr":
            Q: NDArray[np.floating]
            R: NDArray[np.floating]
            Q, R = la.qr(mat)  # type: ignore[misc]

            return format_json({"Q": Q.tolist(), "R": R.tolist(), "decomposition": decomposition})

        elif decomposition == "cholesky":
            if mat.shape[0] != mat.shape[1]:
                raise ValueError(
                    f"Cholesky decomposition requires square matrix. Got shape: {mat.shape}"
                )

            # Check if matrix is symmetric
            if not np.allclose(mat, mat.T):
                raise ValueError("Cholesky decomposition requires symmetric matrix")

            try:
                L = la.cholesky(mat, lower=True)
                return format_json(
                    {"L": L.tolist(), "decomposition": decomposition, "note": "A = L * L^T"}
                )
            except np.linalg.LinAlgError:
                raise ValueError("Matrix is not positive definite")

        elif decomposition == "lu":
            P, L, U = la.lu(mat)  # type: ignore[misc]

            return format_json(
                {
                    "P": P.tolist(),
                    "L": L.tolist(),
                    "U": U.tolist(),
                    "decomposition": decomposition,
                    "note": "A = P * L * U",
                }
            )

        else:
            raise ValueError(f"Unknown decomposition: {decomposition}")

    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Matrix decomposition failed: {str(e)}")


# Import format_json for decomposition results
from ..core import format_json  # noqa: E402
