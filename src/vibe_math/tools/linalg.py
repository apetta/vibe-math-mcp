"""Linear algebra tools using NumPy and SciPy."""

from typing import Annotated, List, Literal, Union, cast
from pydantic import Field
from mcp.types import ToolAnnotations
import json
import numpy as np
from numpy.typing import NDArray
import scipy.linalg as la

from ..server import mcp
from ..core import format_result, format_array_result, list_to_numpy, numpy_to_list


@mcp.tool(
    name="matrix_operations",
    description="""Core matrix operations using NumPy BLAS.

Examples:

MATRIX MULTIPLICATION:
    operation="multiply", matrix1=[[1,2],[3,4]], matrix2=[[5,6],[7,8]]
    Result: [[19,22],[43,50]]

MATRIX INVERSE:
    operation="inverse", matrix1=[[1,2],[3,4]]
    Result: [[-2,1],[1.5,-0.5]]

TRANSPOSE:
    operation="transpose", matrix1=[[1,2],[3,4]]
    Result: [[1,3],[2,4]]

DETERMINANT:
    operation="determinant", matrix1=[[1,2],[3,4]]
    Result: -2.0

TRACE:
    operation="trace", matrix1=[[1,2],[3,4]]
    Result: 5.0 (1+4)""",
    annotations=ToolAnnotations(
        title="Matrix Operations",
        readOnlyHint=True,
        idempotentHint=True,
    ),
)
async def matrix_operations(
    operation: Annotated[Literal["multiply", "inverse", "transpose", "determinant", "trace"], Field(description="Matrix operation")],
    matrix1: Annotated[List[List[float]], Field(description="First matrix (e.g., [[1,2],[3,4]])")],
    matrix2: Annotated[Union[str, List[List[float]], None], Field(description="Second matrix for multiplication")] = None,
) -> str:
    """Core matrix operations."""
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
    name="solve_linear_system",
    description="""Solve systems of linear equations (Ax = b) using SciPy's optimised solver.

Examples:

SQUARE SYSTEM (2 equations, 2 unknowns):
    coefficients=[[2,3],[1,1]], constants=[8,3], method="direct"
    Solves: 2x+3y=8, x+y=3
    Result: [x=1, y=2]

OVERDETERMINED SYSTEM (3 equations, 2 unknowns):
    coefficients=[[1,2],[3,4],[5,6]], constants=[5,6,7], method="least_squares"
    Finds best-fit x minimizing ||Ax-b||
    Result: [x≈-6, y≈5.5]

3x3 SYSTEM:
    coefficients=[[2,1,-1],[1,3,2],[-1,2,1]], constants=[8,13,5], method="direct"
    Result: [x=3, y=2, z=1]""",
    annotations=ToolAnnotations(
        title="Linear System Solver",
        readOnlyHint=True,
        idempotentHint=True,
    ),
)
async def solve_linear_system(
    coefficients: Annotated[List[List[float]], Field(description="Coefficient matrix A in Ax=b system (2D list, e.g., [[2,3],[1,1]])")],
    constants: Annotated[List[float], Field(description="Constants vector b in Ax=b system (1D list, e.g., [8,3])")],
    method: Annotated[Literal["direct", "least_squares"], Field(description="Solution method: direct=exact (square systems), least_squares=overdetermined systems")] = "direct",
) -> str:
    """Solve linear systems Ax=b using SciPy. Direct method for square systems, least squares for overdetermined. More stable than matrix inversion."""
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
            x, residuals, rank, _ = la.lstsq(A, b)  # type: ignore[misc]
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
    name="matrix_decomposition",
    description="""Matrix decompositions: eigenvalues/vectors, SVD, QR, Cholesky, LU.

Examples:

EIGENVALUE DECOMPOSITION:
    matrix=[[4,2],[1,3]], decomposition="eigen"
    Result: {eigenvalues: [5, 2], eigenvectors: [[0.89,0.45],[0.71,-0.71]]}

SINGULAR VALUE DECOMPOSITION (SVD):
    matrix=[[1,2],[3,4],[5,6]], decomposition="svd"
    Result: {U: 3×3, singular_values: [9.5, 0.77], Vt: 2×2}

QR FACTORISATION:
    matrix=[[1,2],[3,4]], decomposition="qr"
    Result: {Q: orthogonal, R: upper triangular}

CHOLESKY (symmetric positive definite):
    matrix=[[4,2],[2,3]], decomposition="cholesky"
    Result: {L: [[2,0],[1,1.41]]} where A=LL^T

LU DECOMPOSITION:
    matrix=[[2,1],[4,3]], decomposition="lu"
    Result: {P: permutation, L: lower, U: upper} where A=PLU""",
    annotations=ToolAnnotations(
        title="Matrix Decomposition",
        readOnlyHint=True,
        idempotentHint=True,
    ),
)
async def matrix_decomposition(
    matrix: Annotated[List[List[float]], Field(description="Matrix to decompose as 2D nested list (e.g., [[4,2],[1,3]])")],
    decomposition: Annotated[Literal["eigen", "svd", "qr", "cholesky", "lu"], Field(description="Decomposition type: eigen=eigenvalues/vectors, svd=singular value, qr=QR, cholesky=symmetric positive definite, lu=LU factorisation")],
) -> str:
    """Matrix decompositions using SciPy: eigen (λ,v), SVD (UΣV^T), QR (orthogonal×triangular), Cholesky (LL^T), LU (PLU). For analysis, solving, and numerical stability."""
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
