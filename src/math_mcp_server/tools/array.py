"""Array calculation tools using Polars for optimal performance."""

from typing import Annotated, List, Literal, Union, cast
from pydantic import Field
from mcp.types import ToolAnnotations
import json
import numpy as np

from ..server import mcp
from ..core import format_result, format_array_result, list_to_polars, polars_to_list, list_to_numpy


@mcp.tool(
    name="array_operations",
    description="""Perform element-wise operations on arrays using Polars.

Supports array-array and array-scalar operations.

Examples:

SCALAR MULTIPLICATION:
    operation="multiply", array1=[[1,2],[3,4]], array2=2
    Result: [[2,4],[6,8]]

ARRAY ADDITION:
    operation="add", array1=[[1,2]], array2=[[3,4]]
    Result: [[4,6]]

POWER OPERATION:
    operation="power", array1=[[2,3]], array2=2
    Result: [[4,9]]

ARRAY DIVISION:
    operation="divide", array1=[[10,20],[30,40]], array2=[[2,4],[5,8]]
    Result: [[5,5],[6,5]]""",
    annotations=ToolAnnotations(
        title="Array Operations",
        readOnlyHint=True,
        idempotentHint=True,
    ),
)
async def array_operations(
    operation: Annotated[
        Literal["add", "subtract", "multiply", "divide", "power"],
        Field(description="Element-wise operation to perform"),
    ],
    array1: Annotated[List[List[float]], Field(description="First 2D array (e.g., [[1,2],[3,4]])")],
    array2: Annotated[
        Union[str, List[List[float]], float],
        Field(description="Second array, scalar, or JSON string"),
    ],
) -> str:
    """Element-wise array operations."""
    try:
        # Handle XML serialization: parse stringified JSON
        if isinstance(array2, str):
            try:
                array2 = cast(List[List[float]], json.loads(array2))
            except (json.JSONDecodeError, ValueError):
                array2 = cast(float, float(array2))

        df1 = list_to_polars(array1)

        is_scalar = isinstance(array2, (int, float))

        if operation == "add":
            result_df = df1 + array2 if is_scalar else df1 + list_to_polars(array2)
        elif operation == "subtract":
            result_df = df1 - array2 if is_scalar else df1 - list_to_polars(array2)
        elif operation == "multiply":
            result_df = df1 * array2 if is_scalar else df1 * list_to_polars(array2)
        elif operation == "divide":
            if is_scalar and array2 == 0:
                raise ValueError("Division by zero")
            result_df = df1 / array2 if is_scalar else df1 / list_to_polars(array2)
        elif operation == "power":
            # Use NumPy for reliable power operations (Polars doesn't support ** operator)
            arr1 = df1.to_numpy()
            if is_scalar:
                result_arr = arr1**array2
            else:
                arr2 = list_to_polars(array2).to_numpy()
                result_arr = arr1**arr2
            result_df = list_to_polars(result_arr.tolist())
        else:
            raise ValueError(f"Unknown operation: {operation}")

        result = polars_to_list(result_df)

        return format_array_result(
            result, {"operation": operation, "shape": f"{len(result)}×{len(result[0])}"}
        )
    except Exception as e:
        raise ValueError(f"Array operation failed: {str(e)}")


@mcp.tool(
    name="array_statistics",
    description="""Calculate statistical measures on arrays using Polars.

Supports computation across entire array, rows, or columns.

Examples:

COLUMN-WISE MEANS:
    data=[[1,2,3],[4,5,6]], operations=["mean"], axis=0
    Result: [2.5, 3.5, 4.5] (average of each column)

ROW-WISE MEANS:
    data=[[1,2,3],[4,5,6]], operations=["mean"], axis=1
    Result: [2.0, 5.0] (average of each row)

OVERALL STATISTICS:
    data=[[1,2,3],[4,5,6]], operations=["mean","std"], axis=None
    Result: {mean: 3.5, std: 1.71}

MULTIPLE STATISTICS:
    data=[[1,2,3],[4,5,6]], operations=["min","max","mean"], axis=0
    Result: {min: [1,2,3], max: [4,5,6], mean: [2.5,3.5,4.5]}""",
    annotations=ToolAnnotations(
        title="Array Statistics",
        readOnlyHint=True,
        idempotentHint=True,
    ),
)
async def array_statistics(
    data: Annotated[List[List[float]], Field(description="2D array (e.g., [[1,2,3],[4,5,6]])")],
    operations: Annotated[
        List[Literal["mean", "median", "std", "min", "max", "sum"]],
        Field(description="Statistics to compute (e.g., ['mean','std'])"),
    ],
    axis: Annotated[
        int | None, Field(description="Axis: 0=column-wise, 1=row-wise, None=overall")
    ] = None,
) -> str:
    """Calculate array statistics."""
    try:
        df = list_to_polars(data)

        results = {}

        for op in operations:
            if axis is None:
                # Overall statistics across all values
                all_values = df.to_numpy().flatten()
                if op == "mean":
                    results[op] = float(np.mean(all_values))
                elif op == "median":
                    results[op] = float(np.median(all_values))
                elif op == "std":
                    results[op] = float(np.std(all_values, ddof=1))
                elif op == "min":
                    results[op] = float(np.min(all_values))
                elif op == "max":
                    results[op] = float(np.max(all_values))
                elif op == "sum":
                    results[op] = float(np.sum(all_values))
            elif axis == 0:
                # Column-wise statistics
                if op == "mean":
                    results[op] = df.mean().to_numpy()[0].tolist()
                elif op == "median":
                    results[op] = df.median().to_numpy()[0].tolist()
                elif op == "std":
                    results[op] = df.std().to_numpy()[0].tolist()
                elif op == "min":
                    results[op] = df.min().to_numpy()[0].tolist()
                elif op == "max":
                    results[op] = df.max().to_numpy()[0].tolist()
                elif op == "sum":
                    results[op] = df.sum().to_numpy()[0].tolist()
            elif axis == 1:
                # Row-wise statistics
                arr = df.to_numpy()
                if op == "mean":
                    results[op] = np.mean(arr, axis=1).tolist()
                elif op == "median":
                    results[op] = np.median(arr, axis=1).tolist()
                elif op == "std":
                    results[op] = np.std(arr, axis=1, ddof=1).tolist()
                elif op == "min":
                    results[op] = np.min(arr, axis=1).tolist()
                elif op == "max":
                    results[op] = np.max(arr, axis=1).tolist()
                elif op == "sum":
                    results[op] = np.sum(arr, axis=1).tolist()

        return format_result(results, {"shape": f"{len(data)}×{len(data[0])}", "axis": axis})
    except Exception as e:
        raise ValueError(f"Statistics calculation failed: {str(e)}")


@mcp.tool(
    name="array_aggregate",
    description="""Perform aggregation operations on 1D arrays.

Examples:

SUMPRODUCT:
    operation="sumproduct", array1=[1,2,3], array2=[4,5,6]
    Result: 32 (1×4 + 2×5 + 3×6)

WEIGHTED AVERAGE:
    operation="weighted_average", array1=[10,20,30], weights=[1,2,3]
    Result: 23.33... ((10×1 + 20×2 + 30×3) / (1+2+3))

DOT PRODUCT:
    operation="dot_product", array1=[1,2], array2=[3,4]
    Result: 11 (1×3 + 2×4)

GRADE CALCULATION:
    operation="weighted_average", array1=[85,92,78], weights=[0.3,0.5,0.2]
    Result: 86.5""",
    annotations=ToolAnnotations(
        title="Array Aggregation",
        readOnlyHint=True,
        idempotentHint=True,
    ),
)
async def array_aggregate(
    operation: Annotated[
        Literal["sumproduct", "weighted_average", "dot_product"],
        Field(description="Aggregation operation"),
    ],
    array1: Annotated[List[float], Field(description="First 1D array (e.g., [1,2,3])")],
    array2: Annotated[
        Union[str, List[float], None],
        Field(description="Second 1D array for sumproduct/dot_product"),
    ] = None,
    weights: Annotated[
        Union[str, List[float], None],
        Field(description="Weights for weighted_average (e.g., [1,2,3])"),
    ] = None,
) -> str:
    """Aggregate 1D arrays."""
    try:
        # Parse stringified JSON from XML serialization
        if isinstance(array2, str):
            array2 = cast(List[float], json.loads(array2))
        if isinstance(weights, str):
            weights = cast(List[float], json.loads(weights))

        arr1 = np.array(array1, dtype=float)

        if operation == "sumproduct" or operation == "dot_product":
            if array2 is None:
                raise ValueError(f"{operation} requires array2")
            arr2 = np.array(array2, dtype=float)
            if len(arr1) != len(arr2):
                raise ValueError(f"Arrays must have same length. Got {len(arr1)} and {len(arr2)}")
            result = float(np.dot(arr1, arr2))

        elif operation == "weighted_average":
            if weights is None:
                raise ValueError("weighted_average requires weights")
            w = np.array(weights, dtype=float)
            if len(arr1) != len(w):
                raise ValueError(
                    f"Array and weights must have same length. Got {len(arr1)} and {len(w)}"
                )
            result = float(np.average(arr1, weights=w))

        else:
            raise ValueError(f"Unknown operation: {operation}")

        return format_result(result, {"operation": operation})
    except Exception as e:
        raise ValueError(f"Aggregation failed: {str(e)}")


@mcp.tool(
    name="array_transform",
    description="""Transform arrays for ML preprocessing and data normalization.

Transformations:
    - normalize: L2 normalization (unit vector)
    - standardize: Z-score (mean=0, std=1)
    - minmax_scale: Scale to [0,1] range
    - log_transform: Natural log transform

Examples:

L2 NORMALIZATION:
    data=[[3,4]], transform="normalize"
    Result: [[0.6,0.8]] (3²+4²=25, √25=5, 3/5=0.6, 4/5=0.8)

STANDARDIZATION (Z-SCORE):
    data=[[1,2],[3,4]], transform="standardize"
    Result: Values with mean=0, std=1

MIN-MAX SCALING:
    data=[[1,2],[3,4]], transform="minmax_scale"
    Result: [[0,0.33],[0.67,1]] (scaled to [0,1])

LOG TRANSFORM:
    data=[[1,10,100]], transform="log_transform"
    Result: [[0,2.3,4.6]] (natural log)""",
    annotations=ToolAnnotations(
        title="Array Transformation",
        readOnlyHint=True,
        idempotentHint=True,
    ),
)
async def array_transform(
    data: Annotated[
        List[List[float]], Field(description="2D array to transform (e.g., [[1,2],[3,4]])")
    ],
    transform: Annotated[
        Literal["normalize", "standardize", "minmax_scale", "log_transform"],
        Field(description="Transformation type"),
    ],
    axis: Annotated[
        int | None, Field(description="Axis: 0=column-wise, 1=row-wise, None=overall")
    ] = None,
) -> str:
    """Transform arrays."""
    try:
        arr = list_to_numpy(data)

        if transform == "normalize":
            # L2 normalization
            if axis is None:
                norm = np.linalg.norm(arr)
                result = (arr / norm if norm != 0 else arr).tolist()
            elif axis == 0:
                norms = np.linalg.norm(arr, axis=0, keepdims=True)
                result = (arr / np.where(norms != 0, norms, 1)).tolist()
            else:
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                result = (arr / np.where(norms != 0, norms, 1)).tolist()

        elif transform == "standardize":
            # Z-score standardization
            if axis is None:
                mean = np.mean(arr)
                std = np.std(arr, ddof=1)
                result = ((arr - mean) / std if std != 0 else arr - mean).tolist()
            elif axis == 0:
                mean = np.mean(arr, axis=0, keepdims=True)
                std = np.std(arr, axis=0, ddof=1, keepdims=True)
                result = ((arr - mean) / np.where(std != 0, std, 1)).tolist()
            else:
                mean = np.mean(arr, axis=1, keepdims=True)
                std = np.std(arr, axis=1, ddof=1, keepdims=True)
                result = ((arr - mean) / np.where(std != 0, std, 1)).tolist()

        elif transform == "minmax_scale":
            # Min-Max scaling to [0, 1]
            if axis is None:
                min_val = np.min(arr)
                max_val = np.max(arr)
                range_val = max_val - min_val
                result = ((arr - min_val) / range_val if range_val != 0 else arr - min_val).tolist()
            elif axis == 0:
                min_val = np.min(arr, axis=0, keepdims=True)
                max_val = np.max(arr, axis=0, keepdims=True)
                range_val = max_val - min_val
                result = ((arr - min_val) / np.where(range_val != 0, range_val, 1)).tolist()
            else:
                min_val = np.min(arr, axis=1, keepdims=True)
                max_val = np.max(arr, axis=1, keepdims=True)
                range_val = max_val - min_val
                result = ((arr - min_val) / np.where(range_val != 0, range_val, 1)).tolist()

        elif transform == "log_transform":
            # Natural log transform (handles negatives by using log1p)
            result = np.log1p(np.abs(arr) * np.sign(arr)).tolist()

        else:
            raise ValueError(f"Unknown transform: {transform}")

        return format_array_result(result, {"transform": transform, "axis": axis})
    except Exception as e:
        raise ValueError(f"Transformation failed: {str(e)}")
