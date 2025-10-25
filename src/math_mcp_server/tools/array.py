"""Array calculation tools using Polars for optimal performance."""

from typing import List, Literal, Optional, Union
from mcp.types import ToolAnnotations
import numpy as np

from ..server import mcp
from ..core import format_result, format_array_result, list_to_polars, polars_to_list, list_to_numpy


@mcp.tool(
    name="math_array_operations",
    description="Perform element-wise operations on arrays (add, subtract, multiply, divide, power).",
    annotations=ToolAnnotations(
        title="Array Operations",
        readOnlyHint=True,
        idempotentHint=True,
    ),
)
async def array_operations(
    operation: Literal["add", "subtract", "multiply", "divide", "power"],
    array1: List[List[float]],
    array2: Union[List[List[float]], float],
) -> str:
    """
    Perform element-wise operations using Polars for speed.

    Examples:
        - operation="multiply", array1=[[1,2],[3,4]], array2=2 → [[2,4],[6,8]]
        - operation="add", array1=[[1,2]], array2=[[3,4]] → [[4,6]]

    Args:
        operation: Operation type (add, subtract, multiply, divide, power)
        array1: First array (2D list)
        array2: Second array (2D list) or scalar value

    Returns:
        JSON with result array
    """
    try:
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
    name="math_array_statistics",
    description="Calculate statistical measures on arrays (mean, median, std, min, max, sum) with optional axis selection.",
    annotations=ToolAnnotations(
        title="Array Statistics",
        readOnlyHint=True,
        idempotentHint=True,
    ),
)
async def array_statistics(
    data: List[List[float]],
    operations: List[Literal["mean", "median", "std", "min", "max", "sum"]],
    axis: Optional[int] = None,
) -> str:
    """
    Compute statistics using Polars for optimal performance.

    Examples:
        - data=[[1,2,3],[4,5,6]], operations=["mean"], axis=0 → column means
        - data=[[1,2,3],[4,5,6]], operations=["mean","std"], axis=None → overall stats

    Args:
        data: 2D array as nested list
        operations: Statistics to compute
        axis: 0 for column-wise, 1 for row-wise, None for overall

    Returns:
        JSON with computed statistics
    """
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
    name="math_array_aggregate",
    description="Perform aggregation operations: sumproduct, weighted average, or dot product.",
    annotations=ToolAnnotations(
        title="Array Aggregation",
        readOnlyHint=True,
        idempotentHint=True,
    ),
)
async def array_aggregate(
    operation: Literal["sumproduct", "weighted_average", "dot_product"],
    array1: List[float],
    array2: Optional[List[float]] = None,
    weights: Optional[List[float]] = None,
) -> str:
    """
    Perform advanced aggregation operations.

    Examples:
        - operation="sumproduct", array1=[1,2,3], array2=[4,5,6] → 32 (1*4 + 2*5 + 3*6)
        - operation="weighted_average", array1=[10,20,30], weights=[1,2,3] → 23.33...
        - operation="dot_product", array1=[1,2], array2=[3,4] → 11

    Args:
        operation: Aggregation type
        array1: First array
        array2: Second array (for sumproduct/dot_product)
        weights: Weights (for weighted_average)

    Returns:
        JSON with aggregated result
    """
    try:
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
    name="math_array_transform",
    description="Transform arrays: normalise, standardise, min-max scale, or log transform.",
    annotations=ToolAnnotations(
        title="Array Transformation",
        readOnlyHint=True,
        idempotentHint=True,
    ),
)
async def array_transform(
    data: List[List[float]],
    transform: Literal["normalize", "standardize", "minmax_scale", "log_transform"],
    axis: Optional[int] = None,
) -> str:
    """
    Apply transformations to arrays.

    Examples:
        - data=[[1,2],[3,4]], transform="normalize" → L2 normalized values
        - data=[[1,2],[3,4]], transform="standardize" → z-score standardized
        - data=[[1,2],[3,4]], transform="minmax_scale" → scaled to [0,1]

    Args:
        data: 2D array as nested list
        transform: Transformation type
        axis: 0 for column-wise, None for overall

    Returns:
        JSON with transformed array
    """
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
