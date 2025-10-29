"""Core utilities for the Math MCP server."""

from .validators import (
    CalculateInput,
    ArrayInput,
    validate_matrix_square,
    validate_arrays_compatible,
)
from .formatters import format_json, format_result, format_array_result, format_error
from .converters import (
    list_to_polars,
    polars_to_list,
    polars_to_pandas,
    list_to_numpy,
    numpy_to_list,
)
from .batch_models import BatchOperation, OperationResult, BatchSummary, BatchResponse
from .batch_executor import BatchExecutor
from .result_resolver import ResultResolver

__all__ = [
    # Validators
    "CalculateInput",
    "ArrayInput",
    "validate_matrix_square",
    "validate_arrays_compatible",
    # Formatters
    "format_json",
    "format_result",
    "format_array_result",
    "format_error",
    # Converters
    "list_to_polars",
    "polars_to_list",
    "polars_to_pandas",
    "list_to_numpy",
    "numpy_to_list",
    # Batch execution
    "BatchOperation",
    "OperationResult",
    "BatchSummary",
    "BatchResponse",
    "BatchExecutor",
    "ResultResolver",
]
