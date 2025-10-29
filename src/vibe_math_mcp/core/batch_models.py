"""Pydantic models for batch operations with comprehensive validation."""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator
from uuid import uuid4


class BatchOperation(BaseModel):
    """Single operation within a batch request.

    Represents one tool call with its arguments, dependencies, and metadata.
    Operations are executed according to their dependencies and the selected execution mode.
    """

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique operation identifier (auto-generated UUID if not provided)",
        min_length=1,
        max_length=200,
    )

    tool: str = Field(
        description="Tool name (must match one of the 19 available mathematical tools)",
        min_length=1,
        max_length=100,
    )

    arguments: Dict[str, Any] = Field(
        description="Tool arguments as key-value pairs matching the tool's parameter signature"
    )

    context: Optional[str] = Field(
        default=None,
        description="Operation-specific context annotation (e.g., 'Bond A valuation')",
        max_length=1000,
    )

    label: Optional[str] = Field(
        default=None,
        description="Human-readable label for this operation (displayed in results)",
        max_length=200,
    )

    timeout_ms: Optional[int] = Field(
        default=None,
        description="Operation-specific timeout in milliseconds (100ms - 300s)",
        ge=100,
        le=300000,
    )

    @field_validator('tool')
    @classmethod
    def validate_tool_name(cls, v: str) -> str:
        """Validate tool exists in registry.

        This will be checked at runtime when the tool registry is available.
        Static validation happens in the batch_execute tool itself.
        """
        return v

    @field_validator('id')
    @classmethod
    def validate_id_format(cls, v: str) -> str:
        """Validate operation ID format (no special chars that break references)."""
        import re

        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError(
                f"Operation ID '{v}' contains invalid characters. "
                "Only letters, numbers, underscores, and hyphens are allowed."
            )
        return v


class OperationResult(BaseModel):
    """Result of a single operation execution.

    Contains the operation's output, status, execution metadata, and any errors.
    """

    id: str = Field(description="Operation identifier matching the request")

    tool: str = Field(description="Tool that was executed")

    status: Literal["success", "error", "timeout"] = Field(
        description="Execution status: success, error, or timeout"
    )

    result: Optional[Dict[str, Any]] = Field(
        default=None, description="Tool's result data (present if status=success)"
    )

    error: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Error details (present if status=error or status=timeout)",
    )

    execution_time_ms: float = Field(
        description="Time taken to execute this operation in milliseconds", ge=0
    )

    wave: int = Field(
        description="Execution wave number (0 = no dependencies, 1+ = depends on earlier waves)",
        ge=0,
    )

    dependencies: List[str] = Field(
        default_factory=list, description="List of operation IDs this operation depended on"
    )

    label: Optional[str] = Field(
        default=None, description="Human-readable label from the request"
    )


class BatchSummary(BaseModel):
    """Summary statistics for the entire batch execution."""

    total_operations: int = Field(description="Total number of operations in the batch", ge=0)

    succeeded: int = Field(description="Number of operations that completed successfully", ge=0)

    failed: int = Field(description="Number of operations that failed or timed out", ge=0)

    total_execution_time_ms: float = Field(
        description="Total wall-clock time for the entire batch execution", ge=0
    )

    execution_mode: Literal["sequential", "parallel", "auto"] = Field(
        description="Execution mode used for this batch"
    )

    num_waves: int = Field(
        description="Number of execution waves (only for auto mode with dependencies)", ge=0
    )

    max_concurrent: int = Field(
        description="Maximum concurrent operations allowed", ge=1
    )


class BatchResponse(BaseModel):
    """Complete batch execution response.

    Note: The 'context' field (batch-level) is injected by CustomMCP's
    transformation layer and appears at the top level of the JSON response.
    """

    results: List[OperationResult] = Field(
        description="Results for each operation in execution order"
    )

    summary: BatchSummary = Field(description="Batch execution summary statistics")
