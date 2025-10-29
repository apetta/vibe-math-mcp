"""Batch execution engine with dependency-aware parallelization using DAG."""

import asyncio
import json
import time
from graphlib import TopologicalSorter, CycleError
from typing import Any, Dict, List, Literal, Set

from .batch_models import BatchOperation, OperationResult, BatchSummary, BatchResponse
from .result_resolver import ResultResolver


class BatchExecutor:
    """Execute batch operations with intelligent dependency management.

    Supports three execution modes:
    - sequential: Operations execute in order specified
    - parallel: All operations execute concurrently (ignoring dependencies)
    - auto: Build DAG from dependencies and execute in optimal wave-based manner

    Uses Python's graphlib.TopologicalSorter for dependency resolution and
    asyncio for parallel execution within each wave.
    """

    def __init__(
        self,
        operations: List[BatchOperation],
        tool_registry: Dict[str, Any],  # Tool functions (async callables)
        mode: Literal["sequential", "parallel", "auto"] = "auto",
        max_concurrent: int = 5,
        stop_on_error: bool = False,
    ):
        """Initialise batch executor.

        Args:
            operations: List of operations to execute
            tool_registry: Map of tool_name -> async function
            mode: Execution mode (sequential, parallel, auto)
            max_concurrent: Maximum concurrent operations
            stop_on_error: Whether to halt on first error
        """
        self.operations = {op.id: op for op in operations}
        self.tool_registry = tool_registry
        self.mode: Literal["sequential", "parallel", "auto"] = mode
        self.max_concurrent = max_concurrent
        self.stop_on_error = stop_on_error

        # Results storage
        self.results: Dict[str, Dict[str, Any]] = {}  # For dependency resolution
        self.operation_results: List[OperationResult] = []  # Final results
        self.errors: Dict[str, Exception] = {}

        # Timing
        self.start_time: float = 0
        self.num_waves: int = 0

    async def execute(self) -> BatchResponse:
        """Execute all operations and return complete batch response.

        Returns:
            BatchResponse with results and summary
        """
        self.start_time = time.time()

        # Execute based on mode
        if self.mode == "sequential":
            await self._execute_sequential()
        elif self.mode == "parallel":
            await self._execute_parallel()
        else:  # auto
            await self._execute_auto()

        # Build response
        return self._build_response()

    async def _execute_sequential(self) -> None:
        """Execute operations in exact order specified (index order)."""
        # Sort by creation order (Python 3.7+ dict maintains insertion order)
        op_ids = list(self.operations.keys())

        for wave_num, op_id in enumerate(op_ids):
            op = self.operations[op_id]
            result = await self._execute_operation(op, wave=wave_num)
            self.operation_results.append(result)

            if result.status == "error" and self.stop_on_error:
                break

        self.num_waves = len(self.operation_results)

    async def _execute_parallel(self) -> None:
        """Execute all operations in parallel (ignore dependencies)."""
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def bounded_execute(op: BatchOperation) -> OperationResult:
            async with semaphore:
                return await self._execute_operation(op, wave=0)

        # Create tasks for all operations
        tasks = [bounded_execute(op) for op in self.operations.values()]

        # Execute all in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for result in results:
            if isinstance(result, Exception):
                # Shouldn't happen as _execute_operation catches exceptions
                # but handle just in case
                continue
            if isinstance(result, OperationResult):
                self.operation_results.append(result)

        self.num_waves = 1

    async def _execute_auto(self) -> None:
        """Execute with dependency-aware parallelization using DAG.

        Uses TopologicalSorter to identify execution waves.
        Operations within a wave execute in parallel.
        """
        # Build dependency graph
        try:
            sorter = self._build_dependency_graph()
            sorter.prepare()
        except CycleError as e:
            # Extract cycle information
            cycle = e.args[1] if len(e.args) > 1 else []
            raise ValueError(
                f"Circular dependency detected in operations: {' -> '.join(cycle)}. "
                "Operations cannot depend on themselves directly or indirectly."
            )

        wave_num = 0
        semaphore = asyncio.Semaphore(self.max_concurrent)

        # Execute wave by wave
        while sorter.is_active():
            ready_ids = list(sorter.get_ready())

            if not ready_ids:
                # This shouldn't happen with a valid DAG, but be defensive
                break

            # Execute current wave in parallel
            async def bounded_execute(op_id: str) -> OperationResult:
                async with semaphore:
                    op = self.operations[op_id]
                    return await self._execute_operation(op, wave=wave_num)

            tasks = [bounded_execute(op_id) for op_id in ready_ids]
            wave_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process wave results
            should_stop = False
            for op_id, result in zip(ready_ids, wave_results):
                if isinstance(result, Exception):
                    # Unexpected exception (shouldn't happen)
                    self.errors[op_id] = result
                    if self.stop_on_error:
                        should_stop = True
                elif isinstance(result, OperationResult):
                    self.operation_results.append(result)

                    if result.status == "error" and self.stop_on_error:
                        should_stop = True

                # Mark operation as done for topological sorter
                sorter.done(op_id)

            if should_stop:
                break

            wave_num += 1

        self.num_waves = wave_num if self.operation_results else 0

    def _build_dependency_graph(self) -> TopologicalSorter:
        """Build DAG from operation dependencies.

        Returns:
            TopologicalSorter configured with operation dependencies

        Raises:
            ValueError: If dependencies reference non-existent operations
        """
        graph: Dict[str, List[str]] = {}

        for op_id, op in self.operations.items():
            # Scan arguments for $refs to detect dependencies
            deps = self._extract_refs_from_value(op.arguments)

            # Convert set to list for TopologicalSorter
            graph[op_id] = list(deps)

        # Validate all dependencies exist
        all_op_ids = set(self.operations.keys())
        for op_id, dep_list in graph.items():
            invalid_deps = set(dep_list) - all_op_ids
            if invalid_deps:
                raise ValueError(
                    f"Operation '{op_id}' has dependencies on non-existent operations: "
                    f"{', '.join(sorted(invalid_deps))}. "
                    f"Available operation IDs: {', '.join(sorted(all_op_ids))}"
                )

        return TopologicalSorter(graph)

    def _extract_refs_from_value(self, value: Any) -> Set[str]:
        """Recursively extract $operation_id references from any value."""
        refs: Set[str] = set()

        if isinstance(value, str) and value.startswith('$'):
            # Extract operation ID from $op_id or $op_id.path
            op_id = value.split('.')[0][1:]  # Remove $ and take first part
            refs.add(op_id)
        elif isinstance(value, dict):
            for v in value.values():
                refs.update(self._extract_refs_from_value(v))
        elif isinstance(value, list):
            for item in value:
                refs.update(self._extract_refs_from_value(item))

        return refs

    async def _execute_operation(
        self, op: BatchOperation, wave: int
    ) -> OperationResult:
        """Execute a single operation with timing and error handling.

        Args:
            op: Operation to execute
            wave: Execution wave number (for metadata)

        Returns:
            OperationResult with status, result/error, and metadata
        """
        start_time = time.time()

        try:
            # Resolve arguments with dependencies
            resolved_args = self._resolve_arguments(op)

            # Get wrapped tool instance (not raw function)
            if op.tool not in self.tool_registry:
                raise ValueError(
                    f"Tool '{op.tool}' not found in registry. "
                    f"Available tools: {', '.join(sorted(self.tool_registry.keys()))}"
                )

            tool = self.tool_registry[op.tool]

            # Execute tool.run() with arguments dict
            if op.timeout_ms:
                tool_result = await asyncio.wait_for(
                    tool.run(resolved_args), timeout=op.timeout_ms / 1000
                )
            else:
                tool_result = await tool.run(resolved_args)

            # Extract text content from ToolResult
            from mcp.types import TextContent
            if tool_result.content and isinstance(tool_result.content[0], TextContent):
                raw_result = tool_result.content[0].text
            else:
                raise ValueError(
                    f"Unexpected tool result format from {op.tool}. "
                    f"Expected TextContent, got {type(tool_result.content[0]) if tool_result.content else 'no content'}"
                )

            # Parse JSON result
            result_data = json.loads(raw_result)

            # Inject operation-level context if provided
            if op.context:
                result_data['context'] = op.context

            # Store result for dependency resolution
            self.results[op.id] = result_data

            # Calculate execution time
            execution_time = (time.time() - start_time) * 1000

            return OperationResult(
                id=op.id,
                tool=op.tool,
                status="success",
                result=result_data,
                execution_time_ms=execution_time,
                wave=wave,
                dependencies=list(self._extract_refs_from_value(op.arguments)),
                label=op.label,
            )

        except asyncio.TimeoutError:
            execution_time = (time.time() - start_time) * 1000
            return OperationResult(
                id=op.id,
                tool=op.tool,
                status="timeout",
                error={
                    "type": "TimeoutError",
                    "message": f"Operation exceeded {op.timeout_ms}ms timeout",
                    "tool": op.tool,
                },
                execution_time_ms=execution_time,
                wave=wave,
                dependencies=list(self._extract_refs_from_value(op.arguments)),
                label=op.label,
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.errors[op.id] = e

            return OperationResult(
                id=op.id,
                tool=op.tool,
                status="error",
                error={
                    "type": type(e).__name__,
                    "message": str(e),
                    "tool": op.tool,
                },
                execution_time_ms=execution_time,
                wave=wave,
                dependencies=list(self._extract_refs_from_value(op.arguments)),
                label=op.label,
            )

    def _resolve_arguments(self, op: BatchOperation) -> Dict[str, Any]:
        """Resolve operation arguments with result references.

        Resolves $refs in arguments.
        Handles precedence for context and output_mode parameters.

        Args:
            op: Operation to resolve arguments for

        Returns:
            Fully resolved arguments dictionary

        Raises:
            ValueError: If references cannot be resolved
        """
        resolver = ResultResolver(self.results)

        # Start with base arguments
        resolved = op.arguments.copy()

        # Handle context precedence: operation-level > arguments-level
        # If operation has context at operation level, remove from arguments
        # (operation-level takes precedence and will be injected after execution)
        if op.context and 'context' in resolved:
            del resolved['context']

        # Always remove output_mode from arguments to prevent double transformation
        # The batch-level output_mode controls the entire response format
        if 'output_mode' in resolved:
            del resolved['output_mode']

        # Resolve all $refs in argument values
        resolved = resolver.resolve(resolved)

        return resolved

    def _build_response(self) -> BatchResponse:
        """Build complete batch response with results and summary.

        Returns:
            BatchResponse with all results and execution statistics
        """
        total_time = (time.time() - self.start_time) * 1000

        # Count successes and failures
        succeeded = sum(1 for r in self.operation_results if r.status == "success")
        failed = sum(1 for r in self.operation_results if r.status in ["error", "timeout"])

        summary = BatchSummary(
            total_operations=len(self.operations),
            succeeded=succeeded,
            failed=failed,
            total_execution_time_ms=total_time,
            execution_mode=self.mode,
            num_waves=self.num_waves,
            max_concurrent=self.max_concurrent,
        )

        return BatchResponse(results=self.operation_results, summary=summary)
