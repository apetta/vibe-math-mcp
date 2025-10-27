"""Comprehensive tests for batch execution functionality."""

import json
import pytest
from vibe_math.core.batch_models import BatchOperation, OperationResult
from vibe_math.core.result_resolver import ResultResolver
from vibe_math.core.batch_executor import BatchExecutor


class TestResultResolver:
    """Test the JSONPath-like result resolution system."""

    def test_simple_reference(self):
        """Test resolving $op_id reference."""
        results = {"op1": {"result": 42, "metadata": {"rate": 0.05}}}
        resolver = ResultResolver(results)

        resolved = resolver.resolve("$op1")
        assert resolved == {"result": 42, "metadata": {"rate": 0.05}}

    def test_path_navigation(self):
        """Test resolving $op_id.path references."""
        results = {"op1": {"result": 42, "metadata": {"rate": 0.05}}}
        resolver = ResultResolver(results)

        assert resolver.resolve("$op1.result") == 42
        assert resolver.resolve("$op1.metadata.rate") == 0.05

    def test_array_indexing(self):
        """Test array indexing in references."""
        results = {"op1": {"values": [[1, 2], [3, 4]]}}
        resolver = ResultResolver(results)

        assert resolver.resolve("$op1.values[0]") == [1, 2]
        assert resolver.resolve("$op1.values[1][0]") == 3

    def test_dict_resolution(self):
        """Test recursive resolution in dictionaries."""
        results = {"op1": {"result": 10}, "op2": {"result": 20}}
        resolver = ResultResolver(results)

        resolved = resolver.resolve({"x": "$op1.result", "y": "$op2.result"})
        assert resolved == {"x": 10, "y": 20}

    def test_list_resolution(self):
        """Test recursive resolution in lists."""
        results = {"op1": {"result": 10}, "op2": {"result": 20}}
        resolver = ResultResolver(results)

        resolved = resolver.resolve(["$op1.result", "$op2.result", 30])
        assert resolved == [10, 20, 30]

    def test_nested_resolution(self):
        """Test deeply nested reference resolution."""
        results = {
            "op1": {"result": 10},
            "op2": {"result": {"nested": {"value": 42}}},
        }
        resolver = ResultResolver(results)

        resolved = resolver.resolve({
            "variables": {"x": "$op1.result", "y": "$op2.result.nested.value"}
        })
        assert resolved == {"variables": {"x": 10, "y": 42}}

    def test_unknown_operation_error(self):
        """Test error on reference to non-existent operation."""
        results = {"op1": {"result": 42}}
        resolver = ResultResolver(results)

        with pytest.raises(ValueError, match="unknown operation 'op2'"):
            resolver.resolve("$op2.result")

    def test_invalid_path_error(self):
        """Test error on invalid path navigation."""
        results = {"op1": {"result": 42}}
        resolver = ResultResolver(results)

        with pytest.raises(ValueError, match="not found"):
            resolver.resolve("$op1.nonexistent")

    def test_invalid_syntax_error(self):
        """Test error on invalid reference syntax."""
        results = {"op1": {"result": 42}}
        resolver = ResultResolver(results)

        with pytest.raises(ValueError, match="Invalid reference syntax"):
            resolver.resolve("$")

        with pytest.raises(ValueError, match="Invalid reference syntax"):
            resolver.resolve("$ op1.result")  # Space in operation ID


class TestBatchModels:
    """Test Pydantic models for batch operations."""

    def test_batch_operation_defaults(self):
        """Test BatchOperation with default values."""
        op = BatchOperation(tool="calculate", arguments={"expression": "2 + 2"})

        assert op.tool == "calculate"
        assert op.arguments == {"expression": "2 + 2"}
        assert op.context is None
        assert op.label is None
        assert op.depends_on == []
        assert op.result_mapping is None
        assert op.timeout_ms is None
        assert len(op.id) > 0  # UUID generated

    def test_batch_operation_custom_id(self):
        """Test BatchOperation with custom ID."""
        op = BatchOperation(id="my_calc", tool="calculate", arguments={"expression": "2 + 2"})

        assert op.id == "my_calc"

    def test_batch_operation_invalid_id(self):
        """Test validation of operation ID format."""
        with pytest.raises(ValueError, match="invalid characters"):
            BatchOperation(id="my calc!", tool="calculate", arguments={})

    def test_batch_operation_duplicate_dependencies(self):
        """Test validation prevents duplicate dependencies."""
        with pytest.raises(ValueError, match="Duplicate dependencies"):
            BatchOperation(
                tool="calculate", arguments={}, depends_on=["op1", "op1", "op2"]
            )

    def test_operation_result_success(self):
        """Test OperationResult for successful operation."""
        result = OperationResult(
            id="op1",
            tool="calculate",
            status="success",
            result={"result": 42},
            execution_time_ms=15.5,
            wave=0,
        )

        assert result.status == "success"
        assert result.result == {"result": 42}
        assert result.error is None

    def test_operation_result_error(self):
        """Test OperationResult for failed operation."""
        result = OperationResult(
            id="op1",
            tool="calculate",
            status="error",
            error={"type": "ValueError", "message": "Invalid expression"},
            execution_time_ms=5.0,
            wave=0,
        )

        assert result.status == "error"
        assert result.result is None
        assert result.error
        assert result.error["type"] == "ValueError"


@pytest.mark.asyncio
class TestBatchExecutor:
    """Test the batch executor with DAG-based parallelization."""

    async def test_sequential_execution(self):
        """Test sequential execution mode."""
        # Create mock tool registry
        async def mock_calculate(**kwargs):
            expr = kwargs.get("expression", "0")
            if expr == "2 + 2":
                return json.dumps({"result": 4})
            elif expr == "x * 2":
                # This should have x=4 from first operation
                x = kwargs.get("variables", {}).get("x", 0)
                return json.dumps({"result": x * 2})
            return json.dumps({"result": 0})

        tool_registry = {"calculate": mock_calculate}

        # Create operations with dependency
        operations = [
            BatchOperation(id="op1", tool="calculate", arguments={"expression": "2 + 2"}),
            BatchOperation(
                id="op2",
                tool="calculate",
                arguments={"expression": "x * 2", "variables": {"x": "$op1.result"}},
                depends_on=["op1"],
            ),
        ]

        executor = BatchExecutor(
            operations=operations,
            tool_registry=tool_registry,
            mode="sequential",
        )

        response = await executor.execute()

        assert len(response.results) == 2
        assert response.results[0].id == "op1"
        assert response.results[0].result
        assert response.results[0].result["result"] == 4
        assert response.results[1].id == "op2"
        assert response.results[1].result
        assert response.results[1].result["result"] == 8  # 4 * 2

    async def test_parallel_execution(self):
        """Test parallel execution mode (ignores dependencies)."""

        async def mock_tool(**kwargs):
            return json.dumps({"result": 1})

        tool_registry = {"calculate": mock_tool}

        operations = [
            BatchOperation(id=f"op{i}", tool="calculate", arguments={}) for i in range(3)
        ]

        executor = BatchExecutor(
            operations=operations,
            tool_registry=tool_registry,
            mode="parallel",
            max_concurrent=2,
        )

        response = await executor.execute()

        assert len(response.results) == 3
        assert response.summary.num_waves == 1
        # All operations ran in parallel (wave 0)
        assert all(r.wave == 0 for r in response.results)

    async def test_auto_mode_dependency_detection(self):
        """Test auto mode detects dependencies from result_mapping."""

        async def mock_calc(**kwargs):
            expr = kwargs.get("expression", "")
            if expr == "2 + 2":
                return json.dumps({"result": 4})
            elif expr == "3 + 3":
                return json.dumps({"result": 6})
            else:
                # Final operation: should get x=4, y=6
                x = kwargs.get("variables", {}).get("x", 0)
                y = kwargs.get("variables", {}).get("y", 0)
                return json.dumps({"result": x + y})

        tool_registry = {"calculate": mock_calc}

        operations = [
            BatchOperation(id="op1", tool="calculate", arguments={"expression": "2 + 2"}),
            BatchOperation(id="op2", tool="calculate", arguments={"expression": "3 + 3"}),
            BatchOperation(
                id="op3",
                tool="calculate",
                arguments={"expression": "x + y"},
                result_mapping={"variables": {"x": "$op1.result", "y": "$op2.result"}},
                # No explicit depends_on - should be inferred from result_mapping
            ),
        ]

        executor = BatchExecutor(
            operations=operations, tool_registry=tool_registry, mode="auto"
        )

        response = await executor.execute()

        assert len(response.results) == 3
        assert response.summary.num_waves == 2

        # op1 and op2 should be in wave 0 (parallel)
        assert response.results[0].wave == 0
        assert response.results[1].wave == 0

        # op3 should be in wave 1 (depends on op1 and op2)
        assert response.results[2].wave == 1
        assert response.results[2].result
        assert response.results[2].result["result"] == 10  # 4 + 6

    async def test_circular_dependency_detection(self):
        """Test that circular dependencies are detected and raise error."""

        async def mock_tool(**kwargs):
            return json.dumps({"result": 1})

        tool_registry = {"calculate": mock_tool}

        operations = [
            BatchOperation(id="op1", tool="calculate", arguments={}, depends_on=["op2"]),
            BatchOperation(id="op2", tool="calculate", arguments={}, depends_on=["op1"]),
        ]

        executor = BatchExecutor(
            operations=operations, tool_registry=tool_registry, mode="auto"
        )

        with pytest.raises(ValueError, match="Circular dependency"):
            await executor.execute()

    async def test_missing_dependency_error(self):
        """Test error when operation depends on non-existent operation."""

        async def mock_tool(**kwargs):
            return json.dumps({"result": 1})

        tool_registry = {"calculate": mock_tool}

        operations = [
            BatchOperation(
                id="op1", tool="calculate", arguments={}, depends_on=["nonexistent"]
            ),
        ]

        executor = BatchExecutor(
            operations=operations, tool_registry=tool_registry, mode="auto"
        )

        with pytest.raises(ValueError, match="non-existent operations"):
            await executor.execute()

    async def test_stop_on_error_true(self):
        """Test that execution stops on first error when stop_on_error=True."""

        async def mock_tool(**kwargs):
            op_id = kwargs.get("_op_id", "")
            if op_id == "op1":
                raise ValueError("Test error")
            return json.dumps({"result": 1})

        tool_registry = {"calculate": mock_tool}

        operations = [
            BatchOperation(id="op1", tool="calculate", arguments={"_op_id": "op1"}),
            BatchOperation(id="op2", tool="calculate", arguments={"_op_id": "op2"}),
        ]

        executor = BatchExecutor(
            operations=operations,
            tool_registry=tool_registry,
            mode="sequential",
            stop_on_error=True,
        )

        response = await executor.execute()

        # Only op1 should have executed (and failed)
        assert len(response.results) == 1
        assert response.results[0].status == "error"
        assert response.summary.failed == 1

    async def test_stop_on_error_false(self):
        """Test that execution continues on error when stop_on_error=False."""

        async def mock_tool(**kwargs):
            op_id = kwargs.get("_op_id", "")
            if op_id == "op1":
                raise ValueError("Test error")
            return json.dumps({"result": 1})

        tool_registry = {"calculate": mock_tool}

        operations = [
            BatchOperation(id="op1", tool="calculate", arguments={"_op_id": "op1"}),
            BatchOperation(id="op2", tool="calculate", arguments={"_op_id": "op2"}),
        ]

        executor = BatchExecutor(
            operations=operations,
            tool_registry=tool_registry,
            mode="sequential",
            stop_on_error=False,
        )

        response = await executor.execute()

        # Both operations should have executed
        assert len(response.results) == 2
        assert response.results[0].status == "error"
        assert response.results[1].status == "success"
        assert response.summary.succeeded == 1
        assert response.summary.failed == 1

    async def test_operation_timeout(self):
        """Test operation-level timeout handling."""
        import asyncio

        async def slow_tool(**kwargs):
            await asyncio.sleep(0.2)  # 200ms delay
            return json.dumps({"result": 1})

        tool_registry = {"calculate": slow_tool}

        operations = [
            BatchOperation(
                id="op1", tool="calculate", arguments={}, timeout_ms=100  # 100ms timeout
            ),
        ]

        executor = BatchExecutor(
            operations=operations, tool_registry=tool_registry, mode="sequential"
        )

        response = await executor.execute()

        assert len(response.results) == 1
        assert response.results[0].status == "timeout"
        assert response.results[0].error
        assert "timeout" in response.results[0].error["message"].lower()

    async def test_context_injection_per_operation(self):
        """Test that operation-level context is injected into results."""

        async def mock_tool(**kwargs):
            return json.dumps({"result": 42})

        tool_registry = {"calculate": mock_tool}

        operations = [
            BatchOperation(
                id="op1",
                tool="calculate",
                arguments={},
                context="Operation-specific context",
            ),
        ]

        executor = BatchExecutor(
            operations=operations, tool_registry=tool_registry, mode="sequential"
        )

        response = await executor.execute()

        assert response.results[0].result
        assert response.results[0].result["context"] == "Operation-specific context"

    async def test_label_passthrough(self):
        """Test that operation labels pass through to results."""

        async def mock_tool(**kwargs):
            return json.dumps({"result": 1})

        tool_registry = {"calculate": mock_tool}

        operations = [
            BatchOperation(
                id="op1", tool="calculate", arguments={}, label="Calculate bond PV"
            ),
        ]

        executor = BatchExecutor(
            operations=operations, tool_registry=tool_registry, mode="sequential"
        )

        response = await executor.execute()

        assert response.results[0].label == "Calculate bond PV"


@pytest.mark.asyncio
class TestBatchIntegration:
    """Integration tests using the actual batch_execute tool."""

    async def test_batch_execute_simple(self, mcp_client):
        """Test simple batch execution with calculate tool."""
        result = await mcp_client.call_tool(
            "batch_execute",
            {
                "operations": [
                    {"id": "calc1", "tool": "calculate", "arguments": {"expression": "2 + 2"}},
                    {
                        "id": "calc2",
                        "tool": "percentage",
                        "arguments": {"operation": "of", "value": 100, "percentage": 15},
                    },
                ]
            },
        )

        data = json.loads(result.content[0].text)

        # Debug: print full data if failed
        if data.get("summary", {}).get("failed", 0) > 0:
            import pprint
            print("\n=== BATCH RESPONSE ===")
            pprint.pprint(data)
            print("======================\n")

        assert "results" in data
        assert "summary" in data
        assert len(data["results"]) == 2
        assert data["summary"]["succeeded"] == 2, f"Expected 2 succeeded, got {data['summary']}"
        assert data["summary"]["failed"] == 0

    async def test_batch_execute_with_dependencies(self, mcp_client):
        """Test batch with dependencies and result referencing."""
        result = await mcp_client.call_tool(
            "batch_execute",
            {
                "operations": [
                    {"id": "calc1", "tool": "calculate", "arguments": {"expression": "10 + 5"}},
                    {
                        "id": "calc2",
                        "tool": "calculate",
                        "arguments": {
                            "expression": "x * 2",
                            "variables": {"x": "$calc1.result"},
                        },
                        "depends_on": ["calc1"],
                    },
                ]
            },
        )

        data = json.loads(result.content[0].text)

        assert data["results"][0]["result"]["result"] == 15
        assert data["results"][1]["result"]["result"] == 30  # 15 * 2
        assert data["summary"]["num_waves"] == 2

    async def test_batch_execute_invalid_tool(self, mcp_client):
        """Test error handling for invalid tool name."""
        result = await mcp_client.call_tool(
            "batch_execute",
            {"operations": [{"id": "bad", "tool": "nonexistent", "arguments": {}}]},
        )

        data = json.loads(result.content[0].text)

        assert "error" in data
        assert "nonexistent" in data["error"]["message"]
