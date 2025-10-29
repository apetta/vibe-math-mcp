"""Comprehensive tests for batch execution functionality."""

import json
import pytest
from vibe_math_mcp.core.batch_models import BatchOperation, OperationResult
from vibe_math_mcp.core.result_resolver import ResultResolver


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
        results = {"op1": {"result": [[1, 2], [3, 4]]}}
        resolver = ResultResolver(results)

        assert resolver.resolve("$op1.result[0]") == [1, 2]
        assert resolver.resolve("$op1.result[1][0]") == 3

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

    async def test_sequential_execution(self, mcp_client):
        """Test sequential execution mode."""
        result = await mcp_client.call_tool(
            "batch_execute",
            {
                "operations": [
                    {"id": "op1", "tool": "calculate", "arguments": {"expression": "2 + 2"}},
                    {
                        "id": "op2",
                        "tool": "calculate",
                        "arguments": {"expression": "x * 2", "variables": {"x": "$op1.result"}},
                    },
                ],
                "execution_mode": "sequential",
            },
        )

        data = json.loads(result.content[0].text)

        assert len(data["results"]) == 2
        assert data["results"][0]["id"] == "op1"
        assert data["results"][0]["result"]["result"] == 4.0
        assert data["results"][1]["id"] == "op2"
        assert data["results"][1]["result"]["result"] == 8.0  # 4 * 2

    async def test_parallel_execution(self, mcp_client):
        """Test parallel execution mode (all operations run in wave 0)."""
        result = await mcp_client.call_tool(
            "batch_execute",
            {
                "operations": [
                    {"id": "op0", "tool": "calculate", "arguments": {"expression": "1 + 1"}},
                    {"id": "op1", "tool": "calculate", "arguments": {"expression": "2 + 2"}},
                    {"id": "op2", "tool": "calculate", "arguments": {"expression": "3 + 3"}},
                ],
                "execution_mode": "parallel",
                "max_concurrent": 2,
            },
        )

        data = json.loads(result.content[0].text)

        assert len(data["results"]) == 3
        assert data["summary"]["num_waves"] == 1
        # All operations ran in parallel (wave 0)
        assert all(r["wave"] == 0 for r in data["results"])

    async def test_auto_mode_dependency_detection(self, mcp_client):
        """Test auto mode detects dependencies from $refs in arguments."""
        result = await mcp_client.call_tool(
            "batch_execute",
            {
                "operations": [
                    {"id": "op1", "tool": "calculate", "arguments": {"expression": "2 + 2"}},
                    {"id": "op2", "tool": "calculate", "arguments": {"expression": "3 + 3"}},
                    {
                        "id": "op3",
                        "tool": "calculate",
                        "arguments": {
                            "expression": "x + y",
                            "variables": {"x": "$op1.result", "y": "$op2.result"},
                        },
                        # Dependencies automatically inferred from $refs in arguments
                    },
                ],
                "execution_mode": "auto",
            },
        )

        data = json.loads(result.content[0].text)

        assert len(data["results"]) == 3
        assert data["summary"]["num_waves"] == 2

        # op1 and op2 should be in wave 0 (parallel)
        assert data["results"][0]["wave"] == 0
        assert data["results"][1]["wave"] == 0

        # op3 should be in wave 1 (depends on op1 and op2)
        assert data["results"][2]["wave"] == 1
        assert data["results"][2]["result"]["result"] == 10.0  # 4 + 6

    async def test_circular_dependency_detection(self, mcp_client):
        """Test that circular dependencies are detected and raise error."""
        result = await mcp_client.call_tool(
            "batch_execute",
            {
                "operations": [
                    {
                        "id": "op1",
                        "tool": "calculate",
                        "arguments": {"expression": "x + 1", "variables": {"x": "$op2.result"}},
                    },
                    {
                        "id": "op2",
                        "tool": "calculate",
                        "arguments": {"expression": "y + 1", "variables": {"y": "$op1.result"}},
                    },
                ],
                "execution_mode": "auto",
            },
        )

        data = json.loads(result.content[0].text)

        # Should return error response
        assert "error" in data
        assert "Circular dependency" in data["error"]["message"]

    async def test_missing_dependency_error(self, mcp_client):
        """Test error when operation depends on non-existent operation."""
        result = await mcp_client.call_tool(
            "batch_execute",
            {
                "operations": [
                    {
                        "id": "op1",
                        "tool": "calculate",
                        "arguments": {
                            "expression": "x + 1",
                            "variables": {"x": "$nonexistent.result"},
                        },
                    },
                ],
                "execution_mode": "auto",
            },
        )

        data = json.loads(result.content[0].text)

        # Should return error response
        assert "error" in data
        assert "non-existent operations" in data["error"]["message"]

    async def test_stop_on_error_true(self, mcp_client):
        """Test that execution stops on first error when stop_on_error=True."""
        result = await mcp_client.call_tool(
            "batch_execute",
            {
                "operations": [
                    {
                        "id": "op1",
                        "tool": "calculate",
                        "arguments": {
                            "expression": "undefined_variable",  # Will cause error
                        },
                    },
                    {
                        "id": "op2",
                        "tool": "calculate",
                        "arguments": {"expression": "2 + 2"},
                    },
                ],
                "execution_mode": "sequential",
                "stop_on_error": True,
            },
        )

        data = json.loads(result.content[0].text)

        # Only op1 should have executed (and failed)
        assert len(data["results"]) == 1
        assert data["results"][0]["status"] == "error"
        assert data["summary"]["failed"] == 1

    async def test_stop_on_error_false(self, mcp_client):
        """Test that execution continues on error when stop_on_error=False."""
        result = await mcp_client.call_tool(
            "batch_execute",
            {
                "operations": [
                    {
                        "id": "op1",
                        "tool": "calculate",
                        "arguments": {
                            "expression": "undefined_variable",  # Will cause error
                        },
                    },
                    {
                        "id": "op2",
                        "tool": "calculate",
                        "arguments": {"expression": "2 + 2"},
                    },
                ],
                "execution_mode": "sequential",
                "stop_on_error": False,
            },
        )

        data = json.loads(result.content[0].text)

        # Both operations should have executed
        assert len(data["results"]) == 2
        assert data["results"][0]["status"] == "error"
        assert data["results"][1]["status"] == "success"
        assert data["summary"]["succeeded"] == 1
        assert data["summary"]["failed"] == 1

    @pytest.mark.skip(
        reason="Timeout testing requires slow operations; all math tools are too fast to timeout reliably via mcp_client"
    )
    async def test_operation_timeout(self):
        """Test operation-level timeout handling.

        Note: This test is skipped because testing timeouts via mcp_client is not viable.
        Math operations execute too quickly (< 100ms minimum timeout), making timeout testing unreliable.
        Timeout functionality is tested indirectly via the batch executor implementation.
        """
        pass

    async def test_context_injection_per_operation(self, mcp_client):
        """Test that operation-level context is injected into results."""
        result = await mcp_client.call_tool(
            "batch_execute",
            {
                "operations": [
                    {
                        "id": "op1",
                        "tool": "calculate",
                        "arguments": {"expression": "2 + 2"},
                        "context": "Operation-specific context",
                    },
                ],
                "execution_mode": "sequential",
            },
        )

        data = json.loads(result.content[0].text)

        # Context should be in result
        assert data["results"][0]["result"]
        assert data["results"][0]["result"]["context"] == "Operation-specific context"

    async def test_label_passthrough(self, mcp_client):
        """Test that operation labels pass through to results."""
        result = await mcp_client.call_tool(
            "batch_execute",
            {
                "operations": [
                    {
                        "id": "op1",
                        "tool": "calculate",
                        "arguments": {"expression": "2 + 2"},
                        "label": "Calculate bond PV",
                    },
                ],
                "execution_mode": "sequential",
            },
        )

        data = json.loads(result.content[0].text)

        # Label should pass through
        assert data["results"][0]["label"] == "Calculate bond PV"


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

    async def test_batch_matrix_decomposition_value_mode(self, mcp_client):
        """Test matrix_decomposition in batch with value output mode."""
        result = await mcp_client.call_tool(
            "batch_execute",
            {
                "operations": [
                    {
                        "id": "svd",
                        "tool": "matrix_decomposition",
                        "arguments": {"matrix": [[1, 2], [3, 4]], "decomposition": "svd"},
                    }
                ],
                "output_mode": "value",
            },
        )

        data = json.loads(result.content[0].text)

        # In value mode, should get flat mapping
        assert "svd" in data
        assert isinstance(data["svd"], dict)
        assert "U" in data["svd"]
        assert "singular_values" in data["svd"]
        assert "Vt" in data["svd"]

    async def test_batch_derivative_with_chaining(self, mcp_client):
        """Test derivative in batch with result chaining using value_at_point."""
        result = await mcp_client.call_tool(
            "batch_execute",
            {
                "operations": [
                    {
                        "id": "deriv",
                        "tool": "derivative",
                        "arguments": {"expression": "x^2", "variable": "x", "point": 3},
                    },
                    {
                        "id": "calc",
                        "tool": "calculate",
                        "arguments": {
                            "expression": "x * 10",
                            "variables": {"x": "$deriv.value_at_point"},
                        },
                    },
                ],
            },
        )

        data = json.loads(result.content[0].text)

        # Derivative of x^2 is 2x, at x=3 is 6
        # Then 6 * 10 = 60
        assert data["results"][0]["status"] == "success"
        assert data["results"][1]["status"] == "success"
        # Derivative result should be "2*x"
        assert "2*x" in data["results"][0]["result"]["result"] or "2x" in data["results"][0]["result"]["result"]
        # Value at point should be 6
        assert data["results"][0]["result"]["value_at_point"] == 6.0
        # Calculation result should be 60
        assert data["results"][1]["result"]["result"] == 60.0
        assert data["summary"]["num_waves"] == 2

    async def test_batch_limits_series_minimal_mode(self, mcp_client):
        """Test limits_series in batch with minimal output mode."""
        result = await mcp_client.call_tool(
            "batch_execute",
            {
                "operations": [
                    {
                        "id": "limit1",
                        "tool": "limits_series",
                        "arguments": {
                            "expression": "sin(x)/x",
                            "variable": "x",
                            "point": 0,
                            "operation": "limit",
                        },
                    },
                    {
                        "id": "series1",
                        "tool": "limits_series",
                        "arguments": {
                            "expression": "exp(x)",
                            "variable": "x",
                            "point": 0,
                            "operation": "series",
                            "order": 4,
                        },
                    },
                ],
                "output_mode": "minimal",
            },
        )

        data = json.loads(result.content[0].text)

        # In minimal mode, should have simplified structure
        assert "results" in data
        assert len(data["results"]) == 2
        assert data["results"][0]["id"] == "limit1"
        assert data["results"][1]["id"] == "series1"
        # Both should have value field (from result)
        assert "value" in data["results"][0]
        assert "value" in data["results"][1]

    async def test_batch_all_three_tools_final_mode(self, mcp_client):
        """Test sequential dependency chain with final mode returning only terminal result."""
        result = await mcp_client.call_tool(
            "batch_execute",
            {
                "operations": [
                    {
                        "id": "calc1",
                        "tool": "calculate",
                        "arguments": {"expression": "10 * 2"},
                    },
                    {
                        "id": "calc2",
                        "tool": "calculate",
                        "arguments": {
                            "expression": "x + 5",
                            "variables": {"x": "$calc1.result"},
                        },
                    },
                    {
                        "id": "final_calc",
                        "tool": "calculate",
                        "arguments": {
                            "expression": "y * 3",
                            "variables": {"y": "$calc2.result"},
                        },
                    },
                ],
                "execution_mode": "auto",
                "output_mode": "final",
            },
        )

        data = json.loads(result.content[0].text)

        # Final mode should return only terminal result for sequential chain
        # calc1: 20, calc2: 25, final_calc: 75
        assert "result" in data
        assert data["result"] == 75.0
        assert "summary" in data
        assert data["summary"]["succeeded"] == 3
