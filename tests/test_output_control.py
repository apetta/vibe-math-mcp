"""Tests for output control functionality (output_mode and extract parameters)."""

from vibe_math.server import (
    transform_single_response,
    transform_batch_response,
    is_sequential_chain,
    find_terminal_operation,
)
from vibe_math.core.result_resolver import ResultResolver


class TestTransformSingleResponse:
    """Test single tool response transformation."""

    def test_full_mode_preserves_everything(self):
        """Test that full mode returns data unchanged."""
        data = {"result": 105.0, "expression": "100 * 1.05", "variables": None}
        result = transform_single_response(data, "full")
        assert result == data

    def test_compact_mode_removes_nulls(self):
        """Test that compact mode removes null/None values."""
        data = {"result": 105.0, "expression": "100 * 1.05", "variables": None}
        result = transform_single_response(data, "compact")
        assert result == {"result": 105.0, "expression": "100 * 1.05"}
        assert "variables" not in result

    def test_minimal_mode_keeps_only_result(self):
        """Test that minimal mode keeps only primary value field."""
        data = {"result": 105.0, "expression": "100 * 1.05", "variables": None}
        result = transform_single_response(data, "minimal")
        assert result == {"result": 105.0}

    def test_minimal_mode_preserves_context(self):
        """Test that minimal mode preserves context field."""
        data = {
            "result": 105.0,
            "expression": "100 * 1.05",
            "context": "Bond A PV"
        }
        result = transform_single_response(data, "minimal")
        assert result == {"result": 105.0, "context": "Bond A PV"}

    def test_value_mode_normalizes_structure(self):
        """Test that value mode normalizes to {value: X} structure."""
        data = {"result": 105.0, "expression": "100 * 1.05", "variables": None}
        result = transform_single_response(data, "value")
        assert result == {"value": 105.0}

    def test_value_mode_with_context(self):
        """Test that value mode preserves context."""
        data = {"result": 105.0, "expression": "100 * 1.05", "context": "Test"}
        result = transform_single_response(data, "value")
        assert result == {"value": 105.0, "context": "Test"}

    def test_minimal_mode_with_array_tool(self):
        """Test minimal mode with array tool response."""
        data = {"result": [[1, 2]], "operation": "add", "shape": "1×2"}
        result = transform_single_response(data, "minimal")
        assert result == {"result": [[1, 2]]}

    def test_value_mode_with_array_tool(self):
        """Test value mode with array tool response."""
        data = {"result": [[1, 2]], "operation": "add", "shape": "1×2"}
        result = transform_single_response(data, "value")
        assert result == {"value": [[1, 2]]}

    def test_stats_tool_minimal_unchanged(self):
        """Test that stats tools are wrapped in result field."""
        data = {"result": {"describe": {"mean": 3.5}, "quartiles": {"Q1": 2.0}}}
        result = transform_single_response(data, "minimal")
        # Stats tools wrap complex objects in result field
        assert result == {"result": {"describe": {"mean": 3.5}, "quartiles": {"Q1": 2.0}}}


class TestTransformBatchResponse:
    """Test batch response transformation."""

    def test_full_mode_returns_complete_response(self):
        """Test that full mode preserves complete batch response."""
        data = {
            "results": [
                {
                    "id": "op1",
                    "tool": "calculate",
                    "status": "success",
                    "result": {"result": 105.0},
                    "wave": 0,
                    "dependencies": []
                },
                {
                    "id": "op2",
                    "tool": "calculate",
                    "status": "success",
                    "result": {"result": 115.5},
                    "wave": 1,
                    "dependencies": ["op1"]
                }
            ],
            "summary": {
                "succeeded": 2,
                "failed": 0,
                "total_execution_time_ms": 0.85
            }
        }
        result = transform_batch_response(data, "full")
        assert result == data

    def test_value_mode_creates_flat_mapping(self):
        """Test that value mode creates {id: value} flat structure."""
        data = {
            "results": [
                {
                    "id": "step1",
                    "tool": "calculate",
                    "status": "success",
                    "result": {"result": 105.0},
                    "wave": 0
                },
                {
                    "id": "step2",
                    "tool": "calculate",
                    "status": "success",
                    "result": {"result": 115.5},
                    "wave": 1
                }
            ],
            "summary": {
                "succeeded": 2,
                "failed": 0,
                "total_execution_time_ms": 0.85
            }
        }
        result = transform_batch_response(data, "value")

        assert "step1" in result
        assert "step2" in result
        assert result["step1"] == 105.0
        assert result["step2"] == 115.5
        assert "summary" in result
        assert result["summary"]["succeeded"] == 2
        assert result["summary"]["failed"] == 0

    def test_minimal_mode_simplifies_operations(self):
        """Test that minimal mode creates simplified operation objects."""
        data = {
            "results": [
                {
                    "id": "op1",
                    "tool": "calculate",
                    "status": "success",
                    "result": {"result": 42.0},
                    "wave": 0,
                    "dependencies": [],
                    "label": None,
                    "execution_time_ms": 1.5
                }
            ],
            "summary": {"succeeded": 1, "failed": 0}
        }
        result = transform_batch_response(data, "minimal")

        assert len(result["results"]) == 1
        op = result["results"][0]
        assert op["id"] == "op1"
        assert op["status"] == "success"
        assert op["value"] == 42.0
        assert op["wave"] == 0
        # Should not include execution_time_ms, dependencies, label
        assert "execution_time_ms" not in op
        assert "tool" not in op

    def test_compact_mode_removes_nulls(self):
        """Test that compact mode removes null fields from operations."""
        data = {
            "results": [
                {
                    "id": "op1",
                    "tool": "calculate",
                    "status": "success",
                    "result": {"result": 42.0},
                    "error": None,
                    "label": None,
                    "wave": 0
                }
            ],
            "summary": {"succeeded": 1, "failed": 0}
        }
        result = transform_batch_response(data, "compact")

        op = result["results"][0]
        assert "error" not in op
        assert "label" not in op
        assert "id" in op
        assert "result" in op

    def test_value_mode_skips_failed_operations(self):
        """Test that value mode only includes successful operations."""
        data = {
            "results": [
                {"id": "op1", "status": "success", "result": {"result": 10}},
                {
                    "id": "op2",
                    "status": "error",
                    "error": {"message": "Division by zero"}
                },
                {"id": "op3", "status": "success", "result": {"result": 30}}
            ],
            "summary": {"succeeded": 2, "failed": 1}
        }
        result = transform_batch_response(data, "value")

        assert "op1" in result
        assert "op3" in result
        assert "op2" not in result  # Failed operation excluded

    def test_minimal_mode_includes_error_message(self):
        """Test that minimal mode includes error messages for failed ops."""
        data = {
            "results": [
                {
                    "id": "op1",
                    "status": "error",
                    "error": {"message": "Invalid expression", "type": "ValueError"},
                    "wave": 0
                }
            ],
            "summary": {"succeeded": 0, "failed": 1}
        }
        result = transform_batch_response(data, "minimal")

        op = result["results"][0]
        assert op["id"] == "op1"
        assert op["status"] == "error"
        assert op["error"] == "Invalid expression"
        assert "value" not in op


class TestFinalMode:
    """Test final output mode for sequential chains."""

    def test_sequential_chain_detection_simple(self):
        """Test detection of simple 2-operation chain."""
        results = [
            {"id": "op1", "status": "success", "dependencies": []},
            {"id": "op2", "status": "success", "dependencies": ["op1"]}
        ]
        assert is_sequential_chain(results) is True

    def test_sequential_chain_detection_long(self):
        """Test detection of 5-operation chain."""
        results = [
            {"id": "op1", "dependencies": []},
            {"id": "op2", "dependencies": ["op1"]},
            {"id": "op3", "dependencies": ["op2"]},
            {"id": "op4", "dependencies": ["op3"]},
            {"id": "op5", "dependencies": ["op4"]}
        ]
        assert is_sequential_chain(results) is True

    def test_branching_not_sequential(self):
        """Test branching DAG is not detected as sequential."""
        results = [
            {"id": "op1", "dependencies": []},
            {"id": "op2", "dependencies": ["op1"]},
            {"id": "op3", "dependencies": ["op1"]}
        ]
        assert is_sequential_chain(results) is False

    def test_parallel_not_sequential(self):
        """Test parallel operations not detected as sequential."""
        results = [
            {"id": "op1", "dependencies": []},
            {"id": "op2", "dependencies": []}
        ]
        assert is_sequential_chain(results) is False

    def test_find_terminal_operation(self):
        """Test finding terminal operation."""
        results = [
            {"id": "op1", "dependencies": []},
            {"id": "op2", "dependencies": ["op1"]},
            {"id": "op3", "dependencies": ["op2"]}
        ]
        assert find_terminal_operation(results) == "op3"

    def test_final_mode_returns_terminal_success(self):
        """Test final mode with successful terminal operation."""
        data = {
            "results": [
                {"id": "op1", "status": "success", "result": {"result": 100.0}, "dependencies": []},
                {"id": "op2", "status": "success", "result": {"result": 110.0}, "dependencies": ["op1"]},
                {"id": "op3", "status": "success", "result": {"result": 90.5}, "dependencies": ["op2"]}
            ],
            "summary": {"succeeded": 3, "failed": 0, "total_execution_time_ms": 1.5}
        }
        result = transform_batch_response(data, "final")

        assert "result" in result
        assert result["result"] == 90.5
        assert result["summary"]["succeeded"] == 3
        assert result["summary"]["failed"] == 0

    def test_final_mode_returns_terminal_error(self):
        """Test final mode with failed terminal operation."""
        data = {
            "results": [
                {"id": "op1", "status": "success", "result": {"result": 100.0}, "dependencies": []},
                {
                    "id": "op2",
                    "status": "error",
                    "error": {"message": "Division by zero", "type": "ZeroDivisionError"},
                    "dependencies": ["op1"]
                }
            ],
            "summary": {"succeeded": 1, "failed": 1, "total_execution_time_ms": 1.2}
        }
        result = transform_batch_response(data, "final")

        assert "error" in result
        assert result["error"] == "Division by zero"
        assert result["summary"]["succeeded"] == 1
        assert result["summary"]["failed"] == 1

    def test_final_mode_fallback_to_value(self):
        """Test final mode falls back to value mode for non-sequential."""
        data = {
            "results": [
                {"id": "op1", "status": "success", "result": {"result": 100.0}, "dependencies": []},
                {"id": "op2", "status": "success", "result": {"result": 200.0}, "dependencies": ["op1"]},
                {"id": "op3", "status": "success", "result": {"result": 300.0}, "dependencies": ["op1"]}
            ],
            "summary": {"succeeded": 3, "failed": 0, "total_execution_time_ms": 2.0}
        }
        result = transform_batch_response(data, "final")

        assert "op1" in result
        assert "op2" in result
        assert "op3" in result
        assert result["op1"] == 100.0

    def test_final_mode_single_operation(self):
        """Test final mode with single operation."""
        data = {
            "results": [
                {"id": "op1", "status": "success", "result": {"result": 42.0}, "dependencies": []}
            ],
            "summary": {"succeeded": 1, "failed": 0, "total_execution_time_ms": 0.5}
        }
        result = transform_batch_response(data, "final")

        assert "result" in result
        assert result["result"] == 42.0


class TestContextPreservation:
    """Test context preservation across all output modes."""

    def test_batch_context_in_value_mode(self):
        """Test batch-level context appears in value mode."""
        data = {
            "results": [
                {"id": "op1", "status": "success", "result": {"result": 100.0}, "dependencies": []}
            ],
            "summary": {"succeeded": 1, "failed": 0},
            "context": "Q4 2025 Portfolio"
        }
        result = transform_batch_response(data, "value")

        assert "context" in result
        assert result["context"] == "Q4 2025 Portfolio"
        assert "op1" in result
        assert result["op1"] == 100.0

    def test_batch_context_in_minimal_mode(self):
        """Test batch-level context appears in minimal mode."""
        data = {
            "results": [
                {"id": "op1", "status": "success", "result": {"result": 100.0}, "dependencies": []}
            ],
            "summary": {"succeeded": 1, "failed": 0},
            "context": "Q4 2025 Portfolio"
        }
        result = transform_batch_response(data, "minimal")

        assert "context" in result
        assert result["context"] == "Q4 2025 Portfolio"

    def test_batch_context_in_compact_mode(self):
        """Test batch-level context appears in compact mode."""
        data = {
            "results": [
                {"id": "op1", "status": "success", "result": {"result": 100.0}, "dependencies": []}
            ],
            "summary": {"succeeded": 1, "failed": 0},
            "context": "Q4 2025 Portfolio"
        }
        result = transform_batch_response(data, "compact")

        assert "context" in result
        assert result["context"] == "Q4 2025 Portfolio"

    def test_batch_context_in_final_mode(self):
        """Test batch-level context appears in final mode."""
        data = {
            "results": [
                {"id": "op1", "status": "success", "result": {"result": 100.0}, "dependencies": []}
            ],
            "summary": {"succeeded": 1, "failed": 0},
            "context": "Q4 2025 Portfolio"
        }
        result = transform_batch_response(data, "final")

        assert "context" in result
        assert result["context"] == "Q4 2025 Portfolio"
        assert "result" in result
        assert result["result"] == 100.0

    def test_step_context_in_minimal_mode(self):
        """Test step-level context appears in minimal mode."""
        data = {
            "results": [
                {
                    "id": "bond_a",
                    "status": "success",
                    "result": {"result": 1081.11, "context": "UK Government Bond"},
                    "dependencies": []
                }
            ],
            "summary": {"succeeded": 1, "failed": 0}
        }
        result = transform_batch_response(data, "minimal")

        assert len(result["results"]) == 1
        op = result["results"][0]
        assert "context" in op
        assert op["context"] == "UK Government Bond"

    def test_both_contexts_together(self):
        """Test both batch and step contexts preserved."""
        data = {
            "results": [
                {
                    "id": "bond_a",
                    "status": "success",
                    "result": {"result": 1081.11, "context": "UK Gov Bond"},
                    "dependencies": []
                },
                {
                    "id": "bond_b",
                    "status": "success",
                    "result": {"result": 1000.0, "context": "Corporate Bond"},
                    "dependencies": []
                }
            ],
            "summary": {"succeeded": 2, "failed": 0},
            "context": "Q4 2025 Portfolio Analysis"
        }
        result = transform_batch_response(data, "minimal")

        assert "context" in result
        assert result["context"] == "Q4 2025 Portfolio Analysis"
        assert result["results"][0]["context"] == "UK Gov Bond"
        assert result["results"][1]["context"] == "Corporate Bond"

    def test_no_context_field_when_none(self):
        """Test context field omitted when not provided."""
        data = {
            "results": [
                {"id": "op1", "status": "success", "result": {"result": 100.0}, "dependencies": []}
            ],
            "summary": {"succeeded": 1, "failed": 0}
        }
        result = transform_batch_response(data, "value")

        assert "context" not in result
