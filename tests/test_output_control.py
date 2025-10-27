"""Tests for output control functionality (output_mode and extract parameters)."""

from vibe_math.server import (
    extract_primary_value,
    transform_single_response,
    transform_batch_response,
)


class TestExtractPrimaryValue:
    """Test primary value extraction from different tool responses."""

    def test_extract_from_basic_tool(self):
        """Test extraction from basic tool response (has 'result' field)."""
        data = {"result": 105.0, "expression": "100 * 1.05", "variables": None}
        assert extract_primary_value(data) == 105.0

    def test_extract_from_array_tool(self):
        """Test extraction from array tool response (has 'values' field)."""
        data = {"values": [[1, 2], [3, 4]], "operation": "add", "shape": "2×2"}
        assert extract_primary_value(data) == [[1, 2], [3, 4]]

    def test_extract_from_stats_tool(self):
        """Test extraction from stats tool (multiple result keys)."""
        data = {
            "describe": {"mean": 3.5, "std": 1.71},
            "quartiles": {"Q1": 2.0, "Q3": 4.0}
        }
        # For stats tools with no primary field, return entire object
        assert extract_primary_value(data) == data


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
        data = {"values": [[1, 2]], "operation": "add", "shape": "1×2"}
        result = transform_single_response(data, "minimal")
        assert result == {"values": [[1, 2]]}

    def test_value_mode_with_array_tool(self):
        """Test value mode with array tool response."""
        data = {"values": [[1, 2]], "operation": "add", "shape": "1×2"}
        result = transform_single_response(data, "value")
        assert result == {"value": [[1, 2]]}

    def test_stats_tool_minimal_unchanged(self):
        """Test that stats tools remain unchanged in minimal mode."""
        data = {"describe": {"mean": 3.5}, "quartiles": {"Q1": 2.0}}
        result = transform_single_response(data, "minimal")
        # Stats tools don't have a single primary field, so return all
        assert result == data


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

    def test_extract_filters_operations(self):
        """Test that extract parameter filters results."""
        data = {
            "results": [
                {"id": "op1", "status": "success", "result": {"result": 10}},
                {"id": "op2", "status": "success", "result": {"result": 20}},
                {"id": "op3", "status": "success", "result": {"result": 30}}
            ],
            "summary": {"succeeded": 3, "failed": 0}
        }
        result = transform_batch_response(data, "full", extract=["op1", "op3"])

        assert len(result["results"]) == 2
        assert result["results"][0]["id"] == "op1"
        assert result["results"][1]["id"] == "op3"

    def test_extract_with_value_mode(self):
        """Test extract combined with value mode."""
        data = {
            "results": [
                {"id": "step1", "status": "success", "result": {"result": 105.0}},
                {"id": "step2", "status": "success", "result": {"result": 115.5}},
                {"id": "final", "status": "success", "result": {"result": 90.5}}
            ],
            "summary": {
                "succeeded": 3,
                "failed": 0,
                "total_execution_time_ms": 1.2
            }
        }
        result = transform_batch_response(data, "value", extract=["final"])

        assert "final" in result
        assert result["final"] == 90.5
        assert "step1" not in result
        assert "step2" not in result
        assert "summary" in result

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

    def test_extract_with_array_tool_results(self):
        """Test extract with array tool results."""
        data = {
            "results": [
                {
                    "id": "arr1",
                    "status": "success",
                    "result": {"values": [[1, 2], [3, 4]]}
                },
                {
                    "id": "arr2",
                    "status": "success",
                    "result": {"values": [[5, 6]]}
                }
            ],
            "summary": {"succeeded": 2, "failed": 0}
        }
        result = transform_batch_response(data, "value", extract=["arr1"])

        assert "arr1" in result
        assert result["arr1"] == [[1, 2], [3, 4]]
        assert "arr2" not in result
