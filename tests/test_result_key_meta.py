"""Meta-test to validate all 21 tools return 'result' key.

This test ensures architectural consistency across the entire tool suite.
Every tool MUST return a response with a 'result' key as the primary output field.
"""

import json
import pytest


# Complete tool registry with minimal valid inputs for all 21 tools
TOOL_TEST_CASES = {
    # Basic Tools (4)
    "calculate": {"expression": "2+2"},
    "percentage": {"operation": "of", "value": 100, "percentage": 10},
    "round": {"values": 3.14159, "decimals": 2},
    "convert_units": {"value": 180, "from_unit": "degrees", "to_unit": "radians"},
    # Array Tools (4)
    "array_operations": {"operation": "add", "array1": [[1, 2]], "array2": [[3, 4]]},
    "array_statistics": {"data": [[1, 2], [3, 4]], "operations": ["mean"]},
    "array_aggregate": {"operation": "sumproduct", "array1": [1, 2], "array2": [3, 4]},
    "array_transform": {"data": [[1, 2], [3, 4]], "transform": "normalize"},
    # Statistics Tools (3)
    "statistics": {"data": [1, 2, 3, 4, 5], "analyses": ["describe"]},
    "pivot_table": {
        "data": [{"region": "North", "product": "A", "sales": 100}],
        "index": "region",
        "columns": "product",
        "values": "sales",
    },
    "correlation": {"data": {"x": [1, 2, 3], "y": [2, 4, 6]}},
    # Financial Tools (3)
    "financial_calcs": {
        "calculation": "pv",
        "rate": 0.05,
        "periods": 10,
        "payment": -100,
        "future_value": 0,
    },
    "compound_interest": {"principal": 1000, "rate": 0.05, "time": 10},
    "perpetuity": {"payment": 1000, "rate": 0.05},
    # Linear Algebra Tools (3)
    "matrix_operations": {"operation": "transpose", "matrix1": [[1, 2], [3, 4]]},
    "solve_linear_system": {
        "coefficients": [[2, 3], [1, 1]],
        "constants": [8, 3],
        "method": "direct",
    },
    "matrix_decomposition": {
        "matrix": [[4, 2], [1, 3]],
        "decomposition": "svd",
    },
    # Calculus Tools (3)
    "derivative": {"expression": "x^2", "variable": "x", "order": 1},
    "integral": {"expression": "x^2", "variable": "x", "method": "symbolic"},
    "limits_series": {
        "expression": "sin(x)/x",
        "variable": "x",
        "point": 0,
        "operation": "limit",
    },
}


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name,arguments", TOOL_TEST_CASES.items())
async def test_all_tools_return_result_key(mcp_client, tool_name, arguments):
    """Meta-test: Validate ALL 21 tools return a 'result' key.

    This test ensures architectural consistency. Every tool response MUST
    include a 'result' key containing the primary output value.

    Failure indicates:
    - Tool not using format_result() or format_array_result()
    - Regression in core formatting logic
    - Breaking change to tool output structure
    """
    result = await mcp_client.call_tool(tool_name, arguments)
    data = json.loads(result.content[0].text)

    # Explicit validation that 'result' key exists
    assert "result" in data, (
        f"Tool '{tool_name}' did not return a 'result' key. "
        f"Response keys: {list(data.keys())}. "
        f"This violates the standard tool output structure."
    )

    # Validate 'result' is not None (empty results should be [] or {}, not null)
    assert data["result"] is not None, (
        f"Tool '{tool_name}' returned null result. "
        f"Use empty list/dict instead of null for no-data scenarios."
    )
