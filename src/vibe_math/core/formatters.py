"""Response formatting utilities for JSON and Markdown output."""

import json
from typing import Any, Dict, List, Optional


def format_json(data: Dict[str, Any]) -> str:
    """Format response as clean JSON."""
    return json.dumps(data, indent=2, default=str)


def format_result(value: Any, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Format a single result with optional metadata."""
    result = {"result": value}

    if metadata:
        result.update(metadata)

    return format_json(result)


def format_array_result(values: List[Any], metadata: Optional[Dict[str, Any]] = None) -> str:
    """Format array results."""
    result = {"values": values}

    if metadata:
        result.update(metadata)

    return format_json(result)


def format_error(error_message: str, suggestion: Optional[str] = None) -> str:
    """Format actionable error messages."""
    error_data = {"error": error_message}
    if suggestion:
        error_data["suggestion"] = suggestion
    return format_json(error_data)
