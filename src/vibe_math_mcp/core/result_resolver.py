"""Result resolution for batch operations with JSONPath-like syntax."""

import re
from typing import Any, Dict


class ResultResolver:
    """Resolve $operation_id.path references to actual values.

    Supports:
    - Simple references: $op_id
    - Path navigation: $op_id.result
    - Nested paths: $op_id.metadata.rate
    - Array indexing: $op_id.result[0]
    - Multi-dimensional arrays: $op_id.result[0][1]
    """

    def __init__(self, results: Dict[str, Dict[str, Any]]):
        """Initialise resolver with completed operation results.

        Args:
            results: Map of operation_id -> result dictionary
        """
        self.results = results

    def resolve(self, value: Any) -> Any:
        """Recursively resolve references in value.

        Args:
            value: Any value that might contain $references

        Returns:
            Value with all references resolved to actual values
        """
        if isinstance(value, str) and value.startswith('$'):
            return self._resolve_reference(value)
        elif isinstance(value, dict):
            return {k: self.resolve(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self.resolve(v) for v in value]
        else:
            return value

    def _resolve_reference(self, ref: str) -> Any:
        """Resolve a single $op_id.path reference.

        Args:
            ref: Reference string like $op_id or $op_id.result

        Returns:
            Resolved value from operation result

        Raises:
            ValueError: If reference syntax is invalid or operation not found
        """
        # Pattern: $operation_id or $operation_id.path.to.field
        # Operation ID can contain letters, numbers, underscores, hyphens
        pattern = r'^\$([a-zA-Z0-9_-]+)(?:\.(.+))?$'
        match = re.match(pattern, ref)

        if not match:
            raise ValueError(
                f"Invalid reference syntax: '{ref}'. "
                f"Expected format: $operation_id or $operation_id.path.to.field"
            )

        op_id, path = match.groups()

        # Check operation exists
        if op_id not in self.results:
            available = ', '.join(sorted(self.results.keys()))
            raise ValueError(
                f"Reference to unknown operation '{op_id}' in '{ref}'. "
                f"Available operations: {available if available else 'none'}"
            )

        # Start with full result
        value = self.results[op_id]

        # Navigate path if provided
        if path:
            value = self._navigate_path(value, path, ref)

        return value

    def _navigate_path(self, obj: Any, path: str, original_ref: str) -> Any:
        """Navigate nested object/array structure.

        Supports:
        - Dot notation: result.metadata.rate
        - Array indexing: result[0] or result[0][1]
        - Mixed: metadata.result[0].name

        Args:
            obj: Starting object to navigate from
            path: Navigation path (e.g., "result.metadata.rate")
            original_ref: Original reference string (for error messages)

        Returns:
            Value at the end of the path

        Raises:
            ValueError: If path is invalid or value not found
        """
        # Split path by dots and brackets: "a.b[0].c" -> ["a", "b", "0", "c"]
        # This regex splits on . or [ or ], then filters empty strings
        parts = [p for p in re.split(r'\.|\[|\]', path) if p]

        current = obj
        current_path = []

        for part in parts:
            current_path.append(part)
            path_so_far = '.'.join(current_path)

            if isinstance(current, dict):
                if part not in current:
                    available_keys = ', '.join(f"'{k}'" for k in current.keys())
                    raise ValueError(
                        f"Key '{part}' not found in {original_ref} at path '{path_so_far}'. "
                        f"Available keys: {available_keys if available_keys else 'none (empty dict)'}"
                    )
                current = current[part]

            elif isinstance(current, (list, tuple)):
                try:
                    index = int(part)
                    if index < 0 or index >= len(current):
                        raise IndexError(
                            f"Index {index} out of range [0, {len(current) - 1}]"
                        )
                    current = current[index]
                except ValueError:
                    raise ValueError(
                        f"Invalid array index '{part}' in {original_ref} at path '{path_so_far}'. "
                        f"Expected integer, got '{part}'"
                    )
                except IndexError as e:
                    raise ValueError(
                        f"Array index error in {original_ref} at path '{path_so_far}': {e}"
                    )
            else:
                current_type = type(current).__name__
                raise ValueError(
                    f"Cannot navigate path '{path}' in {original_ref} at '{path_so_far}': "
                    f"reached non-dict/non-list value of type {current_type}"
                )

        return current