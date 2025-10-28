"""Pytest configuration and shared fixtures."""

import pytest
from fastmcp import Client
from vibe_math_mcp import mcp


@pytest.fixture
async def mcp_client():
    """Create in-memory MCP client for testing."""
    async with Client(mcp) as client:
        yield client


@pytest.fixture
def sample_array_2x2():
    """Sample 2×2 array for testing."""
    return [[1.0, 2.0], [3.0, 4.0]]


@pytest.fixture
def sample_array_3x3():
    """Sample 3×3 array for testing."""
    return [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]


@pytest.fixture
def sample_data_list():
    """Sample data list for statistics."""
    return [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
