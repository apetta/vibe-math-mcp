"""Vibe Math - High-performance mathematical operations using Polars and scientific Python."""

import logging
from fastmcp import FastMCP

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastMCP server instance
mcp = FastMCP("vibe-math")

# Import and register all tools (must be after mcp instance creation for decorators)
from .tools import array, basic, calculus, financial, linalg, statistics  # noqa: E402

# Explicitly declare as part of module interface (tools registered via decorators)
__all__ = ["mcp", "basic", "array", "statistics", "financial", "linalg", "calculus"]


def main():
    """Entry point for uvx."""
    mcp.run()


if __name__ == "__main__":
    main()
