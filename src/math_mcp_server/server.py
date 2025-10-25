"""Math MCP Server - High-performance mathematical operations using Polars and scientific Python."""

import logging
from fastmcp import FastMCP

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastMCP server instance
mcp = FastMCP("math-mcp")

# Import and register all tools
from .tools import basic  # noqa: F401, E402
from .tools import array  # noqa: F401, E402
from .tools import statistics  # noqa: F401, E402
from .tools import financial  # noqa: F401, E402
from .tools import linalg  # noqa: F401, E402
from .tools import calculus  # noqa: F401, E402


def main():
    """Entry point for uvx."""
    mcp.run()


if __name__ == "__main__":
    main()
