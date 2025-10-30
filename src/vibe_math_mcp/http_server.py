"""HTTP server entry point for containerized deployment.

This module provides an HTTP transport wrapper around the main MCP server
for use with Smithery's container deployment and streamable HTTP transport.
"""

import os
import uvicorn
from starlette.middleware.cors import CORSMiddleware

from .server import mcp, __version__


def main():
    """Start the HTTP server with CORS support for containerized deployment.

    This function:
    1. Wraps the MCP server with Starlette HTTP app
    2. Configures CORS for cross-origin requests (required for browser clients)
    3. Listens on PORT environment variable (Smithery sets to 8081)
    4. Exposes MCP endpoints at /mcp for streamable HTTP transport
    """

    print(f"Vibe Math MCP Server v{__version__} starting...")
    print("Mode: HTTP (Streamable HTTP Transport)")

    # Create Starlette app from MCP server with HTTP transport
    app = mcp.http_app()

    # CRITICAL: Add CORS middleware for browser-based clients
    # This is required for Smithery and other web-based MCP clients
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins for public server
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],  # Required HTTP methods
        allow_headers=["*"],  # Allow all headers including custom ones
        expose_headers=[
            "mcp-session-id",  # Expose MCP protocol headers
            "mcp-protocol-version",
        ],
        max_age=86400,  # Cache preflight requests for 24 hours
    )

    # Get port from environment variable
    # Smithery sets PORT=8081 in production
    # Use 8080 as default for local development
    port = int(os.environ.get("PORT", 8080))
    print(f"Listening on http://0.0.0.0:{port}")
    print(f"MCP endpoint: http://0.0.0.0:{port}/mcp")
    print("Ready for connections...")

    # Start the server
    uvicorn.run(
        app,
        host="0.0.0.0",  # Listen on all interfaces
        port=port,
        log_level="info",  # Use info level for production
    )


if __name__ == "__main__":
    main()
