# Dockerfile for Vibe Math MCP Server

# Use official uv Docker image with Python 3.12 on Debian
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Set working directory
WORKDIR /app

# Enable bytecode compilation for faster startup
ENV UV_COMPILE_BYTECODE=1

# Copy from cache instead of linking (required for Docker volumes)
ENV UV_LINK_MODE=copy

# Install project dependencies using lockfile
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

# Copy application code
COPY . /app

# Install the application itself
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# Add virtual environment to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Reset entrypoint (don't invoke uv wrapper)
ENTRYPOINT []

# Expose port 8081 (Smithery standard)
EXPOSE 8081

# Run the HTTP server
# Note: The PORT environment variable is set by Smithery to 8081
CMD ["python", "-m", "vibe_math_mcp.http_server"]
