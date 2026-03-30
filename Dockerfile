# TurboRAG — Production Docker Image
# Multi-stage build for a minimal runtime image.
#
# Build:
#   docker build -t turborag .
#
# Run (serve mode):
#   docker run -p 8080:8080 -v ./my_index:/data/index turborag \
#     turborag serve --index /data/index --host 0.0.0.0
#
# Run (MCP mode):
#   docker run -i turborag turborag mcp --index /data/index

# ---------- Stage 1: Builder ----------
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY pyproject.toml README.md ./
COPY src/ src/

# Install with all optional dependencies (excluding dev)
RUN pip install --no-cache-dir --prefix=/install '.[all]'

# Pre-compile the C scoring kernel
RUN gcc -O3 -march=native -funroll-loops -shared -fPIC \
    -o src/turborag/_cscore.so src/turborag/_cscore.c

# ---------- Stage 2: Runtime ----------
FROM python:3.12-slim

# Copy installed packages
COPY --from=builder /install /usr/local

# Copy package source (including pre-compiled C kernel)
WORKDIR /app
COPY --from=builder /build/src/ src/
COPY --from=builder /build/pyproject.toml .
COPY --from=builder /build/README.md .

# Install the package itself in the runtime image
RUN pip install --no-cache-dir --no-deps '.'

# Create a volume mount point for index data
VOLUME ["/data"]

# Default port for serve mode
EXPOSE 8080

# Health check for serve mode
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Default entrypoint
ENTRYPOINT ["turborag"]
CMD ["--help"]
