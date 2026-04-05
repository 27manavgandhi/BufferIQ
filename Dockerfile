# Multi-stage build for BufferIQ backend

# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY backend/ .

# Copy scripts
COPY scripts/ /scripts/
RUN chmod +x /scripts/*.sh

# Set Python path
ENV PYTHONPATH=/app
ENV PATH=/root/.local/bin:$PATH

# Create non-root user
RUN useradd -m -u 1000 bufferiq && \
    chown -R bufferiq:bufferiq /app

USER bufferiq

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD python -c "from bufferiq.core.config import Settings; Settings()" || exit 1

# Default command (overridden in docker-compose)
CMD ["sleep", "infinity"]