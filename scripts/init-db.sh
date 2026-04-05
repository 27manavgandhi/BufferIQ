#!/bin/bash
set -e

echo "Initializing BufferIQ database..."

# Wait for PostgreSQL to be ready
./scripts/wait-for-postgres.sh

# Run Alembic migrations
echo "Running database migrations..."
alembic upgrade head

echo "Database initialization complete!"