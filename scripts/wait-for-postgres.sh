#!/bin/bash
set -e

host="${POSTGRES_HOST:-postgres}"
port="${POSTGRES_PORT:-5432}"
user="${POSTGRES_USER:-bufferiq}"

echo "Waiting for PostgreSQL at $host:$port..."

max_attempts=30
attempt=0

until PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$host" -p "$port" -U "$user" -c '\q' 2>/dev/null; do
  attempt=$((attempt + 1))
  
  if [ $attempt -ge $max_attempts ]; then
    echo "PostgreSQL did not become ready in time"
    exit 1
  fi
  
  echo "PostgreSQL is unavailable (attempt $attempt/$max_attempts) - sleeping"
  sleep 2
done

echo "PostgreSQL is ready!"