# Sync Service Documentation

## Overview

The BufferIQ sync service synchronizes data from the Buffer API to the local database. It supports both initial sync (all data) and incremental sync (only updates).

## Architecture

```text
SyncService
├── Initial Sync - Fetch all organizations, channels, posts
├── Incremental Sync - Fetch only updates since last sync
├── Transformers - Convert API data to database models
├── Progress Tracker - Track sync job status and progress
└── Error Recovery - Resume capability on failure
```

## Features

### Initial Sync
- Fetches all organizations for user
- Fetches all channels for each organization
- Fetches all posts for each channel (paginated)
- Handles large datasets efficiently
- Tracks progress in real-time
- Resume capability on failure

### Incremental Sync
- Fetches only new/updated posts since last sync
- Updates engagement metrics for existing posts
- Minimal API calls
- Fast execution (< 30 seconds typical)

### Pagination
- Cursor-based pagination
- Batch size: 100 items per request
- Memory-efficient streaming
- Progress reporting per batch

### Data Transformation
- Buffer API format → SQLAlchemy models
- Field validation and type conversion
- Null handling and defaults
- Engagement metrics calculation
- Content hash generation for deduplication

### Conflict Resolution
- Upsert strategy (INSERT ON CONFLICT UPDATE)
- Timestamp-based conflict resolution
- Updates only if newer data available

### Progress Tracking
- Database-backed job tracking
- Real-time progress updates
- ETA calculation
- Error logging
- Queryable sync history

## Usage

### CLI Commands

```bash
# Initial sync
python -m bufferiq.cli.sync initial --user-id=1

# Incremental sync
python -m bufferiq.cli.sync incremental --user-id=1

# Check status
python -m bufferiq.cli.sync status --user-id=1

# View history
python -m bufferiq.cli.sync history --user-id=1 --limit=10
```

### Programmatic Usage

```python
from bufferiq.infrastructure.sync import SyncService
from bufferiq.infrastructure.sync import BufferTransformer
from bufferiq.infrastructure.sync import ProgressTracker

# Create service
service = SyncService(session, client, transformer, tracker)

# Run initial sync
job_id = await service.initial_sync(user_id=1)

# Run incremental sync
job_id = await service.incremental_sync(user_id=1)
```

## Database Schema

### Sync Jobs Table

```sql
CREATE TABLE sync_jobs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    sync_type VARCHAR(50) NOT NULL,  -- 'initial', 'incremental'
    status VARCHAR(50) NOT NULL,      -- 'pending', 'running', 'completed', 'failed'
    total_items INTEGER,
    processed_items INTEGER DEFAULT 0,
    failed_items INTEGER DEFAULT 0,
    metadata JSONB,
    error_message TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);
```

## Performance

- **Initial Sync**: Handles 10,000 posts in < 10 minutes
- **Incremental Sync**: Completes in < 30 seconds
- **Memory Usage**: < 200MB for 10,000 posts
- **API Calls**: Batched (100 items per call)
- **Database Writes**: Batched commits every 100 items

## Error Handling

### Handled Errors
- API rate limits (wait and retry)
- Network failures (retry with backoff)
- Database failures (rollback and mark failed)
- Invalid data (log and skip)
- Duplicate posts (upsert)

### Resume Capability
- Checkpoint saving after each batch
- Resume from last successful batch
- Transaction rollback on failure

## Monitoring

### Sync Job Status

```sql
SELECT * FROM sync_jobs
WHERE user_id = 1
ORDER BY created_at DESC
LIMIT 10;
```

### Progress Tracking

```sql
SELECT
    id,
    sync_type,
    status,
    processed_items,
    total_items,
    (processed_items::float / NULLIF(total_items, 0) * 100)::int AS progress_pct
FROM sync_jobs
WHERE status = 'running';
```

## Best Practices

1. **Run incremental sync frequently** (every 15-30 minutes)
2. **Monitor sync job status** via CLI or database
3. **Review failed jobs** and retry as needed
4. **Use initial sync sparingly** (only when needed)
5. **Check API rate limits** before running multiple syncs

## Troubleshooting

### Sync Job Fails
- Check error_message in sync_jobs table
- Verify Buffer API key is valid
- Check network connectivity
- Review application logs

### Slow Performance
- Check database indexes
- Monitor API rate limits
- Review batch size configuration
- Check memory usage

### Missing Data
- Verify user has access to organizations/channels
- Check sync job completed successfully
- Review transformation errors in logs
