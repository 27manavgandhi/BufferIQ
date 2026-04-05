# Database Architecture & Conventions

Complete guide to BufferIQ's database design, conventions, and best practices.

## Overview

BufferIQ uses SQLAlchemy 2.0 with async support for database operations:
- **Development**: SQLite (fast, zero-config)
- **Production**: PostgreSQL 15+ (scalable, feature-rich)
- **Migrations**: Alembic (version-controlled schema changes)

## Architecture

### Connection Management
Application
↓
DatabaseManager (singleton)
↓
AsyncEngine (connection pool)
↓
AsyncSession (transactions)
↓
Database (PostgreSQL/SQLite)

### Connection Pooling

**SQLite (Development)**
- NullPool: No connection pooling
- Reason: SQLite is file-based, pooling not beneficial

**PostgreSQL (Production)**
- QueuePool: Connection reuse
- Pool size: 5 connections
- Max overflow: 10 additional connections
- Pre-ping: Verify connection health
- Recycle: 3600 seconds (1 hour)

## Database Conventions

### Table Naming

- Lowercase with underscores: `users`, `social_posts`
- Plural nouns: `channels` not `channel`
- Join tables: `user_organizations`

### Column Naming

- Lowercase with underscores: `created_at`, `user_id`
- Boolean columns: `is_active`, `has_feature`
- Foreign keys: `{table}_id` (e.g., `user_id`)
- Timestamps: `{action}_at` (e.g., `published_at`)

### Primary Keys

- Auto-incrementing integers: `id`
- Type: `Mapped[int]`
- Constraint: `primary_key=True`

### Foreign Keys

- Always indexed: `index=True`
- Cascade deletes where appropriate: `ondelete="CASCADE"`
- Named constraints: `fk_{table}_{column}`

### Indexes

**Required on:**
- All foreign keys
- Frequently queried columns (status, dates)
- Unique constraints (email, external_ids)

**Composite indexes:**
- For multi-column queries
- Order: High cardinality first

**Example:**
```python
__table_args__ = (
    Index('idx_post_channel_status', 'channel_id', 'status'),
    Index('idx_post_published', 'published_at', 'status'),
)
```

### Timestamps

**Required on all tables:**
- `created_at`: When record was created
- `updated_at`: When record was last modified

**Implementation:**
```python
created_at: Mapped[datetime] = mapped_column(default=func.now())
updated_at: Mapped[datetime] = mapped_column(
    default=func.now(),
    onupdate=func.now()
)
```

### Constraints

**NOT NULL:**
- All required fields
- Use `Optional[Type]` for nullable columns

**UNIQUE:**
- Email addresses
- External IDs (buffer_post_id, buffer_channel_id)
- Natural keys

**CHECK:**
- Enum validation
- Range validation (e.g., confidence 0-1)

## Model Definition Pattern
```python
from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from backend.bufferiq.core.database import Base

class User(Base):
    __tablename__ = "users"
    
    # Primary key
    id: Mapped[int] = mapped_column(primary_key=True)
    
    # Required fields
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255))
    
    # Optional fields
    bio: Mapped[Optional[str]] = mapped_column(Text, default=None)
    
    # Foreign keys
    organization_id: Mapped[int] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"),
        index=True
    )
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        default=func.now(),
        onupdate=func.now()
    )
    
    # Relationships
    organization: Mapped["Organization"] = relationship(back_populates="users")
    
    # Indexes
    __table_args__ = (
        Index('idx_user_org_email', 'organization_id', 'email'),
    )
```

## Query Patterns

### Select
```python
from sqlalchemy import select

async with db_manager.session() as session:
    stmt = select(User).where(User.email == "test@example.com")
    result = await session.execute(stmt)
    user = result.scalar_one_or_none()
```

### Insert
```python
async with db_manager.session() as session:
    user = User(email="test@example.com", name="Test User")
    session.add(user)
    await session.flush()  # Get ID before commit
    user_id = user.id
```

### Update
```python
from sqlalchemy import update

async with db_manager.session() as session:
    stmt = (
        update(User)
        .where(User.id == user_id)
        .values(name="Updated Name")
    )
    await session.execute(stmt)
```

### Delete
```python
from sqlalchemy import delete

async with db_manager.session() as session:
    stmt = delete(User).where(User.id == user_id)
    await session.execute(stmt)
```

### Join
```python
async with db_manager.session() as session:
    stmt = (
        select(User, Organization)
        .join(Organization, User.organization_id == Organization.id)
        .where(Organization.name == "ACME Corp")
    )
    result = await session.execute(stmt)
    users_orgs = result.all()
```

## Transaction Management

### Automatic (Recommended)
```python
async with db_manager.session() as session:
    # Automatic commit on success
    user = User(email="test@example.com")
    session.add(user)
    # Automatic rollback on exception
```

### Manual
```python
async with db_manager.session() as session:
    async with session.begin():
        user = User(email="test@example.com")
        session.add(user)
        # Explicit commit control
```

## Migration Workflow

### 1. Create Models
```python
# backend/bufferiq/domain/models.py
class NewTable(Base):
    __tablename__ = "new_table"
    id: Mapped[int] = mapped_column(primary_key=True)
    # ... fields
```

### 2. Generate Migration
```bash
alembic revision --autogenerate -m "add new_table"
```

### 3. Review Migration
```python
# alembic/versions/XXXX_add_new_table.py
def upgrade() -> None:
    op.create_table(
        'new_table',
        sa.Column('id', sa.Integer(), nullable=False),
        # ... columns
    )
```

### 4. Apply Migration
```bash
alembic upgrade head
```

### 5. Test Rollback
```bash
alembic downgrade -1
alembic upgrade head
```

## Performance Optimization

### Query Optimization

**Use select() over Query API:**
```python
# Good
stmt = select(User).where(User.id == 1)

# Avoid (deprecated in 2.0)
session.query(User).filter_by(id=1)
```

**Load relationships efficiently:**
```python
from sqlalchemy.orm import selectinload

stmt = select(User).options(selectinload(User.posts))
```

**Limit results:**
```python
stmt = select(Post).limit(100).offset(0)
```

### Index Strategy

- Index foreign keys: Always
- Index WHERE clauses: Frequently queried columns
- Index ORDER BY: Sorted columns
- Composite indexes: Multi-column queries

### Batch Operations

**Bulk insert:**
```python
users = [User(email=f"user{i}@example.com") for i in range(1000)]
session.add_all(users)
```

**Bulk update:**
```python
stmt = update(User).where(User.is_active == False).values(deleted_at=func.now())
await session.execute(stmt)
```

## Testing

### Test Database Setup
```python
@pytest.fixture
async def db_session():
    # Use in-memory SQLite for tests
    settings = Settings(database_url="sqlite:///:memory:")
    manager = DatabaseManager(settings)
    await manager.connect()
    await init_database(manager.engine)
    
    async with manager.session() as session:
        yield session
    
    await manager.disconnect()
```

### Transaction Rollback
```python
@pytest.fixture
async def db_session():
    async with db_manager.session() as session:
        async with session.begin():
            yield session
            # Rollback after test
            await session.rollback()
```

## Monitoring

### Health Checks
```python
health_ok = await check_database_health(engine)
if not health_ok:
    logger.error("Database health check failed")
```

### Connection Pool Stats
```python
pool = engine.pool
print(f"Pool size: {pool.size()}")
print(f"Checked out: {pool.checkedout()}")
print(f"Overflow: {pool.overflow()}")
```

### Query Logging

Development only (set in settings):
```python
settings = Settings(debug=True, environment=Environment.DEVELOPMENT)
# Logs all SQL queries
```

## Security

### SQL Injection Prevention

**Always use parameterized queries:**
```python
# Good
stmt = select(User).where(User.email == email)

# Never do this
stmt = text(f"SELECT * FROM users WHERE email = '{email}'")
```

### Sensitive Data

- Encrypt API keys, tokens
- Hash passwords (never store plain text)
- Use environment variables for credentials

## Troubleshooting

### Connection Issues
```python
# Check health
await check_database_health(engine)

# Verify URL
print(engine.url)

# Test connection
async with engine.connect() as conn:
    result = await conn.execute(text("SELECT 1"))
```

### Migration Conflicts
```bash
# Check current state
alembic current
alembic heads

# Merge conflicts
alembic merge heads

# Reset if needed
alembic downgrade base
alembic upgrade head
```

### Pool Exhaustion
```python
# Increase pool size
engine = create_async_engine(
    url,
    pool_size=10,
    max_overflow=20
)
```

---