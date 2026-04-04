# BufferIQ System Architecture

## Overview

BufferIQ implements a strict layered architecture following SOLID principles and domain-driven design patterns. The system is designed for scalability, maintainability, and testability.

## Architectural Principles

### 1. Layered Architecture

Presentation Layer → Application Layer → Domain Layer → Infrastructure Layer

Each layer has clear responsibilities and dependencies flow downward only (Dependency Rule).

### 2. SOLID Principles

**Single Responsibility Principle (SRP)**
- Each module/class has one reason to change
- Example: `Config` class only handles configuration, not validation logic

**Open/Closed Principle (OCP)**
- Open for extension, closed for modification
- Example: Feature extractors can be added without modifying existing code

**Liskov Substitution Principle (LSP)**
- Subtypes are substitutable for base types
- Example: Any `FeatureExtractor` can replace another in the pipeline

**Interface Segregation Principle (ISP)**
- Many specific interfaces > one general interface
- Example: Separate protocols for `Predictor`, `Optimizer`, `Analyzer`

**Dependency Inversion Principle (DIP)**
- Depend on abstractions, not concretions
- Example: Services depend on repository interfaces, not implementations

### 3. Design Patterns

- **Repository Pattern**: Data access abstraction
- **Service Pattern**: Business logic encapsulation
- **Factory Pattern**: Object creation (ML models, feature extractors)
- **Strategy Pattern**: Interchangeable algorithms (ML models)
- **Observer Pattern**: Event-driven updates (model retraining)

## System Components

### Presentation Layer

**MCP Server (TypeScript)**
- Handles Claude Desktop integration
- Translates user intents to API calls
- Formats responses for conversational UI
- No business logic

**Responsibilities**:
- Tool schema definition
- Input validation
- Response formatting
- Error translation

### Application Layer

**FastAPI Services (Python)**
- REST API endpoints
- Request/response handling
- Service orchestration
- Authentication/authorization

**Key Services**:
- `PredictionService`: Engagement prediction orchestration
- `OptimizationService`: Timing optimization logic
- `AnalysisService`: Content intelligence coordination
- `SyncService`: Data synchronization management

**Responsibilities**:
- HTTP routing
- Input validation (Pydantic)
- Service coordination
- Transaction management

### Domain Layer

**Core Business Logic**
- ML models and pipelines
- Feature extraction
- Buffer API client
- Domain entities

**Key Components**:
- `EngagementPredictor`: ML model wrapper
- `TimingOptimizer`: Scheduling algorithm
- `VoiceAnalyzer`: Style analysis
- `BufferClient`: API integration
- `FeatureExtractor`: Feature engineering

**Responsibilities**:
- Business rules
- Domain logic
- Model training/inference
- External API integration

### Infrastructure Layer

**Data & External Services**
- Database (PostgreSQL/SQLite)
- Cache (Redis)
- File system (model storage)
- External APIs (Buffer)

**Key Components**:
- `DatabaseSession`: SQLAlchemy session management
- `CacheManager`: Redis cache wrapper
- `ModelRegistry`: Model versioning and storage
- `MigrationManager`: Alembic migrations

**Responsibilities**:
- Data persistence
- Caching
- External service communication
- Infrastructure concerns

## Data Flow

### Prediction Request Flow
### Prediction Request Flow

User asks Claude for engagement prediction
↓
MCP Server calls POST /api/v1/predict
↓
PredictionService validates input
↓
FeatureExtractor extracts features from post
↓
EngagementPredictor loads model and predicts
↓
PredictionRepository saves prediction
↓
Response formatted and returned to MCP
↓
Claude presents prediction to user


### Data Sync Flow

SyncService triggered (scheduled/manual)
↓
BufferClient fetches posts from Buffer API
↓
Rate limiter enforces 100/15min limit
↓
Posts validated and transformed
↓
Repository batch inserts/updates database
↓
Cache invalidated for affected data
↓
Sync job status updated


## Configuration Management

### Settings Hierarchy

Environment variables (.env)
↓
Config class (type-safe validation)
↓
Dependency injection (services receive config)


### Configuration Layers

- **Development**: SQLite, in-memory cache, debug logging
- **Testing**: In-memory database, mock services
- **Production**: PostgreSQL, Redis, structured logging

## Error Handling Strategy

### Error Categories

1. **Client Errors (4xx)**: Invalid input, not found, unauthorized
2. **Server Errors (5xx)**: Unexpected failures, external service down
3. **Domain Errors**: Business logic violations
4. **Infrastructure Errors**: Database, cache, file system failures

### Error Handling Pattern
```python
try:
    result = await service.operation()
except DomainError as e:
    logger.warning("Business rule violation", error=str(e))
    raise HTTPException(status_code=400, detail=str(e))
except InfrastructureError as e:
    logger.error("Infrastructure failure", error=str(e))
    raise HTTPException(status_code=503, detail="Service temporarily unavailable")
except Exception as e:
    logger.exception("Unexpected error")
    raise HTTPException(status_code=500, detail="Internal server error")
```

## Performance Optimization

### Caching Strategy

- **L1 Cache**: In-memory LRU (fast, single-process)
- **L2 Cache**: Redis (shared, persistent)
- **Cache Keys**: Hierarchical (user:123:posts, user:123:profile)
- **TTL**: Based on data volatility (5min-1hour)

### Database Optimization

- **Indexes**: All foreign keys, frequently queried columns
- **Connection Pooling**: Reuse connections (SQLAlchemy)
- **Batch Operations**: Bulk inserts/updates
- **Pagination**: Cursor-based for large datasets

### API Rate Limiting

- **Buffer API**: 100 requests/15min (enforced by rate limiter)
- **Local API**: Token bucket algorithm (configurable)
- **Backoff**: Exponential backoff on rate limit hits

## Security Considerations

### Data Protection

- **Encryption at Rest**: Sensitive fields encrypted (AES-256)
- **Encryption in Transit**: HTTPS only (TLS 1.3)
- **Secrets Management**: Environment variables, never in code
- **API Keys**: Hashed before storage

### Input Validation

- **Pydantic Models**: Type validation on all inputs
- **SQL Injection**: Prevented by ORM (parameterized queries)
- **XSS**: Not applicable (API only, no HTML rendering)
- **Rate Limiting**: Prevent abuse

## Scalability Design

### Horizontal Scaling

- **Stateless Services**: No shared state between instances
- **Database Connection Pooling**: Multiple app instances share pool
- **Cache Sharing**: Redis as distributed cache
- **Load Balancing**: Ready for reverse proxy (nginx/Caddy)

### Vertical Scaling

- **Async I/O**: Non-blocking operations (aiohttp, asyncpg)
- **Batch Processing**: Reduce per-operation overhead
- **Lazy Loading**: Load data only when needed
- **Memory Efficiency**: Generators for large datasets

## Monitoring & Observability

### Logging

- **Structured Logging**: JSON format for parsing
- **Log Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Context**: Request ID, user ID, correlation ID
- **Retention**: 30 days (configurable)

### Metrics

- **Request Metrics**: Latency, throughput, error rate
- **ML Metrics**: Prediction accuracy, inference time
- **System Metrics**: CPU, memory, disk usage
- **Business Metrics**: Predictions made, users active

### Health Checks

- **Liveness**: Is the service running?
- **Readiness**: Is the service ready to handle requests?
- **Dependency Health**: Database, cache, external APIs

## Testing Strategy

### Test Pyramid
     /\
    /  \  E2E Tests (5%)
   /────\
  / Unit \ Integration Tests (25%)
 /  Tests \
/──────────\ Unit Tests (70%)

### Test Types

- **Unit Tests**: Pure functions, business logic
- **Integration Tests**: Database, cache, external APIs
- **E2E Tests**: Full user workflows (MCP → API → DB)
- **Performance Tests**: Load testing, benchmarking

### Test Patterns

- **Arrange-Act-Assert**: Standard test structure
- **Fixtures**: Reusable test data (pytest fixtures)
- **Mocks**: External dependencies (unittest.mock)
- **Factories**: Test object creation (factory_boy)

## Deployment Architecture

### Development
Developer Machine
├── Python Backend (localhost:8000)
├── PostgreSQL (Docker)
├── Redis (Docker)
└── MCP Server (stdio)

### Production (Future)
Load Balancer (Caddy/nginx)
       ↓
Backend Instances (N)
       ↓
PostgreSQL (Managed Service)
       ↓
Redis (Managed Service)

## Technology Decisions

### Why FastAPI?

- Async-native (high concurrency)
- Automatic OpenAPI docs
- Type validation (Pydantic)
- High performance (comparable to Node.js)

### Why SQLAlchemy 2.0?

- Async support
- Type safety (Python 3.11+)
- Mature ecosystem
- Easy migrations (Alembic)

### Why XGBoost/LightGBM?

- Production-proven
- Fast inference (< 100ms)
- Good accuracy/speed tradeoff
- Not overkill (vs deep learning)

### Why Redis?

- Fast key-value store
- Persistence optional
- Simple API
- Easy Docker deployment

## Future Enhancements

### Phase 2 (Days 31-45)

- Multi-account support
- Webhook system
- Advanced analytics
- A/B testing framework

### Phase 3 (Days 46-60)

- Ensemble models
- Continuous learning
- Real-time predictions
- Mobile optimization

## References

- [SOLID Principles](https://en.wikipedia.org/wiki/SOLID)
- [Clean Architecture](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
- [Domain-Driven Design](https://martinfowler.com/bliki/DomainDrivenDesign.html)
- [FastAPI Best Practices](https://fastapi.tiangolo.com/tutorial/)
- [SQLAlchemy 2.0 Tutorial](https://docs.sqlalchemy.org/en/20/tutorial/)

---