# BufferIQ: The Intelligence Layer for Buffer

[![CI](https://github.com/27manavgandhi/BufferIQ/workflows/CI/badge.svg)](https://github.com/27manavgandhi/BufferIQ/actions)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](https://github.com/27manavgandhi/BufferIQ)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](http://mypy-lang.org/)

**Stop Guessing, Start Knowing**

BufferIQ is a production-grade, ML-powered intelligence platform built on top of Buffer's API that predicts post engagement, optimizes timing, and provides content intelligence through continuous learning.

## 🎯 What BufferIQ Does

- **Engagement Prediction**: Predict post performance before publishing using ML models (R² > 0.75 target)
- **Timing Optimization**: Recommend optimal posting schedules based on audience behavior patterns
- **Content Intelligence**: Analyze voice consistency, identify content gaps, suggest improvements
- **Continuous Learning**: Self-improving models that get better with every post

## 🚀 Quick Start
```powershell
# Clone repository
git clone https://github.com/27manavgandhi/BufferIQ.git
cd BufferIQ

# Setup environment
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt

# Run tests
cd backend
pytest tests/ -v --cov=bufferiq

# Start development
cd ..
docker-compose up -d
```

## 📋 Prerequisites

- Python 3.11+
- Node.js 20 LTS+ (for MCP server)
- Docker & Docker Compose
- Git 2.40+

## 🏗️ Architecture

BufferIQ follows a strict layered architecture with clear separation of concerns:

┌─────────────────────────────────────────┐
│     Presentation Layer (MCP Server)     │
│          TypeScript/Node.js             │
└────────────────┬────────────────────────┘
                 │ REST API
┌────────────────▼────────────────────────┐
│      Application Layer (FastAPI)        │
│  Services: Predict, Optimize, Analyze   │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│          Domain Layer (Core)            │
│   ML Engine, Buffer Client, Models      │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│    Infrastructure Layer (Data)          │
│  PostgreSQL, Redis, File System         │
└─────────────────────────────────────────┘

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed system design.

## 🛠️ Technology Stack

**Backend (Python 3.11+)**
- FastAPI 0.104+ (async web framework)
- SQLAlchemy 2.0+ (ORM with async support)
- XGBoost 2.0+ / LightGBM 4.0+ (ML models)
- spaCy 3.7+ / sentence-transformers 2.2+ (NLP)
- Redis 7.0+ (caching)
- PostgreSQL 15+ (database)

**MCP Server (TypeScript/Node.js)**
- Node.js 20 LTS+
- TypeScript 5.3+
- Anthropic MCP SDK

**Development Tools**
- black (formatting)
- ruff (linting)
- mypy (type checking)
- pytest (testing)
- pre-commit (git hooks)
- GitHub Actions (CI/CD)

## 📊 Project Status

**Current Phase**: Day 1 - Foundation & Architecture
**Progress**: 1/60 days (1.7%)
**Next Milestone**: Day 7 - Exploratory Data Analysis

## 🧪 Testing
```powershell
# Run all tests with coverage
cd backend
pytest tests/ -v --cov=bufferiq --cov-report=term-missing

# Run specific test file
pytest tests/test_config.py -v

# Type checking
mypy bufferiq/ --strict

# Linting
ruff bufferiq/
black bufferiq/ --check
```

## 📈 Quality Standards

- **Test Coverage**: 80%+ (currently 100% on config)
- **Type Safety**: 100% type hints (mypy --strict)
- **Code Style**: black + ruff (zero warnings)
- **Performance**: < 500ms API response time (p95)
- **ML Accuracy**: R² > 0.75 (target)

## 🔒 Design Principles

1. **SOLID Principles**: Single responsibility, dependency inversion throughout
2. **Local-First**: All user data stored locally (privacy-first)
3. **Fail-Safe**: Graceful degradation, comprehensive error handling
4. **Async-Native**: Non-blocking I/O for all network operations
5. **Observable**: Structured logging, metrics, health checks

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development workflow, code standards, and PR process.

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

Built during a 60-day build-in-public journey to demonstrate senior-level engineering capabilities.

**Follow the journey**: [#BufferIQ](https://www.linkedin.com/feed/hashtag/bufferiq) on LinkedIn

## 🔗 Links

- **Documentation**: [docs/](docs/)
- **Setup Guide**: [docs/SETUP.md](docs/SETUP.md)
- **Architecture**: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **Issues**: [GitHub Issues](https://github.com/27manavgandhi/BufferIQ/issues)

---

**Built with 💙 by [Manav Gandhi](https://github.com/27manavgandhi)**
