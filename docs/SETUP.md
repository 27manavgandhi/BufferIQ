# BufferIQ Setup Guide

Complete setup instructions for development environment.

## Prerequisites

Before starting, ensure you have:

- **Python 3.11+** ([Download](https://www.python.org/downloads/))
- **Node.js 20 LTS+** ([Download](https://nodejs.org/))
- **Docker Desktop** ([Download](https://www.docker.com/products/docker-desktop/))
- **Git 2.40+** ([Download](https://git-scm.com/downloads))
- **Visual Studio Code** (recommended) ([Download](https://code.visualstudio.com/))

### Verify Prerequisites
```powershell
python --version  # Should show 3.11 or higher
node --version    # Should show v20 or higher
docker --version  # Should show 24 or higher
git --version     # Should show 2.40 or higher
```

## Initial Setup

### 1. Clone Repository
```powershell
cd C:\path\to\your\projects
git clone https://github.com/27manavgandhi/BufferIQ.git
cd BufferIQ
```

### 2. Create Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**Note**: If you get an execution policy error, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 3. Install Python Dependencies
```powershell
pip install -r backend/requirements.txt
```

This installs:
- FastAPI and dependencies
- Testing tools (pytest, coverage)
- Code quality tools (black, ruff, mypy)
- Pre-commit hooks

### 4. Install Pre-commit Hooks
```powershell
pre-commit install
```

Now, every commit will automatically:
- Format code with black
- Lint with ruff
- Type-check with mypy
- Check for common issues

### 5. Create Environment File
```powershell
copy .env.example .env
```

Edit `.env` and set your values:
```env
# Application
ENVIRONMENT=development
DEBUG=true

# Database
DATABASE_URL=sqlite:///./bufferiq.db

# Buffer API
BUFFER_API_URL=https://graph.buffer.com/graphql
BUFFER_API_KEY=your_buffer_api_key_here

# Cache
REDIS_URL=redis://localhost:6379/0

# ML Models
MODEL_PATH=./models

# Logging
LOG_LEVEL=INFO
```

### 6. Start Infrastructure (Optional)

If you want PostgreSQL and Redis:
```powershell
docker-compose up -d
```

This starts:
- PostgreSQL on port 5432
- Redis on port 6379

**Note**: For Day 1, SQLite is sufficient. PostgreSQL is for future production use.

## Running Tests

### All Tests with Coverage
```powershell
cd backend
pytest tests/ -v --cov=bufferiq --cov-report=term-missing
```

Expected output:
tests/test_config.py::test_settings_from_env PASSED
tests/test_config.py::test_settings_defaults PASSED
tests/test_config.py::test_settings_validation_invalid_env PASSED
...
---------- coverage: platform win32, python 3.11.x -----------
Name                              Stmts   Miss  Cover   Missing
bufferiq/init.py                   1      0   100%
bufferiq/core/init.py              0      0   100%
bufferiq/core/config.py            45     0   100%
TOTAL                              46     0   100%

### Specific Test File
```powershell
pytest tests/test_config.py -v
```

### Watch Mode (Continuous Testing)
```powershell
pip install pytest-watch
ptw tests/
```

## Code Quality Checks

### Format Code
```powershell
black backend/bufferiq/
```

### Lint Code
```powershell
ruff backend/bufferiq/ --fix
```

### Type Check
```powershell
mypy backend/bufferiq/ --strict
```

### Run All Checks
```powershell
# From project root
make lint

# Or manually
black backend/ --check
ruff backend/
mypy backend/ --strict
```

## Validation

Verify everything is working:
```powershell
python -c "from backend.bufferiq.core.config import Settings; s = Settings(); print(f'✅ Config loaded: ENV={s.environment}')"
```

Expected output:
✅ Config loaded: ENV=development

## Development Workflow

### 1. Activate Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
```

### 2. Create Feature Branch
```powershell
git checkout -b feature/day-X-feature-name
```

### 3. Make Changes

Edit code, add tests, update docs.

### 4. Run Tests
```powershell
cd backend
pytest tests/ -v --cov=bufferiq
```

### 5. Check Code Quality
```powershell
black bufferiq/
ruff bufferiq/ --fix
mypy bufferiq/ --strict
```

### 6. Commit Changes

Pre-commit hooks run automatically:
```powershell
git add .
git commit -m "feat: add new feature"
```

If hooks fail, fix issues and commit again.

### 7. Push to GitHub
```powershell
git push origin feature/day-X-feature-name
```

## Troubleshooting

### Python Module Not Found

**Problem**: `ModuleNotFoundError: No module named 'bufferiq'`

**Solution**:
```powershell
# Make sure you're in backend/ directory or PYTHONPATH is set
cd backend
python -c "import bufferiq; print('OK')"
```

### Pre-commit Hooks Failing

**Problem**: Pre-commit hooks fail with formatting issues

**Solution**:
```powershell
# Auto-fix formatting
black backend/
ruff backend/ --fix

# Then commit again
git add .
git commit -m "your message"
```

### Virtual Environment Not Activating

**Problem**: PowerShell execution policy prevents activation

**Solution**:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\venv\Scripts\Activate.ps1
```

### Docker Compose Issues

**Problem**: `docker-compose` command not found

**Solution**:
```powershell
# Use docker compose (v2 syntax)
docker compose up -d
```

### Port Already in Use

**Problem**: Port 5432 or 6379 already in use

**Solution**:
```powershell
# Check what's using the port
netstat -ano | findstr :5432

# Kill the process or change port in docker-compose.yml
```

## IDE Configuration

### VS Code

Install recommended extensions:
- Python (ms-python.python)
- Pylance (ms-python.vscode-pylance)
- Black Formatter (ms-python.black-formatter)
- Ruff (charliermarsh.ruff)
- Better TOML (bungcip.better-toml)

Create `.vscode/settings.json`:
```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/Scripts/python.exe",
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "black",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["backend/tests"],
  "editor.formatOnSave": true,
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    }
  }
}
```

### PyCharm

1. Open Project Settings
2. Set Python Interpreter to `venv/Scripts/python.exe`
3. Enable pytest as test runner
4. Configure black as external tool
5. Enable mypy plugin

## Next Steps

After setup is complete:

1. ✅ Verify all tests pass
2. ✅ Verify code quality checks pass
3. ✅ Read [ARCHITECTURE.md](ARCHITECTURE.md)
4. ✅ Start Day 2: Development Environment

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/27manavgandhi/BufferIQ/issues)
- **Discussions**: [GitHub Discussions](https://github.com/27manavgandhi/BufferIQ/discussions)
- **Documentation**: [docs/](.)

---