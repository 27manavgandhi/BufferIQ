# BufferIQ Health Check Script

Write-Host "🏥 BufferIQ Health Check" -ForegroundColor Cyan
Write-Host "========================`n" -ForegroundColor Cyan

# Check 1: Virtual Environment
Write-Host "1. Checking virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "   ✅ Virtual environment exists" -ForegroundColor Green
} else {
    Write-Host "   ❌ Virtual environment not found" -ForegroundColor Red
    Write-Host "   Run: python -m venv venv" -ForegroundColor White
}

# Check 2: Dependencies
Write-Host "`n2. Checking dependencies..." -ForegroundColor Yellow
$packages = @("pandas", "sqlalchemy", "fastapi", "pytest", "alembic")
foreach ($pkg in $packages) {
    $installed = pip list | Select-String -Pattern "^$pkg "
    if ($installed) {
        Write-Host "   ✅ $pkg installed" -ForegroundColor Green
    } else {
        Write-Host "   ❌ $pkg not installed" -ForegroundColor Red
    }
}

# Check 3: Docker Containers
Write-Host "`n3. Checking Docker containers..." -ForegroundColor Yellow
$containers = docker-compose ps --format json | ConvertFrom-Json
foreach ($container in $containers) {
    if ($container.State -eq "running") {
        Write-Host "   ✅ $($container.Name) is running" -ForegroundColor Green
    } else {
        Write-Host "   ❌ $($container.Name) is not running" -ForegroundColor Red
    }
}

# Check 4: Database Connection
Write-Host "`n4. Checking database connection..." -ForegroundColor Yellow
$env:PGPASSWORD = "bufferiq_dev"
$dbCheck = docker-compose exec postgres psql -U bufferiq -d bufferiq -c "\dt" 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "   ✅ Database connected" -ForegroundColor Green
    $tableCount = ($dbCheck | Select-String -Pattern "public \|").Count
    Write-Host "   ✅ Found $tableCount tables" -ForegroundColor Green
} else {
    Write-Host "   ❌ Database connection failed" -ForegroundColor Red
}

# Check 5: Redis Connection
Write-Host "`n5. Checking Redis connection..." -ForegroundColor Yellow
$redisCheck = docker-compose exec redis redis-cli ping 2>&1
if ($redisCheck -match "PONG") {
    Write-Host "   ✅ Redis connected" -ForegroundColor Green
} else {
    Write-Host "   ❌ Redis connection failed" -ForegroundColor Red
}

# Check 6: Sample Data
Write-Host "`n6. Checking sample data..." -ForegroundColor Yellow
$postCount = docker-compose exec postgres psql -U bufferiq -d bufferiq -t -c "SELECT COUNT(*) FROM posts WHERE status='sent';" 2>&1
if ($postCount -match "\d+") {
    $count = [int]($postCount -replace '\s', '')
    if ($count -gt 0) {
        Write-Host "   ✅ Found $count posts" -ForegroundColor Green
    } else {
        Write-Host "   ⚠️  No posts found" -ForegroundColor Yellow
        Write-Host "   Run: python scripts/generate_sample_data.py" -ForegroundColor White
    }
}

# Check 7: Output Directory
Write-Host "`n7. Checking output directory..." -ForegroundColor Yellow
if (Test-Path "outputs/figures") {
    $figCount = (Get-ChildItem "outputs/figures" -Filter *.png).Count
    Write-Host "   ✅ Output directory exists ($figCount figures)" -ForegroundColor Green
} else {
    Write-Host "   ⚠️  Output directory not found" -ForegroundColor Yellow
    Write-Host "   Run: New-Item -ItemType Directory -Force -Path outputs/figures" -ForegroundColor White
}

# Check 8: Tests
Write-Host "`n8. Running quick test..." -ForegroundColor Yellow
cd backend
$testResult = python -m pytest tests/test_config.py -q 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "   ✅ Tests passing" -ForegroundColor Green
} else {
    Write-Host "   ❌ Tests failing" -ForegroundColor Red
}

Write-Host "`n========================" -ForegroundColor Cyan
Write-Host "Health check complete!" -ForegroundColor Cyan
