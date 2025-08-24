# Makefile for Crypto Arbitrage Trading Platform

.PHONY: help install test lint format clean run demo docker-build docker-up docker-down

# Default target
help:
	@echo "Crypto Arbitrage Trading Platform - Make Commands"
	@echo "================================================"
	@echo ""
	@echo "Development:"
	@echo "  install    - Install Python dependencies"
	@echo "  test       - Run unit tests"
	@echo "  lint       - Run code linting (flake8, mypy)"
	@echo "  format     - Format code (black, isort)"
	@echo "  clean      - Clean temporary files"
	@echo ""
	@echo "Running:"
	@echo "  demo       - Run market data demo"
	@echo "  run        - Run the trading system"
	@echo "  dashboard  - Start Streamlit dashboard"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build  - Build Docker containers"
	@echo "  docker-up     - Start all services"
	@echo "  docker-down   - Stop all services"
	@echo ""
	@echo "Database:"
	@echo "  db-init    - Initialize database"
	@echo "  db-migrate - Run database migrations"

# Development commands
install:
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt
	pip install -e .

test:
	@echo "Running tests..."
	pytest arbi/tests/ -v --cov=arbi --cov-report=html --cov-report=term

lint:
	@echo "Running linting..."
	flake8 arbi/
	mypy arbi/

format:
	@echo "Formatting code..."
	black arbi/
	isort arbi/

clean:
	@echo "Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf dist
	rm -rf build

# Running commands
demo:
	@echo "Starting market data demo..."
	python demo.py

run:
	@echo "Starting trading system..."
	python -m arbi.api.server

dashboard:
	@echo "Starting Streamlit dashboard..."
	streamlit run arbi/ui/dashboard.py

# Docker commands
docker-build:
	@echo "Building Docker containers..."
	docker-compose build

docker-up:
	@echo "Starting all services..."
	docker-compose up -d

docker-down:
	@echo "Stopping all services..."
	docker-compose down

docker-logs:
	@echo "Showing logs..."
	docker-compose logs -f

# Database commands
db-init:
	@echo "Initializing database..."
	python -c "from arbi.core.storage import init_database; init_database()"

db-migrate:
	@echo "Running database migrations..."
	alembic upgrade head

# Environment setup
env-setup:
	@echo "Setting up environment..."
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env file from template. Please edit it with your settings."; \
	else \
		echo ".env file already exists."; \
	fi

# Full setup for new developers
setup: env-setup install db-init
	@echo "Development environment setup complete!"
	@echo "Next steps:"
	@echo "1. Edit .env file with your API keys"
	@echo "2. Run 'make demo' to test the data feed"
	@echo "3. Run 'make run' to start the trading system"

# Production deployment
deploy:
	@echo "Deploying to production..."
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Monitoring
logs:
	@echo "Showing application logs..."
	tail -f logs/arbitrage.log

monitor:
	@echo "Opening monitoring dashboard..."
	@echo "Prometheus: http://localhost:9090"
	@echo "Grafana: http://localhost:3000 (admin/admin123)"
	@echo "API Docs: http://localhost:8000/docs"

# Security scan
security:
	@echo "Running security scan..."
	bandit -r arbi/
	safety check

# Performance profiling
profile:
	@echo "Running performance profiling..."
	python -m cProfile -o profile.stats demo.py
	python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"

# Backup data
backup:
	@echo "Creating backup..."
	mkdir -p backups/$(date +%Y%m%d_%H%M%S)
	cp -r data/ backups/$(date +%Y%m%d_%H%M%S)/
	cp -r logs/ backups/$(date +%Y%m%d_%H%M%S)/

# Check system health
health:
	@echo "Checking system health..."
	curl -f http://localhost:8000/health || echo "API server not running"
	docker-compose ps
