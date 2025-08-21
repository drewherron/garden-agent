.PHONY: help install install-dev test lint format type-check clean run

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install production dependencies
	pip install -r requirements.txt

install-dev:  ## Install development dependencies
	pip install -r requirements.txt -r requirements-dev.txt
	pip install -e .

test:  ## Run tests
	pytest tests/

lint:  ## Run linting
	ruff check src/ tests/
	black --check src/ tests/

format:  ## Format code
	black src/ tests/
	ruff check --fix src/ tests/

type-check:  ## Run type checking
	mypy src/

clean:  ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

run:  ## Run the application
	python -m garden_agent.cli