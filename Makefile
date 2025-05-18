.PHONY: format lint lint-fix all help

help:
	@echo "Available commands:"
	@echo "  make format     - Format code using Black"
	@echo "  make lint       - Check code with Ruff linter"
	@echo "  make lint-fix   - Fix linting issues with Ruff"
	@echo "  make all        - Run all formatting and linting tools"

format:
	@echo "Formatting code with Black..."
	uv run black --line-length 120 .

lint:
	@echo "Checking code with Ruff..."
	uv run ruff check .

lint-fix:
	@echo "Fixing code with Ruff..."
	uv run ruff check --fix .

all: format lint-fix
	@echo "All formatting and linting complete!" 
