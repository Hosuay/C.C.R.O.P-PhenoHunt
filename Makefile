.PHONY: help install test clean lint format

help:
	@echo "PhenoHunter - Development Commands"
	@echo ""
	@echo "make install     - Install package in development mode"
	@echo "make test        - Run tests"
	@echo "make clean       - Clean generated files"
	@echo "make lint        - Run code quality checks"
	@echo "make format      - Format code with black"
	@echo "make examples    - Run example workflows"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -v

test-cli:
	python tests/test_cli.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*~" -delete
	rm -rf build/ dist/ *.egg-info .pytest_cache .coverage htmlcov/
	rm -f /*.csv
	rm -f f1_*.csv f2_*.csv bx*.csv *_hybrid.csv *_population.csv

lint:
	flake8 src/ tests/ --max-line-length=100 --ignore=E501,W503 || true
	pylint src/ --disable=all --enable=E,F || true

format:
	black src/ tests/ examples/ --line-length=100 || echo "black not installed"
	isort src/ tests/ examples/ || echo "isort not installed"

examples:
	@echo "Creating sample data..."
	python examples/create_sample_data.py
	@echo ""
	@echo "Running example workflow..."
	bash examples/example_cli_workflow.sh
