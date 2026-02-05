.PHONY: setup test lint format

setup:
	python -m pip install --upgrade pip
	python -m pip install -e .[dev]

test:
	pytest

lint:
	ruff check .
	mypy src

format:
	ruff format .
