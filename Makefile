.PHONY: help setup install install-dev lint format test build clean

help:
	@echo "Targets:"
	@echo "  install      Install package in editable mode"
	@echo "  install-dev  Install editable package with dev+nli extras"
	@echo "  lint         Run ruff + mypy"
	@echo "  format       Run ruff format"
	@echo "  test         Run pytest wrapper"
	@echo "  build        Build sdist/wheel (if build module installed)"
	@echo "  clean        Remove common build/cache artifacts"

setup: install-dev

install:
	python -m pip install --upgrade pip
	python -m pip install -e .

install-dev:
	python -m pip install --upgrade pip
	python -m pip install -e ".[dev,nli]"

lint:
	python -c "import importlib.util,subprocess,sys; missing=[m for m in ('ruff','mypy') if importlib.util.find_spec(m) is None]; sys.exit((print('Missing tools: '+', '.join(missing)+'. Run: make install-dev') or 1) if missing else subprocess.call([sys.executable,'-m','ruff','check','.']) or subprocess.call([sys.executable,'-m','mypy','src']))"

format:
	python -c "import importlib.util,subprocess,sys; sys.exit((print('Missing tool: ruff. Run: make install-dev') or 1) if importlib.util.find_spec('ruff') is None else subprocess.call([sys.executable,'-m','ruff','format','.']))"

test:
	python scripts/pytest_wrapper.py -q

build:
	python -c "import importlib.util,subprocess,sys; sys.exit((print('Missing tool: build. Install with: python -m pip install build') or 1) if importlib.util.find_spec('build') is None else subprocess.call([sys.executable,'-m','build']))"

clean:
	python -c "import pathlib,shutil; [shutil.rmtree(p, ignore_errors=True) for p in ('build','dist','.pytest_cache','.ruff_cache','.mypy_cache','htmlcov','.tmp')]; [shutil.rmtree(str(p), ignore_errors=True) for p in pathlib.Path('.').glob('*.egg-info')]"
