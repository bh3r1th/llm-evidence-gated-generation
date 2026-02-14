# Contributing to EGA

## Development Setup

### Bash / zsh
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,nli]"
```

### Windows PowerShell
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,nli]"
```

## Lint, Format, Test

```bash
make format
make lint
make test
```

Or explicitly:
```bash
python -m ruff format .
python -m ruff check .
python scripts/pytest_wrapper.py -q
```

## Reproduce Pipeline Demo

### Bash / zsh
```bash
bash examples/run_pipeline_demo.sh
```

### Windows PowerShell
```powershell
python -m ega.cli pipeline `
  --llm-summary-file examples/pipeline_demo/llm_summary.txt `
  --evidence-json examples/pipeline_demo/evidence.json `
  --scores-jsonl examples/pipeline_demo/scores.jsonl `
  --unitizer sentence `
  --partial-allowed
```

## Pull Request Checklist

- [ ] Code is scoped and minimal.
- [ ] `make lint` passes.
- [ ] `make test` passes.
- [ ] New/changed behavior includes tests.
- [ ] Docs updated when flags/contracts changed.
- [ ] Changelog note added if your process uses one (optional).
