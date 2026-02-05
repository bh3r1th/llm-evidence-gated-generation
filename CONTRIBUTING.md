# Contributing to ega

Thanks for your interest in contributing to `ega`.

## Development setup

1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   make setup
   ```

## Quality checks

Before opening a pull request:

```bash
make format
make lint
make test
```

## Scope guidance

EGA is an enforcement/decision layer. Contributions should focus on:

- Evidence-gated answer policy logic.
- Decision representations and enforcement mechanisms.
- Integration points for evidence verification.

Out of scope for this repository:

- Standalone evaluator frameworks.
- Analytics dashboards.
