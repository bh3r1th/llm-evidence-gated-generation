# EGA Release Checklist (Minimal)

Use this for manual v3 release prep.

1. Run tests:
   - `pytest -q`
2. Verify package import/API quickly:
   - `python -c "from ega import verify_answer, PipelineConfig, PolicyConfig; print('ok')"`
3. Build artifacts:
   - `python -m build`
4. Bump version in both canonical package/version locations:
   - `pyproject.toml` (`[project].version`)
   - `src/ega/__init__.py` (`__version__`)
5. Commit version bump and create tag:
   - `git tag -a vX.Y.Z -m "EGA vX.Y.Z"`
6. Draft GitHub release notes and attach/publish package artifacts as needed.
