# EGA Release Checklist (Minimal)

Use this for manual alpha release prep.

1. Run tests:
   - `pytest -q`
2. Verify package import/API quickly:
   - `python -c "from ega import verify_answer, PipelineConfig, PolicyConfig; print('ok')"`
3. Build artifacts:
   - `python -m build`
4. Bump canonical package version:
   - `pyproject.toml` (`[project].version`)
   - `src/ega/version.py` resolves `__version__` from installed metadata or `pyproject.toml`
5. Commit version bump and create tag:
   - `git tag -a vX.Y.Z[-label] -m "EGA vX.Y.Z[-label]"`
6. Update `CHANGELOG.md` with release-date and scope notes.
7. Draft GitHub release notes and attach/publish package artifacts as needed.
