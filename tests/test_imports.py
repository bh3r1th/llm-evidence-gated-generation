"""Basic import skeleton tests for ega."""


def test_package_import() -> None:
    """Package should be importable."""
    import ega

    assert ega.__version__


def test_public_package_import_contract() -> None:
    from ega import PipelineConfig, PolicyConfig, verify_answer

    assert verify_answer is not None
    assert PipelineConfig is not None
    assert PolicyConfig is not None


def test_public_package_exports_are_frozen() -> None:
    import ega

    assert ega.__all__ == ["verify_answer", "PipelineConfig", "PolicyConfig"]
