"""Basic import skeleton tests for ega."""


def test_package_import() -> None:
    """Package should be importable."""
    import ega

    assert ega.__version__
