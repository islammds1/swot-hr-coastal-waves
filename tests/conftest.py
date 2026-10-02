import pytest


@pytest.fixture(autouse=True)
def isolated_paths(tmp_path, monkeypatch):
    """Point every test at an empty temporary root; no real data needed."""
    import swothr.config as c
    monkeypatch.setenv("SWOTHR_PATHS", str(tmp_path / "nope.yaml"))
    monkeypatch.setenv("SWOTHR_ROOT", str(tmp_path))
    monkeypatch.delenv("SWOTHR_CONFIG", raising=False)
    monkeypatch.setattr(c, "REPO_ROOT", tmp_path)
    c.paths.cache_clear()
    yield tmp_path
    c.paths.cache_clear()
