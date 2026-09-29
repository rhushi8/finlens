import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help="Run live integration tests (requires external APIs)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "live: marks test as live integration test (requires external APIs, use --live to run)",
    )
    config.addinivalue_line(
        "markers",
        "unit: marks test as unit test (fast, mocked deps, always safe to run)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--live"):
        return
    skip_live = pytest.mark.skip(reason="need --live option to run live integration tests")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


@pytest.fixture(autouse=True)
def _isolate_disk_cache(tmp_path, monkeypatch):
    import finance_ai.utils.cache as cache_module

    cache_dir = tmp_path / ".cache"
    cache_dir.mkdir()
    monkeypatch.setattr(cache_module, "CACHE_DIR", cache_dir)
    return cache_dir
