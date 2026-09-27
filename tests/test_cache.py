from __future__ import annotations

import pickle
import time

import pytest


@pytest.mark.unit
def test_cached_enforces_ttl_and_recomputes(monkeypatch):
    import finance_ai.utils.cache as cache_module

    calls = {"count": 0}

    @cache_module.cached(ttl_seconds=2)
    def expensive(x: int) -> int:
        calls["count"] += 1
        return x * 10 + calls["count"]

    first = expensive(2)
    second = expensive(2)
    assert first == second
    assert calls["count"] == 1

    original_time = time.time
    monkeypatch.setattr(cache_module.time, "time", lambda: original_time() + 10)

    third = expensive(2)
    assert third != second
    assert calls["count"] == 2


@pytest.mark.unit
def test_cached_invalidates_old_version_entry(tmp_path, monkeypatch):
    import finance_ai.utils.cache as cache_module

    monkeypatch.setattr(cache_module, "CACHE_DIR", tmp_path)

    @cache_module.cached(ttl_seconds=60)
    def fn(x: int) -> int:
        return x + 1

    key = cache_module._cache_key("fn", 1)
    path = cache_module._get_cache_path(key)
    path.write_bytes(
        pickle.dumps(
            {
                "version": 999,
                "ts": time.time(),
                "value": 999,
            }
        )
    )

    value = fn(1)
    assert value == 2
