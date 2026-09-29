from __future__ import annotations

import contextlib
import functools
import hashlib
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

from finance_ai.config import ROOT_DIR

logger = logging.getLogger(__name__)

CACHE_DIR = ROOT_DIR / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
CACHE_VERSION = 4

T = TypeVar("T")

# Unpickle allowlist, so a tampered .cache/ can't run code.
_SAFE_MODULES = frozenset({
    "builtins",
    "datetime",
    "_datetime",
    "decimal",
    "collections",
    "collections.abc",
    "enum",
    "pathlib",
    "pydantic",
    "pydantic.main",
    "pydantic_core",
    "finance_ai.schemas.tools",
    "finance_ai.schemas.agent",
    "finance_ai.schemas.rag",
})


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> type:
        if module in _SAFE_MODULES or module.startswith("finance_ai."):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"Blocked unsafe pickle class: {module}.{name}")


def _cache_key(func_name: str, *args, **kwargs) -> str:
    key_parts = [func_name]
    key_parts.extend(str(arg) for arg in args)
    key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
    key_string = "|".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()


def _get_cache_path(cache_key: str) -> Path:
    return CACHE_DIR / f"{cache_key}.pkl"


def _load_cache_entry(cache_path: Path, ttl_seconds: int) -> Any | None:
    try:
        with open(cache_path, "rb") as f:
            entry = _SafeUnpickler(f).load()
    except Exception as exc:
        logger.debug(f"Cache read failed for {cache_path.name}: {exc}")
        return None

    if not isinstance(entry, dict):
        return None

    ts = entry.get("ts")
    # Old-format, wrong-version or expired entries are dropped.
    if (
        entry.get("version") != CACHE_VERSION
        or not isinstance(ts, (int, float))
        or time.time() - float(ts) >= ttl_seconds
    ):
        with contextlib.suppress(OSError):
            cache_path.unlink(missing_ok=True)
        return None

    return entry.get("value")


def cached(ttl_seconds: int = 3600) -> Callable:
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            key = _cache_key(func.__name__, *args, **kwargs)
            cache_path = _get_cache_path(key)

            if cache_path.exists():
                cached_value = _load_cache_entry(cache_path, ttl_seconds=ttl_seconds)
                if cached_value is not None:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cached_value

            result = func(*args, **kwargs)
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(
                        {
                            "version": CACHE_VERSION,
                            "ts": time.time(),
                            "value": result,
                        },
                        f,
                    )
            except Exception as exc:
                logger.debug(f"Cache write failed for {func.__name__}: {exc}")

            return result

        return wrapper

    return decorator
