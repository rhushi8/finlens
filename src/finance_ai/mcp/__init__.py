from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "FinanceMcpServer": ("finance_ai.mcp.server", "FinanceMcpServer"),
    "run_stdio_server": ("finance_ai.mcp.server", "run_stdio_server"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    return getattr(module, attr_name)


__all__ = list(_EXPORTS)
