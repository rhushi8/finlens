"""MCP-style JSON-RPC over stdio, one JSON object per line."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any, Callable

from finance_ai.agents import route_query
from finance_ai.tools import (
    calculate_financial_metric,
    get_fundamentals,
    get_stock_price,
    search_news,
)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., Any]


def _to_payload(result: Any) -> Any:
    if hasattr(result, "model_dump"):
        return result.model_dump()
    return result


class FinanceMcpServer:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {
            "get_stock_price": ToolSpec(
                name="get_stock_price",
                description="Fetch recent stock OHLCV and movement metrics.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string"},
                        "period": {"type": "string", "default": "1mo"},
                    },
                    "required": ["ticker"],
                },
                handler=get_stock_price,
            ),
            "get_fundamentals": ToolSpec(
                name="get_fundamentals",
                description="Fetch valuation and company-level fundamentals.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string"},
                    },
                    "required": ["ticker"],
                },
                handler=get_fundamentals,
            ),
            "search_news": ToolSpec(
                name="search_news",
                description="Search recent finance news from free sources.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "max_results": {"type": "integer", "default": 5},
                    },
                    "required": ["query"],
                },
                handler=search_news,
            ),
            "calculate_financial_metric": ToolSpec(
                name="calculate_financial_metric",
                description="Run deterministic finance calculations.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "operation": {"type": "string"},
                        "params": {"type": "object"},
                        "inputs": {"type": "object"},
                    },
                    "required": ["operation"],
                },
                handler=calculate_financial_metric,
            ),
            "route_query": ToolSpec(
                name="route_query",
                description="Run full analyst orchestration for a natural-language query.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                    "required": ["query"],
                },
                handler=route_query,
            ),
        }

    def _list_tools(self) -> dict[str, Any]:
        tools = []
        for spec in self._tools.values():
            tools.append(
                {
                    "name": spec.name,
                    "description": spec.description,
                    "inputSchema": spec.input_schema,
                }
            )
        return {"tools": tools}

    def _call_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name", "")
        arguments = params.get("arguments", {})

        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}")

        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be an object")

        spec = self._tools[name]
        if name == "calculate_financial_metric":
            if "params" not in arguments and "inputs" in arguments:
                arguments = dict(arguments)
                arguments["params"] = arguments.pop("inputs")
        result = spec.handler(**arguments)
        payload = _to_payload(result)

        return {
            "content": [
                {
                    "type": "json",
                    "json": payload,
                }
            ],
            "isError": False,
        }

    def _invoke(self, method: str, params: dict[str, Any] | None) -> dict[str, Any]:
        params = params or {}

        if method == "initialize":
            return {
                "protocolVersion": "2025-03-26",
                "serverInfo": {
                    "name": "finance-ai-analyst-mcp",
                    "version": "0.1.0",
                },
                "capabilities": {
                    "tools": {},
                },
            }

        if method == "tools/list":
            return self._list_tools()

        if method == "tools/call":
            return self._call_tool(params)

        raise ValueError(f"Unknown method: {method}")

    def handle_jsonrpc(self, request: dict[str, Any]) -> dict[str, Any] | None:
        if request.get("jsonrpc") != "2.0":
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {"code": -32600, "message": "Invalid Request"},
            }

        rpc_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})

        if method is None:
            return {
                "jsonrpc": "2.0",
                "id": rpc_id,
                "error": {"code": -32600, "message": "Missing method"},
            }

        try:
            result = self._invoke(method, params)
            if rpc_id is None:
                return None
            return {"jsonrpc": "2.0", "id": rpc_id, "result": result}
        except Exception as exc:
            if rpc_id is None:
                return None
            return {
                "jsonrpc": "2.0",
                "id": rpc_id,
                "error": {"code": -32000, "message": str(exc)},
            }


def _read_messages() -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            messages.append(json.loads(line))
        except json.JSONDecodeError:
            messages.append(
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "method": None,
                    "params": {},
                }
            )
    return messages


def run_stdio_server() -> None:
    server = FinanceMcpServer()
    for request in _read_messages():
        response = server.handle_jsonrpc(request)
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()


def main() -> None:
    run_stdio_server()


if __name__ == "__main__":
    main()
