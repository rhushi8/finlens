from finance_ai.mcp import FinanceMcpServer


def test_initialize_returns_capabilities() -> None:
    server = FinanceMcpServer()
    resp = server.handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {},
        }
    )
    assert resp is not None
    assert resp["result"]["serverInfo"]["name"] == "finance-ai-analyst-mcp"
    assert "tools" in resp["result"]["capabilities"]


def test_tools_list_contains_route_query() -> None:
    server = FinanceMcpServer()
    resp = server.handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        }
    )
    assert resp is not None
    names = [tool["name"] for tool in resp["result"]["tools"]]
    assert "route_query" in names


def test_tools_call_calculator() -> None:
    server = FinanceMcpServer()
    resp = server.handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "calculate_financial_metric",
                "arguments": {
                    "operation": "pct_change",
                    "params": {"old_value": 100, "new_value": 120},
                },
            },
        }
    )
    assert resp is not None
    content = resp["result"]["content"]
    assert len(content) == 1
    assert content[0]["json"]["result"] == 20.0


def test_unknown_method_returns_error() -> None:
    server = FinanceMcpServer()
    resp = server.handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "unknown/method",
            "params": {},
        }
    )
    assert resp is not None
    assert "error" in resp
