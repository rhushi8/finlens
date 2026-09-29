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


def test_stdio_answers_each_line_before_reading_the_next(monkeypatch, capsys) -> None:
    import sys

    from finance_ai.mcp import server as server_module

    seen_after_first_line: list[str] = []

    class _OpenStdin:
        def __iter__(self):
            yield '{"jsonrpc": "2.0", "id": 1, "method": "initialize"}\n'
            seen_after_first_line.append(capsys.readouterr().out)
            yield '{"jsonrpc": "2.0", "id": 2, "method": "tools/list"}\n'

    monkeypatch.setattr(sys, "stdin", _OpenStdin())
    server_module.run_stdio_server()
    assert '"id": 1' in seen_after_first_line[0]


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
