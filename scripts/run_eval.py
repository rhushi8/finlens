from __future__ import annotations

from dataclasses import dataclass

from finance_ai.agents import route_query


@dataclass
class EvalCase:
    query: str
    min_confidence: float
    required_tool: str | None = None


CASES = [
    EvalCase("What is the price of AAPL?", min_confidence=0.50, required_tool="get_stock_price"),
    EvalCase("What are the key fundamentals of Microsoft?", min_confidence=0.50, required_tool="get_fundamentals"),
    EvalCase("Summarize the latest news about Nvidia", min_confidence=0.40),
    EvalCase("Compare Apple and Microsoft", min_confidence=0.45),
]


def run() -> int:
    passed = 0
    print("Finance AI Analyst evaluation run")
    print("=" * 45)

    for idx, case in enumerate(CASES, 1):
        answer = route_query(case.query)
        tools = [trace.tool_name for trace in answer.tool_trace]

        checks: list[tuple[str, bool]] = []
        checks.append(("thesis_non_empty", bool(answer.thesis.strip())))
        checks.append(("confidence_min", answer.confidence >= case.min_confidence))
        checks.append(("sources_present", len(answer.sources) > 0 or answer.error is not None))
        if case.required_tool:
            checks.append(("required_tool", case.required_tool in tools))

        case_ok = all(result for _, result in checks)
        passed += int(case_ok)

        print(f"[{idx}] Query: {case.query}")
        print(f"    Confidence: {answer.confidence:.2f} | Error: {answer.error}")
        print(f"    Tools: {tools}")
        for name, result in checks:
            print(f"    - {name}: {'PASS' if result else 'FAIL'}")
        print(f"    Result: {'PASS' if case_ok else 'FAIL'}")

    total = len(CASES)
    print("=" * 45)
    print(f"Summary: {passed}/{total} cases passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(run())
