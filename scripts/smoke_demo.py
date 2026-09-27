from __future__ import annotations

from finance_ai.agents import route_query


DEMO_QUERIES = [
    "What is the price trend of TSLA?",
    "What is the PE ratio of AAPL?",
    "Summarize the latest news about Nvidia",
    "What are the key risks for Apple?",
    "Compare Apple and Microsoft",
]


def main() -> None:
    print("Finance AI Analyst smoke demo")
    print("=" * 40)
    for query in DEMO_QUERIES:
        answer = route_query(query)
        print(f"Q: {query}")
        print(f"A: {answer.thesis}")
        print(f"Confidence: {answer.confidence:.2f}")
        print(f"Tools: {[trace.tool_name for trace in answer.tool_trace]}")
        print(f"Sources: {len(answer.sources)}")
        print("-" * 40)


if __name__ == "__main__":
    main()
