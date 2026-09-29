# AI Financial Analyst

An agentic assistant that answers stock-market questions with sourced evidence
rather than remembered numbers. It decides which tools to call, pulls live
market data, news and documents, scores how well its own answer is supported,
and returns a BUY / HOLD / SELL view with citations and a confidence level.

Educational project. Not investment advice.

## Why

Language models answer financial questions fluently and unreliably. They
hallucinate prices, ratios and headlines, and they give you no way to tell how
well-supported any of it is. That combination of confident tone and
unverifiable content is a bad thing to put next to a money decision. So the
goal here was natural-language analysis that stays tied to real, current data
and is honest about the evidence behind each claim.

## What it does

A planner classifies the intent, price, fundamentals, news, comparison or
market ideas, and routes to the right tools. It runs in `rule`, `hybrid` or
`llm` mode, with a deterministic fallback so it never hard-fails. In hybrid,
the default, the rules decide and the LLM is only asked about queries they
can't place, and any LLM plan naming a tool the router doesn't have is dropped.

Tools run in parallel: live price and fundamentals from yfinance, free news
from GDELT, and an India market scanner covering NIFTY and SENSEX.

Retrieval over documents uses chunking, sentence-transformer embeddings and
FAISS for persistence, with metadata filtering, deduplication and optional
reranking.

Before anything reaches the user, a grounding check scores how well the
evidence supports the answer, calibrates a confidence value, softens the
wording when the evidence is thin, and attaches the citations.

The comparison engine puts two stocks side by side on momentum, P/E, beta,
market cap and dividend yield, then weights those into a BUY/HOLD call with a
bull case and a bear case.

Output is a typed Pydantic `AnalystAnswer` carrying the summary,
recommendation, confidence, citations, warnings, latency and tool traces, which
the Streamlit dashboard renders.

An MCP server exposes the same finance tools over JSON-RPC for other agent
clients.

## How it works

```
query → planner → tool selection → parallel tool calls → RAG retrieval
      → grounded synthesis → grounding check → typed response → UI
```

The model never answers from memory alone. The planner, the tool output, the
retrieved context and the grounding check all sit between the question and the
answer.

## Stack

Python, Streamlit, LangChain for RAG, FAISS, sentence-transformers, yfinance,
GDELT, Ollama for the local model, Pydantic, pytest, Docker and GitHub Actions.

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # source .venv/bin/activate on macOS or Linux
pip install -e ".[dev]"
copy .env.example .env
streamlit run app/streamlit_app.py
```

For local synthesis: `ollama pull qwen2.5:7b-instruct`. Tests run with
`pytest -q`.

## Layout

```
app/                 streamlit UI
src/finance_ai/      planner and router, tools, rag, llm, schemas, utils
tests/               deterministic tests, external services mocked
scripts/             evaluation and smoke scripts
docs/                sample retrieval documents and design notes
Dockerfile           containerisation
.github/workflows/   CI
```

## Known limits

- The planner is heuristic, rules plus a deterministic fallback, rather than
  full model-native planning.
- News quality depends on the free GDELT endpoint. The system degrades
  gracefully when it is down, but it does go down.
- Grounding scores say how well an answer is supported by what was retrieved.
  They say nothing about whether the source itself was right.
