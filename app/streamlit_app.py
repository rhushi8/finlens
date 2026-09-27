from __future__ import annotations

import time

import streamlit as st

from app.ui_components import (
    active_model_name,
    init_state,
    render_answer,
    render_header,
    render_sidebar,
)
from finance_ai.agents import route_query
from finance_ai.rag.service import get_finance_retriever
from finance_ai.schemas.agent import AnalystAnswer
from finance_ai.ui.presenter import build_contextual_suggestions, recommendation_color

st.set_page_config(page_title="AI Financial Analyst", page_icon="$", layout="wide")
HISTORY_PANEL_THRESHOLD = 8
_RATE_LIMIT_WINDOW = 60   # seconds
_RATE_LIMIT_MAX = 10      # queries per window per session


def _check_rate_limit() -> bool:
    now = time.time()
    timestamps = [t for t in st.session_state.get("query_timestamps", []) if now - t < _RATE_LIMIT_WINDOW]
    if len(timestamps) >= _RATE_LIMIT_MAX:
        wait = int(_RATE_LIMIT_WINDOW - (now - timestamps[0])) + 1
        st.warning(f"Rate limit: {_RATE_LIMIT_MAX} queries per {_RATE_LIMIT_WINDOW}s. Try again in {wait}s.")
        return False
    timestamps.append(now)
    st.session_state.query_timestamps = timestamps
    return True


def _build_related_suggestions(query: str, answer: AnalystAnswer) -> list[str]:
    subject = answer.company_name or answer.ticker or "this company"
    return build_contextual_suggestions(
        query=query,
        intent=(answer.intent or "").lower(),
        subject=subject,
        risk_view=answer.risk_view or [],
        news_view=answer.news_view or [],
        stock_view=answer.stock_view or {},
    )


def _run_query(query: str) -> AnalystAnswer:
    with st.spinner("Analyzing market data..."):
        return route_query(
            query,
            model_name=active_model_name(),
            planner_mode=st.session_state.get("planner_mode", "hybrid"),
        )


def _warmup_retriever() -> None:
    if st.session_state.get("retriever_warmed", False):
        return
    try:
        get_finance_retriever()
        st.session_state.retriever_warmed = True
    except Exception:
        # Best-effort. Queries handle retrieval failures anyway.
        st.session_state.retriever_warmed = False


def main() -> None:
    init_state()
    _warmup_retriever()
    render_header()

    sidebar_query = render_sidebar()

    history_messages = st.session_state.history[-HISTORY_PANEL_THRESHOLD:]
    for msg in history_messages:
        with st.chat_message("user"):
            st.write(msg.get("query", ""))
        with st.chat_message("assistant"):
            answer = AnalystAnswer.model_validate(msg.get("answer", {}))
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(answer.summary)
            with col2:
                color = recommendation_color(answer.recommendation)
                st.markdown(f"**:{color}[{answer.recommendation}]**")
            with st.expander("View full analysis"):
                render_answer(answer)

    with st.container():
        with st.form("main_query_form", clear_on_submit=True):
            st.text_input(
                "Ask a follow-up",
                key="main_query",
                placeholder="Ask a financial question about a stock, company, or market",
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("Analyze", type="primary", use_container_width=True)

        form_query = st.session_state.main_query.strip() if submitted and st.session_state.main_query.strip() else None

    query = sidebar_query or form_query

    if query and _check_rate_limit():
        try:
            with st.chat_message("user"):
                st.write(query)

            with st.chat_message("assistant"):
                answer = _run_query(query)
                st.session_state.history.append({"query": query, "answer": answer.model_dump()})
                st.session_state.suggestions = _build_related_suggestions(query, answer)
                render_answer(answer)
        except Exception as exc:
            st.error(f"Unable to fetch data: {exc}")


if __name__ == "__main__":
    main()
