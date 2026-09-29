from __future__ import annotations

import streamlit as st

from app.ui_components.history import render_history
from finance_ai.config import ROOT_DIR, get_settings
from finance_ai.utils.feedback import get_feedback_stats


def default_suggestions() -> list[str]:
    return [
        "What is the price trend of AAPL over 1 month?",
        "Compare Apple and Microsoft fundamentals",
        "Summarize latest news about Nvidia",
        "What are the key risks for Reliance right now?",
        "What stocks to buy in India right now for moderate risk?",
    ]


def init_state() -> None:
    defaults = {
        "history": [],
        "model_profile": "Quality",
        "suggestions": default_suggestions(),
        "pending_query": None,
        "show_technical": False,
        "planner_mode": "hybrid",
        "main_query": "",
        "query_timestamps": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def active_model_name() -> str:
    settings = get_settings()
    return settings.ollama_model if st.session_state.model_profile == "Quality" else "qwen2.5:7b-instruct"


def render_header() -> None:
    st.markdown("# AI Financial Analyst")
    st.caption("Ask anything about stocks, companies, and markets")


def render_sidebar() -> str | None:
    sidebar = st.sidebar
    sidebar.header("Controls")
    sidebar.radio(
        "Inference profile",
        options=["Speed", "Quality"],
        index=1 if st.session_state.get("model_profile") == "Quality" else 0,
        horizontal=True,
        key="model_profile",
    )
    sidebar.caption(f"Model: {active_model_name()}")
    planner_modes = ["hybrid", "llm", "rule"]
    current_planner_mode = st.session_state.get("planner_mode", "hybrid")
    selected_idx = planner_modes.index(current_planner_mode) if current_planner_mode in planner_modes else 0
    sidebar.toggle("Show technical details", key="show_technical")
    sidebar.selectbox("Planner mode", planner_modes, index=selected_idx, key="planner_mode")

    sidebar.divider()
    suggestions_tab, history_tab = sidebar.tabs(["Suggestions", "History"])

    with suggestions_tab:
        suggestions_tab.caption("Updates after every prompt")
        for idx, suggestion in enumerate(st.session_state.suggestions[:6]):
            if suggestions_tab.button(suggestion, key=f"suggestion_{idx}", use_container_width=True):
                st.session_state.pending_query = suggestion
                st.rerun()

    with history_tab:
        render_history()

    sidebar.divider()
    sidebar.subheader("Knowledge Base")
    sidebar.caption("Upload .md, .txt, or .pdf files to add to the RAG index")
    uploaded_files = sidebar.file_uploader(
        "Upload research notes",
        type=["md", "txt", "pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded_files:
        notes_dir = ROOT_DIR / "docs" / "example_notes"
        notes_dir.mkdir(parents=True, exist_ok=True)
        saved: list[str] = []
        for uf in uploaded_files:
            dest = notes_dir / uf.name
            data = uf.getvalue()
            # Streamlit re-sends uploads on every rerun; rewriting would bump
            # mtimes and force a full re-embed each time.
            if dest.exists() and dest.read_bytes() == data:
                continue
            dest.write_bytes(data)
            saved.append(uf.name)
        if saved:
            from finance_ai.rag.service import get_finance_retriever

            get_finance_retriever.cache_clear()
            sidebar.success(f"Saved {len(saved)} file(s). Index rebuilds on next query.")

    sidebar.divider()
    sidebar.subheader("Feedback")
    stats = get_feedback_stats()
    if stats["total"] > 0:
        sidebar.caption(
            f"👍 {stats['positive']}  👎 {stats['negative']}  "
            f"({stats['total']} rated answers)"
        )
    else:
        sidebar.caption("No feedback recorded yet.")

    pending_query = st.session_state.pending_query
    if pending_query:
        st.session_state.pending_query = None
        return pending_query
    return None
