from __future__ import annotations

import streamlit as st

HISTORY_LIST_LIMIT = 8
MAX_LABEL_LENGTH = 42

INTENT_BADGES = {
    "compare": "[Compare]",
    "price": "[Price]",
    "news": "[News]",
    "fundamentals": "[Fundamentals]",
    "market_ideas": "[Ideas]",
    "market_general": "[Market]",
    "rag": "[Context]",
}


def render_history() -> None:
    if not st.session_state.history:
        st.caption("Your earlier chats will appear here.")
        return

    st.caption("Tap a previous question to run it again.")
    for idx, item in enumerate(reversed(st.session_state.history[-HISTORY_LIST_LIMIT:])):
        query = item.get("query", "")
        answer = item.get("answer", {})
        intent = answer.get("intent", "")

        label = query[:MAX_LABEL_LENGTH] + "..." if len(query) > MAX_LABEL_LENGTH else query
        badge = INTENT_BADGES.get(intent, "[?]")
        button_label = f"{badge} {label}"

        if st.button(button_label, key=f"history_item_{idx}", use_container_width=True):
            st.session_state.pending_query = query
            st.rerun()
