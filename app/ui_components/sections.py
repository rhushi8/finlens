"""Answer sections and typed rendering helpers."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from finance_ai.utils.feedback import record_feedback
from finance_ai.ui.presenter import (
    fmt,
    format_metric_value,
    pct,
    price_series_frame,
    recommendation_color,
    scalar_stock_metrics,
)


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def _render_summary_cards(answer) -> None:
    cols = st.columns(4)
    cols[0].metric("Confidence", pct(answer.recommendation_confidence))
    if answer.grounding_score > 0.1:
        cols[1].metric("Grounding", pct(answer.grounding_score))
    cols[2].metric("Latency", f"{answer.latency_ms:.0f} ms")
    if answer.source_count > 0:
        cols[3].metric("Sources", answer.source_count)


def _render_recommendation_panel(answer) -> None:
    color = recommendation_color(answer.recommendation)
    st.markdown(f"### Recommendation: :{color}[{answer.recommendation}] ({pct(answer.recommendation_confidence)})")
    st.write(answer.summary)
    if answer.decision_rationale:
        st.caption(answer.decision_rationale)


def _render_bull_bear(answer) -> None:
    left, right = st.columns(2)
    with left:
        st.subheader("Bull Case")
        if answer.bull_case:
            for point in answer.bull_case:
                st.write(f"- {point}")
        else:
            st.write("- Limited explicit bullish evidence for this query.")
    with right:
        st.subheader("Bear Case")
        if answer.bear_case:
            for point in answer.bear_case:
                st.write(f"- {point}")
        else:
            st.write("- Limited explicit downside evidence for this query.")


def _render_comparison(answer) -> None:
    if not answer.comparison_view:
        st.info("No side-by-side comparison available for this answer.")
        return

    cv = answer.comparison_view
    left_leg = cv.left
    right_leg = cv.right

    color = recommendation_color(cv.recommendation)
    st.markdown(f"**Winner: :{color}[{cv.winner}]** - {cv.winner_reason or 'Balanced'}")
    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"### {left_leg.company_name}")
        st.metric("Price", format_metric_value("current_price", left_leg.price, left_leg.ticker))
        st.metric("1M Change", fmt(left_leg.change_pct_1m, template="{:+.2f}", suffix="%"))
        st.metric("P/E Ratio", fmt(left_leg.pe_ratio, suffix="x"))
        st.metric("Beta", fmt(left_leg.beta))
    with c2:
        st.markdown(f"### {right_leg.company_name}")
        st.metric("Price", format_metric_value("current_price", right_leg.price, right_leg.ticker))
        st.metric("1M Change", fmt(right_leg.change_pct_1m, template="{:+.2f}", suffix="%"))
        st.metric("P/E Ratio", fmt(right_leg.pe_ratio, suffix="x"))
        st.metric("Beta", fmt(right_leg.beta))

    if left_leg.bull_points or right_leg.bull_points:
        st.subheader("Bull Case")
        col1, col2 = st.columns(2)
        with col1:
            st.caption(f"**{left_leg.company_name} Strengths**")
            for point in left_leg.bull_points[:2]:
                st.write(f"- {point}")
        with col2:
            st.caption(f"**{right_leg.company_name} Strengths**")
            for point in right_leg.bull_points[:2]:
                st.write(f"- {point}")

    if left_leg.bear_points or right_leg.bear_points:
        st.subheader("Bear Case")
        col1, col2 = st.columns(2)
        with col1:
            st.caption(f"**{left_leg.company_name} Risks**")
            for point in left_leg.bear_points[:2]:
                st.write(f"- {point}")
        with col2:
            st.caption(f"**{right_leg.company_name} Risks**")
            for point in right_leg.bear_points[:2]:
                st.write(f"- {point}")

    if answer.comparison_view.key_differences:
        st.subheader("Key Differences")
        for item in answer.comparison_view.key_differences:
            st.write(f"- {item}")


def _render_overview(answer) -> None:
    st.subheader("Stock Overview")
    metrics = scalar_stock_metrics(answer.stock_view)
    if metrics:
        cols = st.columns(min(4, len(metrics)))
        for idx, (key, value) in enumerate(metrics.items()):
            cols[idx % len(cols)].metric(
                key.replace("_", " ").title(),
                format_metric_value(key, value, answer.ticker),
            )
    elif answer.comparison_view:
        left_leg = answer.comparison_view.left
        right_leg = answer.comparison_view.right
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(left_leg.company_name, format_metric_value("current_price", left_leg.price, left_leg.ticker))
        with col2:
            st.metric(right_leg.company_name, format_metric_value("current_price", right_leg.price, right_leg.ticker))
        with col3:
            st.metric("Winner", answer.comparison_view.winner or "Balanced")
        st.caption(answer.comparison_view.winner_reason or "Comparison data available below.")
    else:
        st.info("No stock metrics available for this query.")

    series = price_series_frame(answer.stock_view)
    if not series.empty and {"date", "close"}.issubset(series.columns):
        st.line_chart(series.set_index("date")["close"], height=260)


def _render_chart_data(answer) -> None:
    if not answer.chart_data and not answer.stock_view.get("price_series"):
        return

    st.subheader("Visualizations")

    if answer.stock_view.get("price_series"):
        series = price_series_frame(answer.stock_view)
        if not series.empty and {"date", "close"}.issubset(series.columns):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.line_chart(series.set_index("date")[["close"]], height=300)
            with col2:
                st.metric(
                    "Latest Close",
                    format_metric_value("current_price", series["close"].iloc[-1], answer.ticker)
                    if len(series) > 0
                    else "N/A",
                )
                if len(series) > 1:
                    change = series["close"].iloc[-1] - series["close"].iloc[0]
                    pct_change = (change / series["close"].iloc[0] * 100) if series["close"].iloc[0] != 0 else 0
                    st.metric("Change", f"{pct_change:+.2f}%")

    if answer.chart_data:
        for chart_key, chart_value in answer.chart_data.items():
            if isinstance(chart_value, dict):
                numeric_items = [(ticker, value) for ticker, value in chart_value.items() if isinstance(value, (int, float))]
                if numeric_items:
                    frame = pd.DataFrame(numeric_items, columns=["ticker", "value"])
                    st.caption(chart_key.replace("_", " ").title())
                    st.bar_chart(frame.set_index("ticker"))


def _render_news(answer) -> None:
    st.subheader("Key News")
    if answer.news_view:
        for bullet in answer.news_view:
            st.write(f"- {bullet}")
    else:
        st.info("No fresh news items were returned. Try a broader news query.")


def _render_risks(answer) -> None:
    st.subheader("Risks")
    if answer.risk_view:
        for risk in answer.risk_view:
            st.warning(risk)
    else:
        st.info("No explicit risk items were identified.")


def _render_sources(answer) -> None:
    st.subheader("Sources & Evidence")
    if answer.source_items:
        grouped_by_type: dict[str, list] = {}
        for item in answer.source_items:
            source_type = item.source_type or "other"
            grouped_by_type.setdefault(source_type, []).append(item)

        for source_type in sorted(grouped_by_type.keys()):
            label = source_type.replace("_", " ").title()
            with st.expander(f"{label} ({len(grouped_by_type[source_type])})"):
                for idx, item in enumerate(grouped_by_type[source_type], 1):
                    title = item.title or f"Source {idx}"
                    if item.url:
                        st.markdown(f"**[{title}]({item.url})**")
                    else:
                        st.markdown(f"**{title}**")

                    meta: list[str] = []
                    if item.date:
                        meta.append(f"Date: {item.date}")
                    if item.source:
                        meta.append(f"Source: {item.source}")
                    if meta:
                        st.caption(" | ".join(meta))

                    if item.snippet:
                        st.caption(item.snippet)
    elif answer.citations:
        for idx, citation in enumerate(answer.citations, 1):
            title = citation.title or f"Citation {idx}"
            if citation.url:
                st.markdown(f"**[{title}]({citation.url})**")
            else:
                st.markdown(f"**{title}**: {citation.source}")
            if citation.snippet:
                st.caption(citation.snippet)
    else:
        st.info("No sources available.")


def _render_technical(answer) -> None:
    """Show agent reasoning and tool traces."""

    st.subheader("Technical Details")
    st.write(f"**Intent:** {answer.intent}")
    if answer.warnings:
        st.subheader("Warnings")
        for warning in answer.warnings:
            st.warning(warning)
    if answer.tool_calls:
        st.subheader("Tool Execution Trace")
        for call in answer.tool_calls:
            status = "OK" if call.success else "FAILED"
            with st.expander(f"{call.tool_name} | {status} | {call.duration_ms:.0f}ms"):
                st.write("Input")
                st.json(call.input_params)
                st.write("Output")
                st.json(call.output)
                if call.error:
                    st.error(call.error)


def _render_explanation(answer) -> None:
    """Show agent reasoning and confidence breakdown."""

    st.subheader("How This Answer Was Generated")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Confidence", pct(answer.recommendation_confidence))
    with col2:
        st.metric("Grounding", pct(answer.grounding_score))
    with col3:
        st.metric("Sources", answer.source_count)

    st.divider()
    st.subheader("Agent Pipeline")
    if answer.tool_calls:
        steps_taken = _dedupe_preserve_order([call.tool_name for call in answer.tool_calls])
        st.info(f"**Steps:** Plan -> {' -> '.join(steps_taken)} -> Synthesize")
        st.caption(f"Total latency: {answer.latency_ms:.0f}ms | Tools: {len(answer.tool_calls)}")

    st.subheader("Confidence Explanation")
    if answer.grounding_score >= 0.8:
        st.success(f"**High confidence:** {answer.source_count} sources with good diversity")
    elif answer.grounding_score >= 0.5:
        st.warning(f"**Moderate confidence:** {answer.source_count} sources, verify key claims")
    else:
        st.error(f"**Low confidence:** {answer.source_count} sources, recommend additional research")

    if answer.decision_rationale:
        st.subheader("Decision Rationale")
        st.write(answer.decision_rationale)


def _render_feedback(answer) -> None:
    feedback_key = f"feedback_{hash(answer.query)}"
    if st.session_state.get(feedback_key):
        st.caption("Feedback recorded — thanks!")
        return
    col_a, col_b, *_ = st.columns([1, 1, 8])
    if col_a.button("👍", key=f"{feedback_key}_up", help="This answer was helpful"):
        record_feedback(
            query=answer.query,
            ticker=answer.ticker,
            intent=answer.intent,
            recommendation=answer.recommendation,
            grounding_score=answer.grounding_score,
            rating=1,
        )
        st.session_state[feedback_key] = "up"
        st.rerun()
    if col_b.button("👎", key=f"{feedback_key}_down", help="This answer was not helpful"):
        record_feedback(
            query=answer.query,
            ticker=answer.ticker,
            intent=answer.intent,
            recommendation=answer.recommendation,
            grounding_score=answer.grounding_score,
            rating=-1,
        )
        st.session_state[feedback_key] = "down"
        st.rerun()


def render_answer(answer) -> None:
    _render_summary_cards(answer)
    _render_feedback(answer)
    st.divider()
    _render_recommendation_panel(answer)
    _render_bull_bear(answer)
    st.divider()

    tab_names = ["Overview", "Compare", "News", "Risks", "Sources", "Explanation", "Technical"]
    tabs = st.tabs(tab_names)

    if answer.intent == "compare" and answer.comparison_view:
        st.subheader("Comparison Analysis")
        _render_comparison(answer)
        st.divider()

    with tabs[0]:
        _render_overview(answer)
        _render_chart_data(answer)
    with tabs[1]:
        if answer.intent != "compare":
            _render_comparison(answer)
        else:
            st.info("Comparison details shown above at top level.")
    with tabs[2]:
        _render_news(answer)
    with tabs[3]:
        _render_risks(answer)
    with tabs[4]:
        _render_sources(answer)
    with tabs[5]:
        _render_explanation(answer)
    with tabs[6]:
        if st.session_state.show_technical:
            _render_technical(answer)
        else:
            st.info("Enable technical details from the sidebar to inspect execution trace.")
