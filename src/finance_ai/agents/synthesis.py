"""Grounded answer synthesis helpers."""

from __future__ import annotations

import json
import logging
import re

from finance_ai.llm.ollama_client import invoke_ollama
from finance_ai.prompts.finance import build_grounded_thesis_prompt

logger = logging.getLogger(__name__)


def _clean_model_text(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _find_json_object(text: str) -> tuple[dict | None, tuple[int, int] | None]:
    """Locate the first valid JSON object via bracket matching (handles nesting)."""
    start = text.find("{")
    if start == -1:
        return None, None
    depth = 0
    in_string = False
    escape_next = False
    for i, ch in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1]), (start, i + 1)
                except json.JSONDecodeError:
                    return None, None
    return None, None


def _extract_recommendation_json(text: str) -> tuple[str, str, str]:
    """Return (summary_without_json, recommendation, rationale)."""

    normalized = _clean_model_text(text)
    payload, payload_span = _find_json_object(normalized)

    if payload is None or not isinstance(payload, dict) or "recommendation" not in payload:
        return normalized, "HOLD", "Structured recommendation missing from synthesis output."

    recommendation = str(payload.get("recommendation", "HOLD")).upper().strip()
    if recommendation not in {"BUY", "SELL", "HOLD"}:
        recommendation = "HOLD"

    rationale = str(payload.get("rationale", "")).strip() or "Model rationale missing in structured output."
    if payload_span:
        summary = (normalized[: payload_span[0]] + normalized[payload_span[1] :]).strip()
    else:
        summary = normalized

    return summary, recommendation, rationale


def _fallback_thesis(
    query: str,
    subject: str,
    tool_summary: str,
    evidence_lines: list[str],
) -> str:
    evidence_snippets = [line.strip() for line in evidence_lines if line.strip()][:4]
    if not evidence_snippets:
        return (
            f"**Bottom line:** Evidence is too limited to form a strong view on {subject}.\n\n"
            f"- Available context is sparse for: {query}\n"
            f"- Use this as a preliminary signal, not a final decision input\n"
            f"- {tool_summary or 'No supporting tool summary was available'}\n\n"
            "**What to watch:** confirm with fresh filings, earnings commentary, and price follow-through."
        )

    bullets = "\n".join(f"- {snippet}" for snippet in evidence_snippets)
    summary_line = tool_summary.strip() if tool_summary.strip() else "Signals are mixed but actionable."
    return (
        f"**Bottom line:** {summary_line}\n\n"
        f"{bullets}\n\n"
        "**What to watch:** near-term price reaction, revisions to guidance, and new macro headlines."
    )


def synthesize_grounded_response(
    query: str,
    subject: str,
    tool_summary: str,
    evidence_lines: list[str],
    model_name: str | None = None,
    is_comparison: bool = False,
) -> tuple[str, str, str]:
    """Generate grounded summary plus structured recommendation and rationale."""

    prompt = build_grounded_thesis_prompt(
        query=query,
        subject=subject,
        tool_summary=tool_summary,
        evidence_lines=evidence_lines,
        is_comparison=is_comparison,
    )

    try:
        response_text = invoke_ollama(prompt, model_name=model_name)
        if response_text.strip():
            return _extract_recommendation_json(response_text)
    except Exception as exc:
        logger.info("Ollama synthesis unavailable, using fallback: %s", exc)

    fallback_summary = _fallback_thesis(query, subject, tool_summary, evidence_lines)
    return fallback_summary, "HOLD", "Fallback recommendation because structured synthesis was unavailable."