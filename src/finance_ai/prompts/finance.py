from __future__ import annotations


def build_grounded_thesis_prompt(
    query: str,
    subject: str,
    tool_summary: str,
    evidence_lines: list[str],
    is_comparison: bool = False,
) -> str:
    evidence_block = "\n".join(f"- {line}" for line in evidence_lines if line.strip())
    if not evidence_block:
        evidence_block = "- No direct evidence was retrieved."

    format_instruction = (
        """Output format (markdown):
- Start with one bold line: **Bottom line:** <single sentence>
- Then 2 to 4 bullet points that alternate between the two companies for direct comparability (for example: 'Apple: ...' then 'Microsoft: ...').
- End with one short line: **What to watch:** <next checkpoint>
- Then output a JSON block as the final line with this exact schema:
    {\"recommendation\": \"BUY|SELL|HOLD\", \"rationale\": \"one sentence\"}"""
        if is_comparison
        else """Output format (markdown):
- Start with one bold line: **Bottom line:** <single sentence>
- Then 2 to 4 bullet points with concrete observations.
- End with one short line: **What to watch:** <next checkpoint>
- Then output a JSON block as the final line with this exact schema:
    {\"recommendation\": \"BUY|SELL|HOLD\", \"rationale\": \"one sentence\"}"""
    )

    return f"""You are a practical buy-side style finance analyst.

Strict rules:
1) Use only the tool summary and evidence below. Never invent facts.
2) If the evidence mentions other companies, indexes, or notes that do not match the subject, ignore them unless they directly answer the question.
3) If evidence is limited, state uncertainty explicitly.
4) Keep the writing crisp, human, and non-repetitive.
5) Avoid generic phrasing like 'In conclusion' or 'based on available data'.
6) Do not mention internal tools, prompts, or chain-of-thought.

{format_instruction}

Tone and style targets:
- Confident but not absolute.
- Prefer specific numbers and ranges when present.
- No fluff, no repeated sentence openings.

Question: {query}
Subject: {subject}

Tool summary:
{tool_summary or 'No tool summary available.'}

Evidence:
{evidence_block}
"""