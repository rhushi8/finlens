import pytest

from finance_ai.agents.synthesis import synthesize_grounded_response


@pytest.mark.unit
def test_synthesize_grounded_response_parses_structured_recommendation(monkeypatch) -> None:
    import finance_ai.agents.synthesis as synthesis_module

    monkeypatch.setattr(
        synthesis_module,
        "invoke_ollama",
        lambda prompt, model_name=None: (
            "**Bottom line:** Apple has mixed near-term signals.\n"
            "- Services strength offsets some hardware softness.\n"
            "- Margin pressure remains a monitor.\n"
            "**What to watch:** next earnings and guidance update.\n"
            '{"recommendation":"HOLD","rationale":"Mixed growth and risk signals warrant patience."}'
        ),
    )

    summary, recommendation, rationale = synthesize_grounded_response(
        query="What are near-term risks for Apple?",
        subject="Apple",
        tool_summary="Apple margins and services trends.",
        evidence_lines=["Apple sees mixed demand and resilient services."],
        model_name="qwen2.5:14b-instruct",
    )

    assert recommendation == "HOLD"
    assert "Mixed growth" in rationale
    assert "recommendation" not in summary.lower()
    assert "Bottom line" in summary


@pytest.mark.unit
def test_synthesize_grounded_response_defaults_hold_when_json_missing(monkeypatch) -> None:
    import finance_ai.agents.synthesis as synthesis_module

    monkeypatch.setattr(
        synthesis_module,
        "invoke_ollama",
        lambda prompt, model_name=None: (
            "**Bottom line:** Evidence is limited.\n"
            "- Signals are mixed.\n"
            "**What to watch:** macro and earnings."
        ),
    )

    summary, recommendation, rationale = synthesize_grounded_response(
        query="What are near-term risks for Apple?",
        subject="Apple",
        tool_summary="Limited evidence.",
        evidence_lines=["Signals are mixed."],
    )

    assert recommendation == "HOLD"
    assert "missing" in rationale.lower()
    assert "Bottom line" in summary


@pytest.mark.unit
def test_synthesize_grounded_response_uses_first_recommendation_block(monkeypatch) -> None:
    import finance_ai.agents.synthesis as synthesis_module

    monkeypatch.setattr(
        synthesis_module,
        "invoke_ollama",
        lambda prompt, model_name=None: (
            "**Bottom line:** Mixed setup.\n"
            '{"recommendation":"BUY","rationale":"Early block."}\n'
            '{"recommendation":"SELL","rationale":"Later block."}'
        ),
    )

    _summary, recommendation, rationale = synthesize_grounded_response(
        query="What are near-term risks for Apple?",
        subject="Apple",
        tool_summary="Mixed evidence.",
        evidence_lines=["Mixed signals."],
    )

    assert recommendation == "BUY"
    assert rationale == "Early block."


@pytest.mark.unit
def test_synthesize_grounded_response_uses_comparison_prompt_mode(monkeypatch) -> None:
    import finance_ai.agents.synthesis as synthesis_module

    captured_prompt: dict[str, str] = {}

    def _fake_invoke(prompt: str, model_name=None):
        captured_prompt["value"] = prompt
        return (
            "**Bottom line:** Relative setup is mixed.\n"
            '{"recommendation":"HOLD","rationale":"Both names have offsetting strengths."}'
        )

    monkeypatch.setattr(synthesis_module, "invoke_ollama", _fake_invoke)

    _summary, recommendation, _rationale = synthesize_grounded_response(
        query="Compare Apple and Microsoft",
        subject="Apple vs Microsoft",
        tool_summary="Both have mixed momentum and valuation signals.",
        evidence_lines=["Apple: stronger momentum.", "Microsoft: lower beta."],
        is_comparison=True,
    )

    assert recommendation == "HOLD"
    assert "alternate between the two companies" in captured_prompt["value"]
