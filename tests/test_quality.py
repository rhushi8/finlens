from finance_ai.agents.quality import (
    assess_grounding,
    calibrate_confidence,
    enforce_grounded_wording,
)


def test_assess_grounding_full_signal() -> None:
    score, warnings = assess_grounding(
        evidence_lines=["Revenue growth accelerated in Q4."],
        sources=["yfinance: AAPL fundamentals"],
        tool_trace_count=2,
    )
    assert score == 1.0
    assert warnings == []


def test_assess_grounding_missing_components() -> None:
    score, warnings = assess_grounding(
        evidence_lines=[],
        sources=[],
        tool_trace_count=0,
    )
    assert score == 0.0
    assert len(warnings) == 3


def test_enforce_grounded_wording_adds_note_when_low_score() -> None:
    thesis = "Apple appears stable based on available indicators."
    updated = enforce_grounded_wording(
        thesis=thesis,
        warnings=["No evidence lines captured."],
        threshold=0.5,
        score=0.2,
    )
    assert "Grounding note" in updated


def test_calibrate_confidence_zero_on_error() -> None:
    confidence = calibrate_confidence(base_confidence=0.8, grounding_score=0.9, has_error=True)
    assert confidence == 0.0


def test_calibrate_confidence_weighting() -> None:
    confidence = calibrate_confidence(base_confidence=0.7, grounding_score=1.0, has_error=False)
    assert 0.78 <= confidence <= 0.80


def test_assess_grounding_enhanced_signals() -> None:
    score, warnings = assess_grounding(
        evidence_lines=["Apple revenue growth improved.", "Microsoft cloud growth remained strong."],
        sources=["https://example.com/1", "https://example.com/2", "yfinance: AAPL fundamentals"],
        tool_trace_count=3,
        source_types=["news", "news", "fundamentals"],
        query_relevance=0.8,
        unsupported_claims=0,
        expected_claims=2,
    )
    assert score > 0.75
    assert warnings == []
