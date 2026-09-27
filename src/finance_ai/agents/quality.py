from __future__ import annotations


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def assess_grounding(
    evidence_lines: list[str],
    sources: list[str],
    tool_trace_count: int,
    source_types: list[str] | None = None,
    query_relevance: float = 0.0,
    unsupported_claims: int = 0,
    expected_claims: int = 1,
) -> tuple[float, list[str]]:
    enhanced_mode = source_types is not None or query_relevance > 0 or unsupported_claims > 0 or expected_claims != 1

    score = 0.0
    warnings: list[str] = []

    non_empty_evidence = [line for line in evidence_lines if line.strip()]
    non_empty_sources = [source for source in sources if source.strip()]

    if non_empty_evidence:
        if enhanced_mode:
            coverage = _clamp(len(non_empty_evidence) / max(1, expected_claims), 0.0, 1.0)
            score += 0.25 * coverage
            if coverage < 0.5:
                warnings.append("Evidence coverage is light for the number of claims.")
        else:
            score += 0.4
    else:
        warnings.append("No evidence lines captured.")

    source_count = len(non_empty_sources)
    if source_count > 0:
        if enhanced_mode:
            score += 0.25 * _clamp(source_count / 4.0, 0.0, 1.0)
        else:
            score += 0.4
    else:
        warnings.append("No sources were attached.")

    if enhanced_mode:
        distinct_types = len(set([item for item in (source_types or []) if item]))
        if distinct_types:
            score += 0.2 * _clamp(distinct_types / 2.0, 0.0, 1.0)
        else:
            warnings.append("Source diversity is low.")

        score += 0.2 * _clamp(query_relevance, 0.0, 1.0)
        if query_relevance < 0.4:
            warnings.append("Evidence appears weakly related to the query.")

        if unsupported_claims > 0:
            penalty = 0.1 * min(unsupported_claims, 3)
            score -= penalty
            warnings.append("Some claims may be insufficiently supported.")

    if tool_trace_count > 0:
        score += 0.1 if enhanced_mode else 0.2
    else:
        warnings.append("No tool trace available.")

    return _clamp(score, 0.0, 1.0), warnings


def enforce_grounded_wording(
    thesis: str,
    warnings: list[str],
    threshold: float,
    score: float,
) -> str:
    if score >= threshold or not warnings:
        return thesis

    caution = (
        "Grounding note: evidence coverage is limited in this answer "
        "and should be verified with additional sources."
    )
    if caution.lower() in thesis.lower():
        return thesis

    return f"{thesis}\n\n{caution}"


def calibrate_confidence(base_confidence: float, grounding_score: float, has_error: bool) -> float:
    if has_error:
        return 0.0

    weighted = (0.7 * base_confidence) + (0.3 * grounding_score)
    return _clamp(weighted, 0.0, 1.0)
