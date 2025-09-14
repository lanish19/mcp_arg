import pytest
from MCP_example_template.argument_mcp import decompose_argument_structure, detect_argument_patterns

def test_node_span_generation_is_robust():
    """
    Tests that node spans are generated correctly and are not dummy spans,
    even with repeated sentences.
    """
    text = "First sentence. Second sentence. First sentence."
    result = decompose_argument_structure(text, compat="raw")

    nodes = result["structure"]["nodes"]
    assert len(nodes) == 3

    # First sentence
    assert nodes[0]["content"] == "First sentence."
    span1 = nodes[0]["source_text_span"]
    assert isinstance(span1, list)
    assert len(span1) == 2
    assert text[span1[0]:span1[1]] == "First sentence."

    # Second sentence
    assert nodes[1]["content"] == "Second sentence."
    span2 = nodes[1]["source_text_span"]
    assert isinstance(span2, list)
    assert len(span2) == 2
    assert text[span2[0]:span2[1]] == "Second sentence."

    # Repeated first sentence - should have a different span
    assert nodes[2]["content"] == "First sentence."
    span3 = nodes[2]["source_text_span"]
    assert isinstance(span3, list)
    assert len(span3) == 2
    assert text[span3[0]:span3[1]] == "First sentence."

    assert span1 != span3, "Spans for repeated sentences should be different"
    assert span1 == [0, 15]
    assert span3 == [33, 48]


def test_pattern_span_fallback_is_not_dummy():
    """
    Tests that the fallback for pattern spans is the entire text, not a dummy proportional span.
    This text is designed to not have any obvious trigger words for patterns.
    """
    # This text is unlikely to match any specific trigger, forcing a fallback.
    text = "A convoluted narrative regarding fiscal policy."
    result = detect_argument_patterns(text, compat="raw")

    patterns = result["patterns"]
    # We expect some patterns to be found via semantic search, even without triggers
    assert len(patterns) > 0

    for p in patterns:
        assert "source_text_span" in p
        span = p["source_text_span"]
        assert isinstance(span, list)
        assert len(span) == 2

        # Check if the span is the full text span, which is our ultimate fallback.
        # This is better than checking if it's NOT a dummy span, because a fuzzy match might still occur.
        # If any span is NOT the full text, it means a better match was found, which is good.
        # We are primarily concerned with preventing the arbitrary proportional span.
        # The old dummy span was `(0, min(max(20, len(text)//10), len(text)))` which for this text would be `(0, 20)`
        assert span != [0, 20]

        # The new fallback is the full text
        if span == [0, len(text)]:
            print(f"Pattern '{p['label']}' correctly used fallback span.")
        else:
            print(f"Pattern '{p['label']}' found a specific span: {span}")

        # Any valid span must be within the text bounds
        assert 0 <= span[0] < len(text)
        assert 0 < span[1] <= len(text)

def test_evidence_subtype_is_assigned():
    """
    Tests that sentences with evidence-like keywords are assigned the 'Evidence' subtype.
    """
    text = "A study shows that this is effective. Therefore we should do it."
    result = decompose_argument_structure(text, compat="raw")

    nodes = result["structure"]["nodes"]
    assert len(nodes) == 2

    evidence_node = next((n for n in nodes if n["primary_subtype"] == "Evidence"), None)
    claim_node = next((n for n in nodes if n["primary_subtype"] == "Main Claim"), None)

    assert evidence_node is not None, "Evidence node was not identified"
    assert claim_node is not None, "Main Claim node was not identified"

    assert evidence_node["content"] == "A study shows that this is effective."
    assert claim_node["content"] == "Therefore we should do it."
