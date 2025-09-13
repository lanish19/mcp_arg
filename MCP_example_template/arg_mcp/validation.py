from __future__ import annotations

"""Lightweight validation utilities for argument analysis."""

from typing import List, Dict, Tuple

_POSITIVE = {"good", "beneficial", "positive", "improve", "support"}
_NEGATIVE = {"bad", "harm", "negative", "worsen", "oppose"}
_ABSURD = {"square circle", "perpetual motion machine", "humans breathe water"}


def check_factual_consistency(claims: List[str]) -> List[Tuple[str, str]]:
    """Detect contradictions when one sentence negates another."""
    conflicts: List[Tuple[str, str]] = []
    norm = [c.lower().strip() for c in claims]
    for i, a in enumerate(norm):
        for j, b in enumerate(norm):
            if i >= j:
                continue
            if a.replace("not ", "") == b and (" not " in a or " not " in b):
                conflicts.append((claims[i], claims[j]))
    return conflicts


def check_sentiment_coherence(segments: List[str]) -> List[int]:
    """Return indices of segments whose sentiment conflicts with majority."""
    scores = []
    for seg in segments:
        tokens = set(seg.lower().split())
        pos = len(tokens & _POSITIVE)
        neg = len(tokens & _NEGATIVE)
        scores.append(pos - neg)
    avg = sum(scores) / len(scores) if scores else 0
    issues = [i for i, s in enumerate(scores) if (s > 0 and avg < 0) or (s < 0 and avg > 0)]
    return issues


def check_plausibility(assertion: str) -> Tuple[float, str]:
    """Naive plausibility check using an absurd phrase list."""
    low = any(p in assertion.lower() for p in _ABSURD)
    return (0.1, "contains absurd phrase") if low else (0.9, "plausible")
