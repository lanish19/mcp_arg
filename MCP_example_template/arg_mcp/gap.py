from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, DefaultDict


@dataclass
class MissingAssumption:
    text: str
    category: str
    rationale: str
    priority: str  # critical, testable, controversial, other


class DynamicGapAnalyzer:
    """Identify missing assumptions based on detected argument schemes."""

    def __init__(self) -> None:
        self.templates: Dict[str, List[MissingAssumption]] = {
            "Argument from Expert Opinion": [
                MissingAssumption(
                    text="Expertise relevance and scope alignment",
                    category="Authority Preconditions",
                    rationale="Expert’s domain should match the claim’s scope.",
                    priority="critical",
                ),
                MissingAssumption(
                    text="Bias and conflict-of-interest checked",
                    category="Authority Preconditions",
                    rationale="Funding/affiliations may undercut credibility.",
                    priority="testable",
                ),
                MissingAssumption(
                    text="Consensus and replication status known",
                    category="Authority Preconditions",
                    rationale="Single testimony is weaker than replicated consensus.",
                    priority="controversial",
                ),
            ],
            "Argument from Cause to Effect": [
                MissingAssumption(
                    text="No simultaneity or reverse causation",
                    category="Causal ID",
                    rationale="Causal phrasing implies temporal precedence and one-way influence.",
                    priority="critical",
                ),
                MissingAssumption(
                    text="Unmeasured confounding ruled out",
                    category="Causal ID",
                    rationale="Association could be driven by omitted variables; require design or control.",
                    priority="critical",
                ),
                MissingAssumption(
                    text="Mechanism plausibility or pathway specified",
                    category="Mechanistic",
                    rationale="Bridges cause to effect; improves transportability and diagnosis of failures.",
                    priority="testable",
                ),
            ],
            "Argument from Analogy": [
                MissingAssumption(
                    text="Similarity dimensions specified and relevant",
                    category="Analogy Preconditions",
                    rationale="Analogy requires shared, problem-relevant properties.",
                    priority="critical",
                ),
                MissingAssumption(
                    text="Scope and limits of analogy stated",
                    category="Analogy Preconditions",
                    rationale="Prevents over-generalization beyond justified range.",
                    priority="testable",
                ),
            ],
            "Practical Reasoning": [
                MissingAssumption(
                    text="Goals, constraints, and trade-offs explicit",
                    category="Practical Reasoning",
                    rationale="Action claims depend on prioritized values and constraints.",
                    priority="critical",
                ),
            ],
        }

    def analyze(self, patterns: List[Dict[str, Any]]) -> List[MissingAssumption]:
        schemes = {p.get("details", {}).get("scheme") or p.get("label") for p in patterns}
        out: List[MissingAssumption] = []
        for s in schemes:
            for tpl in self.templates.get(s, []):
                out.append(tpl)
        return out


# Backwards compatibility export
GapAnalyzer = DynamicGapAnalyzer

