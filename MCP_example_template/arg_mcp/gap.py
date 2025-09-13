from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class MissingAssumption:
    text: str
    category: str  # e.g., Causal ID, Authority Preconditions, Analogy Preconditions
    rationale: str
    priority: str  # critical, testable, controversial, other


class GapAnalyzer:
    def analyze(self, patterns: List[Dict[str, Any]]) -> List[MissingAssumption]:
        out: List[MissingAssumption] = []
        types = {p.get("pattern_type") for p in patterns}

        if "causal" in types:
            out.append(MissingAssumption(
                text="No Simultaneity/Reverse Causation",
                category="Causal ID",
                rationale="Causal phrasing implies temporal precedence and one-way influence.",
                priority="critical",
            ))
            out.append(MissingAssumption(
                text="Unmeasured Confounding controlled or ruled out",
                category="Causal ID",
                rationale="Association could be driven by omitted variables; require design or control.",
                priority="critical",
            ))
            out.append(MissingAssumption(
                text="Mechanism plausibility or pathway specified",
                category="Mechanistic",
                rationale="Bridges cause to effect; improves transportability and diagnosis of failures.",
                priority="testable",
            ))

        if "authority" in types:
            out.append(MissingAssumption(
                text="Expertise relevance and scope alignment",
                category="Authority Preconditions",
                rationale="Expert’s domain should match the claim’s scope.",
                priority="critical",
            ))
            out.append(MissingAssumption(
                text="Bias and conflict-of-interest checked",
                category="Authority Preconditions",
                rationale="Funding/affiliations may undercut credibility.",
                priority="testable",
            ))
            out.append(MissingAssumption(
                text="Consensus and replication status known",
                category="Authority Preconditions",
                rationale="Single testimony is weaker than replicated consensus.",
                priority="controversial",
            ))

        if "analogical" in types:
            out.append(MissingAssumption(
                text="Similarity dimensions specified and relevant",
                category="Analogy Preconditions",
                rationale="Analogy requires shared, problem-relevant properties.",
                priority="critical",
            ))
            out.append(MissingAssumption(
                text="Scope and limits of analogy stated",
                category="Analogy Preconditions",
                rationale="Prevents over-generalization beyond justified range.",
                priority="testable",
            ))

        if "normative" in types:
            out.append(MissingAssumption(
                text="Goals, constraints, and trade-offs explicit",
                category="Practical Reasoning",
                rationale="Action claims depend on prioritized values and constraints.",
                priority="critical",
            ))

        return out

