from __future__ import annotations

"""Dynamic inference rule engine for argument schemes."""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


@dataclass
class Requirement:
    name: str
    description: str
    probe: Optional[str] = None


@dataclass
class RequirementResult:
    requirement: Requirement
    satisfied: bool
    score: float
    evidence: List[str]
    missing_premise: Optional[str] = None



@dataclass
class MissingAssumption:
    """Legacy container for backward compatibility."""
    text: str
    category: str
    rationale: str
    priority: str  # critical, testable, controversial, other


RequirementCheck = Callable[[str, Dict[str, str]], RequirementResult]


def check_temporal_order(text: str, roles: Dict[str, str]) -> RequirementResult:
    req = Requirement("temporal_order", "Cause must precede effect", probe="timeline_search")
    cause = roles.get("cause", "")
    effect = roles.get("effect", "")
    satisfied = "before" in text.lower() or "precedes" in text.lower()
    missing = None if satisfied else f"Clarify whether '{cause}' occurs before '{effect}'"
    return RequirementResult(req, satisfied, 1.0 if satisfied else 0.2, [text], missing)


def check_mechanism_presence(text: str, roles: Dict[str, str]) -> RequirementResult:
    req = Requirement("mechanism", "Mechanism linking cause and effect")
    satisfied = any(w in text.lower() for w in ["because", "via", "through", "mechanism"])
    missing = None if satisfied else "Specify mechanism linking cause and effect"
    return RequirementResult(req, satisfied, 0.9 if satisfied else 0.3, [text], missing)


def check_confounds(text: str, roles: Dict[str, str]) -> RequirementResult:
    req = Requirement("confound_control", "Confounding factors addressed")
    satisfied = "control" in text.lower() or "random" in text.lower()
    missing = None if satisfied else "Identify and control potential confounders"
    return RequirementResult(req, satisfied, 0.8 if satisfied else 0.2, [text], missing)


def check_expertise(text: str, roles: Dict[str, str]) -> RequirementResult:
    req = Requirement("expertise", "Source has relevant expertise")
    satisfied = any(t in text.lower() for t in ["dr", "prof", "expert"])
    missing = None if satisfied else "Provide credentials for the cited authority"
    return RequirementResult(req, satisfied, 0.8 if satisfied else 0.2, [text], missing)


def check_bias(text: str, roles: Dict[str, str]) -> RequirementResult:
    req = Requirement("bias", "Potential bias disclosed")
    satisfied = "independent" in text.lower() or "unbiased" in text.lower()
    missing = None if satisfied else "Clarify potential biases of the authority"
    return RequirementResult(req, satisfied, 0.7 if satisfied else 0.3, [text], missing)


def check_analogy_similarity(text: str, roles: Dict[str, str]) -> RequirementResult:
    req = Requirement("similarity", "Analogy based on relevant similarities")
    base = roles.get("base", "")
    target = roles.get("target", "")
    satisfied = base and target and any(w in text.lower() for w in ["like", "similar", "just as"])
    missing = None if satisfied else f"Explain why {base} is similar to {target}"
    return RequirementResult(req, satisfied, 0.9 if satisfied else 0.2, [text], missing)


def check_scope(text: str, roles: Dict[str, str]) -> RequirementResult:
    req = Requirement("scope", "Limits of the analogy are stated")
    satisfied = "unlike" in text.lower() or "however" in text.lower()
    missing = None if satisfied else "State disanalogies or scope limits"
    return RequirementResult(req, satisfied, 0.8 if satisfied else 0.3, [text], missing)


SCHEME_CHECKS: Dict[str, List[RequirementCheck]] = {
    "Argument from Cause to Effect": [check_temporal_order, check_mechanism_presence, check_confounds],
    "Argument from Expert Opinion": [check_expertise, check_bias],
    "Argument from Analogy": [check_analogy_similarity, check_scope],
}


@dataclass
class SchemeEvaluation:
    scheme: str
    requirements: List[RequirementResult]
    confidence: float
    generated_assumptions: List[str]


class InferenceEngine:
    def __init__(self, ontology, profile) -> None:
        self.ontology = ontology
        self.profile = profile

    def evaluate_scheme(self, candidate: Dict[str, str]) -> SchemeEvaluation:
        scheme = candidate.get("scheme")
        checks = SCHEME_CHECKS.get(scheme, []) + [self._wrap_extra(r) for r in self.profile.extra_requirements.get(candidate.get("scheme_key", ""), [])]
        results: List[RequirementResult] = []
        for chk in checks:
            results.append(chk(candidate.get("text", ""), candidate.get("roles", {})))
        conf = sum(r.score for r in results) / len(results) if results else 0.0
        assumptions = [r.missing_premise for r in results if not r.satisfied and r.missing_premise]
        return SchemeEvaluation(scheme=scheme or "", requirements=results, confidence=conf, generated_assumptions=assumptions)

    def _wrap_extra(self, name: str) -> RequirementCheck:
        def _inner(text: str, roles: Dict[str, str]) -> RequirementResult:
            req = Requirement(name, f"Domain-specific requirement {name}")
            return RequirementResult(req, False, 0.0, [text], f"Address {name}")
        return _inner



# Backwards compatibility exports
GapAnalyzer = InferenceEngine


# ---- Assumption Generator (pattern-driven) ----
class AssumptionGenerator:
    def __init__(self, ontology=None) -> None:
        self.ontology = ontology

    def _authority(self, pattern: Dict[str, str]) -> List[Dict[str, str]]:
        pid = pattern.get("pattern_id")
        return [
            {
                "text": "The cited experts are qualified in the relevant domain",
                "category": "epistemic",
                "impact": "high",
                "confidence": 0.8,
                "linked_patterns": [pid] if pid else [],
                "tests": [
                    "Check expert credentials and affiliations",
                    "Verify domain relevance of expertise",
                    "Review publication history and recognition",
                ],
            },
            {
                "text": "The experts are free from conflicts of interest",
                "category": "reliability",
                "impact": "high",
                "confidence": 0.7,
                "linked_patterns": [pid] if pid else [],
                "tests": [
                    "Identify funding sources",
                    "Assess organizational or ideological pressures",
                    "Compare to independent expert views",
                ],
            },
            {
                "text": "There is expert consensus on this topic",
                "category": "consensus",
                "impact": "medium",
                "confidence": 0.6,
                "linked_patterns": [pid] if pid else [],
                "tests": [
                    "Survey experts",
                    "Check professional organization positions",
                    "Review meta-analyses",
                ],
            },
        ]

    def _causal(self, pattern: Dict[str, str]) -> List[Dict[str, str]]:
        pid = pattern.get("pattern_id")
        return [
            {
                "text": "The cause precedes the effect temporally",
                "category": "temporal",
                "impact": "critical",
                "confidence": 0.9,
                "linked_patterns": [pid] if pid else [],
                "tests": [
                    "Verify chronological sequence",
                    "Check for reverse causation",
                    "Assess temporal gaps",
                ],
            },
            {
                "text": "No alternative causes adequately explain the effect",
                "category": "alternative_causation",
                "impact": "high",
                "confidence": 0.7,
                "linked_patterns": [pid] if pid else [],
                "tests": [
                    "Identify and test alternatives",
                    "Check for confounders",
                    "Use controlled comparisons",
                ],
            },
            {
                "text": "A plausible mechanism links cause to effect",
                "category": "mechanistic",
                "impact": "high",
                "confidence": 0.8,
                "linked_patterns": [pid] if pid else [],
                "tests": [
                    "Identify intermediate steps",
                    "Test mechanism under conditions",
                    "Compare to analogous mechanisms",
                ],
            },
        ]

    def _analogical(self, pattern: Dict[str, str]) -> List[Dict[str, str]]:
        pid = pattern.get("pattern_id")
        return [
            {
                "text": "The analogy hinges on relevant similarities, not superficial ones",
                "category": "bridging",
                "impact": "medium",
                "confidence": 0.6,
                "linked_patterns": [pid] if pid else [],
                "tests": [
                    "List key relevant similarities",
                    "List critical disanalogies",
                    "Test boundary conditions",
                ],
            }
        ]

    def generate(self, text: str, patterns: List[Dict[str, str]], components: Optional[Dict[str, str]] = None) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        for p in patterns:
            ptype = (p.get("pattern_type") or "").lower()
            if ptype == "authority":
                out.extend(self._authority(p))
            elif ptype == "causal":
                out.extend(self._causal(p))
            elif ptype == "analogical":
                out.extend(self._analogical(p))
        # Simple dedupe by text
        seen = set()
        uniq: List[Dict[str, str]] = []
        for a in out:
            t = a.get("text")
            if t and t not in seen:
                uniq.append(a)
                seen.add(t)
        return uniq
