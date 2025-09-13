from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

from .tfidf import TfidfVectorizer, cosine_similarity

from .ontology import Ontology, OntologyRow


@dataclass
class Pattern:
    pattern_id: str
    pattern_type: str  # causal, authority, analogical, normative, etc.
    label: str
    span: Optional[Tuple[int, int]]
    confidence: float
    details: Dict[str, Any]
    roles: Dict[str, str] = None


class EnhancedPatternDetector:
    """Semantic pattern detector using ontology descriptions instead of regex."""

    def __init__(self, ontology: Ontology) -> None:
        self.ontology = ontology
        self.scheme_rows: List[OntologyRow] = [r for r in ontology.rows if r.dimension == "Argument Scheme"]
        corpus = [" ".join([r.category, r.bucket, r.description, r.example]) for r in self.scheme_rows]
        if corpus:
            self.vectorizer = TfidfVectorizer().fit(corpus)
            self.matrix = self.vectorizer.transform(corpus)
        else:
            self.vectorizer = TfidfVectorizer().fit([""])
            self.matrix = self.vectorizer.transform([""])

    def _map_pattern_type(self, row: OntologyRow) -> str:
        cat = f"{row.category} {row.bucket}".lower()
        if "cause" in cat or "consequence" in cat:
            return "causal"
        if "expert" in cat or "authority" in cat or "testimony" in cat:
            return "authority"
        if "analog" in cat:
            return "analogical"
        if "practical" in cat or "policy" in cat:
            return "normative"
        return "other"

    def detect(self, text: str) -> List[Pattern]:
        if not text.strip():
            return []
        vec = self.vectorizer.transform([text])
        sims = cosine_similarity(vec, self.matrix)[0]
        out: List[Pattern] = []
        for i, score in enumerate(sims):
            if score <= 0:
                continue
            row = self.scheme_rows[i]
            ptype = self._map_pattern_type(row)
            roles: Dict[str, str] = {}
            if ptype == "causal":
                if "because" in text:
                    parts = text.split("because", 1)
                    roles = {"effect": parts[0].strip(), "cause": parts[1].strip()}
                elif "leads to" in text:
                    parts = text.split("leads to", 1)
                    roles = {"cause": parts[0].strip(), "effect": parts[1].strip()}
            if ptype == "analogical" and " like " in text:
                parts = text.split(" like ", 1)
                roles = {"target": parts[0].strip(), "base": parts[1].strip()}

            out.append(
                Pattern(
                    pattern_id=f"scheme_{i}",
                    pattern_type=ptype,
                    label=row.category,
                    span=None,
                    confidence=float(score),
                    details={"scheme": row.category, "score": float(score)},
                    roles=roles,

                )
            )
        out.sort(key=lambda p: p.confidence, reverse=True)
        return out[:10]


# Backwards compatibility export
PatternDetector = EnhancedPatternDetector

