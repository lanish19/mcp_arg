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

    def __init__(self, ontology: Ontology, top_n: int = 5) -> None:
        self.ontology = ontology
        self.top_n = int(top_n) if isinstance(top_n, int) and top_n > 0 else 5
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
            # crude trigger: use the most indicative token for the type
            trigger = None
            if ptype == "causal":
                trigger = "because" if "because" in text else ("leads to" if "leads to" in text else None)
            elif ptype == "authority":
                trigger = "experts" if "expert" in text.lower() or "experts" in text.lower() else None
            elif ptype == "analogical":
                trigger = "like" if " like " in text else None
            elif ptype == "normative":
                trigger = "should" if "should" in text.lower() or "ought" in text.lower() else None

            # Compute a coarse span from trigger if possible
            span = None
            if trigger:
                t_l = text.lower()
                idx = t_l.find(trigger.lower())
                if idx >= 0:
                    span = (idx, idx + len(trigger))

            # Fallback: ensure span is always populated using TF-IDF term present in both row and text
            if span is None:
                try:
                    # Terms present in the ontology row i
                    row_vec = self.matrix[i]
                    row_term_indices = set(row_vec.indices.tolist()) if hasattr(row_vec, "indices") else set()
                    # Terms weighted in the input text
                    text_vec = vec[0]
                    best_term_idx: Optional[int] = None
                    best_weight: float = -1.0
                    if hasattr(text_vec, "indices"):
                        for idx_j, weight in zip(text_vec.indices.tolist(), text_vec.data.tolist()):
                            if idx_j in row_term_indices and weight > best_weight:
                                best_weight = float(weight)
                                best_term_idx = int(idx_j)
                    if best_term_idx is not None:
                        # Map index back to token string
                        vocab_inv = {v: k for k, v in self.vectorizer.vocabulary_.items()}
                        term = vocab_inv.get(best_term_idx, "")
                        if term:
                            t_l = text.lower()
                            j = t_l.find(term.lower())
                            if j >= 0:
                                span = (j, j + len(term))
                except Exception:
                    # Silent fallback continues below
                    pass

            # Absolute fallback to first token in text to guarantee non-null span
            if span is None:
                import re as _re
                m = _re.search(r"\w+", text)
                if m:
                    span = (m.start(), m.end())
                else:
                    span = (0, min(1, len(text)))
            out.append(
                Pattern(
                    pattern_id=f"scheme_{i}",
                    pattern_type=ptype,
                    label=row.category,
                    span=span,
                    confidence=float(score),
                    details={"scheme": row.category, "score": float(score)},
                    roles=roles,

                )
            )
        out.sort(key=lambda p: p.confidence, reverse=True)
        # Keep top N only to reduce noise
        trimmed = out[: self.top_n]
        # Compact details payload
        for p in trimmed:
            if isinstance(p.details, dict):
                p.details = {k: p.details[k] for k in ("scheme", "score") if k in p.details}
        return trimmed


# Backwards compatibility export
PatternDetector = EnhancedPatternDetector

