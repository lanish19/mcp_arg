from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import re


@dataclass
class Pattern:
    pattern_id: str
    pattern_type: str  # causal, authority, analogical, normative, quantifier, temporal, statistical, definition
    label: str
    span: Optional[Tuple[int, int]]
    confidence: float
    details: Dict[str, Any]


class PatternDetector:
    def __init__(self) -> None:
        # Compile regex for core patterns with some paraphrase coverage
        self.re_causal = re.compile(r"\b(because|since|leads to|results in|due to|therefore|thus|so that|causes?)\b", re.I)
        self.re_authority = re.compile(r"\b(experts? (say|agree)|studies? (show|find)|researchers?|doctor|dr\.|prof(essor)?|according to)\b", re.I)
        self.re_analogical = re.compile(r"\b(like|similar to|analog(ous)?|as if|just as)\b", re.I)
        self.re_normative = re.compile(r"\b(should|ought to|must|need to|the best way|we should)\b", re.I)
        self.re_quantifier = re.compile(r"\b(always|never|everyone|no one|all|none|most|many|few)\b", re.I)
        self.re_temporal = re.compile(r"\b(before|after|until|while|prior to|subsequently|eventually|now)\b", re.I)
        self.re_statistical = re.compile(r"\b(percent|%|p-value|statistical(ly)? significant|correlat(es|ion)|regression|sample|random|control group)\b", re.I)
        self.re_definition = re.compile(r"\b(defines? as|means that|by definition|we call|is defined as)\b", re.I)

    def _scan(self, text: str, kind: str, regex: re.Pattern, label: str) -> List[Pattern]:
        out: List[Pattern] = []
        for m in regex.finditer(text):
            conf = 0.7
            if kind in ("quantifier", "temporal"):
                conf = 0.5
            out.append(Pattern(pattern_id=f"pat_{m.start()}_{m.end()}", pattern_type=kind, label=label, span=(m.start(), m.end()), confidence=conf, details={"match": m.group(0)}))
        return out

    def detect(self, text: str) -> List[Pattern]:
        L = text
        pats: List[Pattern] = []
        pats += self._scan(L, "causal", self.re_causal, "Causal Indicator")
        pats += self._scan(L, "authority", self.re_authority, "Authority Marker")
        pats += self._scan(L, "analogical", self.re_analogical, "Analogical Language")
        pats += self._scan(L, "normative", self.re_normative, "Normative/Practical")
        pats += self._scan(L, "quantifier", self.re_quantifier, "Quantifier/Generality")
        pats += self._scan(L, "temporal", self.re_temporal, "Temporal Cue")
        pats += self._scan(L, "statistical", self.re_statistical, "Statistical/Evidence Cue")
        pats += self._scan(L, "definition", self.re_definition, "Definition/Category Cue")
        return pats

