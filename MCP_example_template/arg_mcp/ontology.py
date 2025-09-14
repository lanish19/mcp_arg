from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple
import csv
import os
import math
def _sanitize(s: str) -> str:
    if s is None:
        return ""
    # Normalize quotes and whitespace
    t = s.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    t = " ".join(t.split())
    return t

def _slugify(name: str) -> str:
    import re
    s = (name or "").lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s

# Simple synonyms map for normalization
SYNONYMS: Dict[str, str] = {
    "false trilemma": "false dilemma",
    "trilemma": "false dilemma",
    "appeal to authority": "ad verecundiam",
    "appeal to popularity": "ad populum",
    "post hoc": "post hoc ergo propter hoc",
}

from .tfidf import TfidfVectorizer, cosine_similarity


@dataclass
class OntologyRow:
    dimension: str
    category: str
    bucket: str
    description: str
    example: str
    note: str


def _normalize(s: str) -> str:
    return " ".join(s.lower().strip().split())


def load_ontology(path: str) -> List[OntologyRow]:
    rows: List[OntologyRow] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                OntologyRow(
                    dimension=_sanitize((r.get("Dimension") or "").strip()),
                    category=_sanitize((r.get("Category") or "").strip()),
                    bucket=_sanitize((r.get("Bucket") or "").strip()),
                    description=_sanitize((r.get("Description") or "").strip()),
                    example=_sanitize((r.get("Example") or "").strip()),
                    note=_sanitize((r.get("NOTE") or "").strip()),
                )
            )
    return rows


class Ontology:
    def __init__(self, rows: List[OntologyRow]):
        self.rows = rows
        self.dimensions = sorted({r.dimension for r in rows if r.dimension})
        self._dim_to_cats: Dict[str, List[str]] = {}
        for d in self.dimensions:
            cats = sorted({r.category for r in rows if r.dimension == d and r.category})
            self._dim_to_cats[d] = cats
        # Indexes
        self._bucket_index: Dict[str, List[OntologyRow]] = {}
        self._dim_index: Dict[str, List[OntologyRow]] = {}
        for r in rows:
            bkey = _normalize(r.bucket)
            if bkey:
                self._bucket_index.setdefault(bkey, []).append(r)
            self._dim_index.setdefault(r.dimension, []).append(r)

        # Pre-compute TF-IDF embeddings for semantic similarity
        corpus = [" ".join([r.dimension, r.category, r.bucket, r.description, r.example, r.note]) for r in rows]
        if corpus:
            self._vectorizer = TfidfVectorizer().fit(corpus)
            self._matrix = self._vectorizer.transform(corpus)
        else:
            self._vectorizer = TfidfVectorizer().fit([""])
            self._matrix = self._vectorizer.transform([""])
        self._last_applied_synonyms: List[Tuple[str, str]] = []

    def list_dimensions(self) -> List[str]:
        return self.dimensions

    def list_categories(self, dimension: str) -> List[str]:
        return self._dim_to_cats.get(dimension, [])

    def list_buckets(self, dimension: Optional[str] = None, category: Optional[str] = None) -> List[str]:
        items = self.rows
        if dimension:
            items = [r for r in items if r.dimension == dimension]
        if category:
            items = [r for r in items if r.category == category]
        return sorted({r.bucket for r in items if r.bucket})

    def bucket_detail(self, name: str) -> List[Dict[str, str]]:
        key = _normalize(name)
        out = []
        for r in self._bucket_index.get(key, []):
            out.append({
                "dimension": r.dimension,
                "category": r.category,
                "bucket": r.bucket,
                "description": r.description,
                "example": r.example,
                "note": r.note,
            })
        return out

    def search(self, query: str, dimension: Optional[str] = None, category: Optional[str] = None, bucket: Optional[str] = None) -> List[Dict[str, str]]:
        q_raw = _normalize(query)
        applied: List[Tuple[str, str]] = []
        q = q_raw
        if q_raw in SYNONYMS:
            q = _normalize(SYNONYMS[q_raw])
            applied.append((q_raw, q))
        self._last_applied_synonyms = applied
        out: List[Dict[str, str]] = []
        for r in self.rows:
            if dimension and r.dimension != dimension:
                continue
            if category and r.category != category:
                continue
            if bucket and r.bucket != bucket:
                continue
            blob = _normalize("\n".join([r.dimension, r.category, r.bucket, r.description, r.example, r.note]))
            if q in blob:
                out.append({
                    "dimension": r.dimension,
                    "category": r.category,
                    "bucket": r.bucket,
                    "description": r.description,
                    "example": r.example,
                    "note": r.note,
                })
        return out

    # Semantic similarity using TF-IDF cosine similarity
    def semantic_search(
        self,
        query: str,
        dimensions: Optional[List[str]] = None,
        threshold: float = 0.35,
        max_results: int = 5,
    ) -> List[Dict[str, Any]]:
        if not query.strip():
            return []
        q_raw = query
        applied: List[Tuple[str, str]] = []
        q_use = q_raw
        norm = _normalize(q_raw)
        if norm in SYNONYMS:
            q_use = SYNONYMS[norm]
            applied.append((q_raw, q_use))
        self._last_applied_synonyms = applied
        vec = self._vectorizer.transform([q_use])
        sims = cosine_similarity(vec, self._matrix)[0]
        results: List[Tuple[float, OntologyRow]] = []
        for score, row in zip(sims, self.rows):
            if dimensions and row.dimension not in dimensions:
                continue
            if score >= threshold:
                results.append((float(score), row))
        results.sort(key=lambda x: x[0], reverse=True)
        out: List[Dict[str, Any]] = []
        for score, r in results[:max_results]:
            out.append({
                "score": round(float(score), 4),
                "dimension": r.dimension,
                "category": r.category,
                "bucket": r.bucket,
                "description": r.description,
                "example": r.example,
            })
        return out

    def last_applied_synonyms(self) -> List[Tuple[str, str]]:
        return list(self._last_applied_synonyms)

    def scheme_requirements(self, scheme: str) -> List[str]:
        mapping = {
            "Argument from Cause to Effect": ["temporal_order", "mechanism", "confound_control"],
            "Argument from Analogy": ["similarity", "scope"],
            "Argument from Expert Opinion": ["expertise", "bias", "consensus"],
            "Practical Reasoning": ["goal_specified"],
        }
        return mapping.get(scheme, [])


@dataclass
class ToolRow:
    when: str
    name: str
    purpose: str
    how: str


def load_tool_catalog(path: str) -> List[ToolRow]:
    rows: List[ToolRow] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                ToolRow(
                    when=(r.get("When to Use") or "").strip(),
                    name=(r.get("Tool Name") or "").strip(),
                    purpose=(r.get("Purpose") or "").strip(),
                    how=(r.get("How") or "").strip(),
                )
            )
    return rows


class ToolCatalog:
    def __init__(self, rows: List[ToolRow]):
        self.rows = rows
        self._name_index: Dict[str, ToolRow] = {r.name.lower(): r for r in rows if r.name}
        self._slug_index: Dict[str, ToolRow] = {_slugify(r.name): r for r in rows if r.name}
        # derive tags per tool
        self._tags: Dict[str, List[str]] = {}
        for r in rows:
            tags: List[str] = []
            n = (r.name or "").lower()
            w = (r.when or "").lower() + " " + (r.purpose or "").lower() + " " + (r.how or "").lower()
            # Phase/theme heuristics
            if any(k in w for k in ["assumption", "linchpin", "audit"]):
                tags.append("assumption")
            if any(k in w for k in ["causal", "confound", "mechanism", "quasi-experimental"]):
                tags.append("causal")
            if any(k in w for k in ["toulmin", "warrant", "claim"]):
                tags.append("structure")
            if any(k in w for k in ["triangulation", "replication", "validation"]):
                tags.append("validation")
            if any(k in n for k in ["dilemma", "disassembler"]):
                tags.append("dilemma")
            self._tags[_slugify(r.name)] = sorted(list(set(tags)))
        # Build semantic index over tools (name + purpose + how + when)
        docs = [
            _normalize(" ".join([t.name or "", t.purpose or "", t.how or "", t.when or ""]))
            for t in rows
        ]
        if docs:
            self._tool_vectorizer = TfidfVectorizer().fit(docs)
            self._tool_matrix = self._tool_vectorizer.transform(docs)
        else:
            self._tool_vectorizer = TfidfVectorizer().fit([""])
            self._tool_matrix = self._tool_vectorizer.transform([""])

    def list(self) -> List[str]:
        return sorted([r.name for r in self._name_index.values()])

    def get(self, name: str) -> Optional[Dict[str, str]]:
        if not name:
            return None
        r = self._name_index.get(name.lower()) or self._slug_index.get(_slugify(name))
        if not r:
            return None
        slug = _slugify(r.name)
        return {"name": r.name, "when": r.when, "purpose": r.purpose, "how": r.how, "slug": slug, "tags": self._tags.get(slug, [])}

    def search(self, query: str) -> List[Dict[str, str]]:
        q = _normalize(query)
        # tag filter syntax: tags:any:a,b  OR tags:all:x,y
        tag_any: List[str] = []
        tag_all: List[str] = []
        if q.startswith("tags:"):
            try:
                _, expr = q.split(":", 1)
                mode, rest = expr.split(":", 1)
                tags = [t.strip() for t in rest.split(",") if t.strip()]
                if mode == "any":
                    tag_any = tags
                elif mode == "all":
                    tag_all = tags
            except Exception:
                pass
        out: List[Dict[str, str]] = []
        for r in self.rows:
            slug = _slugify(r.name)
            tags = self._tags.get(slug, [])
            if tag_any or tag_all:
                if tag_any and not any(t in tags for t in tag_any):
                    continue
                if tag_all and not all(t in tags for t in tag_all):
                    continue
                out.append({"name": r.name, "when": r.when, "purpose": r.purpose, "how": r.how, "slug": slug, "tags": tags})
                continue
            blob = _normalize("\n".join([r.when, r.name, r.purpose, r.how]))
            if q in blob:
                out.append({"name": r.name, "when": r.when, "purpose": r.purpose, "how": r.how, "slug": slug, "tags": tags})
        return out

    def semantic_search_tools(self, query: str, threshold: float = 0.1, max_results: int = 10) -> List[Dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            return []
        q = _normalize(query)
        vec = self._tool_vectorizer.transform([q])
        sims = cosine_similarity(vec, self._tool_matrix)[0]
        scored: List[Tuple[float, ToolRow]] = []
        for score, row in zip(sims, self.rows):
            if score >= threshold:
                scored.append((float(score), row))
        scored.sort(key=lambda x: x[0], reverse=True)
        out: List[Dict[str, Any]] = []
        for score, r in scored[:max_results]:
            slug = _slugify(r.name)
            out.append({
                "score": round(float(score), 4),
                "name": r.name,
                "when": r.when,
                "purpose": r.purpose,
                "how": r.how,
                "slug": slug,
                "tags": self._tags.get(slug, []),
            })
        return out

    def dump(self, page: int = 1, per_page: int = 50, fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        page = max(1, int(page or 1))
        per_page = max(1, min(200, int(per_page or 50)))
        start = (page - 1) * per_page
        end = start + per_page
        cols = fields or ["name", "when", "purpose", "how", "slug", "tags"]
        out: List[Dict[str, Any]] = []
        for r in self.rows[start:end]:
            slug = _slugify(r.name)
            row = {
                "name": r.name,
                "when": r.when,
                "purpose": r.purpose,
                "how": r.how,
                "slug": slug,
                "tags": self._tags.get(slug, []),
            }
            out.append({k: row[k] for k in cols if k in row})
        return out
