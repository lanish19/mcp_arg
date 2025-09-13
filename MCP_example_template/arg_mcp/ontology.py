from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple
import csv
import os
import math

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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
                    dimension=(r.get("Dimension") or "").strip(),
                    category=(r.get("Category") or "").strip(),
                    bucket=(r.get("Bucket") or "").strip(),
                    description=(r.get("Description") or "").strip(),
                    example=(r.get("Example") or "").strip(),
                    note=(r.get("NOTE") or "").strip(),
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
        q = _normalize(query)
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
        threshold: float = 0.2,
        max_results: int = 10,
    ) -> List[Dict[str, Any]]:
        if not query.strip():
            return []
        vec = self._vectorizer.transform([query])
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

    def list(self) -> List[str]:
        return sorted(self._name_index.keys())

    def get(self, name: str) -> Optional[Dict[str, str]]:
        r = self._name_index.get(name.lower())
        if not r:
            return None
        return {"name": r.name, "when": r.when, "purpose": r.purpose, "how": r.how}

    def search(self, query: str) -> List[Dict[str, str]]:
        q = _normalize(query)
        out: List[Dict[str, str]] = []
        for r in self.rows:
            blob = _normalize("\n".join([r.when, r.name, r.purpose, r.how]))
            if q in blob:
                out.append({"name": r.name, "when": r.when, "purpose": r.purpose, "how": r.how})
        return out

