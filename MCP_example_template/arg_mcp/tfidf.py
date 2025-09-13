from __future__ import annotations

from math import log, sqrt
from typing import Dict, List
import re


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


class TfidfVectorizer:
    """Lightweight TF-IDF vectorizer with a minimal scikit-learn-like API.

    - fit(corpus) -> self
    - transform(docs) -> List[Dict[int, float]] (sparse vectors indexed by vocab id)
    """

    def __init__(self) -> None:
        self.vocabulary_: Dict[str, int] = {}
        self._idf: Dict[int, float] = {}

    def fit(self, corpus: List[str]) -> "TfidfVectorizer":
        # Build vocabulary and document frequencies
        vocab: Dict[str, int] = {}
        df_counts: Dict[int, int] = {}
        next_idx = 0
        for doc in corpus:
            tokens = _tokenize(doc)
            seen: Dict[int, bool] = {}
            for tok in tokens:
                if tok not in vocab:
                    vocab[tok] = next_idx
                    next_idx += 1
                idx = vocab[tok]
                if idx not in seen:
                    df_counts[idx] = df_counts.get(idx, 0) + 1
                    seen[idx] = True
        self.vocabulary_ = vocab
        n_docs = max(1, len(corpus))
        # Smooth IDF similar to sklearn: log((1 + n)/(1 + df)) + 1
        self._idf = {idx: log((1.0 + n_docs) / (1.0 + df)) + 1.0 for idx, df in df_counts.items()}
        return self

    def transform(self, docs: List[str]) -> List[Dict[int, float]]:
        vectors: List[Dict[int, float]] = []
        for doc in docs:
            tokens = _tokenize(doc)
            if not tokens:
                vectors.append({})
                continue
            tf_counts: Dict[int, int] = {}
            for tok in tokens:
                idx = self.vocabulary_.get(tok)
                if idx is None:
                    continue
                tf_counts[idx] = tf_counts.get(idx, 0) + 1
            length = float(len(tokens))
            vec: Dict[int, float] = {}
            for idx, count in tf_counts.items():
                idf = self._idf.get(idx, 1.0)
                tf = count / length
                vec[idx] = tf * idf
            vectors.append(vec)
        return vectors


def cosine_similarity(vecs: List[Dict[int, float]], matrix: List[Dict[int, float]]) -> List[List[float]]:
    """Compute cosine similarity between each vec in vecs and each row in matrix.

    Inputs are sparse dicts mapping index->value. Returns a 2D list of similarities.
    """
    # Precompute norms for matrix rows
    def norm(v: Dict[int, float]) -> float:
        return sqrt(sum(val * val for val in v.values())) or 1.0

    matrix_norms = [norm(row) for row in matrix]

    results: List[List[float]] = []
    for v in vecs:
        vnorm = norm(v)
        sims_row: List[float] = []
        v_items = v.items()
        for row, rnorm in zip(matrix, matrix_norms):
            # dot product over intersection of indices
            if len(v) < len(row):
                dot = sum(val * row.get(idx, 0.0) for idx, val in v_items)
            else:
                dot = sum(row_val * v.get(idx, 0.0) for idx, row_val in row.items())
            sims_row.append(dot / (vnorm * rnorm))
        results.append(sims_row)
    return results


