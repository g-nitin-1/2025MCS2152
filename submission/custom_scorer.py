"""
submission/custom_scorer.py — the combined scorer (assignment Section 4.1,
"where separation in the leaderboard tends to happen").

A linear blend of three signals, all computed from the same inverted index:

    final = BM25(k1, b)
          + W_COVERAGE * coverage(q, d)
          + W_VSM      * cosine(q, d)

where coverage(q, d) is the summed IDF of the *distinct* query terms that
document d matches at all. BM25 already rewards matching a rare term, but
it saturates per-term; coverage is deliberately flat per term, so it
separates a document matching four of five query terms weakly from one
matching a single query term strongly. On question-style topics ("what
drugs have been active against SARS-CoV or SARS-CoV-2 in animal studies")
that distinction matters.

Weights live in submission/params.py and are swept on the dev set. With
both weights at 0 this reduces exactly to BM25, so the blend can never do
worse than the scorer it extends without that showing up in the sweep.
"""
from collections import Counter
from typing import List, Optional, Tuple

import numpy as np

from submission import bm25, boolean_vsm, params
from submission.indexer import InvertedIndex, tokenize

_INDEX: Optional[InvertedIndex] = None
_IDF: Optional[np.ndarray] = None


def build(index: InvertedIndex) -> None:
    """Called from retrieve.load_index(). bm25.build()/boolean_vsm.build()
    are called there too; this only adds the coverage IDF table."""
    global _INDEX, _IDF
    _INDEX = index
    df = np.diff(index.offsets).astype(np.float64)
    _IDF = np.log(float(index.N) / np.maximum(df, 1.0))


def _coverage(query: str) -> np.ndarray:
    assert _INDEX is not None and _IDF is not None
    index = _INDEX
    cov = np.zeros(index.N, dtype=np.float32)
    for term in Counter(tokenize(query)):  # distinct terms only
        ti = index.vocab.get(term)
        if ti is None:
            continue
        s, e = index.offsets[ti], index.offsets[ti + 1]
        cov[index.post_docs[s:e]] += _IDF[ti]
    return cov


def accumulate(query: str, k1: Optional[float] = None, b: Optional[float] = None,
               w_vsm: Optional[float] = None, w_coverage: Optional[float] = None) -> np.ndarray:
    k1 = params.BM25_K1 if k1 is None else k1
    b = params.BM25_B if b is None else b
    w_vsm = params.W_VSM if w_vsm is None else w_vsm
    w_coverage = params.W_COVERAGE if w_coverage is None else w_coverage

    scores = bm25.accumulate(query, k1=k1, b=b)
    if w_coverage:
        scores += w_coverage * _coverage(query)
    if w_vsm:
        scores += w_vsm * boolean_vsm.accumulate(query)
    return scores


def score(query: str, k: int, **kwargs) -> List[Tuple[str, float]]:
    """Up to k (doc_id, score) pairs under the blended scoring function."""
    return bm25.top_k(accumulate(query, **kwargs), k)
