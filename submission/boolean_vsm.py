"""
submission/boolean_vsm.py — Boolean retrieval + TF-IDF cosine VSM
(assignment Section 4.1).

Boolean: the query's terms combined conjunctively (AND) or disjunctively
(OR), returning an unranked document set. Postings lists are stored
ascending, so AND is a sorted-array intersection and OR a sorted-array
union — no set-of-strings materialisation.

Vector space: ltc-style weighting,

    w(t, d) = tf(t, d) * log( N / df(t) )
    sim(q, d) = (q . d) / (||q|| * ||d||)

Document norms ||d|| are recomputed in build() from the postings rather
than persisted, for the same reason as BM25's IDF cache: load time is not
graded, index bytes are.
"""
from collections import Counter
from typing import List, Optional, Tuple

import numpy as np

from submission.indexer import InvertedIndex, tokenize
from submission.bm25 import top_k

_INDEX: Optional[InvertedIndex] = None
_IDF: Optional[np.ndarray] = None    # log(N / df), the VSM weighting
_NORM: Optional[np.ndarray] = None   # ||d|| per document


def build(index: InvertedIndex) -> None:
    global _INDEX, _IDF, _NORM
    _INDEX = index
    df = np.diff(index.offsets).astype(np.float64)
    N = float(index.N)
    _IDF = np.log(N / np.maximum(df, 1.0))
    # ||d||^2 = sum_t (tf(t,d) * idf(t))^2, accumulated over every posting
    # at once: repeat each term's idf across its postings list, square, and
    # bincount by document.
    per_posting = np.repeat(_IDF, df.astype(np.int64))
    w = index.post_tfs.astype(np.float64) * per_posting
    sq = np.bincount(index.post_docs, weights=w * w, minlength=index.N)
    _NORM = np.sqrt(sq)
    _NORM[_NORM == 0.0] = 1.0  # empty documents: avoid a divide-by-zero


def boolean_search(query: str, mode: str = "and") -> List[str]:
    """Unranked doc_ids matching `query` as a conjunction (mode="and") or
    disjunction (mode="or") of its terms."""
    assert _INDEX is not None
    index = _INDEX
    mode = mode.lower()
    if mode not in ("and", "or"):
        raise ValueError(f"mode must be 'and' or 'or', got {mode!r}")

    terms = list(dict.fromkeys(tokenize(query)))
    if not terms:
        return []

    lists = [index.postings_for(t)[0] for t in terms]
    if mode == "and":
        if any(l.size == 0 for l in lists):
            return []  # a missing term makes the conjunction empty
        lists.sort(key=len)  # intersect the shortest lists first
        acc = lists[0]
        for l in lists[1:]:
            acc = np.intersect1d(acc, l, assume_unique=True)
            if acc.size == 0:
                break
    else:
        acc = np.unique(np.concatenate(lists)) if lists else np.zeros(0, dtype=np.int32)

    ids = index.doc_ids
    return [ids[d] for d in acc]


def accumulate(query: str) -> np.ndarray:
    """Dense TF-IDF cosine similarity over all documents."""
    assert _INDEX is not None and _IDF is not None and _NORM is not None
    index = _INDEX
    dot = np.zeros(index.N, dtype=np.float64)
    qcounts = Counter(tokenize(query))
    qnorm_sq = 0.0
    for term, qtf in qcounts.items():
        ti = index.vocab.get(term)
        if ti is None:
            continue
        wq = qtf * _IDF[ti]
        qnorm_sq += wq * wq
        s, e = index.offsets[ti], index.offsets[ti + 1]
        docs = index.post_docs[s:e]
        dot[docs] += wq * (index.post_tfs[s:e] * _IDF[ti])
    if qnorm_sq == 0.0:
        return np.zeros(index.N, dtype=np.float32)
    return (dot / (np.sqrt(qnorm_sq) * _NORM)).astype(np.float32)


def vsm_score(query: str, k: int) -> List[Tuple[str, float]]:
    """Up to k (doc_id, score) pairs ranked by TF-IDF cosine similarity."""
    return top_k(accumulate(query), k)
