"""
submission/retrieve.py — THE REQUIRED COMPETITION ENTRYPOINT.

The grading harness only ever imports and calls the three functions below.
Their names and signatures are fixed by the assignment (Section 5) — they
are not renamed, their signatures are unchanged, and they stay in this
file.

    build_index(corpus_path, index_dir) -> None
        Streams corpus.jsonl, builds the inverted index, and writes it to
        index_dir as a single zlib-compressed, VByte-coded file. Timed as
        the index-build-time efficiency metric; the resulting on-disk size
        is the index-size score.

    load_index(index_dir) -> None
        Runs in a fresh process with nothing but index_dir. Reconstructs
        the index and rebuilds the per-term/per-document caches the
        scorers need (IDF tables, document norms). Those caches are
        deliberately NOT persisted: index load time is not a scored
        component, index bytes are, so recomputing is strictly cheaper.

    retrieve(query, k) -> List[Tuple[str, float]]
        Answers one query from the loaded index. Which scorer is served is
        set by params.SCORER, so switching the competition entry between
        BM25, VSM, and the blend is a one-line change and the sweep script
        can drive it directly.

Design notes for the report / oral defense
------------------------------------------
- Persistence is real: build_index writes everything and load_index reads
  only index_dir. Nothing crosses the process boundary in memory.
- retrieve() is deterministic. Ties in score break by ascending internal
  doc-id (see bm25.top_k), so the same query always returns the same list.
- No doc_id can repeat in a result list: scores are accumulated into one
  dense vector indexed by internal doc-id, so a document is a single slot
  no matter how many query terms hit it.
"""
import os
from typing import List, Optional, Tuple

from submission import bm25, boolean_vsm, custom_scorer, params
from submission.corpus_utils import stream_corpus
from submission.indexer import InvertedIndex

# ---------------------------------------------------------------------------
# Module-level state. load_index() populates this; retrieve() reads it.
# build_index() runs in a SEPARATE process and nothing it holds in memory
# survives into load_index().
# ---------------------------------------------------------------------------
_INDEX: Optional[InvertedIndex] = None


def build_index(corpus_path: str, index_dir: str) -> None:
    """Build the inverted index from `corpus_path` and persist it to
    `index_dir`."""
    os.makedirs(index_dir, exist_ok=True)
    index = InvertedIndex()
    index.build(stream_corpus(corpus_path))
    index.save(index_dir)


def load_index(index_dir: str) -> None:
    """Reconstruct the index from `index_dir` alone and prime every
    scorer's query-time caches."""
    global _INDEX
    _INDEX = InvertedIndex.load(index_dir)
    bm25.build(_INDEX)
    boolean_vsm.build(_INDEX)
    custom_scorer.build(_INDEX)


def retrieve(query: str, k: int = 10) -> List[Tuple[str, float]]:
    """Return up to k (doc_id, score) pairs for `query`, best first."""
    if _INDEX is None:
        raise RuntimeError(
            "retrieve() called before load_index(); the harness always "
            "calls build_index(corpus_path, index_dir) and then "
            "load_index(index_dir) — in that order, in two separate "
            "processes — before any retrieve() calls. If you're testing "
            "manually, do the same."
        )
    if k <= 0:
        return []

    scorer = params.SCORER
    if scorer == "bm25":
        return bm25.score(query, k, k1=params.BM25_K1, b=params.BM25_B)
    if scorer == "vsm":
        return boolean_vsm.vsm_score(query, k)
    return custom_scorer.score(query, k)
