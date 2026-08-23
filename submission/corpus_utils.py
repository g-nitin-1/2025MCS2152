"""
submission/corpus_utils.py

Small, shared, non-graded utility for reading the corpus format used
throughout this assignment. You are welcome to use this as-is — reading a
JSONL file is plumbing, not the ranking logic the assignment grades — or
replace it if your index needs a different loading strategy (e.g.
streaming instead of loading everything into memory).
"""
import json
from typing import Iterator, List, Tuple


def load_corpus(path: str) -> List[Tuple[str, str]]:
    """Read a corpus.jsonl file (one {"doc_id": ..., "text": ...} object
    per line) and return a list of (doc_id, text) pairs in file order."""
    docs: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            docs.append((obj["doc_id"], obj["text"]))
    return docs


def stream_corpus(path: str) -> Iterator[Tuple[str, str]]:
    """Yield (doc_id, text) pairs one at a time, in file order.

    Same contract as load_corpus() without holding the whole corpus in
    memory: the real collection is ~200MB of JSON on disk and several
    times that as live Python strings, all of which is dead weight once a
    document has been tokenised and folded into the postings.
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            yield obj["doc_id"], obj["text"]
