"""
submission/indexer.py — the inverted index (assignment Section 4.1).

Built from scratch: no search/indexing library is used for postings
construction or scoring. NumPy is used as a numeric library only (that is
explicitly permitted), and it is what makes both indexing and query
scoring array-parallel rather than interpreted loops.

Text processing
---------------
`tokenize()` is the single tokenizer every scorer shares: lowercase,
alphanumeric runs, stopword removal, then Porter stemming
(`submission/porter.py`, our own implementation). Keeping one tokenizer
here is what guarantees the Boolean/VSM and BM25 scorers agree with the
index they read.

On-disk format (assignment Section 7, "index size")
---------------------------------------------------
Everything lives in ONE zlib-compressed file, `index.bin`. Nothing that
query time does not need is persisted — in particular the raw document
text is dropped entirely, since BM25 and TF-IDF cosine need only term
frequencies, document lengths, and collection statistics.

Sections, each length-prefixed, all integer streams VByte-coded:

  meta      N, |V|, avgdl, total postings
  doclens   VByte, one per document
  docids    the external doc_id strings, newline-joined
  vocab     front-coded (sorted terms share long prefixes, which matters
            a lot for a medical vocabulary: "coronavirus", "coronaviruses",
            "coronaviral" ... collapse to a shared prefix plus a suffix)
  df        VByte, one per term
  postings  doc-ids delta-gapped (ascending, so gaps are small and most
            fit in one byte) then VByte
  tfs       VByte of (tf - 1), so the overwhelmingly common tf==1 costs a
            single zero byte and zlib then crushes the resulting runs

The VByte codec is vectorised over NumPy rather than looped in Python —
encoding ~20M postings a byte at a time in the interpreter would other-
wise dominate index build time, which is itself a graded component.
"""
import os
import zlib
from array import array
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from submission.porter import stem

INDEX_FILENAME = "index.bin"

_MAGIC = b"IDX1"

# Re-exported so existing references to indexer.STOPWORDS keep working;
# the list itself lives in stopwords.py so the compiled tokenizer can share
# it without an import cycle.
from submission.stopwords import STOPWORDS  # noqa: E402,F401


def _build_translation_table() -> Dict[int, Optional[str]]:
    table: Dict[int, Optional[str]] = {}
    for cp in range(256):
        if not chr(cp).isalnum():
            table[cp] = " "
    return table


_TRANSLATION = str.maketrans(_build_translation_table())


def _py_tokenize(text: str) -> List[str]:
    """Reference tokenizer: lowercase -> alphanumeric runs -> stopword
    removal -> Porter stem. Used directly when the compiled tokenizer is
    not available, and as the parity oracle for it in tests."""
    out = []
    for tok in text.lower().translate(_TRANSLATION).split():
        if tok in STOPWORDS or len(tok) < 2:
            continue
        out.append(stem(tok))
    return out


def _py_term_counts(text: str) -> Dict[str, int]:
    return dict(Counter(_py_tokenize(text)))


# The hot loop is compiled if the extension was built (see setup.py, and
# the Dockerfile / CI workflow that run it). This is a pure speed
# optimisation for a graded component: the compiled tokenizer is asserted
# byte-identical to _py_tokenize in tests/test_tokenizer_parity.py, so the
# index and every score are the same either way. If the build was skipped,
# or Cython/a compiler was unavailable, or the .so was built for a
# different Python ABI, the import simply fails and we fall back.
try:
    from submission._fasttok import term_counts as _term_counts, tokenize as _tokenize
    USING_COMPILED_TOKENIZER = True
except ImportError:  # pragma: no cover - depends on whether the build ran
    _tokenize, _term_counts = _py_tokenize, _py_term_counts
    USING_COMPILED_TOKENIZER = False


def tokenize(text: str) -> List[str]:
    """The shared tokenizer every scorer and the index must agree on."""
    return _tokenize(text)


def term_counts(text: str) -> Dict[str, int]:
    """{term: frequency} for one document, in a single pass."""
    return _term_counts(text)


# ---------------------------------------------------------------------------
# Vectorised VByte codec.
# ---------------------------------------------------------------------------

def vbyte_encode(values: np.ndarray) -> bytes:
    """VByte-encode a non-negative int array. The high bit marks the FINAL
    byte of each value. Fully vectorised: a fixed five passes regardless of
    how many values there are."""
    v = np.asarray(values, dtype=np.uint64)
    if v.size == 0:
        return b""
    nbytes = np.ones(v.size, dtype=np.int64)
    for shift in (7, 14, 21, 28):
        nbytes += (v >= (np.uint64(1) << np.uint64(shift)))
    ends = np.cumsum(nbytes)
    starts = ends - nbytes
    out = np.zeros(int(ends[-1]), dtype=np.uint8)
    for j in range(int(nbytes.max())):
        sel = nbytes > j
        out[starts[sel] + j] = ((v[sel] >> np.uint64(7 * j)) & np.uint64(127)).astype(np.uint8)
    out[ends - 1] |= 128
    return out.tobytes()


def vbyte_decode(data: bytes, count: int) -> np.ndarray:
    """Inverse of vbyte_encode. Vectorised; returns an int64 array of
    `count` values."""
    if count == 0:
        return np.zeros(0, dtype=np.int64)
    b = np.frombuffer(data, dtype=np.uint8)
    is_last = b >= 128
    payload = (b & 127).astype(np.float64)
    group = np.cumsum(is_last) - is_last  # 0-based value index per byte
    ends = np.flatnonzero(is_last)
    starts = np.empty(ends.size, dtype=np.int64)
    starts[0] = 0
    starts[1:] = ends[:-1] + 1
    pos = np.arange(b.size, dtype=np.int64) - starts[group]
    weighted = payload * np.exp2(7 * pos)  # exact: all values < 2**53
    return np.bincount(group, weights=weighted, minlength=count)[:count].astype(np.int64)


def _pack(sections: Iterable[bytes]) -> bytes:
    parts = []
    for s in sections:
        parts.append(len(s).to_bytes(8, "little"))
        parts.append(s)
    return b"".join(parts)


def _unpack(blob: bytes, n: int) -> List[bytes]:
    out = []
    off = 0
    for _ in range(n):
        length = int.from_bytes(blob[off:off + 8], "little")
        off += 8
        out.append(blob[off:off + length])
        off += length
    return out


def _front_code(terms: List[str]) -> bytes:
    """Front-coding: each term is stored as (shared prefix length with the
    previous term, suffix). Sorted vocabularies share long prefixes."""
    prefixes = array("I")
    suffixes = []
    prev = ""
    for t in terms:
        n = 0
        limit = min(len(prev), len(t), 255)
        while n < limit and prev[n] == t[n]:
            n += 1
        prefixes.append(n)
        suffixes.append(t[n:])
        prev = t
    suffix_blob = "\n".join(suffixes).encode("utf-8")
    return _pack([vbyte_encode(np.frombuffer(prefixes, dtype=np.uint32)), suffix_blob])


def _front_decode(blob: bytes, n_terms: int) -> List[str]:
    prefix_blob, suffix_blob = _unpack(blob, 2)
    prefixes = vbyte_decode(prefix_blob, n_terms)
    suffixes = suffix_blob.decode("utf-8").split("\n")
    terms = []
    prev = ""
    for p, suf in zip(prefixes, suffixes):
        t = prev[:p] + suf
        terms.append(t)
        prev = t
    return terms


class _PostingsView:
    """Backwards-compatible `index.postings[term] -> {doc_id: tf}` view.

    The real storage is the CSR-style NumPy triple below; materialising a
    dict-of-dicts for 171K documents would cost gigabytes for no benefit.
    This view reconstructs a single term's dict on demand, so the shape
    documented in the starter skeleton still works for tests and debugging.
    """

    def __init__(self, index: "InvertedIndex"):
        self._index = index

    def __getitem__(self, term: str) -> Dict[str, int]:
        docids, tfs = self._index.postings_for(term)
        ids = self._index.doc_ids
        return {ids[d]: int(t) for d, t in zip(docids, tfs)}

    def __contains__(self, term: str) -> bool:
        return term in self._index.vocab

    def __len__(self) -> int:
        return len(self._index.vocab)

    def __iter__(self):
        return iter(self._index.vocab)

    def keys(self):
        return self._index.vocab.keys()


class InvertedIndex:
    """Inverted index over the corpus.

    After `build()` or `load()` the postings live in three parallel arrays
    (a CSR / compressed-sparse-row layout):

        offsets[i] : where term i's postings start
        post_docs  : internal integer doc-ids, ascending within each term
        post_tfs   : matching term frequencies

    which lets a scorer pull a whole postings list as a NumPy slice and
    score it with vector arithmetic instead of a Python loop.
    """

    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.terms: List[str] = []
        self.doc_ids: List[str] = []
        self.doc_len: np.ndarray = np.zeros(0, dtype=np.int32)
        self.offsets: np.ndarray = np.zeros(1, dtype=np.int64)
        self.post_docs: np.ndarray = np.zeros(0, dtype=np.int32)
        self.post_tfs: np.ndarray = np.zeros(0, dtype=np.int32)
        self.N: int = 0
        self.avg_doc_len: float = 0.0
        self.postings = _PostingsView(self)

    # -- build ------------------------------------------------------------

    def build(self, corpus: Iterable[Tuple[str, str]]) -> None:
        """corpus: (doc_id, text) pairs in file order.

        Accepts any iterable, not just a list, so build_index() can stream
        a 199MB corpus.jsonl past this loop instead of materialising every
        document string in memory first."""
        acc: Dict[str, array] = {}
        doc_ids: List[str] = []
        doc_len = array("i")

        for internal_id, (doc_id, text) in enumerate(corpus):
            counts = term_counts(text)
            doc_ids.append(doc_id)
            doc_len.append(sum(counts.values()))
            for term, tf in counts.items():
                bucket = acc.get(term)
                if bucket is None:
                    bucket = acc[term] = array("i")
                bucket.append(internal_id)
                bucket.append(tf)

        self.doc_ids = doc_ids
        self.doc_len = np.frombuffer(doc_len, dtype=np.int32).copy()
        self.N = len(doc_ids)
        self.avg_doc_len = float(self.doc_len.mean()) if self.N else 0.0

        self.terms = sorted(acc)
        self.vocab = {t: i for i, t in enumerate(self.terms)}

        counts = np.fromiter((len(acc[t]) // 2 for t in self.terms), dtype=np.int64, count=len(self.terms))
        self.offsets = np.zeros(len(self.terms) + 1, dtype=np.int64)
        np.cumsum(counts, out=self.offsets[1:])

        total = int(self.offsets[-1])
        self.post_docs = np.empty(total, dtype=np.int32)
        self.post_tfs = np.empty(total, dtype=np.int32)
        for i, t in enumerate(self.terms):
            flat = np.frombuffer(acc[t], dtype=np.int32)
            s, e = self.offsets[i], self.offsets[i + 1]
            self.post_docs[s:e] = flat[0::2]
            self.post_tfs[s:e] = flat[1::2]

    def document_frequency(self, term: str) -> int:
        """Number of documents containing `term` at least once."""
        i = self.vocab.get(term)
        if i is None:
            return 0
        return int(self.offsets[i + 1] - self.offsets[i])

    def postings_for(self, term: str) -> Tuple[np.ndarray, np.ndarray]:
        """(internal doc-ids, term frequencies) for `term`, doc-ids
        ascending. Empty arrays if the term is not in the vocabulary."""
        i = self.vocab.get(term)
        if i is None:
            return self.post_docs[:0], self.post_tfs[:0]
        s, e = self.offsets[i], self.offsets[i + 1]
        return self.post_docs[s:e], self.post_tfs[s:e]

    # -- persistence ------------------------------------------------------

    def save(self, index_dir: str) -> None:
        os.makedirs(index_dir, exist_ok=True)
        n_terms = len(self.terms)
        total = int(self.offsets[-1])

        meta = _pack([
            self.N.to_bytes(8, "little"),
            n_terms.to_bytes(8, "little"),
            total.to_bytes(8, "little"),
            np.float64(self.avg_doc_len).tobytes(),
        ])

        df = np.diff(self.offsets)

        # Delta-gap doc-ids within each postings list. The first entry of
        # each list stays absolute; the rest become (this - previous),
        # which is small and therefore usually a single VByte byte.
        gaps = self.post_docs.astype(np.int64).copy()
        if total:
            gaps[1:] -= self.post_docs[:-1]
            gaps[self.offsets[:-1][df > 0]] = self.post_docs[self.offsets[:-1][df > 0]]

        blob = _pack([
            meta,
            vbyte_encode(self.doc_len),
            "\n".join(self.doc_ids).encode("utf-8"),
            _front_code(self.terms),
            vbyte_encode(df),
            vbyte_encode(gaps),
            vbyte_encode(self.post_tfs.astype(np.int64) - 1),
        ])
        with open(os.path.join(index_dir, INDEX_FILENAME), "wb") as f:
            f.write(_MAGIC)
            f.write(zlib.compress(blob, 6))

    @classmethod
    def load(cls, index_dir: str) -> "InvertedIndex":
        with open(os.path.join(index_dir, INDEX_FILENAME), "rb") as f:
            raw = f.read()
        if raw[:4] != _MAGIC:
            raise ValueError(f"{INDEX_FILENAME} is not a valid index (bad magic)")
        blob = zlib.decompress(raw[4:])
        meta_b, doclen_b, docid_b, vocab_b, df_b, gaps_b, tf_b = _unpack(blob, 7)

        n_b, nt_b, tot_b, avg_b = _unpack(meta_b, 4)
        idx = cls()
        idx.N = int.from_bytes(n_b, "little")
        n_terms = int.from_bytes(nt_b, "little")
        total = int.from_bytes(tot_b, "little")
        idx.avg_doc_len = float(np.frombuffer(avg_b, dtype=np.float64)[0])

        idx.doc_len = vbyte_decode(doclen_b, idx.N).astype(np.int32)
        idx.doc_ids = docid_b.decode("utf-8").split("\n") if idx.N else []
        idx.terms = _front_decode(vocab_b, n_terms)
        idx.vocab = {t: i for i, t in enumerate(idx.terms)}

        df = vbyte_decode(df_b, n_terms)
        idx.offsets = np.zeros(n_terms + 1, dtype=np.int64)
        np.cumsum(df, out=idx.offsets[1:])

        gaps = vbyte_decode(gaps_b, total)
        # Undo the delta-gapping: prefix-sum within each postings list.
        # A global cumsum then subtracting each list's running base is the
        # vectorised equivalent of a per-list Python loop.
        csum = np.cumsum(gaps)
        starts = idx.offsets[:-1]
        base = np.zeros(n_terms, dtype=np.int64)
        nonempty = df > 0
        base[nonempty] = csum[starts[nonempty]] - gaps[starts[nonempty]]
        idx.post_docs = (csum - np.repeat(base, df)).astype(np.int32)
        idx.post_tfs = (vbyte_decode(tf_b, total) + 1).astype(np.int32)
        return idx
