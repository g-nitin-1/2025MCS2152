# distutils: language = c++
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False
"""
submission/_fasttok.pyx — the tokenizer's hot loop, compiled as C++.

Why this file exists
--------------------
Profiling the index build showed 11.3s of a 17.8s build going to text
processing, and it is the one stage NumPy cannot vectorise: it is a
character-by-character scan over ~200MB of text producing ~30M tokens,
each of which then needs a stopword test and a stem. In pure Python that
is tens of millions of interpreter round-trips.

Everything else in the build (postings assembly, VByte coding, zlib) is
already running in C underneath NumPy, so this is the only place a
compiled extension buys anything. Index build time is a graded component
(assignment Section 7), which is what makes it worth compiling.

What it does NOT do
-------------------
It does not reimplement Porter. The stemmer stays in submission/porter.py,
in Python, and is called through the transform cache below — so it runs
once per DISTINCT term (~207K times) rather than once per token occurrence
(~30M times). Reimplementing Porter in Cython would have risked diverging
from the Python one that is verified against Porter's published test
vocabulary, for no measurable gain.

Exactness
---------
This must produce byte-identical output to indexer._py_tokenize, or the
index and every score change. The character classification below
replicates that function exactly, including its two subtle behaviours:
  - lowercasing is done by Python's str.lower() on the whole string, so
    Unicode-aware case folding is unchanged;
  - only code points BELOW 256 are classified; anything >= 256 is kept as
    part of a token, exactly as the pure-Python version's 256-entry
    translation table does by leaving them untouched.
tests/test_tokenizer_parity.py asserts this over the real corpus.

The evaluator builds this module from inside ``submission/`` using
``python setup.py build_ext --inplace``. No generated C file or compiled
shared object is part of the submission archive.
"""
from submission.porter import stem as _porter_stem
from submission.stopwords import STOPWORDS

# Character class table for code points < 256: 1 = part of a token,
# 0 = separator. Built from Python's own str.isalnum() so it agrees with
# the pure-Python tokenizer by construction rather than by hand-copying.
cdef unsigned char _IS_TOKEN_CHAR[256]

cdef _init_table():
    cdef int cp
    for cp in range(256):
        _IS_TOKEN_CHAR[cp] = 1 if chr(cp).isalnum() else 0

_init_table()

# raw token -> stemmed form, or None if the token is dropped (stopword or
# too short). One dict lookup per token occurrence replaces a stopword-set
# probe, a length test and a separate stem cache probe.
cdef dict _XFORM = {}


cdef inline object _transform(object raw):
    """Map a raw token to its indexed form, or None to drop it."""
    cdef object out = _XFORM.get(raw)
    if out is not None or raw in _XFORM:
        return out
    if raw in STOPWORDS or len(raw) < 2:
        _XFORM[raw] = None
        return None
    out = _porter_stem(raw)
    _XFORM[raw] = out
    return out


def term_counts(text):
    """{term: frequency} for one document — tokenize and count in a single
    pass, which is what build() actually needs (it never wants the token
    list itself)."""
    cdef unicode low = text.lower()
    cdef Py_ssize_t n = len(low), i = 0, start = -1
    cdef Py_UCS4 ch
    cdef dict counts = {}
    cdef object term
    cdef bint keep

    for i in range(n):
        ch = low[i]
        keep = _IS_TOKEN_CHAR[ch] if ch < 256 else 1
        if keep:
            if start < 0:
                start = i
        elif start >= 0:
            term = _transform(low[start:i])
            if term is not None:
                counts[term] = counts.get(term, 0) + 1
            start = -1
    if start >= 0:
        term = _transform(low[start:n])
        if term is not None:
            counts[term] = counts.get(term, 0) + 1
    return counts


def tokenize(text):
    """The token list, in order — used by the query path and kept so the
    compiled and pure-Python tokenizers expose the same API."""
    cdef unicode low = text.lower()
    cdef Py_ssize_t n = len(low), i = 0, start = -1
    cdef Py_UCS4 ch
    cdef list out = []
    cdef object term
    cdef bint keep

    for i in range(n):
        ch = low[i]
        keep = _IS_TOKEN_CHAR[ch] if ch < 256 else 1
        if keep:
            if start < 0:
                start = i
        elif start >= 0:
            term = _transform(low[start:i])
            if term is not None:
                out.append(term)
            start = -1
    if start >= 0:
        term = _transform(low[start:n])
        if term is not None:
            out.append(term)
    return out
