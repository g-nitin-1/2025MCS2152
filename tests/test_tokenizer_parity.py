"""
Parity between the compiled tokenizer and the pure-Python reference.

submission/_fasttok.pyx exists purely to make index building faster. It is
only legitimate if it is indistinguishable from the Python tokenizer it
replaces: if the two ever disagree, the index and every retrieval score
silently change depending on whether the extension happened to compile.
These tests are what make the compiled path safe to ship with a fallback.
"""
import json
import os

import pytest

from submission import indexer
from submission.indexer import _py_term_counts, _py_tokenize

pytestmark = pytest.mark.skipif(
    not indexer.USING_COMPILED_TOKENIZER,
    reason="compiled tokenizer not built; the pure-Python path is in use",
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ADVERSARIAL = [
    "",
    "   ",
    "a",
    "I",
    "the and or but",                      # all stopwords
    "SARS-CoV-2",                          # punctuation inside a token
    "COVID-19, 2020; (asymptomatic).",
    "hyphen-separated multi-word tokens",
    "trailing punctuation!!!",
    "!!!leading punctuation",
    "MiXeD CaSe WoRdS",
    "tabs\tand\nnewlines\r\nhere",
    "numbers 12345 and 3.14 and 1e10",
    "café naïve résumé",                   # latin-1 accents, code points < 256
    "Ωμέγα δοκιμή",                        # code points >= 256
    "中文字符测试",                          # CJK, code points >= 256
    "emoji 🦠 test",                        # astral plane
    "under_score and dotted.name",
    "a" * 500,
    "running runs ran runner runnings",    # stemming collisions
    "ss sses ies ied",                     # Porter step-1a edge cases
]


@pytest.mark.parametrize("text", ADVERSARIAL)
def test_parity_on_adversarial_strings(text):
    assert indexer.tokenize(text) == _py_tokenize(text)
    assert indexer.term_counts(text) == _py_term_counts(text)


def test_parity_on_the_toy_corpus():
    path = os.path.join(REPO, "data", "toy", "corpus.jsonl")
    with open(path, encoding="utf-8") as f:
        for line in f:
            text = json.loads(line)["text"]
            assert indexer.tokenize(text) == _py_tokenize(text)
            assert indexer.term_counts(text) == _py_term_counts(text)


def test_term_counts_agrees_with_counting_tokenize_output():
    """term_counts() must be exactly Counter(tokenize()) -- build() relies
    on this to skip materialising the token list."""
    from collections import Counter

    for text in ADVERSARIAL:
        assert indexer.term_counts(text) == dict(Counter(indexer.tokenize(text)))


def test_doc_length_is_total_token_count():
    """build() records document length as sum(counts.values()); that must
    equal len(tokenize(text)), since BM25 length normalisation depends on
    it."""
    for text in ADVERSARIAL:
        assert sum(indexer.term_counts(text).values()) == len(indexer.tokenize(text))
