"""
submission/porter.py — Porter (1980) stemmer, implemented from scratch.

Written out here rather than pulled from NLTK so the submission has no
dependency beyond what the grading image installs from requirements.txt
(assignment Section 5, "Containerisation"). The algorithm is Porter's
original published one; the implementation is ours.

Reference: M.F. Porter, "An algorithm for suffix stripping", Program
14(3), 1980.
"""
from typing import Dict

_VOWELS = frozenset("aeiou")


def _is_consonant(word: str, i: int) -> bool:
    ch = word[i]
    if ch in _VOWELS:
        return False
    if ch == "y":
        # 'y' is a consonant unless preceded by a consonant.
        return i == 0 or not _is_consonant(word, i - 1)
    return True


def _measure(stem: str) -> int:
    """Porter's m: the number of VC sequences in `stem`."""
    m = 0
    prev_was_vowel = False
    for i in range(len(stem)):
        if _is_consonant(stem, i):
            if prev_was_vowel:
                m += 1
            prev_was_vowel = False
        else:
            prev_was_vowel = True
    return m


def _contains_vowel(stem: str) -> bool:
    return any(not _is_consonant(stem, i) for i in range(len(stem)))


def _ends_double_consonant(word: str) -> bool:
    return (
        len(word) >= 2
        and word[-1] == word[-2]
        and _is_consonant(word, len(word) - 1)
    )


def _ends_cvc(word: str) -> bool:
    """*o: stem ends consonant-vowel-consonant, last consonant not w/x/y."""
    if len(word) < 3:
        return False
    n = len(word)
    return (
        _is_consonant(word, n - 3)
        and not _is_consonant(word, n - 2)
        and _is_consonant(word, n - 1)
        and word[-1] not in "wxy"
    )


_STEP2 = [
    ("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"),
    ("izer", "ize"), ("bli", "ble"), ("alli", "al"), ("entli", "ent"),
    ("eli", "e"), ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
    ("ator", "ate"), ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
    ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"),
    ("logi", "log"),
]

_STEP3 = [
    ("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"),
    ("ical", "ic"), ("ful", ""), ("ness", ""),
]

_STEP4 = [
    "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement",
    "ment", "ent", "ou", "ism", "ate", "iti", "ous", "ive", "ize",
]


def _step1a(word: str) -> str:
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s"):
        return word[:-1]
    return word


def _step1b(word: str) -> str:
    if word.endswith("eed"):
        if _measure(word[:-3]) > 0:
            return word[:-1]
        return word
    hit = False
    if word.endswith("ed") and _contains_vowel(word[:-2]):
        word = word[:-2]
        hit = True
    elif word.endswith("ing") and _contains_vowel(word[:-3]):
        word = word[:-3]
        hit = True
    if hit:
        if word.endswith(("at", "bl", "iz")):
            return word + "e"
        if _ends_double_consonant(word) and not word.endswith(("l", "s", "z")):
            return word[:-1]
        if _measure(word) == 1 and _ends_cvc(word):
            return word + "e"
    return word


def _step1c(word: str) -> str:
    if word.endswith("y") and _contains_vowel(word[:-1]):
        return word[:-1] + "i"
    return word


def _apply(word: str, rules, min_m: int) -> str:
    for suffix, replacement in rules:
        if word.endswith(suffix):
            stem = word[: -len(suffix)]
            if _measure(stem) > min_m:
                return stem + replacement
            return word
    return word


def _step4(word: str) -> str:
    for suffix in _STEP4:
        if word.endswith(suffix):
            stem = word[: -len(suffix)]
            if _measure(stem) > 1:
                return stem
            return word
    if word.endswith("ion"):
        stem = word[:-3]
        if _measure(stem) > 1 and stem.endswith(("s", "t")):
            return stem
    return word


def _step5(word: str) -> str:
    if word.endswith("e"):
        stem = word[:-1]
        m = _measure(stem)
        if m > 1 or (m == 1 and not _ends_cvc(stem)):
            word = stem
    if word.endswith("ll") and _measure(word) > 1:
        word = word[:-1]
    return word


_cache: Dict[str, str] = {}


def stem(word: str) -> str:
    """Porter-stem `word` (assumed already lowercased). Memoised — the
    same vocabulary terms recur constantly across a 171K-document corpus,
    and stemming is otherwise a measurable slice of index build time."""
    cached = _cache.get(word)
    if cached is not None:
        return cached
    if len(word) <= 2:
        _cache[word] = word
        return word
    w = _step1c(_step1b(_step1a(word)))
    w = _apply(w, _STEP2, 0)
    w = _apply(w, _STEP3, 0)
    w = _step4(w)
    w = _step5(w)
    _cache[word] = w
    return w
