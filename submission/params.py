"""
submission/params.py — every tuned constant in one place.

Kept separate so a parameter sweep rewrites exactly one file, and so the
report (and the oral defense) can point at a single place where the tuned
values live rather than at magic numbers scattered through the scorers.

All values here were chosen by sweeping on the released DEV topics
(data/full/queries_dev.tsv + qrels_dev.txt). The held-out topics are never
used for tuning -- that is the whole point of the two-tier design in
assignment Section 7.
"""

# --- BM25 -------------------------------------------------------------
# Swept on dev; see scripts/sweep.py and the report's nDCG@10-vs-parameter
# plot. Textbook defaults are k1=1.2, b=0.75.
# Selected by 5-fold cross-validation over the 50 dev topics, NOT by taking
# the raw dev argmax: the grid peak (k1=2.2, b=0.55, dev nDCG@10=0.6636) is
# optimistic by +0.0156 against the honest CV estimate of 0.6480, and it won
# only 43 of 200 folds. These are the centroid of the CV selection
# distribution, which is stable across shuffles.
BM25_K1 = 2.1
BM25_B = 0.55

# --- Blend (submission/custom_scorer.py) ------------------------------
# Weight on the TF-IDF cosine signal, which lives on a 0..1 scale.
W_VSM = 8.0
# Weight on coordination: the summed IDF of the DISTINCT query terms a
# document matches at all. Rewards covering more of the query, and
# weights rare query terms more than common ones.
W_COVERAGE = 0.5

# Which scorer retrieve() actually serves. One of "bm25", "vsm", "blend".
SCORER = "blend"
