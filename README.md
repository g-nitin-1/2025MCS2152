# A1 — Sparse Retrieval Arena: 2025MCS2152 Submission

This repository contains the Phase 1 submission for **Assignment 1:
Sparse Retrieval Arena**. If anything here conflicts with the assignment
specification or a later instructor clarification, those instructions
govern grading and submission rules; this repository documents the
implementation and reproducible results being submitted.

## What you're building

This submission is a from-scratch inverted-index retrieval engine with
Boolean/vector-space and BM25 scorers. Its final ranking is a tuned blend
of BM25, distinct-query-term coverage, and TF-IDF cosine similarity.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate     # or your preferred env manager
pip install -r requirements.txt
cd submission && python setup.py build_ext --inplace && cd ..

# Run conformance tests and the full harness against the toy set.
bash scripts/smoke_test.sh
```

You should see a report ending in something like:

```
nDCG@10:              0.8793
MAP@10:               0.8250
...
Index size on disk:   730.0B  (730 bytes)
...
Provisional score (80% weight, nDCG@10 + MAP@10): 0.6980
(Remaining ±10% efficiency modifier and 0-10% index-size score are
 class-relative — applied when course staff aggregate the full leaderboard.)
Local reference nDCG@10: 0.8625  (beat it: True)
(This is YOUR local comparison run, not the official grading baseline — see Section 7.)
```

The toy result is a conformance and regression check, not a predictor of
held-out leaderboard rank. The released TREC-COVID dev topics are used for
local tuning and Phase 1 baseline comparison; course staff substitute their
held-out evaluation data when producing the class leaderboard.

## A note on the baseline

Two comparison runs are relevant:

1. **Toy reference run** — `data/toy/reference_bm25_run_dev.trec`, used
   only for the local smoke test.
2. **Released full-dev reference run** —
   `data/full/baseline_run_dev.trec`, downloaded with the public
   TREC-COVID data and used for the Phase 1 comparison.

The current submission beats both provided comparisons. Neither local run
reveals performance on the private held-out collection used for the class
leaderboard, so parameters are selected only from released dev data.

## Where to write code

Everything you implement lives in `submission/`:

| File | What goes here |
|---|---|
| `submission/retrieve.py` | **The required entrypoint** (`build_index`, `load_index`, `retrieve`) and scorer selection. Its public function signatures must not change. |
| `submission/indexer.py` | Your inverted index: postings, document lengths, collection stats, plus `save()`/`load()` for on-disk persistence. |
| `submission/boolean_vsm.py` | Boolean AND/OR retrieval + TF-IDF cosine vector-space ranking. |
| `submission/bm25.py` | BM25 with tunable `k1`, `b`. |
| `submission/custom_scorer.py` | The submitted BM25 + coverage + cosine blend. |
| `submission/setup.py` | Evaluator-compatible build definition for the C++ tokenizer extension. |

Every file above has a docstring with the relevant formula and a
reference back to the assignment section it satisfies — read those before
you start.

**You may not use an existing search/indexing library** (Lucene,
Elasticsearch, Pyserini, Whoosh, `rank_bm25`, etc.) inside `submission/`.
Standard libraries for tokenisation/stemming (e.g. NLTK) and numeric
libraries (NumPy) are fine — and you're not limited to pure Python either:
a C/C++ extension you compile yourself (Cython, already in
`requirements.txt`, or pybind11) is fine too, see
`docs/SUBMISSION_INTERFACE.md`, "Compiled extensions". See the
assignment's Academic Integrity section for the full policy, including
AI-use disclosure and code
provenance requirements.

## Running the harness yourself

```bash
python -m harness.run_harness \
  --corpus data/toy/corpus.jsonl \
  --queries data/toy/queries_dev.tsv \
  --qrels data/toy/qrels_dev.txt \
  --baseline-run data/toy/reference_bm25_run_dev.trec \
  --run-out runs/dev_run.trec \
  --report-out runs/dev_report.json
```

This exercises the same three-function interface and metric computation
used during evaluation. Course infrastructure supplies its trusted
harness and the appropriate corpus, queries, and qrels. See
`harness/metrics.py` for exactly how
nDCG@10, MAP@10, MRR, and P@k are computed, and `harness/leaderboard.py` for
how they combine into your leaderboard score.

Under the hood, this one command spawns `build_index()` and
`load_index()`/`retrieve()` as two separate subprocesses of itself, with
a `--index-dir` on disk in between — see the module docstring at the top
of `harness/run_harness.py` for why, and `docs/SUBMISSION_INTERFACE.md`
for the full three-function contract.

To test against the real assignment corpus instead of the toy set, run
`python scripts/download_full_corpus.py` first (see `data/README.md`),
then point `--corpus`/`--queries`/`--qrels` at `data/full/` instead.

## Before you push: run the smoke test

```bash
bash scripts/smoke_test.sh
```

This runs the same interface-conformance tests, metrics unit tests, and
full harness pass that CI runs on every push
(`.github/workflows/conformance.yml`). Fix anything it flags before your
conformance freeze (48 hours before the deadline — see
`docs/SUBMISSION_INTERFACE.md`).

## Repository layout

```
.
├── data/
│   ├── toy/                 # small hand-built set for fast local dev (ships here)
│   ├── README.md            # data format + how to get the real corpus
│   └── full/                # created by scripts/download_full_corpus.py (gitignored)
├── submission/               # <-- you write code here
│   └── setup.py              # optional compiled-extension build definition
├── harness/                  # scoring code (read-only reference; don't need to edit)
├── tests/                    # conformance + metrics unit tests
├── scripts/
│   ├── download_full_corpus.py
│   └── smoke_test.sh
├── docs/
│   ├── SUBMISSION_INTERFACE.md   # the exact, binding interface contract
│   └── DOCKER_SUBMISSION.md      # what the Dockerfile is for, and the grading trust boundary
├── Dockerfile                 # how course staff run every submission
└── .github/workflows/conformance.yml   # what runs on every push
```

## This submission (2025MCS2152)

### Reproducing the index and the leaderboard run

```bash
pip install -r requirements.txt
cd submission && python setup.py build_ext --inplace && cd ..
python scripts/download_full_corpus.py          # -> data/full/ (beir/trec-covid, 171,332 docs)

python -m harness.run_harness \
  --corpus  data/full/corpus.jsonl \
  --queries data/full/queries_dev.tsv \
  --qrels   data/full/qrels_dev.txt \
  --run-out runs/full_dev_run.trec \
  --report-out runs/full_dev_report.json
```

Index build is deterministic: two independent builds of the same corpus
produce a byte-identical `index.bin`.

### Results on the released dev topics (50 topics)

| Scorer | nDCG@10 | MAP@10 |
|---|---|---|
| Reference BM25, textbook k1=1.2 b=0.75 | 0.4279 | 0.0089 |
| TF-IDF cosine (VSM) | 0.4433 | 0.0103 |
| BM25, textbook k1=1.2 b=0.75 | 0.6307 | 0.0155 |
| BM25, tuned k1=2.1 b=0.55 | 0.6619 | 0.0167 |
| **Blend (submitted): BM25 + 0.5*coverage + 8.0*cosine** | **0.6808** | **0.0171** |

MAP@10 is normalised by the true relevant count, which averages 493 per
topic on this collection, so a perfect top-10 caps at ~0.021; 0.0171 is
about 82% of the achievable maximum.

Recorded with the compiled tokenizer: index build 12.8s, load 0.9s,
**index 15.8MB** on disk (from a 199MB corpus), mean query latency 3.4ms.
Timings are machine-dependent; ranking metrics and index bytes are
deterministic.

### Design summary

- **Indexer** (`submission/indexer.py`): one zlib-compressed file. Doc-ids
  delta-gapped then VByte-coded, term frequencies stored as `tf-1`,
  vocabulary front-coded. Raw document text is not persisted — BM25 and
  cosine need only term frequencies, document lengths, and collection
  statistics. The VByte codec is vectorised over NumPy, so encoding ~12M
  postings does not dominate the graded build time.
- **Compiled hot loop** (`submission/_fasttok.pyx`): Cython generates a
  C++ extension for tokenisation and term counting; grading builds it via
  `submission/setup.py`, with no precompiled binary committed.
- **Stemmer** (`submission/porter.py`): our own Porter (1980)
  implementation, verified 23,531/23,531 against Porter's published test
  vocabulary. Written out rather than taken from NLTK so the submission
  needs nothing beyond `requirements.txt`.
- **Scorers**: BM25 with tunable `k1`/`b` (`bm25.py`), Boolean AND/OR plus
  TF-IDF cosine (`boolean_vsm.py`), and the submitted linear blend
  (`custom_scorer.py`). All read the same index and the same tokenizer.
- **Caches are never persisted.** IDF tables and document norms are
  recomputed in `load_index()`. Index load time is not a graded component
  but index bytes are, so recomputing is strictly cheaper than storing.
- **Tuning** (`scripts/sweep.py`): all parameters were selected on the dev
  topics only. `k1`/`b` were chosen by 5-fold cross-validation rather than
  by the dev argmax — the grid peak (k1=2.2, b=0.55, 0.6636) is optimistic
  by +0.0156 against the CV estimate of 0.6480 and won only 43 of 200
  folds, so the submitted values are the centroid of the CV selection
  distribution.

## Getting help

Discussing high-level strategy with classmates is fine. Sharing code, a
tuned parameter file, or your `submission/` implementation is not — see
the assignment's Academic Integrity section, and remember every team sits
a short oral defense after the leaderboard closes where you'll be asked
to explain and modify your own submission live.
