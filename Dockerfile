# Minimal container image for reproducible local and CI execution
# (assignment Section 5, "Containerisation").
FROM python:3.11-slim

WORKDIR /repo

# C/C++ toolchain — present so a submission that compiles part of itself
# as a Cython or pybind11 extension (see docs/SUBMISSION_INTERFACE.md,
# "Compiled extensions") builds correctly here and in course staff's
# grading image (instructor-tools/Dockerfile.grading, kept in lockstep
# with this one). Installed at image-build time only; the grading
# container still runs with --network none at scoring time.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the optional compiled tokenizer from the location and working
# directory used by CI and grading. A one-time compile must not be charged
# against the index-build-time metric by happening inside build_index().
#
# This step is an optimisation, not a requirement. If it is removed, or
# fails, submission/indexer.py falls back to its pure-Python tokenizer and
# produces a byte-identical index -- see tests/test_tokenizer_parity.py.
RUN if [ -f submission/setup.py ]; then \
        cd submission && python setup.py build_ext --inplace; \
    fi

# Default command: run the interface conformance + smoke-test suite
# against the toy set. Course staff override CMD to point at the real
# corpus/topics/qrels for scoring.
CMD ["python", "-m", "harness.run_harness", \
     "--corpus", "data/toy/corpus.jsonl", \
     "--queries", "data/toy/queries_dev.tsv", \
     "--qrels", "data/toy/qrels_dev.txt", \
     "--baseline-run", "data/toy/reference_bm25_run_dev.trec", \
     "--run-out", "runs/dev_run.trec", \
     "--report-out", "runs/dev_report.json"]
