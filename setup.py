"""
setup.py — builds the compiled tokenizer extension.

Per the assignment (Section 5, "Compiled extensions") and
docs/SUBMISSION_INTERFACE.md, a C/C++ extension must be built at IMAGE
BUILD time, not inside build_index() -- anything build_index() does is
charged against the index-build-time efficiency metric, and a one-time
compile is not indexing work.

    python setup.py build_ext --inplace

The Dockerfile and the CI workflow both run this. If it is skipped, or if
Cython or a compiler is unavailable, the submission still works:
submission/indexer.py falls back to the pure-Python tokenizer and produces
a byte-identical index, just more slowly. Nothing about correctness or the
interface contract depends on this extension existing.
"""
from setuptools import Extension, setup

try:
    from Cython.Build import cythonize
except ImportError:  # no Cython -> ship pure Python, see module docstring
    cythonize = None

extensions = [Extension("submission._fasttok", ["submission/_fasttok.pyx"])]

setup(
    name="a1-sparse-retrieval",
    ext_modules=cythonize(extensions, language_level="3") if cythonize else [],
    zip_safe=False,
)
