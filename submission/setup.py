"""Build the optional compiled tokenizer from inside ``submission/``.

CI and grading run this file as:

    cd submission && python setup.py build_ext --inplace

The extension is therefore named ``_fasttok`` and its source path is
relative to this directory.  Python imports it as ``submission._fasttok``
when the repository root is on ``sys.path``.
"""
from setuptools import Extension, setup

try:
    from Cython.Build import cythonize
except ImportError:  # no Cython -> the pure-Python fallback remains usable
    cythonize = None


extensions = [Extension("_fasttok", ["_fasttok.pyx"], language="c++")]

setup(
    name="a1-sparse-retrieval-fasttok",
    ext_modules=cythonize(extensions, language_level="3") if cythonize else [],
    zip_safe=False,
)
