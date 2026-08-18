# Sphinx configuration. Build with:  python -m sphinx doc doc/_build/html
project = "XRT Toolkit"
author = "S. Kashani, Y. Haouchat"
release = "2.0.0"

extensions = [
    "sphinx.ext.autodoc",      # pull docstrings from the code
    "sphinx.ext.napoleon",     # understand numpydoc-style sections
    "sphinx.ext.mathjax",      # render the :math: roles
    "sphinx.ext.viewcode",     # "source" links next to each object
    "sphinx.ext.intersphinx",  # link numpy/python types
]

# Importing the package needs a CUDA device; on a machine or CI runner without
# one, mock the GPU dependencies so the docstrings can still be extracted.
autodoc_mock_imports = []
try:
    import xrt_toolkit  # noqa: F401
except Exception:
    autodoc_mock_imports = ["drjit", "cupy", "torch", "astra"]

autodoc_member_order = "bysource"
autodoc_typehints = "none"        # signatures stay readable; types are in the docstrings
napoleon_use_rtype = False
add_module_names = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

html_theme = "furo"
html_title = f"{project} {release}"

# The library docstrings use LaTeX macros; define the ones they rely on.
mathjax3_config = {
    "tex": {
        "macros": {
            "bbx": r"\mathbf{x}", "bbq": r"\mathbf{q}", "bbt": r"\mathbf{t}",
            "bbn": r"\mathbf{n}", "bbE": r"\mathbf{E}", "bbe": r"\mathbf{e}",
            "bbZ": r"\mathbb{Z}", "bR": r"\mathbb{R}", "bbQ": r"\mathbf{Q}",
            "bbDelta": r"\boldsymbol{\Delta}", "bbZero": r"\mathbf{0}",
            "bbH": r"\mathbf{H}", "bbb": r"\mathbf{b}", "bbT": r"\mathbf{T}",
            "discreteRange": [r"\{#1, \ldots, #2\}", 2],
        }
    }
}
