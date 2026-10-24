import datetime as dt
import importlib.metadata as im
import os
import sys


sys.path.insert(0, os.path.abspath("../src"))  # src-layout project structure

# Project information ---------------------------------------------------------
cfg = im.metadata("xrt_toolkit")
project = "XRT Toolkit"
author = cfg["Author-email"].replace('"', "")
copyright = f"{dt.date.today().year}, {author}"

# Compute legible version info.
version = cfg["Version"]  # <semver>[.devXXX][+<git-hash>]
version = version.strip().split("+")[0]  # restrict to <semver>[.devXXX]
release = version
# -----------------------------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Theme options ---------------------------------------------------------------
html_theme = "bizstyle"
html_sidebars = {
    "**": [
        "globaltoc.html",
        "relations.html",
        "searchbox.html",
    ]
}

# Extension options -----------------------------------------------------------
autodoc_member_order = "bysource"
autodoc_typehints = "none"

intersphinx_mapping = {
    "drjit": ("https://drjit.readthedocs.io/en/latest/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
}

mathjax3_config = dict(
    tex=dict(
        macros={
            # Vectors & Matrices ----------------------------------------------
            "bbA": r"{\bf{A}}",
            "bbb": r"{\bf{b}}",
            "bbe": r"{\bf{e}}",
            "bbE": r"{\bf{E}}",
            "bbH": r"{\bf{H}}",
            "bbI": r"{\bf{I}}",
            "bbm": r"{\bf{m}}",
            "bbn": r"{\bf{n}}",
            "bbq": r"{\bf{q}}",
            "bbQ": r"{\bf{Q}}",
            "bbt": r"{\bf{t}}",
            "bbu": r"{\bf{u}}",
            "bbx": r"{\bf{x}}",
            "bbDelta": r"{\boldsymbol{\Delta}}",
            "bbZero": r"{\bf{0}}",
            # Number Spaces ---------------------------------------------------
            "bR": r"{\mathbb{R}}",
            # Misc ------------------------------------------------------------
            "discreteRange": [  # {#1_1,...,#2_1} x ... x {#1_D,...,#2_D}
                r"\lbrack\!\lbrack #1 \rbrack\!\rbrack",
                1,
            ],
            # Math Operators --------------------------------------------------
            "diag": r"\operatorname*{diag}",
            "xrt": r"\mathcal{P}",
        },
    ),
)

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

bibtex_bibfiles = ["bibliography.bib"]
bibtex_reference_style = "label"
