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
    "myst_nb",                 # render the tutorial notebooks
    "sphinx_copybutton",       # copy button on every code block
    "sphinx_design",           # cards and grids on the landing page
]

# Notebooks are shown with the outputs they were committed with: the builder has
# no GPU, so executing them here is neither possible nor desirable.
nb_execution_mode = "off"
exclude_patterns = ["_build", "jupyter_execute"]
myst_enable_extensions = ["dollarmath", "amsmath"]
suppress_warnings = ["mystnb.unknown_mime_type"]

# Work from a source checkout without installing the package first.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# autodoc imports the package to read its docstrings. Dr.Jit itself imports
# fine without a CUDA device (only creating arrays needs one) and the
# GPU-dependent module-level work is guarded, so a builder with no GPU — Read
# the Docs, GitHub Actions — works as long as the real dependencies are
# installed. `autodoc_mock_imports` is deliberately NOT used as a substitute:
# mocking Dr.Jit breaks the module-level type unions (`dr.ArrayBase | bool`),
# autodoc then imports nothing, and the API page silently comes out empty.
# `fail_on_warning` in .readthedocs.yaml turns that failure mode into a red
# build instead of a published empty page.

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
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "source_repository": "https://github.com/SepandKashani/xrt_toolkit/",
    "source_branch": "v2",
    "source_directory": "doc/",
    "light_css_variables": {
        "color-brand-primary": "#0e7490",
        "color-brand-content": "#0e7490",
        "color-brand-visited": "#155e75",
        "font-stack": "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', "
                      "Helvetica, Arial, sans-serif",
        "font-stack--monospace": "'JetBrains Mono', SFMono-Regular, Menlo, "
                                 "Consolas, Monaco, 'Liberation Mono', monospace",
    },
    "dark_css_variables": {
        "color-brand-primary": "#22d3ee",
        "color-brand-content": "#38bdf8",
        "color-brand-visited": "#67e8f9",
    },
}

pygments_style = "tango"
pygments_dark_style = "github-dark"
copybutton_exclude = ".linenos, .gp, .go"      # skip prompts and output lines

# The library docstrings use LaTeX macros; define the ones they rely on.
mathjax3_config = {
    "tex": {
        "macros": {
            "bbx": r"\mathbf{x}", "bbq": r"\mathbf{q}", "bbt": r"\mathbf{t}",
            "bbn": r"\mathbf{n}", "bbE": r"\mathbf{E}", "bbe": r"\mathbf{e}",
            "bbZ": r"\mathbb{Z}", "bR": r"\mathbb{R}", "bbQ": r"\mathbf{Q}",
            "bbDelta": r"\boldsymbol{\Delta}", "bbZero": r"\mathbf{0}",
            "bbH": r"\mathbf{H}", "bbb": r"\mathbf{b}", "bbT": r"\mathbf{T}",
            "bbA": r"\mathbf{A}", "bbu": r"\mathbf{u}", "bbm": r"\mathbf{m}",
            "diag": r"\operatorname{diag}",
            "xrt": r"\mathcal{P}",
            "discreteRange": [r"\left\{#1, \ldots, #2\right\}", 2],
        }
    }
}
