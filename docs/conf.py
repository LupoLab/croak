"""Sphinx configuration for the croak documentation.

Build locally with::

    uv sync --group docs
    uv run sphinx-build -b html docs docs/_build/html
"""

from __future__ import annotations

import croak

# -- Project information -----------------------------------------------------
project = "croak"
author = "John C. Travers and the LUPO research group"
copyright = "2026, John C. Travers and contributors"  # noqa: A001
release = croak.__version__
version = release

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_nb",  # MyST Markdown + executed notebooks (pulls in myst_parser)
    "sphinx.ext.napoleon",  # NumPy-style docstrings
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "**.ipynb_checkpoints", "Thumbs.db", ".DS_Store"]

# -- MyST / MyST-NB ----------------------------------------------------------
myst_enable_extensions = [
    "amsmath",
    "dollarmath",
    "colon_fence",
    "deflist",
    "fieldlist",
    "html_image",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3

# Execute the tutorial notebooks at build time (cached). A failing cell fails
# the build, so the docs can never drift from the code.
nb_execution_mode = "auto"
nb_execution_timeout = 600
nb_execution_raise_on_error = True
nb_merge_streams = True

# -- autodoc / autosummary ---------------------------------------------------
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_member_order = "bysource"

napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_rtype = False
napoleon_use_param = True

# -- intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
}
intersphinx_disabled_reftypes = ["*"]  # only resolve explicit :external: refs

# -- HTML output -------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = "croak"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "github_url": "https://github.com/LupoLab/croak",
    "navigation_with_keys": True,
    "show_toc_level": 2,
    "use_edit_page_button": False,
    "navbar_align": "left",
    "header_links_before_dropdown": 6,
}

# MathJax: render the physics-convention transforms and Wirtinger algebra.
mathjax3_config = {
    "tex": {
        "macros": {
            "Et": r"\tilde E",
        }
    }
}
