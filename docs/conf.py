# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

from typing import Any, Dict

import os
import sys

# import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath("../"))

from miv import get_version

# -- Project information -----------------------------------------------------

project = "Mind-in-Vitro MiV-OS"
copyright = "2022, GazzolaLab"
author = "Gazzola Lab"

# The full version, including alpha/beta/rc tags
release = get_version()

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx_autodoc_typehints",
    #'sphinx.ext.napoleon',
    "sphinx.ext.viewcode",
    "sphinx_togglebutton",
    "sphinx_copybutton",
    "sphinx_rtd_theme",
    "sphinx.ext.mathjax",
    "numpydoc",
    # "myst_parser", # Moving onto jupyter-notebook style
    "myst_nb",
]

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    # "fieldlist",
    "html_admonition",
    "html_image",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "README.md",  # File reserved to explain how documentationing works.
]

autodoc_default_options = {
    # "members": False,
    "member-order": "bysource",
    "special-members": "",
    "exclude-members": "__weakref__",
    "show-inheritance": True,
}
autosectionlabel_prefix_document = True
autosummary_generate = True
autosummary_generate_overwrite = False

source_parsers: Dict[str, str] = {}
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".myst": "myst-nb",
    ".ipynb": "myst-nb",
}

master_doc = "index"
# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/GazzolaLab/MiV-OS",
    "use_repository_button": True,
}
html_title = "MiV-OS"
# html_logo = ""
# pygments_style = "sphinx"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static", "_static/assets"]
html_css_files = ["css/*", "css/logo.css"]

# -- Options for numpydoc ---------------------------------------------------
numpydoc_show_class_members = False

# -- Options for myst-nb ---------------------------------------------------
nb_execution_mode = "off"
