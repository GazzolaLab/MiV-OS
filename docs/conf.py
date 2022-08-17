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

sys.path.insert(0, os.path.abspath("../"))

from miv import get_version

# -- Project information -----------------------------------------------------

project = "Mind-in-Vitro"
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


html_theme = "pydata_sphinx_theme"
html_logo = "https://raw.githubusercontent.com/skim0119/MiV-OS/assets_logo/docs/_static/assets/logo1.svg"
html_favicon = "https://raw.githubusercontent.com/skim0119/MiV-OS/assets_logo/docs/_static/assets/logo1.svg"
html_sourcelink_suffix = ""
html_theme_options = {
    "logo": {
        # "link": "https://mindinvitro.illinois.edu/",
        "text": "Open Software"
    },
    # Navbar Configuration
    "navbar_start": ["navbar-logo", "miv-switcher.html"],
    "navbar_center": ["navbar-nav"],
    # "navbar_end": ["navbar-icon-links"],
    # Header Link
    "external_links": [
        # {"name": "link-one-name", "url": "https://<link-one>"},
    ],
    "icon_links": [
        {
            # Label for this link
            "name": "GitHub",
            "url": "https://github.com/GazzolaLab/MiV-OS",  # required
            # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
            # The type of image to be used (see below for details)
            "icon": "fab fa-github-square",
            "type": "fontawesome",
        }
    ],
    # Sidebar Configuraion
    "left_sidebar_end": [],
    # Theme (https://help.farbox.com/pygments.html)
    # "pygment_light_style": "default",
    # "pygment_dark_style": "native",
}
html_title = "MiV-OS"

html_context = {"default_mode": "dark"}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static", "_static/assets"]
# html_css_files = ["css/*", "css/logo.css"]

# -- Options for numpydoc ---------------------------------------------------
numpydoc_show_class_members = False

# -- Options for myst-nb ---------------------------------------------------
nb_execution_mode = "auto"
