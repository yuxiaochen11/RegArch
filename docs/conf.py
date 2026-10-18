# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "RegArch"
copyright = "2025–2026, Xiaochen Yu"
author = "Xiaochen Yu"
release = "0.1.0"


import sys
from pathlib import Path

CONF_DIR = Path(__file__).resolve().parent
REPO_ROOT = CONF_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

extensions = [
    "myst_parser",
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
autodoc_member_order = "bysource"

static_dir = CONF_DIR / "_static"
static_dir.mkdir(exist_ok=True)

html_logo = "_static/logo.png"
html_theme = "furo"

html_theme_options = {
    "sidebar_hide_name": True,
    "light_css_variables": {
        "color-brand-primary": "#357473",
        "color-brand-content": "#357473",
    },
}
html_static_path = ["_static"]
