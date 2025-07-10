# Configuration file for the Sphinx documentation builder.
#
# This file adopts the theme and basic settings used by the Lightx2v docs
# but keeps the llmc-specific information from the original configuration.
# -----------------------------------------------------------------------------

import os
import sys
from typing import List

# -- Path setup --------------------------------------------------------------
# Add project root (two levels up) so autodoc can find the modules.
ROOT_DIR = os.path.abspath(os.path.join(__file__, "../../.."))
sys.path.append(ROOT_DIR)

# -- Project information -----------------------------------------------------
project = "llmc"
copyright = "2024, llmc contributors"
author = "ModelTC"
release = "1.0.0"

# GitHub repository ----------------------------------------------------------
github_url = "https://github.com/ModelTC/llmc"

html_context = {
    "display_github": True,
    "github_user": author,
    "github_repo": "llmc",
    "github_version": "main",
    "conf_py_path": "/docs/en/source/",  # Path in the checkout to the docs root
}

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.githubpages",
    "sphinx.ext.autosectionlabel",
    "sphinxcontrib.katex",
    "sphinxcontrib.contentui",
]

templates_path: List[str] = ["_templates"]
exclude_patterns: List[str] = []

language = "en"

# Exclude the prompt "$" when copying code blocks --------------------------
copybutton_prompt_text = r"\$ "
copybutton_prompt_is_regexp = True

# -- Options for HTML output -------------------------------------------------
html_title = project
html_theme = "sphinx_book_theme"
html_logo = "images/logo/llmc.svg"
html_static_path = ["_static"]

# Theme options compatible with sphinx_book_theme / pydata-sphinx-theme
html_theme_options = {
    "path_to_docs": "docs/en/source",
    "repository_url": github_url,
    "use_repository_button": True,
    "logo": {
        "text": "LLMC",
        "image_light": "images/logo/llmc.svg",
        "image_dark": "images/logo/llmc.svg",
    },
    "doc_items": {
        "paper": "https://arxiv.org/abs/2405.06001",
        "institution": "https://github.com/ModelTC",
    },
}

# -- Intersphinx mapping (optional) -----------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", {}),
    "sphinx": ("https://www.sphinx-doc.org/en/master", {}),
}

# -- Mock heavy external dependencies ---------------------------------------
autodoc_mock_imports = [
    "torch",
    "transformers",
    "sentencepiece",
    "tensorizer",
]

# Remove base-class note in generated docs ----------------------------------
from sphinx.ext import autodoc  # noqa: E402, isort: skip

class MockedClassDocumenter(autodoc.ClassDocumenter):
    """Remove note about base class when a class is derived from object."""

    def add_line(self, line: str, source: str, *lineno: int) -> None:
        if line == "   Bases: :py:class:`object`":
            return
        super().add_line(line, source, *lineno)

autodoc.ClassDocumenter = MockedClassDocumenter

# -- Customisation hooks -----------------------------------------------------

def setup(app):
    """Optional Sphinx setup hooks."""
    pass
