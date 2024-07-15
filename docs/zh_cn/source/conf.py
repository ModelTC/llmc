# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "llmc"
copyright = "2024, llmc contributors"
author = "ModelTC"
release = "1.0.0"

github_url = f"https://github.com/ModelTC/llmc"

html_context = {
    "display_github": True,
    "github_user": author,
    "github_repo": "llmc",
    "github_version": "main",
    "conf_py_path": "/docs/zh_cn/source/",  # Path in the checkout to the docs root
}
html_theme_options = {
    "github_url": github_url,
    "doc_items": {
        "paper": "https://arxiv.org/abs/2405.06001",
        "institution": "https://github.com/ModelTC",
    },
    "logo": "images/logo/llmc.svg",
    "logo_dark": "images/logo/llmc.svg",
    "logo_icon": "images/logo/llmc.svg",
}


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinxcontrib.contentui",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx-prompt",
    "sphinxcontrib.jquery",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinxcontrib.katex",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = []

language = "cn"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


html_theme = "trojanzoo_sphinx_theme"

html_static_path = ["_static"]

source_suffix = [".rst", ".md"]
