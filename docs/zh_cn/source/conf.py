# Configuration file for the Sphinx documentation builder (中文文档).
# -----------------------------------------------------------------------------
# 参考 Lightx2v 样式，把原先 trojanzoo_sphinx_theme 改为 sphinx_book_theme，
# 并修正 logo 配置格式。

import os
import sys
from typing import List

# -- Path setup --------------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(__file__, "../../.."))
sys.path.append(ROOT_DIR)

# -- 项目信息 ---------------------------------------------------------------
project = "llmc"
copyright = "2024, llmc contributors"
author = "ModelTC"
release = "1.0.0"

# GitHub 信息 ---------------------------------------------------------------
github_url = "https://github.com/ModelTC/llmc"

html_context = {
    "display_github": True,
    "github_user": author,
    "github_repo": "llmc",
    "github_version": "main",
    "conf_py_path": "/docs/zh_cn/source/",  # 文档根路径
}

# -- 通用配置 ----------------------------------------------------------------
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

language = "zh_CN"

# 复制代码块时去除shell提示符 ---------------------------------------------
copybutton_prompt_text = r"\$ "
copybutton_prompt_is_regexp = True

# -- HTML 输出选项 -----------------------------------------------------------
html_title = project
html_theme = "sphinx_book_theme"
html_logo = "images/logo/llmc.svg"
html_static_path = ["_static"]

html_theme_options = {
    "path_to_docs": "docs/zh_cn/source",
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

# -- Intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", {}),
    "sphinx": ("https://www.sphinx-doc.org/en/master", {}),
}

# -- Mock 外部依赖 -----------------------------------------------------------
autodoc_mock_imports = [
    "torch",
    "transformers",
    "sentencepiece",
    "tensorizer",
]

# -- 自定义处理 -------------------------------------------------------------
from sphinx.ext import autodoc  # noqa: E402, isort: skip

class MockedClassDocumenter(autodoc.ClassDocumenter):
    """移除“Bases: object”行。"""

    def add_line(self, line: str, source: str, *lineno: int) -> None:
        if line == "   Bases: :py:class:`object`":
            return
        super().add_line(line, source, *lineno)

autodoc.ClassDocumenter = MockedClassDocumenter

# -- 额外钩子 ---------------------------------------------------------------

def setup(app):
    """可选的 Sphinx setup。"""
    pass
