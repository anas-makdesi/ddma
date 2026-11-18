"""Sphinx configuration for the ddma documentation.

Build:  pip install -e ".[docs]" && sphinx-build -W --keep-going -b html docs docs/_build/html
The tutorials (tutorials/*.md, MyST notebooks) are executed during the build.
"""

import os
import sys
from pathlib import Path

import ddma

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent / "_ext"))

project = "ddma"
author = "Anas Makdesi, Antoine Girard, Laurent Fribourg, Majid Zamani, Saber Jafarpour"
copyright = "2020–2023 CentraleSupélec; 2026 Anas Makdesi"
version = release = ddma.__version__

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "matlab_api",
    "ddma_docstrings",
    "sphinx_design",
    "sphinx_copybutton",
]

# ---------------------------------------------------------------- sources
source_suffix = {".rst": "restructuredtext", ".md": "myst-nb"}
# the Markdown files at the top of docs/ are linked from the README; the guides
# include them, so they are not built twice
exclude_patterns = ["_build", "jupyter_execute", "**/.ipynb_checkpoints", "rl_safety_filter.json",
                    "CHANGELOG.md", "INVARIANCE.md", "MONITOR.md", "PARITY.md", "PERFORMANCE.md",
                    "REPRODUCTION.md", "ROADMAP.md"]
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist", "attrs_inline"]
myst_heading_anchors = 3

# ---------------------------------------------------------------- notebooks
# Executed on every build (CI included); a failing cell fails the build.
nb_execution_mode = os.environ.get("DDMA_NB_EXECUTION", "cache")
nb_execution_timeout = 600
nb_execution_raise_on_error = True
nb_execution_show_tb = True
nb_merge_streams = True

# ---------------------------------------------------------------- API (Python)
autodoc_member_order = "bysource"
autodoc_inherit_docstrings = False
autodoc_default_options = {"members": True}
# the docstrings describe shapes and types in words; annotations would only add noise
autodoc_typehints = "none"
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_rtype = False
napoleon_preprocess_types = False
add_module_names = False

# ---------------------------------------------------------------- API (MATLAB)
# api/matlab.md is generated from the help text of matlab/+ddma/*.m (_ext/matlab_api.py)
matlab_src_dir = str(ROOT / "matlab")

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "gymnasium": ("https://gymnasium.farama.org", None),
}
intersphinx_timeout = 10

# ---------------------------------------------------------------- HTML
html_theme = "furo"
html_title = f"ddma {release}"
html_baseurl = "https://anas-makdesi.github.io/ddma/"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "source_repository": "https://github.com/anas-makdesi/ddma",
    "source_branch": "main",
    "source_directory": "docs/",
    "light_css_variables": {"color-brand-primary": "#2a7a6f", "color-brand-content": "#2a7a6f"},
    "dark_css_variables": {"color-brand-primary": "#5fc4b4", "color-brand-content": "#5fc4b4"},
}
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True


# ---------------------------------------------------------------- links between the guides
# docs/*.md link to each other by file name (so the links work on GitHub); here those
# files are included by the guides, so point the links at the guide pages.
_GUIDES = {"PARITY.md": "guides/parity", "PERFORMANCE.md": "guides/performance",
           "INVARIANCE.md": "guides/envelopes", "MONITOR.md": "guides/monitor",
           "REPRODUCTION.md": "guides/reproduction",
           "CHANGELOG.md": "about/changelog", "ROADMAP.md": "about/roadmap"}


def _guide_links(app, doctree):
    from sphinx.addnodes import pending_xref

    for node in doctree.findall(pending_xref):
        target = node.get("reftarget", "").split("/")[-1].split("#")[0]
        if target in _GUIDES:
            node["refdomain"], node["reftype"] = "std", "doc"
            node["reftarget"] = "/" + _GUIDES[target]
            node["refexplicit"] = True


def setup(app):
    app.connect("doctree-read", _guide_links)
