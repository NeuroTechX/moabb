#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import inspect
import os
import os.path as op
import sys

import matplotlib


matplotlib.use("agg")

from datetime import datetime

import sphinx_gallery  # noqa
from numpydoc import docscrape, numpydoc  # noqa
from sphinx_gallery.sorting import ExplicitOrder, FileNameSortKey  # noqa

import moabb  # noqa


sys.path.insert(0, os.path.abspath("../"))
sys.path.insert(0, os.path.abspath("../../"))


matplotlib.use("Agg")
# -- Project information -----------------------------------------------------

project = "moabb"
year = datetime.now().year
copyright = f"2018-{year} MOABB contributors"
author = "Alexandre Barachant, Vinay Jayaram, Sylvain Chevallier"

# The short X.Y version
version = moabb.__version__
# The full version, including alpha/beta/rc tags
release = f"{moabb.__version__}-dev"


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = "2.0"

curdir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(curdir, "sphinxext")))

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.githubpages",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.intersphinx",
    "sphinx.ext.imgmath",
    "sphinx.ext.napoleon",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
    "gh_substitutions",
    "myst_parser",
    "numpydoc",
    "sphinx_favicon",
    "sphinxcontrib.jquery",
]


def linkcode_resolve(domain, info):  # noqa: C901
    """Determine the URL corresponding to a Python object.

    Parameters
    ----------
    domain : str
        Only useful when "py".
    info : dict
        With keys "module" and "fullname".

    Returns
    -------
    url : str
        The code URL.

    Notes
    -----
    This has been adapted to deal with our "verbose" decorator.
    Adapted from SciPy (doc/source/conf.py).
    """
    repo = "https://github.com/NeuroTechX/moabb"

    if domain != "py":
        return None
    if not info["module"]:
        return None

    modname = info["module"]
    fullname = info["fullname"]

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
        except Exception:
            return None

    try:
        fn = inspect.getsourcefile(obj)
    except Exception:
        fn = None
    if not fn:
        try:
            fn = inspect.getsourcefile(sys.modules[obj.__module__])
        except Exception:
            fn = None
    if not fn:
        return None
    fn = op.relpath(fn, start=op.dirname(moabb.__file__))
    fn = "/".join(op.normpath(fn).split(os.sep))  # in case on Windows

    try:
        source, lineno = inspect.getsourcelines(obj)
    except Exception:
        lineno = None

    if lineno:
        linespec = "#L%d-L%d" % (lineno, lineno + len(source) - 1)
    else:
        linespec = ""

    # if "dev" in moabb.__version__:
    #     kind = "develop"
    # else:
    #     kind = "master"
    return f"{repo}/blob/develop/moabb/{fn}{linespec}"


# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
curdir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(curdir, "..", "moabb")))
sys.path.append(os.path.abspath(os.path.join(curdir, "sphinxext")))

sphinx_gallery_conf = {
    "examples_dirs": ["../../examples"],
    "gallery_dirs": ["auto_examples"],
    "doc_module": ("moabb", "mne"),
    "backreferences_dir": "generated",
    "show_memory": True,
    "reference_url": dict(moabb=None),
    "filename_pattern": "(/plot_|/tutorial_)",
    "default_thumb_file": "../images/M.png",
    "subsection_order": ExplicitOrder(
        [
            "../../examples/tutorials",
            "../../examples/paradigm_examples",
            "../../examples/data_management_and_configuration",
            "../../examples/how_to_benchmark",
            "../../examples/advanced_examples",
            "../../examples/learning_curve",
        ]
    ),
    "within_subsection_order": "FileNameSortKey",
}


autodoc_default_options = {"inherited-members": False}
autodoc_default_flags = {"inherited-members": None}
autosummary_generate = True

numpydoc_show_class_members = False

exclude_patterns = ["build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = [".rst", ".md"]

# The master toctree document.
master_doc = "index"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.


html_theme = "pydata_sphinx_theme"
switcher_version_match = "dev" if release.endswith("dev0") else version
# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.

html_theme_options = {
    "icon_links": [
        dict(
            name="GitHub",
            url="https://github.com/NeuroTechX/moabb",
            icon="fa-brands fa-square-github",
        ),
    ],
    "github_url": "https://github.com/NeuroTechX/moabb",
    "icon_links_label": "External Links",  # for screen reader
    "use_edit_page_button": False,
    "navigation_with_keys": False,
    "collapse_navigation": False,
    "navigation_depth": -1,
    "show_toc_level": 1,
    "nosidebar": True,
    "navbar_end": ["theme-switcher"],
    "announcement": "https://raw.githubusercontent.com/neurotechx/moabb/develop/docs/source/_templates/custom-template.html",
    "show_version_warning_banner": True,
    "analytics": dict(google_analytics_id="G-5WJBKDMSTE"),
    "pygments_light_style": "tango",
    "pygments_dark_style": "monokai",
    "logo": {
        "image_light": "moabb_light.svg",
        "image_dark": "moabb_dark.svg",
    },
}

html_sidebars = {
    "whats_new": [],
    "paper_results": [],
    "dataset_summary": [],
    "api": [],
}

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "images/moabb_logo.svg"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_css_files = [
    "https://raw.githubusercontent.com/neurotechx/moabb/refs/heads/develop/docs/source/_static/css/custom.css",
    "https://cdn.datatables.net/v/dt/dt-2.0.4/b-3.0.2/b-html5-3.0.2/datatables.min.css",
]

html_js_files = [
    "https://code.jquery.com/jquery-3.7.1.min.js",
    "https://cdn.datatables.net/v/dt/dt-2.0.4/b-3.0.2/b-html5-3.0.2/datatables.min.js",
]

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False
html_copy_source = False

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = False

# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "moabbdoc"

# accommodate different logo shapes (width values in rem)
xs = "2"
sm = "2.5"
md = "3"
lg = "4.5"
xl = "5"
xxl = "6"

html_context = {
    "build_dev_html": bool(int(os.environ.get("BUILD_DEV_HTML", False))),
    "default_mode": "light",
    "icon_links_label": "Quick Links",  # for screen reader
    "show_toc_level": 1,
    "github_user": "neurotechx",
    "github_repo": "moabb",
    "github_version": "develop",
    "doc_path": "docs",
}


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ("letterpaper" or "a4paper").
    #
    # "papersize": "letterpaper",
    # The font size ("10pt", "11pt" or "12pt").
    #
    # "pointsize": "10pt",
    # Additional stuff for the LaTeX preamble.
    #
    # "preamble": "",
    # Latex figure (float) alignment
    #
    # "figure_align": "htbp",
}

latex_logo = "images/moabb_logo.svg"
latex_toplevel_sectioning = "part"

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, "moabb.tex", "moabb Documentation", "Alexandre Barachant", "manual"),
]


# -- Fontawesome support -----------------------------------------------------

# here the "fab" and "fas" refer to "brand" and "solid" (determines which font
# file to look in). "fw" indicates fixed width.
brand_icons = ("apple", "linux", "windows", "discourse", "python")
fixed_icons = (
    # homepage:
    "book",
    "code-branch",
    "newspaper",
    "question-circle",
    "quote-left",
    # contrib guide:
    "bug",
    "comment",
    "hand-sparkles",
    "magic",
    "pencil-alt",
    "remove-format",
    "universal-access",
    "discourse",
    "python",
)
other_icons = (
    "hand-paper",
    "question",
    "rocket",
    "server",
    "code",
    "desktop",
    "terminal",
    "cloud-download-alt",
    "wrench",
    "hourglass",
)
icons = dict()
for icon in brand_icons + fixed_icons + other_icons:
    font = ("fab" if icon in brand_icons else "fas",)  # brand or solid font
    fw = ("fa-fw",) if icon in fixed_icons else ()  # fixed-width
    icons[icon] = font + fw

prolog = ""
for icon, classes in icons.items():
    prolog += f"""
.. |{icon}| raw:: html

    <i class="{" ".join(classes)} fa-{icon}"></i>
"""

prolog += """
.. |fix-bug| raw:: html

    <span class="fa-stack small-stack">
        <i class="fas fa-bug fa-stack-1x"></i>
        <i class="fas fa-ban fa-stack-2x"></i>
    </span>
"""

prolog += """
.. |ensp| unicode:: U+2002 .. EN SPACE
"""


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "moabb", "moabb Documentation", [author], 1)]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "moabb",
        "moabb Documentation",
        author,
        "moabb",
        "Mother of all BCI benchmarks.",
        "Miscellaneous",
    ),
]

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/{.major}".format(sys.version_info), None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "sklearn": ("http://scikit-learn.org/stable", None),
    "mne": ("http://mne.tools/stable", None),
    "skorch": ("https://skorch.readthedocs.io/en/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "moabb": ("https://neurotechx.github.io/moabb/", None),
}

# -- Options for sphinx-gallery ----------------------------------------------
favicons = [
    {
        "rel": "moabb icon",
        "sizes": "180x180",
        "href": "moabb_logo.png",  # use a local file in _static
    },
    {"rel": "icon", "href": "favicon.svg", "type": "image/svg+xml"},
    {"rel": "icon", "sizes": "144x144", "href": "favicon-144.png", "type": "image/png"},
    {"rel": "mask-icon", "href": "favicon_mask-icon.svg", "color": "#222832"},
    {"rel": "apple-touch-icon", "sizes": "500x500", "href": "favicon-500.png"},
]

# -- Options for MyST --------------------------------------------------------
# Required due to README.md file starting at H2 not H1
suppress_warnings = ["myst.header"]
