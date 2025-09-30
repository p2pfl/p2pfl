#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2022 Pedro Guijas Bravo.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

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

"""Sphinx configuration file for p2pfl documentation."""

import os
import sys
from importlib.metadata import version as get_version

sys.path.insert(0, os.path.abspath("../../p2pfl"))


# -- Project information -----------------------------------------------------

project = "p2pfl"

copyright = "2022, Pedro Guijas Bravo"
author = "Pedro Guijas Bravo"

html_logo = "logo.png"


# The full version, including alpha/beta/rc tags
release = get_version("p2pfl")
version = release


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinxcontrib.mermaid",
    "sphinx_design",
    "sphinx_autodoc_typehints",
]

# Autodoc options
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": False,
    "special-members": False,
    "inherited-members": False,
    "show-inheritance": True,
}

# Type hints configuration (using sphinx-autodoc-typehints extension v3.2.0)
autodoc_typehints = "description"  # Show type hints in parameter descriptions
autodoc_typehints_description_target = "documented"  # Only add type hints to documented parameters
typehints_fully_qualified = False  # Use short type names (e.g., 'str' instead of 'builtins.str')
always_document_param_types = True  # Always document parameter types

# Mock imports for packages that are not needed for documentation
autodoc_mock_imports = [
    "grpc_tools",
    "grpc",
    "google",
    "google.protobuf",
    "tensorflow",
    "torch",
    "torchvision",
    "torchmetrics",
    "lightning",
    "keras",
    "flax",
    "optax",
    "opendp",
    "wandb",
    "ray",
    "jax",
    "pytorch_lightning",
    "p2pfl.communication.protocols.protobuff.proto.node_pb2",
    "p2pfl.communication.protocols.protobuff.proto.node_pb2_grpc",
]

# Todos
todo_include_todos = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
# exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"
html_favicon = "favicon.ico"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_theme_options = {
    "sidebar_hide_name": True,
}
