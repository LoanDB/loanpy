# Configuration file for the Sphinx documentation builder.

import os
import sys

sys.path.insert(0, os.path.abspath('..'))

project = 'LoanPy'
copyright = '2023, Viktor Martinović'
author = 'Viktor Martinović'
version = '3.0'
release = '3.0'

html_theme = 'sphinx_rtd_theme'
extensions = ['sphinx.ext.autodoc','sphinxcontrib.examples']
autodoc_mock_imports = ['sphinxcontrib.examples']

# Links
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
# https://www.sphinx-doc.org/en/master/usage/configuration.html
