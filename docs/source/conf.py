# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'PySSC'
copyright = '2018, Lacasa'
author = 'Fabien Lacasa'

release = '0.1'
version = '0.1.0'

# -- General configuration

import mock
import sys
import os 

sys.path.insert(0, os.path.abspath('../..'))


 
MOCK_MODULES = ['numpy', 
                'scipy',
                'scipy.integrate', 
                'matplotlib', 
                'matplotlib.pyplot', 
                'scipy.interpolate', 
                'classy', 
                'healpy'
                'scipy.special']

for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'nbsphinx',
    'sphinx_gallery.load_style']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']
autodoc_mock_imports = ['numpy', 'scipy', 'math', 'healpy', 'classy']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'
