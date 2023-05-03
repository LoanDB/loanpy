.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.7893906.svg
   :target: https://doi.org/10.5281/zenodo.7893906

.. image:: https://dl.circleci.com/status-badge/img/gh/LoanpyDataHub/loanpy/tree/2.0.1.svg?style=svg
       :target: https://dl.circleci.com/status-badge/redirect/gh/LoanpyDataHub/loanpy/tree/2.0.1

.. image:: https://coveralls.io/repos/github/martino-vic/loanpy/badge.svg?branch=2.0.1
   :target: https://coveralls.io/github/martino-vic/loanpy?branch=2.0.1

.. image:: https://readthedocs.org/projects/loanpy/badge/?version=latest
   :target: https://loanpy.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://img.shields.io/pypi/v/loanpy.svg
  :target: https://pypi.org/project/loanpy/
  :alt: PyPI Latest Version



LoanPy: A Comprehensive Linguistic Toolkit
==========================================

loanpy is a linguistic toolkit developed during the course of my PhD thesis, providing solutions for various linguistic tasks, such as:

- Mining correspondence patterns from etymological data on the level of phonology and phonotactics
- Predicting loanword adaptation (lateral/horizontal transfers) based on heuristics and etymological data
- Historical reconstruction of words (vertical transfers) based on etymological data
- Evaluating the quality of the predictive models
- Searching for potential (old) loanwords between two languages

This toolkit has been designed with ease of use and compatibility in mind, offering a standalone, easy-to-setup, and cross-platform solution that works with Python 3.7 or higher.

Upcoming Release: Version 3
---------------------------

Stay tuned for the upcoming release of loanpy version 3, which is scheduled for May 2023.

Installation
------------

Latest stable version:

::

    $ python -m pip install loanpy

Development version:

::

    $ python -m pip install git+https://github.com/martino-vic/loanpy.git@2.0.1

Documentation
-------------

- `Read the docs <https://loanpy.readthedocs.io/en/latest/home.html>`_

Citation
--------

If you use loanpy in your research or project, please cite the following:

::

   @Misc{Martinovic2022,
     author    = {Viktor Martinovi{\'c}},
     title     = {loanpy},
     year      = {2022},
     doi       = {10.5281/zenodo.6628976},
     publisher = {Zenodo},
   }

License
-------

LoanPy is released under the MIT License.

Compatible Input Data
---------------------

LoanPy supports the following datasets:

- `streitberggothic <https://github.com/LoanpyDataHub/streitberggothic>`_,
- `koeblergothic <https://github.com/LoanpyDataHub/koeblergothic>`_
- `gerstnerhungarian <https://github.com/LoanpyDataHub/gerstnerhungarian>`_
- `ronataswestoldturkic <https://github.com/LoanpyDataHub/ronataswestoldturkic>`_

Main Changes Compared to Version 2
----------------------------------

- Standalone: No external dependencies needed.
- Easy setup: Just download or clone the repository and start using it.
- Cross-platform: Works on any platform that supports Python.
- Python Compatibility: Works with Python 3.7 or higher.
- CLDF Integration: Fully compatible with `CLDF data standards <https://cldf.clld.org/>`_.
