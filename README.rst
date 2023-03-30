.. image:: https://zenodo.org/badge/428920599.svg
   :target: https://zenodo.org/record/6628976

.. image:: https://dl.circleci.com/status-badge/img/gh/martino-vic/loanpy/tree/2%2E0%2E1.svg?style=svg
   :target: https://dl.circleci.com/status-badge/redirect/gh/martino-vic/loanpy/tree/2%2E0%2E1

.. image:: https://coveralls.io/repos/github/martino-vic/loanpy/badge.svg?branch=2.0.1
   :target: https://coveralls.io/github/martino-vic/loanpy?branch=2.0.1

.. image:: https://readthedocs.org/projects/loanpy/badge/?version=latest
   :target: https://loanpy.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://img.shields.io/pypi/v/your_package_name.svg
  :target: https://pypi.org/project/loanpy/
  :alt: PyPI Latest Version



loanpy: A Comprehensive Linguistic Toolkit
==========================================

loanpy is a powerful linguistic toolkit developed during the course of my PhD thesis, providing solutions for various linguistic tasks, such as:

- extracting correspondence patterns from etymological data on the level of phonology and phonotactics
- Predicting loanword adaptation (lateral/horizontal transfers) based on heuristics and etymological data
- Historical reconstruction of words (vertical transfers) based on etymological data
- Searching for potential (old) loanwords between two languages

This versatile toolkit has been designed with ease of use and compatibility in mind, offering a standalone, easy-to-setup, and cross-platform solution that works with Python 3.7 or higher.

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

- [Read the docs](https://loanpy.readthedocs.io/en/latest/documentation.html)
- [Tutorial](https://loanpy.readthedocs.io/en/latest/tutorial.html)

Citation
--------

If you use loanpy in your research or project, please cite the following:

Viktor Martinović. (2022). loanpy. Zenodo. https://doi.org/10.5281/zenodo.6628976

BibTex:

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

loanpy is released under the MIT License.

Compatible Input Data
---------------------

loanpy supports the following datasets:

- `Gothic <https://github.com/martino-vic/streitberggothic>`_
- `Hungarian <https://github.com/martino-vic/gerstnerhungarian>`_
- `Proto-Bolgar <https://github.com/martino-vic/ronataswestoldturkic>`_

Main Changes Compared to Version 2
----------------------------------

- Standalone: No external dependencies needed.
- Easy setup: Just download or clone the repository and start using it.
- Cross-platform: Works on any platform that supports Python.
- Python Compatibility: Works with Python 3.7 or higher.
- CLDF Integration: Fully integrated in the [CLDF ecosystem](https://cldf.clld.org/).
