loanpy
======

|tests| |docs| |python| |pypi|

Linguistic toolkit for **loanword detection** and **sound change** (version 4).
Pure Python 3.9+ with **no runtime dependencies**.

.. |tests| image:: https://github.com/loanwordbank/loanpy/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/loanwordbank/loanpy/actions/workflows/tests.yml
   :alt: Tests

.. |docs| image:: https://readthedocs.org/projects/loanpy/badge/?version=latest
   :target: https://loanpy.readthedocs.io/en/latest/
   :alt: Documentation

.. |python| image:: https://img.shields.io/pypi/pyversions/loanpy
   :alt: Python versions

.. |pypi| image:: https://img.shields.io/pypi/v/loanpy
   :target: https://pypi.org/project/loanpy/
   :alt: PyPI

Features
--------

* **Segment clustering** — ``Cluster.cv``, ``Cluster.glides``, ``Cluster.gaps``
* **Alignment** — ``Uralign.hu``, ``Uralign.get_score``
* **Sound correspondences** — ``get_sound_correspondences``
* **Adaptation** — ``Adapt`` (substitution + phonotactic repair)
* **Edit distance** — insertion/deletion distance, DP matrix, shortest path
* **Phonotactics** — ``expand_phonotactics``, ``get_closest_phonotactics``

Installation
------------

From PyPI (when published)::

    pip install loanpy

From GitHub (release tag)::

    pip install "loanpy @ git+https://github.com/loanwordbank/loanpy.git@v4.0.0"

Development::

    git clone https://github.com/loanwordbank/loanpy.git
    cd loanpy
    pip install -e ".[test,docs]"
    pytest

Documentation
-------------

Full API reference and tutorials: https://loanpy.readthedocs.io/

Build locally::

    pip install -e ".[docs]"
    cd docs && make html

Quick example
-------------

::

    from loanpy import Cluster, get_sound_correspondences, Uralign

    clusters = Cluster.cv(["f", "l", "a"], ["C", "C", "V"])
    # ['f.l', 'a']

Typical use
-----------

* **CLDF dataset pipelines** — ``cldfbench`` modules call ``Cluster`` and
  ``Uralign`` when writing ``forms.csv`` / ``cognates.csv`` columns.
* **Loanword detection studies** — combine ``Adapt``, ``Uralign``, and
  ``get_sound_correspondences`` over aligned cognate tables (see the project
  documentation for a full workflow).

Package layout
--------------

============================= ================================================
Module                        Main symbols
============================= ================================================
``loanpy.cluster``            ``Cluster``
``loanpy.uralign``            ``Uralign``
``loanpy.adapt``              ``Adapt``
``loanpy.correspondences``    ``get_sound_correspondences``
``loanpy.edit``               edit distance utilities
``loanpy.phonotactics``       ``expand_phonotactics``, ``get_closest_phonotactics``
============================= ================================================

Re-exported from ``loanpy`` (see ``loanpy.__all__``).

Testing
-------

::

    pytest --cov=loanpy --cov-report=term-missing

See ``docs/testing.rst`` for coverage notes.

License
-------

MIT — see ``LICENSE``.

Changelog
---------

See ``CHANGELOG.md``.

Citation
--------

If you use loanpy in research, cite the GitHub release and the Zenodo DOI
(added when you publish version 4.0.0 on Zenodo).
