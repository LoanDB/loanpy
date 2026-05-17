Testing and coverage
====================

The test suite lives in ``tests/`` and uses `pytest <https://docs.pytest.org/>`_.

Run locally
-----------

.. code-block:: bash

   pip install -e ".[test]"
   pytest

With coverage report:

.. code-block:: bash

   pytest --cov=loanpy --cov-report=term-missing

HTML report:

.. code-block:: bash

   pytest --cov=loanpy --cov-report=html
   # open htmlcov/index.html

What is covered
---------------

Tests exercise the public API end-to-end:

* **edit** — LCS-based distance, DP matrix, shortest path, edit operations
* **cluster** — CV clustering, glide clustering, gap collapsing
* **uralign** — sequential alignment and scorer aggregation
* **correspondences** — alternating-row validation and correspondence mining
* **phonotactics** — template expansion and nearest-template selection
* **adapt** — substitution learning, mapping, and repair
* **package** — version string and ``__all__`` exports

Continuous integration
----------------------

GitHub Actions installs **loanpy** (from the latest GitHub release when available,
otherwise from the repository checkout) and runs ``pytest`` with coverage on
Python 3.9–3.12. See ``.github/workflows/tests.yml``.

Coverage expectations
---------------------

On the current 4.0 test suite, line coverage on ``loanpy/`` is about **95%**
(run ``pytest --cov=loanpy`` locally for an exact figure). Coverage is
informational: linguistic pipelines should still be validated on real CLDF
datasets in downstream projects.
