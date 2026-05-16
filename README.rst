Loanpy
======

Linguistic toolkit for loanword detection and sound change (version 4).

Package layout
--------------

- ``loanpy.cluster`` — ``Cluster`` (CV grouping, glide clustering, gap handling)
- ``loanpy.uralign`` — ``Uralign`` (Hungarian–proto-Uralic alignment and scoring)
- ``loanpy.adapt`` — ``Adapt`` (substitution and phonotactic repair)
- ``loanpy.correspondences`` — ``get_correspondences``, ``get_sound_correspondences``
- ``loanpy.edit`` — edit distance, matrices, and edit operations
- ``loanpy.phonotactics`` — ``expand_phonotactics``, ``get_closest_phonotactics``

The top-level package re-exports ``Adapt``, ``Cluster``, ``Uralign``,
``get_correspondences``, and ``get_sound_correspondences``.

Installation
------------

::

    pip install loanpy

Development::

    pip install -e .

Sound correspondences from CLDF cognates
----------------------------------------

``get_sound_correspondences`` reads a ``cognates.csv`` file (and the sibling
``forms.csv``), filters training pairs, and returns correspondence statistics.
The ``freq`` dict is used as a Uralign scorer (keys like ``ɟ < j.ŋ``).

To also write a TOML file under ``scorers/``::

    from pathlib import Path
    from loanpy import get_sound_correspondences

    get_sound_correspondences(
        "data/UEW-hu/cldf/cognates.csv",
        "Uralign",
        scorer_out_dir=Path("analysis/IndoIranian-Hungarian/scorers"),
    )

``make_results.py`` in the Indo-Iranian–Hungarian analysis imports
``get_sound_correspondences`` from ``loanpy.correspondences`` and builds scorers
in memory at runtime.

License
-------

MIT.
