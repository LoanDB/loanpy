LOANPY
======

| |loanpy|
| |Build Status|

loanpy is a tool for historical linguists. It extracts sound changes and constraints from an etymological dictionary, generates pseudo-roots for L1, pseudo- sound-substitutions for L2, searches for phonetically identical lexeme-pairs and ranks them according to semantic similarity.

Installation
~~~~~~~~~~~~

.. code:: sh

    $ python -m pip install loanpy

Getting started
~~~~~~~~~~~~~~~

.. code:: sh

    >>> from loanpy import loanfinder as lf

Download and unpack 3 Gigabytes of `pretrained Google-News
vectors <https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit>`__. Move *GoogleNews-vectors-negative300.bin* to the folder "data", the full path to which can be retrieved via:

.. code:: sh

    >>> import os
    >>> print(os.path.dirname(lf.__file__)+r"\data")

Following code will compare a set of Gothic words (data/dfgot.csv) with Hungarian words (data/zaicz.csv) and evaluate which elements are the most likely candidates for loanwords.
The result can be viewed in data/results/matches.csv:

::

    >>> lf.loandf()

Data Sources
_________________
`Gábor Zaicz's  Hungarian etymological dictionary from 2006 <https://regi.tankonyvtar.hu/hu/tartalom/tinta/TAMOP-4_2_5-09_Etimologiai_szotar/adatok.html>`__

`Gerhard Köbler's Gothic database <https://koeblergerhard.de/wikiling/?f=got>`__

`Hungarian Academy of Science's online version of Uralisches Etymologisches Wörterbuch <http://uralonet.nytud.hu>`__

License
-------

Academic Free License (AFL) (Creative Commons Attribution 4.0
International)

.. |loanpy| image:: https://github.com/martino-vic/loanpy/blob/master/logo_resizeimage.svg
   :target: https://pypi.org/project/loanpy/
.. |Build Status| image:: https://about.zenodo.org/static/img/logos/zenodo-gradient-square.svg
   :target: https://zenodo.org/record/4100594#.X5RgbIgzaUk