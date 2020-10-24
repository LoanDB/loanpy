============
Installation
============

::

    $ python -m pip install loanpy
	
.. image:: PyPI_logo.svg
   :target: https://pypi.org/project/loanpy/
	

================
Documentation
================


.. image:: white_logo_dark_background.jpg
   :target: https://martino-vic.github.io/loanpy/index.html
   
    
====================
Citation
====================

.. image:: zenodo-gradient-200.png
   :target: https://zenodo.org/record/4127115#.YHCQwej7SLQ   

============
Description
============

loanpy is a toolkit for historical linguists.
It extracts sound changes from an etymological dictionary.
It reconstructs hypothetical roots of modern L1 words.
It creates hypothetical adaptions of L2 words into proto-L1.
It searches for potential loanwords by first finding phonetic matches
and then calculating their semantic similarity.


Data Sources
~~~~~~~~~~~~~~~~~~~~~~~

- **dfhun_zaicz_backup.csv**: data frame based on the `Hungarian etymological dictionary (Zaicz 2006) <https://regi.tankonyvtar.hu/hu/tartalom/tinta/TAMOP-4_2_5-09_Etimologiai_szotar/adatok.html>`__

- **dfgot_wikiling_backup.csv**: data frame based on `Wikiling <https://koeblergerhard.de/wikiling/?f=got>`__

- **dfgot_wiktionary_backup.csv**: data frame based on `Wiktionary <https://en.wiktionary.org/wiki/Category:Gothic_lemmas>`__

- **dfuralonet.csv**: data frame based on the `Uralonet <http://uralonet.nytud.hu>`__

- **substi.csv**: Sound substitutions based on `Information-theoretic causal inference of lexical flow (Dellert 2017)  <https://langsci-press.org/catalog/book/233>`__

- **wordvectornames.xlsx**: Names of pretrained word vector models from `gensim-data  <https://github.com/RaRe-Technologies/gensim-data>`__


Dependencies
~~~~~~~~~~~~~~~~~~~

- `gensim 4.0.1  <https://pypi.org/project/gensim/>`__

- `ipatok 0.2.0  <https://pypi.org/project/ipatok/>`__

- `levenshtein 0.12.0 <https://pypi.org/project/levenshtein/>`__

- `pandas 1.2.4 <https://pypi.org/project/pandas/>`__


License
~~~~~~~~~~~~~~~~

Academic Free License (AFL)



=======
Git
=======

.. image:: Octocat.png
   :target: https://github.com/martino-vic/loanpy
   :scale: 30%