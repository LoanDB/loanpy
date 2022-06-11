Installation
============

::

    $ python -m pip install loanpy

Documentation
==============

https://martino-vic.github.io/loanpy/documentation.html

Citation
==========

https://zenodo.org/record/4127115#.YHCQwej7SLQ

License
==========

Academic Free License (AFL)

Description
============

loanpy is a toolkit for solving various linguistic tasks such as:

* predicting loanword adaptation (lateral transfers)

* historical reconstruction of words (vertical transfers)

* searching for potential (old) loanwords between two languages.


Compatible input data
======================

- `Gothic <https://github.com/martino-vic/streitberggothic>`_
- `Hungarian <https://github.com/martino-vic/gerstnerhungarian>`_
- `West Old Turkic <https://github.com/martino-vic/ronatasbertawot>`_

Main changes compared to version 1
======================================================

* Mange data flow through classes instead of gloabal variables
* reconstructor.py and adapter.py reorganised into adrc.py and qfysc.py
* new module sanity.py to evaluate results and optimise parameters
* Sound and phonotactic substitutions based on etymological data combined with feature vectors from the PanPhon library
* Implementation of CLDF standard
* new, linpy-based alignment method
* multiple paths when repairing phonotactics, with help of the networkx library
* tested with real-world data: current model seems to accurately predict English loanword adaptation in Maori



Dependencies
==============

::

    appdirs==1.4.4
    attrs==21.4.0
    clldutils==3.12.0
    colorlog==6.6.0
    csvw==2.0.0
    cycler==0.11.0
    editdistance==0.6.0
    fonttools==4.33.3
    gensim==4.2.0
    ipatok==0.4.1
    isodate==0.6.1
    kiwisolver==1.4.2
    latexcodec==2.0.1
    lingpy==2.6.9
    matplotlib==3.5.2
    munkres==1.1.4
    networkx==2.8.3
    numpy==1.22.4
    packaging==21.3
    pandas==1.4.2
    panphon==0.20.0
    Pillow==9.1.1
    pybtex==0.24.0
    pycldf==1.26.1
    pyparsing==3.0.9
    python-dateutil==2.8.2
    pytz==2022.1
    PyYAML==6.0
    regex==2022.6.2
    rfc3986==1.5.0
    scipy==1.8.1
    six==1.16.0
    smart-open==6.0.0
    tabulate==0.8.9
    tqdm==4.64.0
    unicodecsv==0.14.1
    uritemplate==4.1.1
    pytest==7.1.2
