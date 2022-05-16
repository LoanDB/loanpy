"""
.. image:: PyPI_logo.svg
   :target: https://pypi.org/project/loanpy/
.. image:: zenodo-gradient-200.png
   :target: https://zenodo.org/record/4127115#.YHCQwej7SLQ
.. image:: Octocat.png
   :scale: 35%
   :target: https://github.com/martino-vic/loanpy
   

|



loanpy is a toolkit for historical linguists. It can be used to:\n\

    - extract sound change and substitution patterns from etymological \
dictionaries. 
    - reconstruct hypothetical proto-forms of present-day words
    - simulate loanword adaptation
    - search for potential loanwords between two word lists

All examples \
shown here can also be found in loanpy/data/examples.ipynb

.. note::
    .. list-table::
       :widths: 25 25
       :header-rows: 1

       * - Symbol
         - Meaning
       * - V
         - any vowel
       * - C
         - any consonant
       * - ȣ̈
         - any front vowel
       * - ȣ
         - any back vowel


.. note::
   Files that are saved as .csv are UTF-8 encoded without BOM.
   To view them in Excel, either the encoding has to be changed, e.g.
   by opening the file in Notepad++, selecting "encoding",
   "UTF-8 BOM", "save". Or the data has to be imported to Excel via "Data",
   "Get Data", "From File", "From Text/CSV", selecting the file, "Open",
   from dropdown "File Origin" selecting "65001: Unicode (UTF-8)", "Load".


.. note::
   Loading word vectors can use a lot of RAM. If a memory error appears,
   closing all other tabs, windows and background processes
   and using a memory-saving browser, e.g. Opera, can help.

"""
