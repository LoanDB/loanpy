"""
.. image:: PyPI_logo.svg
   :target: https://pypi.org/project/loanpy/
.. image:: zenodo-gradient-200.png
   :target: https://zenodo.org/record/4127115#.YHCQwej7SLQ

loanpy is a toolkit for historical linguists. \
It extracts sound changes from an etymological \
dictionary. It reconstructs hypothetical roots of \
modern L1 words. It creates hypothetical adaptions \
of L2 words into proto-L1. It searches for potential \
loanwords by first finding phonetic matches \
and then calculating their semantic similarity. All examples \
shown here can also be found in loanpy/data/examples.ipynb

.. note::
    .. list-table::
       :widths: 25 25
       :header-rows: 1

       * - Symbol
         - Meaning
       * - ͽ
         - any vowel
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
   and using a memory-saving browser, like e.g. Opera, is recommended.

"""
