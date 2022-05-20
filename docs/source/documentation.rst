Intro
=================================

.. automodule:: __init__

To be used
=================================

qfysc.py
-----------
.. automodule:: qfysc

Qfy
~~~~~~~~~~
.. autoclass:: Qfy

get sound correspondences
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automethod:: Qfy.get_sound_corresp

sanity.py
-----------
.. automodule:: sanity

evaluate all
~~~~~~~~~~~~~~~~
.. autofunction:: eval_all

adrc.py
--------------
.. automodule:: adrc

Adrc
~~~~~~~~
.. autoclass:: Adrc

adapt
~~~~~~~
.. automethod:: Adrc.adapt

reconstruct
~~~~~~~~~~~~~
.. automethod:: Adrc.reconstruct

loanfinder.py
---------------------------------

.. automodule:: loanfinder

Search
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: Search

loans
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automethod:: Search.loans


For internal calls
==========================

helpers.py
------------
.. automodule:: loanpy.helpers
   :members:
   :undoc-members:
   :show-inheritance:

qfysc.py
-----------

read mode
~~~~~~~~~~~
.. autofunction:: loanpy.qfysc.read_mode

read connector
~~~~~~~~~~~~~~~~~
.. autofunction:: loanpy.qfysc.read_connector

read sound change dictionary base
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: loanpy.qfysc.read_scdictbase

align
~~~~~~~~~~~
.. automethod:: loanpy.qfysc.Qfy.align

align with lingpy
~~~~~~~~~~~~~~~~~~~~
.. automethod:: loanpy.qfysc.Qfy.align_lingpy

align clusterwise
~~~~~~~~~~~~~~~~~~~~~
.. automethod:: loanpy.qfysc.Qfy.align_clusterwise

get structure correspondences
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automethod:: loanpy.qfysc.Qfy.get_struc_corresp

sanity.py
-----------

check cache
~~~~~~~~~~~~~~~~~~
.. autofunction:: loanpy.sanity.check_cache

get soundchanges
~~~~~~~~~~~~~~~~~~
.. autofunction:: loanpy.sanity.get_sc

evaluate one
~~~~~~~~~~~~~~~~~~
.. autofunction:: loanpy.sanity.eval_one

make statistics
~~~~~~~~~~~~~~~~~~
.. autofunction:: loanpy.sanity.make_stat

get true positive rate and false positive rate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: loanpy.sanity.gettprfpr

write to cache
~~~~~~~~~~~~~~~~~~
.. autofunction:: loanpy.sanity.write_to_cache

adrc.py
---------

read sound change dictionary list
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: loanpy.adrc.read_scdictlist

move soundchanges
~~~~~~~~~~~~~~~~~~
.. autofunction:: loanpy.adrc.move_sc

get difference
~~~~~~~~~~~~~~~~~
.. automethod:: loanpy.adrc.Adrc.get_diff

read soundchanges
~~~~~~~~~~~~~~~~~
.. automethod:: loanpy.adrc.Adrc.read_sc

adapt structure
~~~~~~~~~~~~~~~~~
.. automethod:: loanpy.adrc.Adrc.adapt_struc

get normalised sum of examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automethod:: loanpy.adrc.Adrc.get_nse

loanfinder.py
-------------------

read data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: loanpy.loanfinder.read_data

generator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: loanpy.loanfinder.gen

phonmatch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automethod:: loanpy.loanfinder.Search.phonmatch

postprocess
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automethod:: loanpy.loanfinder.Search.postprocess

likeliestphonmatch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automethod:: loanpy.loanfinder.Search.likeliestphonmatch

phonmatch_small
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automethod:: loanpy.loanfinder.Search.phonmatch_small

merge_with_rest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automethod:: loanpy.loanfinder.Search.merge_with_rest
