Intro
=================================

.. automodule:: __init__

For usage
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

sanity.py
-----------
.. automodule:: sanity

evaluate all
~~~~~~~~~~~~~~~~
.. autofunction:: eval_all

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

read sound correspondence dictionary base
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

get phonotactic correspondences
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automethod:: loanpy.qfysc.Qfy.get_phonotactics_corresp

adrc.py
---------

read sound correspondence dictionary list
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: loanpy.adrc.read_scdictlist

move sound correspondences
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: loanpy.adrc.move_sc

get difference
~~~~~~~~~~~~~~~~~
.. automethod:: loanpy.adrc.Adrc.get_diff

read sound correspondences
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automethod:: loanpy.adrc.Adrc.read_sc

repair phonotactics
~~~~~~~~~~~~~~~~~~~~~
.. automethod:: loanpy.adrc.Adrc.repair_phonotactics

get normalised sum of examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automethod:: loanpy.adrc.Adrc.get_nse

sanity.py
-----------

cache
~~~~~~~~~~~~~~~~~~
.. autofunction:: loanpy.sanity.cache

loop_thru_data
~~~~~~~~~~~~~~~~~~
.. autofunction:: loanpy.sanity.loop_thru_data

evaluate one
~~~~~~~~~~~~~~~~~~
.. autofunction:: loanpy.sanity.eval_one

evaluate adapt
~~~~~~~~~~~~~~~~~~
.. autofunction:: loanpy.sanity.eval_adapt

evaluate reconstruct
~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: loanpy.sanity.eval_recon

get non-crossvalidated sound correspondences
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: loanpy.sanity.get_noncrossval_sc

get crossvalidated data
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: loanpy.sanity.get_crossval_data

postprocess
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: loanpy.sanity.postprocess

postprocess2
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: loanpy.sanity.postprocess2

get normalised sum of examples for data frame
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: loanpy.sanity.get_nse4df

phonotactics predicted?
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: loanpy.sanity.phonotactics_predicted

get distances
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: loanpy.sanity.get_dist

make statistics
~~~~~~~~~~~~~~~~~~
.. autofunction:: loanpy.sanity.make_stat

get true positive rate false positive rate, optimum
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: loanpy.sanity.get_tpr_fpr_opt

plot roc-curve
~~~~~~~~~~~~~~~~~~
.. autofunction:: loanpy.sanity.plot_roc

check cache
~~~~~~~~~~~~~~~~~~
.. autofunction:: loanpy.sanity.check_cache

write to cache
~~~~~~~~~~~~~~~~~~
.. autofunction:: loanpy.sanity.write_to_cache

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
