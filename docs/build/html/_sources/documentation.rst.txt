Intro
=================================

.. automodule:: __init__

For usage
=================================
This section describes classes, methods, and functions \
that the user calls. It deals with the \
the extraction, evaluation, and application of \
patterns of correspondence \
between phonemes, phoneme clusters, \
and phonotactic profiles. In addition, it shows how to search for \
semantically and phonologically matching pairs between two lists of words.

qfysc.py
-----------
.. automodule:: qfysc

Quantify
~~~~~~~~~~
.. autoclass:: Qfy

get sound correspondences
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automethod:: Qfy.get_sound_corresp

adrc.py
--------------
.. automodule:: adrc

Adapt/Reconstruct
~~~~~~~~~~~~~~~~~~
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
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: Search

loans
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automethod:: Search.loans


For internal calls
==========================

This section covers classes, methods, and functions, which are mostly \
called internally and are not intended to be called by the user. \
Some functions, like loanpy.helpers.make_cvfb, do have to be called \
by a user, but only under \
special and rare circumstances, some others, like \
loanpy.helpers.Etym.make_scdictbase only \
once per investigated target language. \
And yet some others, like loanpy.helpers.editops, \
loanpy.helpers.Etym.has_harmony, or loanpy.helpers.Etym.repair_harmony \
might turn out as a useful tool in a completely different context.

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

loop through data
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

get non-cross-validated sound correspondences
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: loanpy.sanity.get_noncrossval_sc

get cross-validated data
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: loanpy.sanity.get_crossval_data

post-process
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: loanpy.sanity.postprocess

post-process 2
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

get true positive rate, false positive rate, optimum
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: loanpy.sanity.get_tpr_fpr_opt

plot ROC-curve
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
~~~~~~~~~~~
.. autofunction:: loanpy.loanfinder.read_data

generator
~~~~~~~~~~
.. autofunction:: loanpy.loanfinder.gen

phonological matching
~~~~~~~~~~~~~~~~~~~~~~
.. automethod:: loanpy.loanfinder.Search.phonmatch

post-process
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automethod:: loanpy.loanfinder.Search.postprocess

likeliest phonological match
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automethod:: loanpy.loanfinder.Search.likeliestphonmatch

phonological match small
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automethod:: loanpy.loanfinder.Search.phonmatch_small

merge with rest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automethod:: loanpy.loanfinder.Search.merge_with_rest
