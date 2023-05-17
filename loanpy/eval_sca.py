# -*- coding: utf-8 -*-
"""
This module focuses on evaluating the quality of adapted and reconstructed
words. It processes the input data,
which consists of tokenised IPA source and target strings, as well as
prosodic strings, and extracts and applies correspondences to predict the best
possible adaptations or reconstructions. The module then
calculates the accuracy of the predictions by counting the relative number of
false positives (how many guesses) vs true positives.
Overall, this module aims to facilitate a deeper understanding of
loanword adaptation and historical sound change processes
by quantifying the success rate of predictive models.
"""

import re
from typing import Dict, List, Tuple

from loanpy.scapplier import Adrc
from loanpy.scminer import get_correspondences, get_prosodic_inventory

def eval_all(
        intable: List[List[str]],
        heur: Dict[str, List[str]],
        adapt: bool,
        guess_list: List[int],
        pros: bool = False
        ) -> List[Tuple[int, int]]:
    """
    #. Input a loanpy-compatible table containing etymological data.
    #. Start a nested for-loop for
    #. The first loop goes through the number of guesses (~ false positives)
    #. The second loop performs `leave-one-out cross validation
       <https://en.wikipedia.org/wiki/Cross-validation_(statistics)#Leave-one-out_cross-validation>`_
       with ``loanpy.eval_sca.eval_one``.
    #. The output is a list of tuples containing the relative number of
       true positives vs. relative number of false positives

    :param intable: The input tsv-table. Space-separated tokenised IPA source
                    and target strings must be in column “ALIGNMENT”, prosodic
                    strings in column “PROSODY”.
    :type intable: list of lists
    :param heur: The path to the heuristic sound correspondences file,
                 e.g. "heur.json", which was created with
                 ``loanpy.scminer.get_heur``.
    :type heur: str or pathlike object, optional
    :param adapt: Set to ``True`` to make predictions with
                  ``loanpy.scapplier.Adrc.adapt``, set to ``False`` to
                  make predictions with
                  ``loanpy.scapplier.Adrc.reconstruct``.
    :type adapt: bool
    :param guess_list: The list of number of guesses to evaluate.
    :type guess_list: list of int
    :param pros: Wheter phonotactic repairs should be applied
    :type pros: bool, default=False
    :return: A list of tuples of integer-pairs
             representing false positives vs true positives
    :rtype: list of tuples of integers

    `Run in Google Colab >> <https://colab.research.google.com/drive/1JlHKfdff_yjCO8yvxiKV9xoRAiEPgarM#scrollTo=9TJGhnf5Ysmk&line=3&uniqifier=1>`__

    .. code-block:: python

        >>> from loanpy.eval_sca import eval_all
        >>> intable = [  ['ID', 'COGID', 'DOCULECT', 'ALIGNMENT', 'PROSODY'],
        ...   ['0', '1', 'H', 'k i k i', 'VC'],
        ...   ['1', '1', 'EAH', 'k i g i', 'VCVCV'],
        ...   ['2', '2', 'H', 'i k k i', 'VCV'],
        ...   ['3', '2', 'EAH', 'i g k i', 'VCCVC']
        ... ]
        >>>
        >>> eval_all(intable, "", False, [1, 2, 3])
        [(0.33, 0.0), (0.67, 1.0), (1.0, 1.0)]
    """

    true_positives = []

    # Iterate through the guess list and calculate evaluation results
    for num_guesses in guess_list:
        eval_result = eval_one(intable, heur, adapt, num_guesses, pros)
        true_positives.append(eval_result)  # already normalised

    # Normalize guess list values
    normalised_guess_list = [round(i / guess_list[-1], 2) for i in guess_list]

    # Combine normalized guess list with true positive results
    fp_vs_tp = list(zip(normalised_guess_list, true_positives))

    return fp_vs_tp


def eval_one(
        intable: List[List[str]],
        heur: Dict[str, List[str]],
        adapt: bool,
        howmany: int,
        pros: bool = False
        ) -> float:
    """
    Called by ``loanpy.eval.eval_all``.
    Loops through the loanpy-compatible etymological input-table and
    performs `leave-one-out cross validation
    <https://en.wikipedia.org/wiki/Cross-validation_(statistics)#Leave-one-out_cross-validation>`_.
    The result is how many words were correctly predicted, relative to the
    total number of predictions made.

    :param intable: The input tsv-table. Space-separated tokenised IPA source
                    and target strings must be in column “ALIGNMENT”, prosodic
                    strings in column “PROSODY”.
    :type intable: list of lists
    :param heur: The path to the heuristic sound correspondences file,
                 e.g. "heur.json", which was created with
                 ``loanpy.scminer.get_heur``.
    :type heur: str or pathlike object, optional
    :param adapt: Set to ``True`` to make predictions with
                  ``loanpy.scapplier.Adrc.adapt``, set to ``False`` to
                  make predictions with
                  ``loanpy.scapplier.Adrc.reconstruct``.
    :type adapt: bool
    :param howmany: Howmany guesses should be made. Treated as false positives.
    :type howmany: list
    :param pros: Wheter phonotactic repairs should be applied
    :type pros: bool, default=False

    :return: A tuple with the ratio of successful predictions
             (rounded to 2 decimal places).
    :rtype: float

    `Run in Google Colab >> <https://colab.research.google.com/drive/1JlHKfdff_yjCO8yvxiKV9xoRAiEPgarM#scrollTo=mxCV1_xiWZpu&line=11&uniqifier=1>`__

    .. code-block:: python

        >>> from loanpy.eval_sca import eval_one
        >>> intable = [  # regular sound correspondences
        ...     ['ID', 'COGID', 'DOCULECT', 'ALIGNMENT', 'PROSODY'],
        ...     ['0', '1', 'H', 'k i k i', 'VC'],
        ...     ['1', '1', 'EAH', 'g i g i', 'VCVCV'],
        ...     ['2', '2', 'H', 'i k k i', 'VCV'],
        ...     ['3', '2', 'EAH', 'i g g i', 'VCCVC']
        ... ]
        >>> eval_one(intable, "", False, 1)
        1.0

        >>> intable = [  # not enough regular sound correspondences
        ...   ['ID', 'COGID', 'DOCULECT', 'ALIGNMENT', 'PROSODY'],
        ...   ['0', '1', 'H', 'k i k i', 'VC'],
        ...   ['1', '1', 'EAH', 'g i g i', 'VCVCV'],
        ...   ['2', '2', 'H', 'b u b a', 'VCV'],
        ...   ['3', '2', 'EAH', 'p u p a', 'VCCVC']
        ... ]
        >>> eval_one(intable, "", False, 1)
        0.0

        >>> intable = [  # irregular sound correspondences
        ...   ['ID', 'COGID', 'DOCULECT', 'ALIGNMENT', 'PROSODY'],
        ...   ['0', '1', 'H', 'k i k i', 'VC'],
        ...   ['1', '1', 'EAH', 'k i g i', 'VCVCV'],
        ...   ['2', '2', 'H', 'i k k i', 'VCV'],
        ...   ['3', '2', 'EAH', 'i g k i', 'VCCVC']
        ... ]
        >>> eval_one(intable, "", False, 1)
        0.0

        >>> intable = [  # irregular sound correspondences
        ...   ['ID', 'COGID', 'DOCULECT', 'ALIGNMENT', 'PROSODY'],
        ...   ['0', '1', 'H', 'k i k i', 'VC'],
        ...   ['1', '1', 'EAH', 'k i g i', 'VCVCV'],
        ...   ['2', '2', 'H', 'i k k i', 'VCV'],
        ...   ['3', '2', 'EAH', 'i g k i', 'VCCVC']
        ... ]
        >>> eval_one(intable, "", False, 2)  # increase rate of false positives
        1.0

    """

    out = []
    h = {i: intable[0].index(i) for i in intable[0]}
    for i in range(1, len(intable), 2):  # 1 bc skip header
        srcrow, tgtrow = intable.pop(i), intable.pop(i)  # leave one out
        src = srcrow[h["ALIGNMENT"]]   # define left-outs as test input
        tgt = "".join(re.sub("[-. ]", "", tgtrow[h["ALIGNMENT"]]))
        src_pros = srcrow[h["PROSODY"]] if pros else ""
        adrc = Adrc()   # initiate adapt-reconstruct class
        adrc.set_sc(get_correspondences(intable, heur))  # extract info from traing data
        adrc.set_prosodic_inventory(get_prosodic_inventory(intable))  # extract prosodic_inventory
        if adapt:
            try:
                ad = adrc.adapt(src, howmany, src_pros)
                out.append(tgt in ad)
            except KeyError:  # bugfix issue #50
                out.append(False)
        else:
            rc = adrc.reconstruct(src, howmany)
            out.append(bool(re.match(rc, tgt)))
        intable.insert(i, tgtrow)
        intable.insert(i, srcrow)

    return round(len([i for i in out if i]) / len(out), 2)
