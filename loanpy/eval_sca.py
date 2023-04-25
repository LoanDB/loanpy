# -*- coding: utf-8 -*-
"""
This module focuses on evaluating the quality of adapted and reconstructed
words in a linguistic dataset by leveraging data-driven and heuristic
prosodic and phonetic correspondences. It processes the input data,
which consists of tokenised IPA source and target strings, as well as
prosodic strings, and applies the correspondences to predict the best
possible adaptations or reconstructions. The module then
calculates the accuracy of the predictions by generating a table of
false positives (how many guesses) vs true positives,
providing insights into the effectiveness of this method.
Additionally, the module offers the option to apply phonotactic repairs,
allowing for more refined analysis of the linguistic data.
Overall, this module aims to facilitate a deeper understanding of
loanword adaptation and historical sound change processes
by quantifying the success rate of predictive models.
"""

import re
from typing import Dict, List, Tuple

from loanpy.scapplier import Adrc
from loanpy.scminer import get_correspondences, get_inventory

def eval_all(
        intable: List[List[str]],
        heur: Dict[str, List[str]],
        adapt: bool,
        guess_list: List[int],
        pros: bool = False
        ) -> List[Tuple[int, int]]:
    """
    #. Input a loanpy-compatible table containing etymological data.
    #. Start a neseted for-loop
    #. The first loop goes through the number of guesses (~ false positives)
    #. The second loops through the input-table and calculates the relative
       number of true positives.
    #. The output is a list of tuples containing the relative number of
       true positives vs. false positives

    :param intable: The input tsv-table, edited with the Edictor.
    :type intable: list of lists
    :param heur: The heuristic prosodic correspondences.
    :type heur: list
    :param adapt: Whether words are adapted or reconstructed.
    :type adapt: bool
    :param guess_list: The list of number of guesses to evaluate.
    :type guess_list: list of int
    :param pros: Wheter phonotactic repairs should be applied
    :type pros: bool, default=False
    :return: A list of tuples of integer-pairs
             representing false positives vs true positives
    :rtype: tuple
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
        ) -> Tuple[float]:
    """
    Called by loanpy.eval.eval_all.
    loops through the loanpy-compatible etymological input-table and
    performs `leave-one-out cross validation
    <https://en.wikipedia.org/wiki/Cross-validation_(statistics)#Leave-one-out_cross-validation>`_.
    The result is how many words were correctly predicted, relative to the
    length of the table

    :param intable: The input tsv-table, edited with the Edictor.
                    Tokenised IPA source and target strings must be
                    in column "ALIGNMENT". Prosodic strings in col "PROSODY".
    :type intable: list of lists
    :param heur: The heuristic sound and prosodic correspondences.
                 Created with loanpy.recover.get_correspondences
    :type heur: dict
    :param adapt: Whether words are adapted or reconstructed.
    :type adapt: bool
    :param howmany: Howmany guesses should be made. Treated as false positives.
    :type howmany: list
    :param pros: Whether phonotactic/prosodic repairs should apply
    :type pros: bool, default=False
    :return: A tuple with the ratio of successful adaptations/reconstructions
             (rounded to 2 decimal places).
    :rtype: tuple
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
        adrc.set_inventory(get_inventory(intable))  # extract inventory
        if adapt:
            ad = adrc.adapt(src, howmany, src_pros)
            out.append(tgt in ad)
        else:
            rc = adrc.reconstruct(src, howmany)
            out.append(bool(re.match(rc, tgt)))
        intable.insert(i, tgtrow)
        intable.insert(i, srcrow)

    return round(len([i for i in out if i]) / len(out), 2)
