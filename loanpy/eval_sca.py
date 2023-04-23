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

from loanpy.scapplier import Adrc
from loanpy.scminer import get_correspondences, get_inventory

def eval_all(edicted, heur, adapt, guess_list, pros=False):
    """
    Get a table of False Positives (how many guesses) vs True Positives.

    :param edicted: The input tsv-table, edited with the Edictor.
    :type edicted: list of lists
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
        eval_result = eval_one(edicted, heur, adapt, num_guesses, pros)
        true_positives.append(eval_result)  # already normalised

    # Normalize guess list values
    normalised_guess_list = [round(i / guess_list[-1], 2) for i in guess_list]

    # Combine normalized guess list with true positive results
    fp_vs_tp = list(zip(normalised_guess_list, true_positives))

    return fp_vs_tp


def eval_one(edicted, heur, adapt, howmany, pros=False):
    """
    Evaluate the quality of the adapted and reconstructed words.
    Called by loanpy.eval.eval_all.

    :param edicted: The input tsv-table, edited with the Edictor.
                    Tokenised IPA source and target strings must be
                    in column "ALIGNMENT". Prosodic strings in col "PROSODY".
    :type edicted: list of lists
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
    h = {i: edicted[0].index(i) for i in edicted[0]}
    for i in range(1, len(edicted), 2):  # 1 bc skip header
        srcrow, tgtrow = edicted.pop(i), edicted.pop(i)
        src = srcrow[h["ALIGNMENT"]]
        tgt = "".join(re.sub("[-. ]", "", tgtrow[h["ALIGNMENT"]]))
        src_pros = srcrow[h["PROSODY"]] if pros else ""
        adrc = Adrc()
        adrc.sc = get_correspondences(edicted, heur)
        adrc.inventory = get_inventory(edicted)
        if adapt:
            ad = adrc.adapt(src, howmany, src_pros)
            #print("tgt: ", tgt, "src: ", src, ", ad: ", ad)
            out.append(tgt in ad)
        else:
            rc = adrc.reconstruct(src, howmany)
            #if not bool(re.match(rc, tgt)):
            #    print(howmany, src, rc, tgt)
            out.append(bool(re.match(rc, tgt)))
        edicted.insert(i, tgtrow)
        edicted.insert(i, srcrow)
    return round(len([i for i in out if i])/len(out), 2)
