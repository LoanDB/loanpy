# -*- coding: utf-8 -*-
"""Check how sane the model is by evaluating predictions."""
import re

from loanpy.apply import Adrc
from loanpy.recover import get_correspondences, get_inventory

def eval_all(edicted, heur, adapt, guess_list, pros=False):
    """
    Get a table of False Positives (how many guesses) vs True Positives.

    :param edicted: The input tsv-table, edited with the Edictor.
    :type edicted: list of lists
    :param heur: The heuristic sound and prosodic correspondences.
    :type heur: dict
    :param adapt: Whether words are adapted or reconstructed.
    :type adapt: bool
    :param guess_list: The list of guesses to evaluate.
    :type guess_list: list
    :param args: Additional arguments for
                 loanpy.apply.Adrc.adapt and loanpy.apply.Adrc.reconstruct
    :type args: list
    :return: A tuple containing a list of tuples representing FP vs TP
             and a list of workflow results.
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

    :param edicted: The input tsv-table, edited with the Edictor.
    :type edicted: list of lists
    :param heur: The heuristic sound and prosodic correspondences.
    :type heur: dict
    :param adapt: Whether words are adapted or reconstructed.
    :type adapt: bool
    :param guess_list: The list of guesses to evaluate.
    :type guess_list: list
    :param args: Additional arguments for
                 loanpy.apply.Adrc.adapt and loanpy.apply.Adrc.reconstruct
    :type args: list
    :return: A tuple with the ratio of successful adaptations/reconstructions
             (rounded to 2 decimal places) and a list of workflows.
    :rtype: tuple
    """
    out = []
    for i in range(1, len(edicted), 2):  # 1 bc skip header
        srcrow, tgtrow = edicted.pop(i), edicted.pop(i)
        src, tgt = srcrow[3], "".join(tgtrow[3].replace("-", "").replace(" ", "").replace(".", ""))
        src_pros = srcrow[4] if pros else ""
        adrc = Adrc()
        adrc.sc = get_correspondences(edicted, heur)
        adrc.inventory = get_inventory(edicted)
        if adapt:
            ad = adrc.adapt(src, howmany, src_pros).split(", ")
            #print("tgt: ", tgt, "src: ", src, ", ad: ", ad)
            out.append(tgt in ad)
        else:
            rc = adrc.reconstruct(src, howmany)
            #print(rc, tgt)
            out.append(bool(re.match(rc, tgt)))
        edicted.insert(i, tgtrow)
        edicted.insert(i, srcrow)
    return round(len([i for i in out if i])/len(out), 2)
