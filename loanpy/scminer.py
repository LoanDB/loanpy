# -*- coding: utf-8 -*-
"""
The sound correspondence miner module contains several functions to
extract and manipulate linguistic data stored in tab-separated tables.
The main function is ``get_correspondences``, which extracts sound and prosodic
correspondences from the table and returns them as six dictionaries, each
with corresponding frequencies and COGID values. The module also includes
``uralign``, a function that aligns Uralic input strings based on custom
rules, and ``get_heur``, which computes a heuristic mapping between phonemes
in a target language's phoneme inventory and all phonemes in the IPA sound
system based on the Euclidean distance of their feature vectors. Finally,
``get_prosodic_inventory`` extracts all types of prosodic structures
from a target language in a given etymological table.
"""

import json
import math
from collections import defaultdict, Counter
from pathlib import Path
from typing import List, Dict

from loanpy.utils import read_ipa_all

def get_correspondences(
        table: List[List[str]], heur: Dict[str, List[str]] = ""
        ) -> List[Dict]:
    """
    Get sound and prosodic correspondences from a given table string.

    :param table: A list of lists representing an etymological table.
                  It must contain columns
                  named ``ALIGNMENT``, ``PROSODY``, and ``COGID``.
    :type table: list of lists

    :param heur: Optional dictionary containing heuristic correspondences
                 to be merged with the output. Defaults to an empty string.
    :type heur: dictionary with IPA characters as keys and a list of
                phonemes of a language's phoneme inventory ranked
                according to feature vector similarity as values.

    :return: A list of six dictionaries containing correspondences
             and their frequencies:

             1) Sound correspondences.
             2) Frequency of sound correspondences.
             3) COGID values for sound correspondences.
             4) Prosodic correspondences.
             5) Frequency of prosodic correspondences.
             6) COGID values for prosodic correspondences.

    :rtype: list of six dictionaries

    `Run in Google Colab >> <https://colab.research.google.com/drive/1JlHKfdff_yjCO8yvxiKV9xoRAiEPgarM#scrollTo=JMX_mrNhr92B&line=7&uniqifier=1>`__

    .. code-block:: python

        >>> from loanpy.scminer import get_correspondences
        >>> input_table = [
        ...     ['ID', 'COGID', 'DOCULECT', 'ALIGNMENT', 'PROSODY'],
        ...     ['0', '1', 'LG1', 'a b', 'VC'],
        ...     ['1', '1', 'LG2', 'c d', 'CC']
        ... ]
        >>> get_correspondences(input_table)
        [{'a': ['c'], 'b': ['d']},
         {'a c': 1, 'b d': 1},
         {'a c': [1], 'b d': [1]},
         {'VC': ['CC']},
         {'VC CC': 1},
         {'VC CC': [1]}]

    """

    header, table_data = table[0], table[1:]
    cols = {col: i for i, col in enumerate(header)}
    out = [defaultdict(list) for _ in range(6)]  # not *2!

    for i in range(0, len(table_data), 2):
        row1, row2 = table_data[i], table_data[i+1]
        for i, j in zip(
            row1[cols["ALIGNMENT"]].split(" "), row2[cols["ALIGNMENT"]].split(" ")
        ):
            out[0][i].append(j)
            out[1][f"{i} {j}"].append(1)
            out[2][f"{i} {j}"].append(int(row2[cols["COGID"]]))

        cv1, cv2 = row1[cols["PROSODY"]], row2[cols["PROSODY"]]
        out[3][cv1].append(cv2)
        out[4][f"{cv1} {cv2}"].append(1)
        out[5][f"{cv1} {cv2}"].append(int(row2[cols["COGID"]]))

    for i in [0, 3]: # sort by freq
        out[i] = {k: [j[0] for j in Counter(out[i][k]).most_common()] for k in out[i]}
    for i in [1, 4]: # sort by freq
        out[i] = {k: len(out[i][k]) for k in out[i]}
    for i in [2, 5]: # sort by freq
        out[i] = {k: list(dict.fromkeys(out[i][k])) for k in out[i]}

    if heur:
        for k in heur:
            if k in out[0]:
                out[0][k].extend(heur[k])
                out[0][k] = list(dict.fromkeys(out[0][k]))
            else:
                out[0][k] = heur[k]

    return out

def uralign(left: str, right: str) -> str:
    """
    Aligns the left and right input strings based on a custom alignment
    for Hungarian-preHungarian.

    The function splits the input strings by space, aligns them one by one
    and squeezes the remainder of the longer string into one single block,
    which can be seen as a suffix.

    It then returns the aligned strings, joined by a newline character.

    :param left: The left input string with space-separated IPA-sounds.
    :type left: str
    :param right: The right input string with space-separated IPA-sounds.
    :type right: str
    :return: The aligned left and right strings, separated by
             a newline character.
    :rtype: str

    `Run in Google Colab >> <https://colab.research.google.com/drive/1JlHKfdff_yjCO8yvxiKV9xoRAiEPgarM#scrollTo=OIKmNGkssXDZ&line=2&uniqifier=1>`__

    .. code-block:: python

        >>> from loanpy.scminer import uralign
        >>> print(uralign("a b c", "f g h i j k").replace(" ", "\t"))
        #a	b	c#	-#
        f	g	h	ijk

    """

    left, right = left.split(" "), right.split(" ")
    # tag word initial & final cluster, only in left
    left[0], left[-1] = "#" + left[0], left[-1] + "#"

    # go sequentially and squeeze the leftover together to one suffix
    # e.g. "a,b","c,d,e,f,g->"a,b,-#","c,d,efg
    diff = abs(len(right) - len(left))
    if len(left) < len(right):
        left.append("-#")
        right = right[:-diff] + ["".join(right[-diff:])]
    elif len(left) > len(right):
        left = left[:-diff] + ["+"] + ["".join(left[-diff:])]
    else:
        left, right = left + ["-#"], right + ["-"]

    return f'{" ".join(left)}\n{" ".join(right)}'

def get_heur(tgtlg: str) -> Dict[str, List[str]]:
    """
    Rank the phonemes of a target langauge's phoneme inventory according
    to feature vector similarity to all IPA sounds.
    Relies on ``./cldf/.transcription-report.json`` which contains the
    phoneme inventory. The file ``ipa_all.csv`` contains all IPA sounds
    and their feature vectors and is shipped together with loanpy.

    :param tgtlg: The ID of the target language, as defined in
                  ``etc/languages.tsv`` in a CLDF repository.
    :type tgtlg: str
    :returns: A dictionary with IPA phonemes as keys and a list of
              closest target language phonemes as values.
    :rtype: dict

    :raises FileNotFoundError: If the data file or the transcription
                               report file is not found.

    `Run in Google Colab >> <https://colab.research.google.com/drive/1JlHKfdff_yjCO8yvxiKV9xoRAiEPgarM#scrollTo=jfYaOtaAsfxz&line=3&uniqifier=1>`__

    .. code-block:: python

        >>> from loanpy.scminer import get_heur
        >>> get_heur("eng")
        {'˩': ['a', 'b'],
         '˨': ['a', 'b'],
         '˧': ['a', 'b'],
         '˦': ['a', 'b'],
         '˥': ['a', 'b'],
         ...

    """
    ipa_all = read_ipa_all()
    vectors = {row[0]: [int(i) for i in row[1:]] for row in ipa_all[1:]}

    report_path = Path.cwd() / 'cldf/.transcription-report.json'
    with report_path.open("r") as f:
        phoneme_inventory = json.load(f)["by_language"][tgtlg]["segments"]

    # sort phoneme_inventory phonemes by euclidean distance to every ipa sound
    heur = {}
    for row in ipa_all[1:]:
        distances = []
        for phoneme in phoneme_inventory:
            try:  # Bugfix for Bislama diphthong "ae": not in ipa_all!
                v1, v2 = vectors[phoneme], vectors[row[0]]
            except KeyError:
                continue
            # measure euclidean distance between feature vectors
            distances.append(
            math.sqrt(sum((v1[i] - v2[i]) ** 2 for i in range(len(v1))))
            )
        dist_and_phon = sorted(zip(distances, phoneme_inventory))
        heur[row[0]] = [i[1] for i in dist_and_phon]

    return heur


def get_prosodic_inventory(table: List[List[str]]) -> List[str]:
    """
    Extracts all types of prosodic structures (e.g. "CVCV")
    from rows with an uneven ID (i.e. where data of target language is
    located) of the given table.

    :param table: A table where every row is a list.
    :type table: list of lists

    :return: A list of prosodic structures (e.g. "CVCV") that occur in the
             target languages (i.e. in the uneven rows)
    :rtype: list

    `Run in Google Colab >> <https://colab.research.google.com/drive/1JlHKfdff_yjCO8yvxiKV9xoRAiEPgarM#scrollTo=MaLK4PJIG5VN&line=2&uniqifier=1>`__

    .. code-block:: python

        >>> from loanpy.scminer import get_prosodic_inventory
        >>>
        >>> data = [  ['ID', 'COGID', 'DOCULECT', 'ALIGNMENT', 'PROSODY'],
        ...   [0, 1, 'H', '#aː t͡ʃ# -#', 'VC'],
        ...   [1, 1, 'EAH', 'a.ɣ.a t͡ʃ i', 'VCVCV'],
        ...   [2, 2, 'H', '#aː ɟ uː#', 'VCV'],
        ...   [3, 2, 'EAH', 'a l.d a.ɣ', 'VCCVC'],
        ...   [4, 3, 'H', '#ɒ j n', 'VCC'],
        ...   [5, 3, 'EAH', 'a j.a n', 'VCVC']
        ... ]
        >>> get_prosodic_inventory(data)
        ['VCVCV', 'VCCVC', 'VCVC']

    """
    headers = table.pop(0)
    h = {i: headers.index(i) for i in headers}
    out = list(set([row[h["PROSODY"]] for row in table[1::2]]))
    table.insert(0, headers)
    return out
