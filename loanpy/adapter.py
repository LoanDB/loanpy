"""
Adapt words of L2 so that they do not violate constraints of L1.
"""

import ast
import os
from collections import defaultdict
from itertools import count, product
import math
import string

from Levenshtein import apply_edit, editops
from ipatok import tokenise
import pandas as pd

from loanpy.helpers import (adaptharmony, editdistancewith2ops,
                            ipa2clusters, word2struc)

os.chdir(os.path.join(os.path.dirname(__file__), "data"))

substidict = {}
allowedphonotactics = []
scvalues = set()


def launch(substicsv="substi.csv", dfetymology="dfuralonet.csv",
           timelayer="", soundchangedict="scdict.txt"):
    """
    Define global variables necessary for loanpy.loanfinder.adapt

    :param substicsv:
            The name of the csv-file that contains the sound substitutions.
    :type substicsv: str, default="substi.csv"

    :param dfetymology:
            The name of the etymological dictionary of L1 stored as a csv. \
Needed only if parameter struc=True. If this is not needed set \
dfetymology="".
    :type dfetymology: str, default="dfuralonet.csv"

    :param timelayer:
            Indicate from which historical layer sound changes \
should be extracted, Uralic ("U"), Finno-Ugric ("FU"), or Ugric ("Ug"). \
By default all three are taken into consideration simultaneously.
    :type timelayer:  {"", "U", "FU", "Ug"}, default=""

    :param soundchangedict:
            The dictionary of sound changes of L1, extracted from the \
etymological dictionary by loanpy.reconstructor.dfetymology2dict. Needed \
if parameter only_documented_clusters=True. If this is not needed set \
soundchangedict="".
    :type soundchangedict:  str, "scdict.txt"

    :return: None, but defines global variables, explained below
    :rtype: NoneType

    :var substidict: dictionary based on substi.csv
    :vartype substidict: global, dictionary, Keys: str, values: list of str
    :var allowedphonotactics: list of allowed phonotactic structures, \
based on a set, so its elements are unordered.
    :vartype allowedphonotactics: global, list of str
    :var scvalues: set of IPA-strings that represent consonant and vowel \
clusters that are documented in proto-L1
    :vartype scvalues: global, set of str

    :Example:

    >>> from loanpy import adapter as ad
    >>> ad.launch()
    >>> ad.substidict
    {'a': ['ͽ', 'ȣ̈', 'ɑ', 'æ'],
     'b': ['p', 'w'],
     'd': ['ð', 't'],
     ...}

    >>> from loanpy import adapter as ad
    >>> ad.launch()
    >>> ad.allowedphonotactics
    ['CVCCV',
     'CV',
     'VCVCV',
     'CVCCVCCV',
     'CVCVCCV',
     'CVCV',
     'VCCCV',
     'V',
     'CVCVCV',
     'VCCVCV',
     'VCV',
     'CVCCCV',
     'VCCVCVCV',
     'CVCVCC',
     'VCCV',
     'CVCCVCV']
    ad.allowedphonotactics is unordered b/c it is based on a set

    >>> from loanpy import adapter as ad
    >>> ad.launch()
    >>> ad.scvalues
    {'0',
     'cx',
     'e',
     ...
     }
     note that scvalues is a set, so its elements are unordered
    """

    dfsubsti = pd.read_csv(substicsv, encoding="utf-8")
    global substidict
    substidict = dict(zip(dfsubsti["L2_phons"],
                          dfsubsti["L1_substi"].str.split(", ")))

    if dfetymology != "":
        dfety = pd.read_csv(dfetymology,
                            encoding="utf-8", usecols=["Old", "Lan"])
        if timelayer != "":
            dfety = dfety[dfety.Lan == timelayer].reset_index(drop=True)
        global allowedphonotactics
        allowedphonotactics = list(set([word2struc(i) for i in dfety["Old"]]))

    if soundchangedict != "":
        with open(soundchangedict, "r", encoding="utf-8") as f:
            soundchangedict = ast.literal_eval(f.read())
            global scvalues
            scvalues = set([item for sublist in soundchangedict.values()
                            for item in sublist])


def adapt(ipastring, howmany=1, struc=True, vowelharmony=True,
          only_documented_clusters=True):
    """
    Takes an L2 word and adapts it to the constraints of L1

    :param ipastring:
        An L2 word consisting of characters of the \
International Phonetic Alphabet (IPA).
    :type ipastring: str

    :param howmany: Indicate how many adaptions to return.
    :type howmany: int, default=float("inf")

    :param struc: Indicate whether words should be adapted to \
phonotactic constraints.
    :type struc: bool, default=True

    :param vowelharmony: Indicate whether words should be adapted \
to vowelharmonic constraints
    :type vowelharmony: bool, default=True

    :param only_documented_clusters: Indicate whether only those words should \
be accepted that consist exclusively of clusters that are documented in \
the etymological dictionary of L1.
    :type only_documented_clusters: bool, default=True

    :return: The most likely adaptations into proto-L1
    :rtype: str

    :Example:

    >>> from loanpy import adapter as ad
    >>> ad.launch()
    >>> ad.adapt("wulɸɪla", howmany=float("inf"), struc=False,
                  only_documented_clusters=False, vowelharmony=False)
    'wͽlwͽlͽ, wͽlwͽlȣ̈, wͽlwͽlɑ, wͽlwͽlæ, wͽlwȣ̈lͽ, wͽlwȣ̈lȣ̈, wͽlwȣ̈lɑ, wͽlwȣ̈læ, wͽlwilͽ, wͽlwilȣ̈, wͽlwilɑ, wͽlwilæ, wͽlwelͽ, wͽlwelȣ̈, wͽlwelɑ, wͽlwelæ, wͽlwjlͽ, wͽlwjlȣ̈, wͽlwjlɑ, wͽlwjlæ, wȣlwͽlͽ, wȣlwͽlȣ̈, wȣlwͽlɑ, wȣlwͽlæ, wȣlwȣ̈lͽ, wȣlwȣ̈lȣ̈, wȣlwȣ̈lɑ, wȣlwȣ̈læ, wȣlwilͽ, wȣlwilȣ̈, wȣlwilɑ, wȣlwilæ, wȣlwelͽ, wȣlwelȣ̈, wȣlwelɑ, wȣlwelæ, wȣlwjlͽ, wȣlwjlȣ̈, wȣlwjlɑ, wȣlwjlæ, wulwͽlͽ, wulwͽlȣ̈, wulwͽlɑ, wulwͽlæ, wulwȣ̈lͽ, wulwȣ̈lȣ̈, wulwȣ̈lɑ, wulwȣ̈læ, wulwilͽ, wulwilȣ̈, wulwilɑ, wulwilæ, wulwelͽ, wulwelȣ̈, wulwelɑ, wulwelæ, wulwjlͽ, wulwjlȣ̈, wulwjlɑ, wulwjlæ'

    >>> from loanpy import adapter as ad
    >>> ad.launch()
    >>> ad.allowedphonotactics = ["CVCVCVCV","CVCV"]
    >>> ad.adapt("wulɸɪla", howmany=2, struc=True,
                  vowelharmony=False, only_documented_clusters=False)
    'wͽlVwͽlͽ, wȣlVwͽlͽ

    >>> from loanpy import adapter as ad
    >>> ad.launch()
    >>> ad.adapt("wulɸɪla", howmany=5, struc=False,
                  vowelharmony=True, only_documented_clusters=False)
    'wͽlwͽlͽ, wͽlwͽlȣ̈, wͽlwȣ̈lͽ, wͽlwȣ̈lȣ̈, welwͽlͽ, welwͽlȣ̈, welwȣ̈lͽ, welwȣ̈lȣ̈'

    >>> from loanpy import adapter as ad
    >>> ad.launch()
    >>> ad.adapt("wulɸɪla", howmany=5, struc=False,
                  vowelharmony=True, only_documented_clusters=True)
    'wͽlwͽlͽ, wͽlwͽlȣ̈, wͽlwȣ̈lͽ, wͽlwȣ̈lȣ̈, welwͽlͽ, welwͽlȣ̈, welwȣ̈lͽ, welwȣ̈lȣ̈'

    """

    ipalist = tokenise(ipastring)

    if struc is True:
        L2struc = word2struc(ipastring)
        adaptedstruc = sorted(list(zip([editdistancewith2ops(L2struc, i)
                                        for i in allowedphonotactics],
                                       allowedphonotactics)))[0][1]
        letters = string.ascii_letters[:len(ipalist)]  # 1 char = 1 ipa char!
        adhocdict = dict(zip(letters, ipalist))
        ipastring = apply_edit(editops(L2struc, adaptedstruc),
                               letters, adaptedstruc)
        ipalist = [adhocdict[i] if i in adhocdict else i for i in ipastring]

    if vowelharmony is True:
        ipalist = adaptharmony(ipalist)

    if not all(phon in substidict for phon in ipalist):
        return (", ".join([i for i in ipalist if i not in substidict]) +
                " not in substi.csv")

    idxdict = defaultdict(count(1).__next__)
    idxlist = [idxdict[cluster]-1 for cluster in ipalist]
    ipalist = list(dict.fromkeys(ipalist))

    substi = [substidict[i] for i in ipalist if i in substidict]

    if howmany >= math.prod([len(i) for i in substi]):
        args = substi
    else:
        args = [[i[0]] for i in substi]
        whichsound, whichsubsti = 0, 1
        while math.prod([len(i) for i in args]) < howmany:

            if whichsubsti < len(substi[whichsound]):
                args[whichsound].append(substi[whichsound][whichsubsti])

            if whichsound == len(substi)-1:
                whichsubsti += 1
                whichsound = 0
            else:
                whichsound += 1

    out = ["".join([subst[i] for i in idxlist]) for subst in product(*args)]

    if only_documented_clusters is True:
        out = [i for i in out if all(j in scvalues for j in ipa2clusters(i))]
        if out == []:
            return "every substituted word contains \
at least one cluster undocumented in proto-L1"

    return ", ".join(out)
