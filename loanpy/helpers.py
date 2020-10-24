"""
Functions that are called by other modules.
Global variables cns, vow, front and back are based on
panphon (https://github.com/dmort27/panphon, last access 08.apr.2021): \
David R. Mortensen, Patrick Littell, Akash Bharadwaj, Kartik Goyal, \
Chris Dyer, Lori Levin (2016). "PanPhon: A Resource for Mapping IPA \
Segments to Articulatory Feature Vectors." Proceedings of COLING 2016, \
the 26th International Conference on Computational Linguistics: \
Technical Papers, pages 3475–3484, Osaka, Japan, December 11-17 2016.)

"""


import datetime
from itertools import product
import os
import re
import sys

import gensim.downloader as api
from ipatok import tokenise
import pandas as pd


cns = 'jwʘǀǃǂǁk͡pɡ͡bcɡkqɖɟɠɢʄʈʛbb͡ddd̪pp͡ttt̪ɓɗb͡βk͡xp͡ɸq͡χɡ͡ɣɢ͡ʁc͡çd͡ʒt͡ʃɖ͡ʐɟ͡ʝʈ͡ʂb͡vd̪͡z̪d̪͡ðd̪͡ɮ̪d͡z\
d͡ɮd͡ʑp͡ft̪͡s̪t̪͡ɬ̪t̪͡θt͡st͡ɕt͡ɬxçħɣʁʂʃʐʒʕʝχfss̪vzz̪ðɸβθɧɕɬɬ̪ɮʑɱŋɳɴmnn̪ɲʀʙʟɭɽʎrr̪ɫɺɾhll̪ɦðʲt͡ʃʲnʲ\
ʃʲlʲCl̥m̥n̥r̥hʷkʷjːwːʘːǀːǃːǂːǁːk͡pːɡ͡bːcːɡːkːqːɖːɟːɠːɢːʄːʈːʛːbːb͡dːdːd̪ːpːp͡tːtːt̪ːɓːɗːb͡βːk͡xː\
p͡ɸːq͡χːɡ͡ɣːɢ͡ʁːc͡çːd͡ʒːt͡ʃːɖ͡ʐːɟ͡ʝːʈ͡ʂːb͡vːd̪͡z̪ːd̪͡ðːd̪͡ɮ̪ːd͡zːd͡ɮːd͡ʑːp͡fːt̪͡s̪ːt̪͡ɬ̪ːt̪͡θːt͡sːt͡ɕːt͡ɬːxːçːħː\
ɣːʁːʂːʃːʐːʒːʕːʝːχːfːsːs̪ːvːzːz̪ːðːɸːβːθːɧːɕːɬːɬ̪ːɮːʑːɱːŋːɳːɴːmːnːn̪ːɲːʀːʙːʟːɭːɽː\
ʎːrːr̪ːɫːɺːɾːhːlːl̪ːɦːðʲːt͡ʃʲːnʲːʃʲːlʲːCːl̥ːm̥ːn̥ːr̥ːhʷːkʷː'
vow = 'ɑɘɞɤɵʉaeiouyæøœɒɔəɘɵɞɜɛɨɪɯɶʊɐʌʏʔɥɰʋʍɹɻ¨ȣȣ̈ɑːɘːɞːɤːɵːʉːaːeːiːoːuːyːæːøːœː\
ɒːɔːəːɘːɵːɞːɜːɛːɨːɪːɯːɶːʊːɐːʌːʏːʔːɥːɰːʋːʍːɹːɻːȣ̈ːFBVͽ'
front = 'jcɖɟʄʈbb͡ddd̪pp͡ttt̪ɓɗc͡çd͡ʒt͡ʃɟ͡ʝb͡vd͡zd͡ʑp͡ft͡st͡ɕçʂʃʐʒʝfss̪vzz̪ðɸβθɕʑɳmnn̪ɲʎhll̪ɦɘɞ\
ɵʉaeiyæøœɛɪɶʏʔɥɻ¨ȣ̈jːcːɖːɟːʄːʈːbːb͡dːdːd̪ːpːp͡tːtːt̪ːɓːɗːc͡çːd͡ʒːt͡ʃːɟ͡ʝːb͡vːd͡zːd͡ʑːp͡fːt͡sː\
t͡ɕːçːʂːʃːʐːʒːʝːfːsːs̪ːvːzːz̪ːðːɸːβːθːɕːʑːɳːmːnːn̪ːɲːʎːhːlːl̪ːɦːɘːɞːɵːʉːaːeːiːyːæːøː\
œːɛːɪːɶːʏːʔːɥːɻːȣ̈ːF'
back = 'wɡkqɠɢʛq͡χɢ͡ʁxħɣʁʕχŋɴʀɫɑɤouɒɔəɘɵɞɜɨɯʊɐʌʍɹȣwːɡːkːqːɠːɢːʛːq͡χːɢ͡ʁːxːħːɣːʁːʕːχː\
ŋːɴːʀːɫːɑːɤːoːuːɒːɔːəːɘːɵːɞːɜːɨːɯːʊːɐːʌːʍːɹːB'

cnsvow_regex = "[^"+cns+vow+"]"
model = None

os.chdir(os.path.join(os.path.dirname(__file__), "data"))


def phon2cv(phon):
    """
    Turns a phoneme into "C" for consonant or "V" for vowel. \
    Returns None if it's neither of both.

    :Example:

    >>> from loanpy.helpers import phon2cv
    >>> phon2cv("p")
    "C"
    >>> phon2cv("a")
    "V"

    """
    return "C" if phon in cns else "V" if phon in vow else None


def phon2cv_generator(ipalist):
    """generator function called by word2struc"""
    for i in ipalist:
        yield phon2cv(i)


def word2struc(ipastring):
    """
    Turns a word in IPA into its phonotactic profile.

    :Example:

    >>> from loanpy.helpers import word2struc
    >>> word2struc("mɒɟɒr")
    'CVCVC'

    """
    return "".join(list(phon2cv_generator(tokenise(ipastring))))


def merge(ipalist):
    """
    generator function called by ipa2clusters,
    merges subsequent consonants and vowels to clusters

    """
    tmp = []
    it = iter(ipalist)
    nextit = next(it)
    for phon in ipalist:
        phoncv = phon2cv(phon)
        while phon2cv(nextit) != phoncv:
            yield ''.join(tmp)
            tmp = []
            nextit = next(it)
        tmp.append(phon)
    yield ''.join(tmp)


def ipa2clusters(ipstring):
    """
    Turns a word into a list of consonant and vowel clusters

    :Example:

    >>> from loanpy.helpers import ipa2clusters
    >>> ipa2clusters("roflmao")
    ['r', 'o', 'flm', 'ao']

    """
    return [i for i in merge(tokenise(ipstring)) if i]


def list2regex(sclist):
    """
    Turns a list of phonemes into a regular expression

    :param sclist: a list of phonemes
    :type sclist: list of str

    :return: the same phonemes as a regular expression. Zeros are \
removed and replaced with a question mark at the end.
    :rtype: str

    :Example:

    >>> from loanpy.helpers import list2regex
    >>> list2regex(["b", "k", "v"])
    "(b|k|v)"

    >>> from loanpy.helpers import list2regex
    >>> list2regex(["b", "k", "0", "v"])
    "(b|k|v)?"

    """

    if sclist == ["0"]:
        return ""
    if "0" in sclist:
        sclist = [i for i in sclist if i != "0"]
        return "("+"|".join(sclist) + ")?"

    return "("+"|".join(sclist) + ")"


def editdistancewith2ops(string1, string2):
    """
    Takes two strings and calculates their similarity by \
only allowing two operations: insertion and deletion. \
In line with the "Threshold Principle" by Paradis and LaCharité (1997) \
the distance is weighted in a way that two insertions are cheaper than \
one deletion: "The problem is really not very different from the dilemma \
of a landlord stuck with a limited budget for maintenance and a building \
which no longer meets municipal guidelines. Beyond a certain point, \
renovating is not viable (there are too many steps to be taken) and \
demolition is in order. Similarly, we posit that I) languages have \
a limited budget for adapting ill- formed phonological structures, \
and that 2) the limit for the budget is universally set at two steps, \
beyond which a repair by 'demolition' may apply. In other words, we \
predict that a segment is deleted if (but only if) its rescue is too \
costly in terms of the Threshold Principle" (Preservation and Minimality \
in Loanword Adaptation, \
Author(s): Carole Paradis and Darlene Lacharité, \
Source: Journal of Linguistics , Sep., 1997, Vol. 33, No. 2 (Sep., 1997), \
pp. 379-430, \
Published by: Cambridge University Press, \
Stable URL: http://www.jstor.com/stable/4176422). \
The code is based on a post by ita_c on geeksforgeeks: \
https://www.geeksforgeeks.org/edit-distance-and-lcs-longest-common-subsequence\
 (last access: 11.feb.2021)

    :return: The distance between two input strings
    :rtype: int

    :Example:

    >>> from loanpy.helpers import editdistancewith2ops
    >>> editdistancewith2ops("hey","hey")
    0.0

    >>> from loanpy.helpers import editdistancewith2ops
    >>> editdistancewith2ops("hey","he")
    1.0

    >>> from loanpy.helpers import editdistancewith2ops
    >>> editdistancewith2ops("hey","heyy")
    0.4

    """

    m = len(string1)     # Find longest common subsequence (LCS)
    n = len(string2)
    L = [[0 for x in range(n + 1)]
         for y in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if (i == 0 or j == 0):
                L[i][j] = 0
            elif (string1[i - 1] == string2[j - 1]):
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j],
                              L[i][j - 1])

    lcs = L[m][n]
    # Edit distance is delete operations + insert operations*0.4.
    return (m - lcs) + (n - lcs)*0.4


def gensim_similarity(L1_en, L2_en, return_wordpair=False):
    """
    Pin down the most similar two words from two wordlists, based on gensim: \
@inproceedings{rehurek_lrec, \
title \= \{\{Software Framework for Topic Modelling with Large Corpora\}\}, \
author \= \{Radim \{\\v R\}eh\{\\r u\}\{\\v r\}ek and Petr Sojka\}, \
booktitle \= \{\{Proceedings of the LREC 2010 Workshop on New \
    Challenges for NLP Frameworks\}\}, \
pages \= \{45\-\-50\}, \
year \= 2010, \
month \= May, \
day \= 22, \
publisher \= \{ELRA\}, \
address = \{Valletta, Malta\}, \
note\=\{\\url\{http\:\/\/is\.muni\.cz\/publication\/884893\/en\}\}, \
language\=\{English\} \
\}.

    :param L1_en: the English translations of the L1 word
    :type L1_en: str, words are separated by ", "

    :param L2_en: the English translations of the L2 word
    :type L2_en: str, words are separated by ", "

    :param return_wordpair: Indicate whether the word pair itself should be \
returned too.
    :type return_wordpair: bool, default=False

    :return: The most similar word pair of the two lists
    :rtype: int or tuple of (int and tuple of two str)

    :Example:

    >>> from loanpy.helpers import loadvectors, gensim_similarity
    >>> loadvectors()
    >>> gensim_similarity("house, sing, hello", "cottage, regrettable, car",
                           return_wordpair=False)
    0.54949486

    >>> from loanpy.helpers import loadvectors, gensim_similarity
    >>> loadvectors()
    >>> gensim_similarity("house, sing, hello", "cottage, regrettable, car",
                           return_wordpair=True)
    (0.54949486, ('house', 'cottage'))

    """

    # missing translations = empty cells = nans = floats
    if isinstance(L1_en, float) or isinstance(L2_en, float):
        return -1
    else:
        L1_en, L2_en = L1_en.split(', '), L2_en.split(', ')

    topsim = - 1  # score of the most similar word pair
    wordpair = ()
    for L1 in L1_en:
        for L2 in L2_en:  # calculate semantic similarity of all pairs0
            try:
                modsim = model.similarity(L1, L2)
            except KeyError:  # if word not in KeyedVecors of gensim continue
                continue
            if modsim > topsim:  # replace if more similar than topsim
                topsim = modsim
                wordpair = (L1, L2)
    return (topsim, wordpair) if return_wordpair else topsim


def loadvectors(wordvectors="glove-wiki-gigaword-50"):
    """
    Load pretrained vectors for calculating semantic similarities. \
Find a table of names of pretrained models at \
https://github.com/RaRe-Technologies/gensim-data (last access 09.apr 2021) \
or find it in data/wordvectornames.xlsx.

    :param wordvectors: Name of the vector model to load. \
A list of possible names is found in the column \
"name" in data/wordvectornames.xlsx or at \
https://github.com/RaRe-Technologies/gensim-data (last access 09.apr 2021).
    :type wordvectors: str, default="glove-wiki-gigaword-50"

    :return: None, but vectors are assigned to global variable <module>
    :rtype: NoneType

    .. note: Because it can take some time to load vectors, \
they are only loaded once and assigned \
to the global variable <model>. To reload vectors, set helpers.module=None, \
as shown below.

    :Example:

    >>> # import default vectors
    >>> from loanpy.helpers import loadvectors
    >>> loadvectors()
    None

    >>> # now load some other vectors
    >>> from loanpy import helpers
    >>> helpers.model=None
    >>> helpers.loadvectors("glove-twitter-25")
    None

    """
    global model  # takes a while to load, so will be loaded only once
    if model is None:  # get the vector model
        sys.stdout.write(datetime.datetime.now().
                         strftime("%H:%M: Loading vectors, \
might take a bit...\n"))
        sys.stdout.flush()
        model = api.load(wordvectors)


def progressbar(it, prefix="", size=60, file=sys.stdout):
    """
    Progress bar from \
https\:\/\/stackoverflow\.com\/questions\/3160699\/python\-progress\-bar \
(last access on 18.mar.2021).
    """
    count = len(it)

    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j,
                                         count))
        file.flush()
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()


def filterdf(df, col, occurs_or_bigger, term, write=False,
             name="dffiltered.csv"):
    """
    Filters a data frame according to parameters

    :param df: A pandas data frame e.g. data\dfhun.csv
    :type df: pandas.core.frame.DataFrame

    :param col:
        Indicate in which column to search for criteria according to \
which rows will be dropped
    :type col: str

    :param occurs_or_bigger:
        Indicate which data to **keep** in the table
    :type occurs_or_bigger: bool, default=False

    :param term:
        Indicate which term should or should not occur or which number \
should or should not be overstepped.
    :type term: {str, int}

    :param write:
        If True, None will be returned and the output will be \
written to a csv-file. If False, the data frame will be returned.
    :type write: bool, default=False

    :param name:
        Indicate how the filtered data frame should be named.
    :type name: str, default="dffiltered.csv"

    :return: A table where rows were dropped according to parameters
    :rtype: {pandas.core.frame.DataFrame, None}

    :Example:

    >>> import pandas as pd
    >>> from loanpy.helpers import filterdf
    >>> dfin=pd.read_csv("dfhun_zaicz_backup.csv", encoding="utf-8")
    >>> dffiltered = filterdf(dfin, "L1_suffix", occurs_or_bigger=False, term="\+")
    >>> filterdf(dffiltered, "L1_year", occurs_or_bigger=False, term=1600, write=True, name="example_dfhun_before1600.csv")
    drops rows where column "L1_year" is more than 1600 or L1_suffix is "+". View result in data/example_dfhun_before1600.csv

    >>> import pandas as pd
    >>> from loanpy.helpers import filterdf
    >>> dfin=pd.read_csv("dfhun_zaicz_backup.csv", encoding="utf-8")
    >>> dffiltered = filterdf(dfin, "L1_suffix", occurs_or_bigger=False, term="\+")
    >>> dffiltered = filterdf(dffiltered, "L1_year", occurs_or_bigger=False, term=1600)
    >>> filterdf(dffiltered, "L1_language", occurs_or_bigger=True, term="unknown", write=True, name="example_dfhun_unknown.csv")
    keeps only words of unknown etymology documented before 1600 that are no suffixes. View result in data/example_dfhun_unknown.csv

    """
    out = None
    if isinstance(term, str):
        df = df.fillna('')
        if occurs_or_bigger is False:
            out = df[~df[col].str.contains(term, na=False)]
        else:
            out = df[df[col].str.contains(term, na=False)]
    elif isinstance(term, (float, int)):
        df = df.fillna(0)
        if occurs_or_bigger is False:
            out = df[df[col] <= term]
        else:
            out = df[df[col] > term]
    if write is True:
        out.to_csv(name, encoding="utf-8", index=False)
    else:
        return out


def vow2frontback(vowel, replace=False):
    """
    Turn a vowel into "F" for front or "B" for back. \
    Returns None if it is neither of both.

    :Example:

    >>> from loanpy.helpers import vow2frontback
    >>> vow2frontback("e")
    "F"
    >>> vow2frontback("o")
    "B"

    """
    if replace is True:
        return "B" if vowel in front else "F" if vowel in back else None
    return "F" if vowel in front else "B" if vowel in back else None


def harmony(ipalist):
    """
    Tells if a word has front-back vowel harmony or not

    :param ipalist: the word that should be analysed
    :type ipalist: {list of str, str}

    :rtype: bool

    :Example:

    >>> from loanpy.helpers import harmony
    >>> harmony("bot͡sibot͡si")
    False

    >>> from loanpy.helpers import harmony
    >>> harmony(["t", "ɒ", "r", "k", "ɒ"])
    True

    >>> from loanpy.helpers import harmony
    >>> harmony("ʃɛfylɛʃɛ")
    True

    """
    if isinstance(ipalist, str):
        ipalist = tokenise(ipalist)
    vowels = [i for i in ipalist if i in vow]
    vowels = set([vow2frontback(i) for i in vowels if i not in "Vͽ"])
    return True if vowels in ({"F"}, {"B"}, set()) else False


def adaptharmony(ipalist):
    """
    Adapts the vowel harmony of a word by replacing wrong vowels by a \
placeholder "F" for any front vowel or "B" for any back vowel.

    :param ipalist: a list or a string of phonemes
    :type ipalist: {list of str, str}

    :rtype: list of str

    :Example:

    >>> from loanpy.helpers import adaptharmony
    >>> adaptharmony('kɛsthɛj')
    ['k', 'ɛ', 's', 't', 'h', 'ɛ', 'j']

    >>> from loanpy.helpers import adaptharmony
    >>> adaptharmony(['ɒ', 'l', 'ʃ', 'oː', 'ø', 'r', 'ʃ'])
    ['ɒ', 'l', 'ʃ', 'oː', 'B', 'r', 'ʃ']

    >>> from loanpy.helpers import adaptharmony
    >>> adaptharmony('ʃioːfok')
    ['ʃ', 'B', 'oː', 'f', 'o', 'k']

    >>> from loanpy.helpers import adaptharmony
    >>> adaptharmony(['b', 'ɒ', 'l', 'ɒ', 't', 'o', 'n', 'k', 'ɛ', 'n', 'ɛ', 'ʃ', 'ɛ'])
    ['b', 'ɒ', 'l', 'ɒ', 't', 'o', 'n', 'k', 'B', 'n', 'B', 'ʃ', 'B']

    """
    if isinstance(ipalist, str):
        ipalist = tokenise(ipalist)

    if harmony(ipalist):
        return ipalist

    vowels = [vow2frontback(i) for i in ipalist if i in vow]
    if vowels.count("F") > vowels.count("B"):
        ipalist = [vow2frontback(i, replace=True) if i in vow and i in back
                   else i.replace("V", "F") for i in ipalist]
    elif vowels.count("B") >= vowels.count("F"):
        ipalist = [vow2frontback(i, replace=True) if i in vow and i in front
                   else i for i in ipalist]
    return ipalist
