"""
Contains helper functions and class Etym, which are called internally.
Some of the functions may also be useful for the user in \
other linguistic contexts.

"""
from collections import Counter
from datetime import datetime
from functools import partial
from itertools import product
from logging import getLogger

from gensim.downloader import load
from lingpy.sequence.sound_classes import token2class
from networkx import DiGraph, all_shortest_paths, shortest_path
from numpy import array_equiv, subtract, zeros
from panphon import FeatureTable
from panphon.distance import Distance
from re import split
from tqdm import tqdm


logger = getLogger(__name__)
model = None
ft = FeatureTable()

def cvgaps(str1, str2):
    """
    Input two aligned strings
    Replace "-" by C or V depending on the other sound
    return the new strings as a list
    """
    new = []
    cnt = 0
    for i, j in zip(str1.split(" "), str2.split(" ")):
        if i == "-":
            if ft.word_fts(j)[0].match({"cons": 1}):
                new.append("C")
            elif ft.word_fts(j)[0].match({"cons": -1}):
                new.append("V")
            else:
                new.append(i)
        else:
            new.append(i)

    return [" ".join(new), str2]

def prefilter(data, srclg, tgtlg):
    """
    Keep only cogsets where source and target language occurs.
    """
    data = [i.split(",") for i in data.split("\n")[1:][:-1]]
    cogids = []

    # take only rows with src/tgtlg
    data = [row for row in data if row[2] in {srclg, tgtlg}]
    # get list of cogids and count how often each one occurs
    cogids = Counter([row[9] for row in data])
    # take only cogsets that have 2 entries
    cogids = [i for i in cogids if cogids[i] == 2]  # allowedlist
    data = [row for row in data if row[9] in cogids]

    evalsrctgt(data, srclg, tgtlg)
    return data

def evalsrctgt(data, srclg, tgtlg):
    """
    assert entire table goes srclg-tgtlg-srclg-tgtlg...
    Doculect (=Language_ID) must be in col 2. Col name doesn't matter.
    """
    itertable = iter(data)
    rownr = 0
    while True:
        try:
            assert next(itertable)[2] == srclg
            rownr += 1
            assert next(itertable)[2] == tgtlg
            rownr += 1
        except StopIteration:
            break
        except AssertionError:
            print("Problem in row ", rownr)
            return False

    return True

def evalsamelen(data, srclg, tgtlg):
    """
    Assert that alignments within a cogset have the same length.
    Alignments must be in col 3, col name doesn't matter
    """
    itertable = iter(data)
    rownr = 0
    while True:
        try:
            first = next(itertable)[3].split(" ")
            second = next(itertable)[3].split(" ")
            rownr += 2
            try:
                assert len(first) == len(second)
            except AssertionError:
                print(rownr, "\n", first, "\n", second)
                return False
        except StopIteration:
            break
    return True

def pros(ipastr):
    ipa = split("[ |.]", ipastr)
    out = ""
    for ph in ipa:
        if ft.word_fts(ph)[0].match({"cons": 1}):
            out += "C"
        elif ft.word_fts(ph)[0].match({"cons": -1}):
            out += "V"
    return out

def get_front_back_vowels(segments):
    """
    Take a list of phonemes and replace front vowels with "F"
    and back vowels with "B"
    y"""
    out = []
    for i in segments:
# https://en.wikipedia.org/wiki/Automated_Similarity_Judgment_Program#ASJPcode
        fb = token2class(i, "asjp")
        # exchange front vowels with "F" and back ones with "B".
        if fb in "ieE":  # front vowels
            out.append("F")
        elif fb in "uo":  # back vowels
            out.append("B")
        else:
            out.append(i)
    # return " ".join(out)
    return out


def has_harmony(segments):
    """if "F" AND "B" are in segments, the word has NO vowel harmony"""
    return not all(i in get_front_back_vowels(segments) for i in "FB")


def repair_harmony(ipalist):
    """
    Called by loanpy.adrc.Adrc.adapt. \
Counts how many front and back vowels there are in a word. If there \
are more back than front vowels, all front vowels will be replaced by a "B", \
if there are more front than back vowels, the back vowels will be replaced by \
"F", and if the word has equally \
many front as back vowels, both options will be returned.

    :param ipalist: a list or a string of phonemes
    :type ipalist: list of str | str

    :returns: a tokenised word with repaired vowel harmony
    :rtype: list of list of str

    :Example:

    >>> from loanpy.helpers import Etym
    >>> hp = Etym()
    >>> hp.repair_harmony('kɛsthɛj')
    [['k', 'ɛ', 's', 't', 'h', 'ɛ', 'j']]
    >>> hp.repair_harmony(['ɒ', 'l', 'ʃ', 'oː', 'ø', 'r', 'ʃ'])
    [['ɒ', 'l', 'ʃ', 'oː', 'B', 'r', 'ʃ']]
    >>> hp.repair_harmony(['b', 'eː', 'l', 'ɒ', 't', 'ɛ', 'l', 'ɛ', 'p'])
    [['b', 'eː', 'l', 'F', 't', 'ɛ', 'l', 'ɛ', 'p']]
    >>> hp.repair_harmony(\
['b', 'ɒ', 'l', 'ɒ', 't', 'o', 'n', 'k', 'ɛ', 'n', 'ɛ', 'ʃ', 'ɛ'])
    [['b', 'F', 'l', 'F', 't', 'F', 'n', 'k', 'ɛ', 'n', 'ɛ', 'ʃ', 'ɛ'], \
['b', 'ɒ', 'l', 'ɒ', 't', 'o', 'n', 'k', 'B', 'n', 'B', 'ʃ', 'B']]

    """
    if isinstance(ipalist, str):
        ipalist = ipalist.split(" ")

    if has_harmony(ipalist) and "V" not in ipalist:
        return [ipalist]

    fb_profile = "".join(get_front_back_vowels(ipalist))
    if fb_profile.count("F") > fb_profile.count("B"):
        ipalist = [
            ["F" if token2class(i, "asjp") in "ou" else i for i in ipalist]]
    elif fb_profile.count("B") > fb_profile.count("F"):
        ipalist = [
            ["B" if token2class(i, "asjp") in "ieE" else i for i in ipalist]]
    else:
        ipalist = [
            ["F" if token2class(i, "asjp") in "ou" else i for i in ipalist],
            ["B" if token2class(i, "asjp") in "ieE" else i for i in ipalist]
        ]

    return ipalist


def plug_in_model(word2vec_model):
    """
    Allows to plug in a pre-trained word2vec model into global variable \
loanpy.helpers.model. \
This is for using vectors that can't be loaded with gensim's \
API. Vectors could be plugged in without this function as well, but this way \
debugging is easier. For more information see gensim's documentation, e.g. \
call help(gensim.downloader.load)

    :param word2vec_model: The word2vec model to use
    :type word2vec_model: None | gensim.models.keyedvectors.KeyedVectors

    :returns: global variable <model> gets defined
    :rtype: None (global model = word2vec_model)

    :Example:

    >>> from loanpy import helpers as hp
    >>> from gensim.test.utils import common_texts
    >>> from gensim.models import word2vec
    >>> # no internet needed: load dummy vectors from gensim's test kit folder.
    >>> hp.plug_in_model(word2vec.Word2Vec(common_texts, min_count=1).wv)
    >>> # contains only vectors for "human", "computer", "interface"
    >>> hp.model
    <gensim.models.keyedvectors.KeyedVectors object at 0x7f85fe36d9d0>

    >>> from loanpy import helpers as hp
    >>> from gensim.downloader import load
    >>> # stable internet connection needed to load the following:
    >>> hp.plug_in_model(load("glove-twitter-25"))
    >>> hp.model  # should take only few seconds to load
    <gensim.models.keyedvectors.KeyedVectors object at 0x7ff728663880>

    For more information see gensim's documentation:

    >>> from gensim.download import load
    >>> help(load)

    """

    global model
    model = word2vec_model


def gensim_multiword(recip_transl, donor_transl, return_wordpair=False,
                     wordvectors="word2vec-google-news-300"):
    """
    Called by loanpy.loafinder.Search.loans. Takes two strings as input \
that represent the meanings of a word. Within the strings, \
meanings are separated \
by ", ". It calculates the cosine similarities of the word pairs of the \
Cartesian product and returns the value of the most similar pair. If \
return_wordpair, the words are returned as well. \
If global variable <loanpy.helpers.model> is None, the model provided \
in param <wordvectors> will be loaded and passed on to \
<loanpy.helpers.model>. To reload a model, call loanpy.helpers.plug_in_model.

    :param recip_transl: translations of the recipient word
    :type recip_transl: str, words are separated by ", "

    :param donor_transl: translations of the donor word
    :type donor_transl: str, words are separated by ", "

    :param return_wordpair: Indicate whether the word pair itself should be \
returned too.
    :type return_wordpair: bool, default=False

    :param wordvectors: The name of the pre-trained wordvector model to use. \
For more information see gensim's documentation at \
https://radimrehurek.com/gensim/downloader.html
    :type wordvectors: 'fasttext-wiki-news-subwords-300' | \
'conceptnet-numberbatch-17-06-300' | 'word2vec-ruscorpora-300' | \
'word2vec-google-news-300' | 'glove-wiki-gigaword-50' | \
'glove-wiki-gigaword-100' | 'glove-wiki-gigaword-200' | \
'glove-wiki-gigaword-300' | 'glove-twitter-25' | 'glove-twitter-50' | \
'glove-twitter-100' | 'glove-twitter-200' | \
'__testing_word2vec-matrix-synopsis', default='word2vec-google-news-300'


    :returns: The similarity score of the most similar word pair plus \
the word pair itself if return_wordpair was True.
    :rtype: int or (int, str, str)

    :Example:

    >>> from loanpy.helpers import gensim_multiword
    >>> sense1, sense2 = "hovercraft, full, eels", "nipples, explode, delight"
    >>> gensim_multiword(sense1, sense2, return_wordpair=False)
    0.21005636

    >>> from loanpy.helpers import gensim_multiword
    >>> sense1, sense2 = "drop, panties, William", "cannot, wait, lunchtime"
    >>> gensim_multiword(sense1, sense2, return_wordpair=False)
    (0.18870175, 'drop', 'wait')

    For more infos about available models/datasets run:

    >>> import gensim.downloader as api
    >>> api.info()  # for general information
    >>> list(api.info()['models'])  # to list all models
    """

    # missing translations = empty cells = nans = floats
    if any(not isinstance(arg, str) for arg in [recip_transl, donor_transl]):
        return (-1, f"!{str(type(recip_transl))}!",
                f"!{str(type(donor_transl))}!") if return_wordpair else -1
    else:
        (recip_transl,
         donor_transl) = recip_transl.split(', '), donor_transl.split(', ')

    global model
    if model is None:
        logger.warning(f"\r{datetime.now().strftime('%H:%M')} loading vectors")
        try:
            model = load(wordvectors)
        except MemoryError:
            logger.warning("Memory Error. Close background processes \
and use RAM-saving browser")
        logger.warning(f"\r{datetime.now().strftime('%H:%M')} Vectors loaded")

    topsim, rtr, dtr = -1, "", ""  # lowest possible similarity score

    # throw out words that are not in the model
    recip_transl = [tr for tr in recip_transl if model.has_index_for(tr)]
    donor_transl = [tr for tr in donor_transl if model.has_index_for(tr)]
    if recip_transl == []:
        rtr = "target word not in model"
    if donor_transl == []:
        dtr = "source word not in model"
    if recip_transl == [] or donor_transl == []:
        return (-1, rtr, dtr) if return_wordpair else -1

    for rectr in recip_transl:
        for dontr in donor_transl:  # calculate semantic similarity 4 all pairs
            try:
                modsim = model.similarity(rectr, dontr)
            # if word not in KeyedVecors of gensim continue
            except KeyError:
                continue
            if modsim == 1:
                return (1, rectr, dontr) if return_wordpair else 1
            # replace if more similar than topsim
            if modsim > topsim:
                topsim, rtr, dtr = modsim, rectr, dontr
    return (topsim, rtr, dtr) if return_wordpair else topsim


def edit_distance_with2ops(string1, string2, w_del=100, w_ins=49):
    """
    Called by loanpy.helpers.Etym.rank_closest_phonotactics and \
loanpy.qfysc.Qfy.get_phonotactics_corresp. \
Takes two strings and calculates their similarity by \
only allowing two operations: insertion and deletion. \
In line with the "Threshold Principle" by Carole Paradis and \
Darlene LaCharité (1997) \
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
costly in terms of the Threshold Principle" (p.385, Preservation \
and Minimality \
in Loanword Adaptation, \
Author(s): Carole Paradis and Darlene Lacharité, \
Source: Journal of Linguistics , Sep., 1997, Vol. 33, No. 2 (Sep., 1997), \
pp. 379-430, \
Published by: Cambridge University Press, \
Stable URL: http://www.jstor.com/stable/4176422). \
The code is based on a post by ita_c on \
https://www.geeksforgeeks.org/edit-distance-and-lcs-longest-common-subsequence\
 (last access: June 8th, 2022)

    :param string1: The first of two strings to be compared to each other
    :type string1: str

    :param string2: The second of two strings to be compared to each other
    :type string2: str

    :param w_del: weight (cost) for deleting a phoneme. Default should \
always stay 100, since only relative costs between inserting and deleting \
count.
    :type w_del: int | float, default=100

    :param w_ins: weight (cost) for inserting a phoneme. Default 49 \
is in accordance with the "Threshold Principle": \
2 insertions (2*49=98) are cheaper than a deletion \
(100).
    :type w_ins: int | float, default=49.

    :returns: The distance between two input strings
    :rtype: int | float

    :Example:

    >>> from loanpy.helpers import edit_distance_with2ops
    >>> edit_distance_with2ops("hey","hey")
    0

    >>> from loanpy.helpers import edit_distance_with2ops
    >>> edit_distance_with2ops("hey","he")
    100

    >>> from loanpy.helpers import edit_distance_with2ops
    >>> edit_distance_with2ops("hey","heyy")
    49

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
    # Edit distance is delete operations + insert operations*0.49.
    # costs (=distance) are lower for insertions
    return (m - lcs) * w_del + (n - lcs) * w_ins


def get_mtx(target, source):
    """
    Called by loanpy.helpers.mtx2graph. Similar to \
loanpy.helpers.edit_distance_with2ops but without \
weights (i.e. deletion and insertion \
both always cost one) and the matrix is returned.

    From https://www.youtube.com/watch?v=AY2DZ4a9gyk. \
(Last access: June 8th, 2022) \
Draws a matrix of minimum edit distances between every substring of two \
input strings. The ~secret~ to fill the matrix: \
If two letters are not the same, look at the \
upper and left hand cell, pick the minimum and add one. If they are the same, \
pick the value from the upper left diagonal cell.

    :param target: The target word
    :type target: iterable, e.g. str or list

    :param source: The source word
    :type source: iterable, e.g. str or list

    :returns: A matrix where every cell tells the cost of turning one \
substring to the other (only delete and insert with cost 1 for both)
    :rtype: numpy.ndarray

    :Example:

    >>> from loanpy.helpers import get_mtx
    >>> get_mtx("bcde", "de")
    array([[0., 1., 2., 3., 4.],
       [1., 2., 3., 2., 3.],
       [2., 3., 4., 3., 2.]])
    >>>  # What in reality happened (example from video):
         # deletion costs 1, insertion costs 1, so the distances are:
      # B C D E  # hashtag stands for empty string
    # 0 1 2 3 4  # distance B-#=1, BC-#=2, BCD-#=3, BCDE-#=4
    D 1 2 3 2 3  # distance D-#=1, D-B=2, D-BC=3, D-BCD=2, D-BCDE=3
    E 2 3 4 3 2  # distance DE-#=2, DE-B=3, DE-BC=4, DE-BCD=3, DE-BCDE=2
    # the min. edit distance from BCDE-DE=2: delete B, delete C

    """
    #  build matrix of correct size
    target = ['#'] + [k for k in target]  # add hashtag as starting value
    source = ['#'] + [k for k in source]  # starting value is always zero
    # matrix consists of zeros at first. sol stands for solution.
    sol = zeros((len(source), len(target)))
    # first row of matrix is 1,2,3,4,5,... as long as the target word is
    sol[0] = [j for j in range(len(target))]
    # first column is also 1,2,3,4,5....  as long as the sourcre word is
    sol[:, 0] = [j for j in range(len(source))]
    # Add anchor value
    if target[1] != source[1]:  # if first letters of the 2 words are different
        sol[1, 1] = 2  # set the first value (upper left corner) to 2
    # else it just stays zero

    # loop through the indexes of the two words with a nested loop
    for c in range(1, len(target)):
        for r in range(1, len(source)):
            if target[c] != source[r]:  # when the two letters are different
                # pick minimum of the 2 boxes to the left and above and add 1
                sol[r, c] = min(sol[r - 1, c], sol[r, c - 1]) + 1
            else:  # but if the letters are different
                # pick the letter diagonally up left
                sol[r, c] = sol[r - 1, c - 1]

    # returns the entire matrix. min edit distance in bottom right corner jff.
    return sol


def mtx2graph(s1, s2, w_del=100, w_ins=49):
    """
    Called by loanpy.helpers.editops. \
Takes two strings, draws a distance matrix with \
loanpy.helpers.get_mtx and converts that matrix into a directed graph \
where horizontal edges are given a customisable weight for insertions \
and vertical edges are given a customisable weight for deletions. \
Where it is possible to move diagonally, a diagonal edge is inserted \
and no weight is added, since it means the letter is kept. \
It is necessary to create this type of object to be able to tap \
networkx's all_shortest_paths function in loanpy.helpers.editops

    :param s1: The first of two iterables to be compared to each other
    :type s1: iterable like str or list

    :param s2: The second of two iterables to be compared to each other
    :type s2: iterable like str or list

    :param w_del: The weight (cost) of deletions (vertical edges)
    :type w_del: int | float, default=100

    :param w_ins: The weight (cost) of insertions (horizontal edges). \
Default 49 \
is in accordance with the "Threshold Principle": \
2 insertions (2*49=98) are cheaper than a deletion \
(100).
    :type w_ins: int | float, default=49

    :returns: The directed graph object, its height and its width
    :rtype: (networkx.classes.digraph.DiGraph, int, int)

    :Example:

    >>> from loanpy.helpers import mtx2graph
    >>> mtx2graph("ló", "hó")
    (<networkx.classes.digraph.DiGraph object at 0x7fb8e5758700>, 3, 3)

    """
    mtx = get_mtx(s1, s2)
    s1, s2 = "#" + s1, "#" + s2
    G = DiGraph()
    h, w = mtx.shape

    for r in reversed(range(h)):  # create vertical edges
        for c in reversed(range(w - 1)):
            G.add_edge((r, c + 1), (r, c), weight=w_del)

    for r in reversed(range(h - 1)):  # create horizontal edges
        for c in reversed(range(w)):
            G.add_edge((r + 1, c), (r, c), weight=w_ins)

    for r in reversed(range(h - 1)):  # add diagonal edges where cost=0
        for c in reversed(range(w - 1)):
            if mtx[r + 1, c + 1] == mtx[r, c]:
                if s1[c + 1] == s2[r + 1]:
                    G.add_edge((r + 1, c + 1), (r, c), weight=0)

    return G, h, w


def tuples2editops(op_list, s1, s2):
    """
    Called by loanpy.helpers.editops. \
The path how string1 is converted to string2 is given in form of tuples \
that contain the x and y coordinates of every step through the matrix \
shaped graph. \
This function converts those numerical instructions to human readable ones. \
The x values stand for horizontal movement, y values for vertical ones. \
Vertical movement means deletion, horizontal means insertion. \
Diagonal means the value is kept. \
Moving horizontally and vertically after each other means \
substitution.

    :param op_list: The numeric list of edit operations
    :type op_list: list of tuples of 2 int

    :param s1: The first of two strings to be compared to each other
    :type s1: str

    :param s2: The second of two strings to be compared to each other
    :type s2: str

    :returns: list of human readable edit operations
    :rtype: list of strings

    :Example:

    >>> from loanpy.helpers import tuples2editops
    >>> tuples2editops([(0, 0), (0, 1), (1, 1), (2, 2)], "ló", "hó")
    ['substitute l by h', 'keep ó']
    >>>  # What happened under the hood:
    # (0, 0), (0, 1): move 1 vertically = 1 deletion
    # (0, 1), (1, 1): move 1 horizontally = 1 insertion
    # insertion and deletion after each other equals substitution
    # (1, 1), (2, 2): move 1 diagonally = keep the sound

    """
    s1, s2 = "#" + s1, "#" + s2
    out = []
    for i, todo in enumerate(op_list):
        if i == 0:  # so that i-i won't be out of range
            continue
        # where does the arrow point?
        direction = subtract(todo, op_list[i - 1])
        if array_equiv(direction, [1, 1]):  # if diagonal
            out.append(f"keep {s1[todo[1]]}")
        elif array_equiv(direction, [0, 1]):  # if horizontal
            if i > 1:  # if previous was verical -> substitute
                if array_equiv(
                        subtract(op_list[i - 1], op_list[i - 2]), [1, 0]):
                    out = out[:-1]
                    out.append(f"substitute {s1[todo[1]]} by {s2[todo[0]]}")
                    continue
            out.append(f"delete {s1[todo[1]]}")
        elif array_equiv(direction, [1, 0]):  # if vertical
            if i > 1:  # if previous was horizontal -> substitute
                if array_equiv(
                        subtract(op_list[i - 1], op_list[i - 2]), [0, 1]):
                    out = out[:-1]
                    out.append(f"substitute {s1[todo[1]]} by {s2[todo[0]]}")
                    continue
            out.append(f"insert {s2[todo[0]]}")

    return out


def editops(s1, s2, howmany_paths=1, w_del=100, w_ins=49):
    """
    Called by loanpy.adrc.Adrc.repair_phonotactics. \
Takes two strings and returns \
all paths of cheapest edit operations between them.

    :param s1: The first of two iterables to be compared to each other
    :type s1: iterable, e.g. str or list

    :param s2: The second of two iterables to be compared to each other
    :type s2: iterable, e.g. str or list

    :param howmany_paths: The number of shortest paths that should be returned.
    :type howmany_paths: int, default=1

    :param w_del: The weight (cost) of deletions
    :type w_del: int | float, default=100

    :param w_ins: The weight (cost) of insertions, Default 49 \
is in accordance with the "Threshold Principle": \
2 insertions (2*49=98) are cheaper than a deletion \
(100).
    :type w_ins: int | float, default=49

    :returns: Human-readable edit operations from string1 to string2
    :rtype: list of tuples of str

    :Example:

    >>> from loanpy.helpers import editops
    >>> editops("Budapest", "Bukarest")
    [('keep B', 'keep u', 'substitute d by k', 'keep a', \
'substitute p by r', \
'keep e', 'keep s', 'keep t')]
    >>> editops("CV", "CCVV")
    [('keep C', 'insert C', 'insert V', 'keep V')]
    >>> editops("CV", "CCVV", howmany_paths=2)
    [('insert C', 'keep C', 'insert V', 'keep V'), \
('insert C', 'keep C', 'keep V', 'insert V')]
    >>> editops("CV", "CCVV", howmany_paths=3)
    [('insert C', 'keep C', 'insert V', 'keep V'), \
('insert C', 'keep C', 'keep V', 'insert V'), \
('keep C', 'insert C', 'insert V', 'keep V')]
    """

    G, h, w = mtx2graph(s1, s2, w_del, w_ins)  # get directed graph
    paths = [shortest_path(G, (h - 1, w - 1), (0, 0),
             weight='weight')] if howmany_paths == 1 else all_shortest_paths(
                 G, (h - 1, w - 1), (0, 0), weight='weight')
    out = [tuples2editops(list(reversed(i)), s1, s2) for i in paths]
    return list(dict.fromkeys(map(tuple, out)))[:howmany_paths]


def apply_edit(word, editops):
    """
    Called by loanpy.adrc.Adrc.repair_phonotactics. \
Applies a list of human readable edit operations to a string.

    :param word: The input word
    :type word: an iterable (e.g. list of phonemes, or string)

    :param editops: list of (human readable) edit operations
    :type editops: list or tuple of strings

    :returns: transformed input word
    :rtype: list of str

    :Example:

    >>> from loanpy.helpers import apply_edit
    >>> apply_edit(["l", "ó"], ('substitute l by h', 'keep ó'))
    ['h', 'ó']
    >>> apply_edit("ló", ('keep C', 'insert C', 'insert V', 'keep V'))
    ['l', 'C', 'V', 'ó']
    >>> apply_edit("ló", ('insert C', 'keep C', 'insert V', 'keep V'))
    ['C', 'l', 'V', 'ó']
    """
    out, letter = [], iter(word)
    for i, op in enumerate(editops):
        if i != len(editops):  # to avoid stopiteration
            if "keep" in op:
                out.append(next(letter))
            elif "delete" in op:
                next(letter)
        if "substitute" in op:
            out.append(op[op.index(" by ") + 4:])
            if i != len(editops) - 1:
                next(letter)
        elif "insert" in op:
            out.append(op[len("insert "):])
    return out


def list2regex(sclist):
    """
    Called by loanpy.adrc.Adrc.reconstruct. \
Turns a list of phonemes into a regular expression.

    :param sclist: a list of phonemes
    :type sclist: list of str

    :returns: The phonemes from the input list separated by a pipe. "-" is \
removed and replaced with a question mark at the end.
    :rtype: str

    :Example:

    >>> from loanpy.helpers import list2regex
    >>> list2regex(["b", "k", "v"])
    '(b|k|v)'

    >>> from loanpy.helpers import list2regex
    >>> list2regex(["b", "k", "-", "v"])
    '(b|k|v)?'

    """

    suff = ")?" if "-" in sclist else ")"
    return "(" + "|".join([i for i in sclist if i != "-"]) + suff


def flatten(nested_list):
    """
    Called by loanpy.adrc.Adrc.repair_phonotactics and loanpy.adrc.Adrc.adapt.
    Flatten a nested list and discard empty strings (to prevent feeding \
empty strings to loanpy.adrc.Adrc.reconstruct, which would throw an Error)

    :param nested_list: a nested list
    :type nested_list: list of lists

    :return: flattened list without empty strings as elements
    :rtype: list

    :Example:

    >>> from loanpy.helpers import flatten
    >>> flatten([["wrd1", "wrd2", ""], ["wrd3", "", ""]])
    ['wrd1', 'wrd2', 'wrd3']
    """
    # empty strings would trigger an error in the tokeniser
    # sometimes both phonemes of a 2-phoneme-word can be replaced with "".
    # so some entire words can be substituted by "". Discard those.
    return [item for sublist in nested_list for item in sublist if item]


def combine_ipalists(wrds):
    """
    Called by loanpy.adrc.Adrc.adapt. \
Combines and flattens a list of lists of sound correspondence lists.

    :param wrds: list of words consisting of lists of sound correspondence \
    lists.
    :type wrds: list of lists of lists of str

    :returns: a list of words without empty strings as elements
    :rtype: list of strings

    :Example:

    >>> from loanpy.helpers import combine_ipalists
    >>> combine_ipalists([[["a", "b"], ["c"], ["d"]], [["e", "f"], \
["g"], ["h"]]])
    ['acd', 'bcd', 'egh', 'fgh']
    """

    return flatten([list(map(" ".join, product(*wrd))) for wrd in wrds])


def get_howmany(step, hm_phonotactics_ceiling, hm_paths_ceiling):
    """
    Called by loanpy.adrc.Adrc.adapt and loanpy.sanity.eval_one. \
Put marbles into three pots, one by one, with following conditions: \
if the product of the number of marbles per pot is higher than step, stop. \
Don't put more marbles in pot 2 and 3 than the ceiling variables indicate. \
The idea is that loanpy.adrc.Adrc.adapt gets a parameter <howmany> that \
indicates the total number of predictions, which flows into this function's \
<step> parameter. This has to be split into three: The first is the number \
of combinations to get from phoneme substitutions. The second the \
maximum number of combinations from the number of different \
phonotactic structures \
chosen, to which to adapt. The third the number of paths through which each \
structure can be be reached. So if 100 predictions should be made for a word \
but not pick more than 2 structures to which to adapt and 2 paths through \
which to reach those structures, 25 predictions have to be made from \
sound substitutions for 2 times 2 different paths to make 100 predictions. \
The loop breaks *after* the product is stepped over because any \
leftover will be sliced away in loanpy.adrc.Adrc.adapt.

    :param step: the product of the three pots. Stop after reached.
    :type step: int

    :param hm_phonotactics_ceiling: The max nr of marbles for pot 2
    :type hm_phonotactics_ceiling: int

    :param hm_paths_ceiling: The max nr of marbles for pot 3
    :type hm_paths_ceiling: int

    :returns: The way the marbles should be split
    :rtype: (int, int, int)

    :Example:

    >>> from loanpy.helpers import get_howmany
    >>> get_howmany(10, 2, 2)
    (3, 2, 2)  # since 3*2*2=12, superfluous 2 will be sliced away later
    >>> get_howmany(100, 9, 2)
    (8, 7, 2)  # since 8*7*2=112, superfluous 12 will be sliced away later

    """
    if hm_phonotactics_ceiling == 0:
        return step, hm_phonotactics_ceiling, hm_paths_ceiling

    x, y, z = 1, 1, 1
    while x * y * z < step:
        x += 1
        if x * y * z >= step:
            return x, y, z
        if y < hm_phonotactics_ceiling:
            y += 1
            if x * y * z >= step:
                return x, y, z
        if z < hm_paths_ceiling:
            z += 1
            if x * y * z >= step:
                return x, y, z

    return x, y, z
