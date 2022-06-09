"""
Contains helper functions and class Etym, which are called internally.
Some of the functions may also be useful for the user in \
other linguistic contexts.

"""

from ast import literal_eval
from collections import Counter
from datetime import datetime
from functools import partial
from itertools import product
from logging import getLogger
from pathlib import Path

from gensim.downloader import load
from ipatok import clusterise, tokenise
from networkx import DiGraph, all_shortest_paths, shortest_path
from numpy import array_equiv, isnan, subtract, zeros
from pandas import DataFrame, read_csv
from panphon.distance import Distance
from tqdm import tqdm

logger = getLogger(__name__)
model = None
tokenise = partial(tokenise, replace=True)
clusterise = partial(clusterise, replace=True)


class InventoryMissingError(Exception):
    """
    Called by lonapy.helpers.Etym.rank_closest and \
loanpy.helpers.Etym.rank_closest_phonotactics if neither forms.csv \
is defined nor the phonotactic/phoneme inventory is plugged in.
    """
    pass


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


def read_cvfb():
    """
    Called by loanpy.helpers.Etym.__init__; \
Reads file cvfb.txt that was generated based on ipa_all.csv \
by loanpy.helpers.make_cvfb. \
It's a tuple of two dictionaries. Keys are same as col "ipa" in ipa_all.csv. \
Values of first dictionary are "C" if consonant and "V" if vowel (6358 keys). \
Values of 2nd dictionary are "F" if front vowel \
and "B" if back vowel (1240 keys). \
Only called by Etym.__init__ to define loanpy.helpers.Etym.phon2cv and \
loanpy.helpers.Ety.vow2fb, \
which in turn is used by loanpy.helpers.Etym.word2phonotactics, \
loanpy.helpers.Etym.has_harmony and others. \
This file could be read directly when importing, but this way things \
feel more stable.

    :returns: two dictionaries, the first defining consonants \
and vowels (cv), \
the second defining front and back vowels (fb).
    :rtype: (dict, dict)

    :Example:

    >>> from loanpy.helpers import read_cvfb
    >>> read_cvfb()
    (two dictionaries of length 6358 and 1240)

    """
    # todo: document the changes from this ipa_all.csv
    # to the original in panphon.
    path = Path(__file__).parent / "cvfb.txt"
    with open(path, "r", encoding="utf-8") as f:
        cvfb = literal_eval(f.read())
    return cvfb[0], cvfb[1]


def read_forms(dff):
    """
    Called by loanpy.helpers.Etym.__init__; Reads forms.csv (cldf), \
keeps only columns "Segement", "Cognacy" and \
"Language ID", drops spaces in Segments to internally re-tokenise later. \
Only called by Etym.__init__ to create local variable dff (data frame forms). \
Returns None if dff is None. So that class can be initiated without args too.

    :param dff: path to forms.csv
    :type dff: pathlib.PosixPath | str | None

    :returns: a workable version of forms.csv as a pandas data frame
    :rtype: pandas.core.frame.DataFrame | None

    :Example:

    >>> from pathlib import Path
    >>> from loanpy.helpers import __file__, read_forms
    >>> path2file = Path(__file__).parent / "tests" / \
"input_files" / "forms.csv"
    >>> read_forms(path2file)
           Language_ID Segments  Cognacy
    0            1      abc        1
    1            2      xyz        1

    """

    if not dff:
        return None
    dff = read_csv(dff, usecols=["Segments", "Cognacy", "Language_ID"])
    dff["Segments"] = [i.replace(" ", "") for i in dff.Segments]
    return dff


def cldf2pd(dfforms, source_language=None, target_language=None):
    """
    Called by loanpy.helpers.Etym.__init__; \
Converts a CLDF based forms.csv to a pandas data frame. \
Returns None if dfforms is None, so class can be initiated without args too. \
Runs through forms.csv and creates a new data frame the following way: \
Checks if a cognate set contains words from both, source and target language. \
If yes: word from source lg goes to column "Source_Form", \
word from target lg goes to column "Target_Form" and \
the number of the cognate set goes to column "Cognacy". \
Note that if a cognate set doesn't contain words from source and target \
language, \
that cognate set is skipped.

    :param dfforms: Takes the output of read_forms() as input
    :type dfforms: pandas.core.frame.DataFrame | None

    :param source_language: The language who's cognates go to \
column "Source_Forms"
    :type source_language: str, default=None

    :param target_language: The language who's cognates go to \
column "Target_Forms"
    :type target_language: str, default=None

    :returns: forms.csv data frame with re-positioned information
    :rtype: pandas.core.frame.DataFrame | None

    :Example:

    >>> from pathlib import Path
    >>> from loanpy.helpers import __file__, cldf2pd, read_forms
    >>> path2forms = Path(__file__).parent / "tests" \
/ "input_files" / "forms.csv"
    >>> forms = read_forms(path2forms)
    >>> cldf2pd(forms, source_language=1, target_language=2)
          Target_Form Source_Form  Cognacy
    0         xyz         abc        1

    """

    if dfforms is None:
        return None
    target_form, source_form, cognacy, dfetymology = [], [], [], DataFrame()

    # bugfix: col Cognacy is sometimes empty. if so, fill it
    if all(isnan(i) for i in dfforms.Cognacy):
        dfforms["Cognacy"] = list(range(len(dfforms)))

    for cog in range(1, int(list(dfforms["Cognacy"])[-1])+1):
        dfformsdrop = dfforms[dfforms["Cognacy"] == cog]
        if all(lg in list(dfformsdrop["Language_ID"])
               for lg in [target_language, source_language]):
            cognacy.append(cog)
            for idx, row in dfformsdrop.iterrows():
                if row["Language_ID"] == target_language:
                    target_form.append(row["Segments"])
                if row["Language_ID"] == source_language:
                    source_form.append(row["Segments"])

    dfetymology["Target_Form"] = target_form
    dfetymology["Source_Form"] = source_form
    dfetymology["Cognacy"] = cognacy
    return dfetymology


def read_dst(dst_msr):
    """
    Called by loanpy.helpers.Etym.__init__; \
Returns a function that calculates the phonological distance between strings \
from panphon.distance.Distance. This will be used to calculate the most \
similar phonemes of the phoneme inventory compared to \
a given phoneme from ipa_all.csv

    :param dst_msr: The name of the distance measure, which has to be \
a method \
of panphon.distance.Distance
    :type dst_msr: None, \
"doglo_prime_distance" | \
"dolgo_prime_distance_div_maxlen" | \
"fast_levenshtein_distance" | \
"fast_levenshtein_distance_div_maxlen" | \
"feature_difference" | \
"feature_edit_distance" | \
"feature_edit_distance_div_maxlen" | \
"hamming_feature_edit_distance" | \
"hamming_feature_edit_distance_div_maxlen" | \
"hamming_substitution_cost" | \
"jt_feature_edit_distance" | \
"jt_feature_edit_distance_div_maxlen" | \
"jt_hamming_feature_edit_distance" | \
"jt_hamming_feature_edit_distance_div_maxlen" | \
"jt_weighted_feature_edit_distance" | \
"jt_weighted_feature_edit_distance_div_maxlen" | \
"levenshtein_distance"

    :returns: a function that calculates the phonological distance between \
IPA-strings
    :rtype: function | None

    :Example:

    >>> from loanpy.helpers import read_dst
    >>> read_dst("fast_levenshtein_distance")
    <bound method Distance.fast_levenshtein_distance of \
<panphon.distance.Distance object at 0x7f7f21fe95b0>>

    For more information see PanPhon's documentation:

    >>> from panphon.distance import Distance
    >>> help(Distance)

    """
    return None if not dst_msr else getattr(Distance(), dst_msr)


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

    return flatten([list(map("".join, product(*wrd))) for wrd in wrds])


def forms2list(dff, target_language):
    """
    Called by loanpy.helpers.Etym.__init__; \
Get a list of words of a language from a forms.csv file.

    :param dff: forms.csv data frame (cldf)
    :type dff: pandas.core.frame.DataFrame

    :param target_language: The language from which the phoneme, \
phoneme cluster and phonotactic inventories will be extracted.
    :type target_language: str

    :returns: a list of words in the target language
    :rtype: list | None

    :Example:

    >>> from pathlib import Path
    >>> from loanpy.helpers import __file__, forms2list, read_forms
    >>> path2forms = Path(__file__).parent / "tests" \
/ "input_files" / "forms.csv"
    >>> forms = read_forms(path2forms)
    >>> forms2list(forms, target_language=2)
    ['xyz']
    """

    return None if dff is None else list(
        dff[dff["Language_ID"] == target_language]["Segments"])


class Etym():
    """
    Class that is based on 2 datasets: The first is static, and \
is ipa_all.csv \
from PanPhon, together with its "spin-off" cvfb.txt. The second has to be \
defined through the init args and will be data extracted from a \
forms.csv of the CLDF standard. \
The class methods eventually all rely on panphon / data / "ipa_all.csv". \
loanpy's ipa_all.csv, has 3 modifications: "ʔ" (glottal stop) \
is defined as neither consonant nor vowel (0), "j", together \
with all its combinations is defined as a consonant, instead of a vowel. \
This can be looked up in differences.csv. \
Additionally, two rows are attached to the bottom, one for \
"C" - which denotes "any consonant" and "V" - for "any vowel". \
Citation of PanPhon: \
David R. Mortensen, Patrick Littell, Akash Bharadwaj, Kartik Goyal, \
Chris Dyer, \
Lori Levin (2016). "PanPhon: A Resource for Mapping IPA Segments to \
Articulatory Feature \
Vectors." Proceedings of COLING 2016, the 26th International \
Conference on Computational \
Linguistics: Technical Papers, pages 3475–3484, Osaka, Japan, \
December 11-17 2016. \
(Based on SPE by Chomsyk&Halle). \
Turn forms.csv into a pandas data frame, \
and extract following information from it: \
phoneme inventory: a set of all phonemes that occur in the \
target language's words, \
cluster inventory: a set of all consonant and vowel clusters \
occurring in the target language, \
phonotactic inventory: a list of phonotactic structures that occur \
in the target language, \
sorted according to their frequency in the data.

    :param forms_csv: a forms.csv of the CLDF data standard
    :type forms_csv: pathlib.PosixPath | str | None, default=None

    :param source_language: The technical source language as defined in \
cldf's etc / "languages.tsv". \
This can be confusing, as sometimes \
the technical source language for the predictions can be the linguistic \
target language, e.g. when backward reconstructing. But when \
adapting loanwords the technical and the linguistic source word are usually \
the same.
    :type source_language: str | None, default=None

    :param target_language: The technical target language as defined in \
cldf's etc / "languages.tsv". For caveats see param source_language
    :type target_language: str | None, default=None

    :param phoneme_inventory: The phoneme inventory \
of the target language. If None, \
it will be extracted \
automatically from forms.csv
    :type phoneme_inventory: None | list | set, default=None

    :param cluster_inventory: All consonant \
and vowel clusters that occur in the target \
language. If None, will be automatically extracted from forms.csv
    :type cluster_inventory: None | list | set, default=None

    :param phonotactic_inventory: All phonotactic structures (e.g. "CVCV") \
that occur in the target \
language. Will be extracted automatically from target language if set to None.
    :type phonotactic_inventory: None | list | set, default=None

    :param distance_measure: The name of the distance measure \
    for loanpy.helpers.Etym.rank_closest. Has to be a \
    method of panphon.distance.Distance. For more details see \
    loanpy.helpers.read_dst.
    :type distance_measure: str, default='weighted_feature_edit_distance'

    :param most_frquent_phonotactics: The how many most frequently \
occurring phonotactic \
structures should be allowed to be part of the phoneme inventory. By default, \
all of them are read.
    :type most_frquent_phonotactics: int, default=9999999

    :Example:

    >>> from pathlib import Path
    >>> from loanpy.helpers import Etym, __file__
    >>> path2forms = Path(__file__).parent / "tests" / \
"input_files" / "forms.csv"
    >>> etym_obj = Etym(forms_csv=path2forms, source_language=1, \
target_language=2)
    >>> etym_obj.phon2cv["k"]
    'C'
    >>> etym_obj.vow2fb["e"]
    'F'
    >>> etym_obj.distance_measure("p", "b")
    0.25
    >>> etym_obj.dfety
          Target_Form Source_Form  Cognacy
    0         xyz         abc        1
    >>> etym_obj.phoneme_inventory
    {'y', 'z', 'x'}
    >>> etym_obj.cluster_inventory
    {'y', 'z', 'x'}
    >>> etym_obj.phonotactic_inventory
    {'CVC'}
    >>> len(etym_obj.__dict__)
    8
    """

    def __init__(self,
                 forms_csv=None,
                 source_language=None,
                 target_language=None,
                 phoneme_inventory=None,
                 cluster_inventory=None,
                 phonotactic_inventory=None,
                 distance_measure="weighted_feature_edit_distance",
                 most_frequent_phonotactics=9999999):

        # independent of other attributes
        self.phon2cv, self.vow2fb = read_cvfb()
        self.distance_measure = read_dst(distance_measure)
        # read data frame forms, turn words of target language to list
        dff = read_forms(forms_csv)
        # conclude dfety from dff
        self.dfety = cldf2pd(dff, source_language, target_language)
        # conclude 3 inventories from forms_target_language
        self.forms_target_language = forms2list(dff, target_language)
        (self.phoneme_inventory, self.cluster_inventory,
         self.phonotactic_inventory) = self.get_inventories(
                                                    phoneme_inventory,
                                                    cluster_inventory,
                                                    phonotactic_inventory,
                                                    most_frequent_phonotactics
                                                    )

    def get_inventories(self,
                        phoneme_inventory=None,
                        cluster_inventory=None,
                        phonotactic_inventory=None,
                        most_frequent_phonotactics=9999999):
        """
        Called by loanpy.helpers.Etym.__init__. \
Returns a tuple of three sets, each representing the inventory \
of the target language at the phoneme, phoneme-cluster, and phonotactic level.

        :param phoneme_inventory: Chance to hard-code the phoneme inventory \
by passing a list or a set. If None, it will be extracted from the \
target language data.
        :type phoneme_inventory: None | set | list | iterable, default=None

        :param cluster_inventory: Chance to hard-code the phoneme-\
cluster inventory by passing a list or a set. If None, it \
will be extracted from the \
target language data.
        :type cluster_inventory: None | set | list | iterable, default=None

        :param phonotactic_inventory: Chance to hard-code the phonotactic \
inventory by passing a list or a set. If None, it \
will be extracted from the target language data.
        :type phonotactic_inventory: None | set | list | iterable, default=None

        :param most_frequent_phonotactics: This many of the most frequently \
occurring phonotactic \
structures will be part of the phoneme inventory.
        :type most_frequent_phonotactics: int, default=9999999

        :returns: Tuple of three sets containing the phoneme, \
phoneme cluster and phonotactic inventories of the target language.
        :rtype: (set, set, set)
        """

        return (self.read_inventory(phoneme_inventory),
                self.read_inventory(cluster_inventory, clusterise),
                self.read_phonotactic_inv(phonotactic_inventory,
                                          most_frequent_phonotactics))

    def read_inventory(self, inv, func=tokenise):
        """
        Called by loanpy.helpers.Etym.__init__; \
    Calculates and returns phoneme inventory from a list of words. \
    Param <inv> is if given inventory should not be calculated but \
manually plugged in.

        :param inv: a set of phonemes that occur in given language.
        :type inv: set

        :param func: The tokeniser to split a word into a list \
        of phonemes or phoneme clusters. \
        Theoretically possible to plug in own tokeniser function here as well.
        :type func: ipatok.tokenise | ipatok.clusterise, \
        default=ipatok.tokenise

        :returns: The phoneme / consonant + vowel cluster \
        inventory of the language
        :rtype: set | None | same as input type

        :Example:

        >>> from loanpy.helpers import Etym, clusterise
        >>> etym_obj = Etym()
        >>> etym_obj.forms_target_language = ["fdedaeda", "badea", "fdddedab"]
        >>> etym_obj.read_inventory(None)
        {'b', 'd', 'f', 'a', 'e'}
        >>> etym_obj.read_inventory(None, clusterise)
        {'b', 'ea', 'd', 'fd', 'a', 'ae', 'fddd', 'e'}

        """
        return inv if inv else set(func("".join(
            self.forms_target_language
            ))) if self.forms_target_language else None

    def read_phonotactic_inv(self,
                             phonotactic_inventory=None, howmany=9999999,
                             print_entire_inv=False):
        """
        Called by loanpy.helpers.Etym.__init__; Returns \
phonotactic inventory of the n \
most frequent phonotactic structures. \
Caveat: The map function seems to swallow errors that would be otherwise \
triggered by collections.Counter. E.g. if a float is used \
(including float("inf")) for param <howmany>, an empty string \
will be returned.

        :param phonotactic_inventory: Possibility to plug in \
the phonotactic inventory manually.
        :type phonotactic_inventory: None | list | set, \
        default=None

        :param howmany: how many most frequent structures should be added to \
phonotactic inventory
        :type howmany: int, default=9999999

        :param print_entire_inv: Indicate if logger should print the \
entire inventory to the console. It is a collections.Counter object with \
phonotactic profiles as keys and their frequencies as values. Inspecting \
this information will help to figure out which integer best to pass to \
param <howmany>. Best is to omit rare ones, but rare is relative to how \
much data is available, therefore this has to be done manually.
        :type print_entire_inv: bool, default=False

        :returns: all possible phonotactic structures documented in the data
        :rtype: list

        :Example:

        >>> from loanpy.helpers import Etym
        >>> etym_obj = Etym()
        >>> etym_obj.forms_target_language = \
["ab", "ab", "aa", "bb", "bb", "bb"]
        >>> etym_obj.read_phonotactic_inv()
        {'VV', 'CC', 'VC'}

        >>> from loanpy.helpers import Etym
        >>> etym_obj = Etym()
        >>> etym_obj.forms_target_language = \
["ab", "ab", "aa", "bb", "bb", "bb"]
        >>> etym_obj.read_phonotactic_inv(howmany=1)
        {'CC'} # b/c that's the nr 1 most frequent structure

        >>> from loanpy.helpers import Etym
        >>> etym_obj = Etym()
        >>> etym_obj.forms_target_language = \
["ab", "ab", "aa", "bb", "bb", "bb"]
        >>> etym_obj.read_phonotactic_inv(howmany=2)
        {'CC', 'VC'} # b/c that's the 2 most frequent structures
        """

        if phonotactic_inventory:
            return phonotactic_inventory
        if self.forms_target_language is None:
            return None
        strucs = list(map(self.word2phonotactics, self.forms_target_language))
        if howmany == 9999999:
            return set(strucs)
        if print_entire_inv is True:
            logger.warning(Counter(strucs))
        return set(map(lambda x: x[0], Counter(strucs).most_common(howmany)))

    def word2phonotactics(self, ipa_in):
        """
        Called by loanpy.helpers.Etym.__init__; \
loanpy.qfysc.Qfy.get_phonotactics_corresp, \
loanpy.adrc.Adrc.adapt, loanpy.adrc.Adrc.repair_phonotactics, \
loanpy.adrc.Adrc.reconstruct, \
and loanpy.santiy.write_workflow.

        Returns the phonotactic profile of an IPA-string.

        :param ipa_in: a string or list consisting of IPA-characters. \
if input is string, it will be tokenised.
        :type ipa_in: str | list

        :returns: the phonotactic profile of the word
        :rtype: str

        :Example:

        >>> from loanpy import helpers
        >>> hp = helpers.Etym()
        >>> hp.word2phonotactics("lol")
        'CVC'

        """
        if isinstance(ipa_in, str):
            ipa_in = tokenise(ipa_in)

        return "".join([self.phon2cv.get(i, "") for i in ipa_in])

    def word2phonotactics_keepcv(self, ipa_in):
        """
        Not called by any function. Returns the phonotactic profile of an \
IPA-string while keeping \
Cs and Vs, otherwise same as loanpy.helpers.Etym.word2phonotactics. \
Originally written for sanity.py but currently not used anywhere.

        :param ipa_in: a string or list consisting of IPA-characters, \
"C"s and "V"s.
        :type ipa_in: list

        :returns: the phonotactic profile of the word
        :rtype: str

        :Example:

        >>> from loanpy import helpers
        >>> hp = helpers.Etym()
        >>> hp.word2phonotactics_keepcv("CloVl")
        'CCVVC'

        """
        return "".join([self.phon2cv.get(i, "") if i not in "CV"
                        else i for i in ipa_in])

    def has_harmony(self, ipalist):
        """
        Called by loanpy.helpers.Etym.repair_harmony and \
loanpy.adrc.Adrc.reconstruct. \
Returns True if there are only front or only back vowels in a word else False.

        :param ipalist: the word that should be analysed
        :type ipalist: list of str | str

        :returns: Does the word have front-back vowel harmony?
        :rtype: bool

        :Example:

        >>> from loanpy import helpers
        >>> hp = helpers.Etym()
        >>> hp.has_harmony("bot͡sibot͡si")
        False

        >>> from loanpy import helpers
        >>> hp = helpers.Etym()
        >>> hp.has_harmony(["t", "ɒ", "r", "k", "ɒ"])
        True

        """
        if isinstance(ipalist, str):
            ipalist = tokenise(ipalist)
        vowels = set(j for j in [self.vow2fb.get(i, "") for i in ipalist] if j)
        return vowels in ({"F"}, {"B"}, set())

    def repair_harmony(self, ipalist):
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
            ipalist = tokenise(ipalist)

        if self.has_harmony(ipalist) and "V" not in ipalist:
            return [ipalist]

        out = [j for j in [self.vow2fb.get(i, "") for i in ipalist] if j]
        if out.count("F") > out.count("B"):
            out = [self.get_fb(ipalist, turnto="F")]
        elif out.count("B") > out.count("F"):
            out = [self.get_fb(ipalist, turnto="B")]
        else:
            out = [self.get_fb(ipalist, "F"), self.get_fb(ipalist, "B")]

        return out

    def get_fb(self, ipalist, turnto="F"):
        """
        Called by loanpy.helpers.Etym.repair_harmony. \
Turns front vowels to back ones if turnto="B", \
but turns back vowels to front ones if turnto="F"

        :param ipalist: a tokenised IPA-string
        :type ipalist: list

        :param turnto: turn back vowels to front ones or vice versa
        :type turnto: "F" | "B", default="F"

        :returns: a tokenised word with some vowels replaced by "F" or "B".
        :rtype: list

        :Example:

        >>> from loanpy import helpers
        >>> hp = helpers.Etym()
        >>> hp.get_fb(['ɒ', 'l', 'ʃ', 'oː', 'ø', 'r', 'ʃ'], turnto="B")
        ['ɒ', 'l', 'ʃ', 'oː', 'B', 'r', 'ʃ']
        """

        checkfor = "F" if turnto == "B" else "B"

        return [turnto if self.vow2fb.get(i, "") == checkfor or i == "V"
                else i for i in ipalist]

    def get_scdictbase(self, write_to=None, most_common=float("inf")):
        """
        Call manually in the beginning. \
Loop through ipa_all.csv and rank most similar phonemes \
from phoneme_inventory. \
Could also turn ipa_all.csv into a square of 6358*6358 phonemes and just use \
that file for all future cases but that would use an estimated 500MB per \
type of distance measure. So more economical to calculate it this way. \
Will still, depending on size of phoneme inventory, \
take up about 2MB. \
The result is returned, assigned to loanpy.helpers.Etym.scdictbase \
and optionally also written. \
Usually there is very little data available \
on sound substitutions and the ones available give only very small insight \
into all possibilities. \
The idea here is therefore that any sound \
that is not contained in a language's phoneme inventory \
will be replaced by \
the most similar available one(s). The available sound substitutions \
based on etymological data can optionally be combined with \
this heuristics in a way that heuristics are only \
applied if for a particular sound (cluster) no more data is available any more.

        :param write_to: If or where the output should be written.
        :type write_to: None | pathlib.PosixPath | str, default=None

        :param most_common: Add only this many most similar phonemes to \
scdictbase. By default, the entire phoneme inventory will be ranked.
        :type most_common: int | float, default=float("inf")

        :returns: A heuristic approach to sound substitution in loanword \
adaptation
        :rtype: dict

        :Example:

        >>> from pathlib import Path
        >>> from loanpy.helpers import Etym, __file__
        >>> etym_obj = Etym(forms_csv=Path(__file__).parent / "tests" / \
"integration" / "input_files" / "forms.csv", source_language=1, \
target_language=2)
        >>> etym_obj.get_scdictbase()
        (returns the entire dictionary with phoneme_inventory \
ranked according to similarity)
        >>> etym_obj.scdictbase["i"]
        ["y", "x", "z"]
        """

        ipa_all = read_csv(Path(__file__).parent / "ipa_all.csv")
        ipa_all["substi"] = [self.rank_closest(ph, most_common)
                             for ph in tqdm(ipa_all["ipa"])]
        scdictbase = dict(zip(ipa_all["ipa"],
                              ipa_all["substi"].str.split(", ")))

        # pick the most unmarked C
        cons_inv = [i for i in self.phoneme_inventory
                    if self.phon2cv.get(i, "") == "C"]
        scdictbase["C"] = self.rank_closest("ə", most_common,
                                            cons_inv).split(", ")
        # pick the most unmarked V
        vow_inv = [i for i in self.phoneme_inventory
                   if self.phon2cv.get(i, "") == "V"]
        scdictbase["V"] = self.rank_closest("ə", most_common,
                                            vow_inv).split(", ")
        scdictbase["F"] = [i for i in self.phoneme_inventory
                           if self.vow2fb.get(i, "") == "F"]
        scdictbase["B"] = [i for i in self.phoneme_inventory
                           if self.vow2fb.get(i, "") == "B"]

        self.scdictbase = scdictbase

        if write_to:
            with open(write_to, "w", encoding="utf-8") as f:
                f.write(str(scdictbase))

        return scdictbase

    def rank_closest(self, ph, howmany=float("inf"), inv=None):
        """
        Called by loanpy.helpers.Etym.get_scdictbase. \
Sort self.phoneme_inventory by distance to input-phoneme.

        :param ph: phoneme to which to rank the phoneme inventory
        :type ph: str (valid chars: col "ipa" in ipa_all.csv)

        :param howmany: howmany of the most similar phonemes to pick \
for the ranking.
        :type howmany: int | float("inf"), default=float("inf")

        :param inv: To plug in phoneme inventory manually if necessary
        :type inv: None | list | set, default=None

        :returns: the phoneme inventory \
ranked by similarity (most similar \
first)
        :rtype: str (elements separates by ", ")

        :Example:

        >>> from pathlib import Path
        >>> from loanpy.helpers import Etym, __file__
        >>> etym_obj = Etym()
        >>> etym_obj.rank_closest(ph="d", inv=["r", "t", "l"], howmany=1)
        't'
        >>> etym_obj = Etym(phoneme_inventory=["a", "b", "c"])
        >>> etym_obj.rank_closest(ph="d")
        'b, c, a'

        """
        if self.phoneme_inventory is None and inv is None:
            raise InventoryMissingError("define phoneme inventory \
or forms.csv")
        if inv is None:
            inv = self.phoneme_inventory

        phons_and_dist = [(i, self.distance_measure(ph, i)) for i in inv]
        return ", ".join(pick_minmax(phons_and_dist, howmany))

    # DONT merge this func into rank_closest. It makes things more complicated
    def rank_closest_phonotactics(self, struc, howmany=9999999, inv=None):
        """
        Called by loanpy.qfysc.Qfy.get_phonotactics_corresp. \
Sort loanpy.helpers.Etym.phonotactic_inventory by distance to given \
phonotactic structure using loanpy.helpers.edit_distance_with2ops \
(cost for insertion: 49, cost for deletion: 100).

        :param struc: The phonotactic structure to which \
to rank the phoneme_inventory
        :type struc: str (consisting of "C"s and "V"s only)

        :param howmany: How many of the most similar \
profiles should be picked
        :type howmany: int | float("inf"), default=float("inf")

        :param inv: To plug in the phonotactic inventory \
manually if necessary
        :type inv: None | list | set, default=None

        :returns: the phonotactic inventory ranked \
by similarity (most similar \
first)
        :rtype: str (elements separates by ", ")

        >>> from pathlib import Path
        >>> from loanpy.helpers import Etym, __file__
        >>> etym_obj = Etym(phonotactic_inventory=\
["CVC", "CVCVV", "CCCC", "VVVVVV"])
        >>> etym_obj.rank_closest_phonotactics(struc="CVCV", howmany=1)
        'CVCVV'
        >>> etym_obj = Etym()
        >>> etym_obj.rank_closest_phonotactics(struc="CVCV", howmany=3,\
inv=["CVC", "CVCVV", "CCCC", "VVVVVV"])
        "CVCVV, CVC, CCCC"

        """
        if self.phonotactic_inventory is None and inv is None:
            raise InventoryMissingError("define phonotactic inventory \
or forms.csv")
        if inv is None:
            inv = self.phonotactic_inventory

        strucs_and_dist = [(i, edit_distance_with2ops(struc, i)) for i in inv]
        return ", ".join(pick_minmax(strucs_and_dist, howmany))


def gensim_multiword(recip_transl, donor_transl, return_wordpair=False,
                     wordvectors="word2vec-google-news-300"):
    """
    Called by loanpy.loafinder.Search.loans. Takes two strings as input \
that represent the meanings of a word. Within the strings, \
meanings are separated \
by ", ". It calculates the cosine similarities of the word pairs of the \
Cartesian product and returns the value of the most similar pair. If \
return_wordpair is True, the words are returned as well. \
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

    if sclist == ["-"]:
        return ""
    if "-" in sclist:
        return "("+"|".join([i for i in sclist if i != "-"]) + ")?"
    return "("+"|".join(sclist) + ")"


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
    target = ['#']+[k for k in target]  # add hashtag as starting value
    source = ['#']+[k for k in source]  # starting value is always zero
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
                sol[r, c] = min(sol[r-1, c], sol[r, c-1]) + 1
            else:  # but if the letters are different
                sol[r, c] = sol[r-1, c-1]  # pick the letter diagonally up left

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
        for c in reversed(range(w-1)):
            G.add_edge((r, c+1), (r, c), weight=w_del)

    for r in reversed(range(h-1)):  # create horizontal edges
        for c in reversed(range(w)):
            G.add_edge((r+1, c), (r, c), weight=w_ins)

    for r in reversed(range(h-1)):  # add diagonal edges where cost=0
        for c in reversed(range(w-1)):
            if mtx[r+1, c+1] == mtx[r, c]:
                if s1[c+1] == s2[r+1]:
                    G.add_edge((r+1, c+1), (r, c), weight=0)

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
    s1, s2 = "#"+s1, "#"+s2
    out = []
    for i, todo in enumerate(op_list):
        if i == 0:  # so that i-i won't be out of range
            continue
        # where does the arrow point?
        direction = subtract(todo, op_list[i-1])
        if array_equiv(direction, [1, 1]):  # if diagonal
            out.append(f"keep {s1[todo[1]]}")
        elif array_equiv(direction, [0, 1]):  # if horizontal
            if i > 1:  # if previous was verical -> substitute
                if array_equiv(subtract(op_list[i-1], op_list[i-2]), [1, 0]):
                    out = out[:-1]
                    out.append(f"substitute {s1[todo[1]]} by {s2[todo[0]]}")
                    continue
            out.append(f"delete {s1[todo[1]]}")
        elif array_equiv(direction, [1, 0]):  # if vertical
            if i > 1:  # if previous was horizontal -> substitute
                if array_equiv(subtract(op_list[i-1], op_list[i-2]), [0, 1]):
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
    paths = [shortest_path(G, (h-1, w-1), (0, 0),
             weight='weight')] if howmany_paths == 1 else all_shortest_paths(
                 G, (h-1, w-1), (0, 0), weight='weight')
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
            out.append(op[op.index(" by ")+4:])
            if i != len(editops)-1:
                next(letter)
        elif "insert" in op:
            out.append(op[len("insert "):])
    return out


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
    while x*y*z < step:
        x += 1
        if x*y*z >= step:
            return x, y, z
        if y < hm_phonotactics_ceiling:
            y += 1
            if x*y*z >= step:
                return x, y, z
        if z < hm_paths_ceiling:
            z += 1
            if x*y*z >= step:
                return x, y, z

    return x, y, z


def pick_minmax(input_and_nr, howmany, func=min, return_all=False):
    """
    Called in loanpy.helpers.Etym.rank_closest and \
loanpy.helpers.Etym.rank_closest_phonotactics. \
Pick only the n smallest numbers from a list. Cheaper than \
sorting the entire list and then taking only the slice needed.

    :param input_and_nr: inventories and distances
    :type input_and_nr: list of tuples

    :param howmany: how many minimums/maximums to pick from input-list
    :type howmany: int

    :param func: Indicate if minimums or maximums should be picked.
    :type func: max | min, default=min

    :param return_all: Indicate whether only the sorted chunk should be \
returned, or if it should be merged with the rest of the unsorted list.
    :type return_all: bool, default=False

    :returns: The indicated number of minimal values
    :rtype: str (separated by ", ")

    :Example:

    >>> from loanpy.helpers import pick_minmax
    >>> pick_minmax([("a", 5), ("b", 7), ("c", 3)], float("inf"))
    ["c", "a", "b"]
    >>> pick_minmax([("a", 5), ("b", 7), ("c", 3)], 1)
    ["c"]
    >>> pick_minmax([("a", 5), ("b", 7), ("c", 3)], 2)
    ["c", "a"]
    """

    # if we want to have at least as many elements as the list is long
    # then we will just have to sort the entire list
    if howmany >= len(input_and_nr) or howmany is True:
        sorted2 = partial(sorted, reverse=True) if func == max else sorted
        return [i[0] for i in sorted2(input_and_nr, key=lambda tup: tup[1])]
    out = []
    # but if we just want a handful of min values
    for i in range(howmany):  # just pick that number of mins thru loop.
        mindisttup = func(input_and_nr, key=lambda tup: tup[1])
        out.append(input_and_nr.pop(input_and_nr.index(mindisttup))[0])
    return out + [i[0] for i in input_and_nr] if return_all else out


def make_cvfb(path2ipa_all, path2out):
    """
    Called manually only once ever. \
    This is how cvfb.txt was created from ipa_all.csv. \
    Good to keep this here, in case ipa_all.csv should change \
    some time. \
    Consonants, vowels, front vowels, back vowels. \
    Transforms the columns of ipa_all.csv to a tuple of two dictionaries. \
    The first returns "C" for consonants and "V" for vowels, \
    the second "F" for front vowels and "B" for back vowels.

    :param path2ipa_all: Location of ipa_all.csv (the input file)
    :type path2ipa_all: pathlib.PosixPath | str

    :param path2out: path to the desired output file.
    :type path2out: pathlib.PosixPath | str

    :returns: Writes a tuple of two dictionaries to indicated path
    :rtype: None
    """
    ipa_all = read_csv(path2ipa_all)

    phon2cv0 = {phoneme: "C" if cons == "+" else
                         "V" if cons == "-" else ""
                for phoneme, cons in zip(ipa_all["ipa"],
                                         ipa_all["cons"])}

    vow2fb0 = {phoneme: "F" if back == "-" and cons == "-" else
                        "B" if back == "+" and cons == "-" else ""
               for phoneme, back, cons in zip(ipa_all["ipa"],
                                              ipa_all["back"],
                                              ipa_all["cons"])}
    vow2fb0["V"] = "V"

    phon2cv = {}
    for i in phon2cv0:
        if phon2cv0[i]:
            phon2cv[i] = phon2cv0[i]

    vow2fb = {}
    for i in vow2fb0:
        if vow2fb0[i]:
            vow2fb[i] = vow2fb0[i]

    with open(path2out, "w") as f:
        f.write(str((phon2cv, vow2fb)))
