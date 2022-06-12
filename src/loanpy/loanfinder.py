"""
Find (old) loanwords between two languages

"""

from ast import literal_eval
from functools import partial
from logging import getLogger

from pandas import DataFrame, Series, concat, read_csv
from tqdm import tqdm
from panphon.distance import Distance

from loanpy.helpers import gensim_multiword
from loanpy.adrc import Adrc

logger = getLogger(__name__)


class NoPhonMatch(Exception):
    pass


def read_data(path2forms, adrc_col):  # explosion means explode
    """
    Reads a column with adapted or reconstructed words in a forms.csv file, \
drops empty elements, drops elements with certain keywords used by \
loanpy.adrc.Adrc.adapt and \
loanpy.adrc.Adrc.reconstruct, such as "not old", "wrong phonotactics", etc. \
Splits elements by ", " and assigns every word its own spot in the \
pandas Series which is returned. Called by loanpy.loanfinder.Search.__init__

    :param path2forms: path to CLDF's forms.csv
    :type path2forms: pathlib.PosixPath | str | None

    :param adrc_col: name of column containing predicted \
adapted or reconstructed words
    :type adrc_col: str

    :return: Series object with one word per element. \
Words can be reg-exes as well
    :rtype: pandas.core.series.Series

    :Example:

    >>> from pathlib import Path
    >>> from loanpy.loanfinder import __file__, read_data
    >>> PATH2READ_DATA = Path(__file__).parent / "tests" / \
"input_files" / "ad_read_data.csv"
    >>> read_data(PATH2READ_DATA, "col1")
    0       a
    1    blub
    1    club
    Name: col1, dtype: object

    """
    # so the class can be initiated even without path2forms
    if path2forms is None:
        return None
    # these red flags are returned by adapt() and reconstruct()
    todrop = "wrong clusters|wrong phonotactics|not old|wrong vowel harmony"
    # reading only 1 column saves RAM. Expensive calculations ahead.
    df_forms = read_csv(path2forms, encoding="utf-8",
                        usecols=[adrc_col]).fillna("")
    # drops columns with red flags
    df_forms = df_forms[~df_forms[adrc_col].str.contains(todrop)]
    # reconstructed words don't have ", " so nothing should happen there
    df_forms[adrc_col] = df_forms[adrc_col].str.split(", ")
    # explode is the pandas Series equivalent of flattening a nested list
    df_forms = df_forms.explode(adrc_col)  # one word per row
    return df_forms[adrc_col]  # a pandas Series object


def gen(iterable1, iterable2, function, prefix="Calculating", *args):
    """
    A generator that applies a function to two iterables, \
incl. tqdm-progress-bar with customisable prefix. \
Called by loanpy.loanfinder.Search.loans to calculate phonological and \
semantic distances.

    :param iterable1: The first iterable, will be zipped with \
iterable2 and and looped through.
    :type iterable1: pathlib.PosixPath | list | iterable

    :param iterable2: The second iterable, will be zipped with \
iterable1 and and looped through.
    :type iterable2: pathlib.PosixPath | list | iterable

    :param function: The function that should be applied to the elements of \
the tuples from the two zipped iterables.
    :type function: function

    :param prefix: The text that should be displayed by the progress-bar
    :type prefix: str, default="Calculating"

    :param args: positional arguments that shall be passed to the function
    :type args: type depends on requirements of function \
    passed to param <function>.

    :return: the outputs of the function passed to param <function>
    :rtype: generator object

    :Example:

    >>> from loanpy.loanfinder import gen
    >>> list(gen([1, 2, 3], [4, 5, 6], lambda x, y: x+y))
    Calculating: 100%|███████████████████████████████████| \
3/3 [00:00<00:00, 7639.90it/s]
    [5, 7, 9]

    >>> from loanpy.loanfinder import gen
    >>> list(gen([1, 2, 3], [4, 5, 6], lambda x, y, z: x+y+z, "running", 1))
    running: 100%|███████████████████████████████████| \
3/3 [00:00<00:00, 7639.90it/s]
    [6, 8, 10]
    """
    for ele1, ele2 in zip(tqdm(iterable1, prefix), iterable2):
        yield function(ele1, ele2, *args)  # can't pass kwargs!


class Search():
    """
        Define the two word lists, the measurements to \
calculate phonological distance and semantic similarity \
and the thresholds below or above which to accept matches.

        :param path2donordf: The path to forms.csv of the \
donor language containing a column of predicted adaptations into \
the recipient language.
        :type path2donordf:  pathlib.PosixPath | str | None, \
default=None

        :param path2recipdf:  The path to forms.csv of the \
recipient language, containing a column of \
predicted backward-reconstructions stored as regular expressions.
        :type path2recipdf: pathlib.PosixPath | str | None, \
default=None

        :param donorcol: The name of the column in the donor \
language's forms.csv containing a column of predicted adaptations into \
the tentative recipient language.
        :type donorcol: str, default="ad"

        :param recipcol: The name of the column in the recipient \
language's forms.csv containing a column of words in that language. When \
searching for old loanwords, this column can consist of regular \
expressions \
that represent backward reconstructions of present-day words.
        :type recipcol: str, default="rc"

        :param phondist: The maximal phonological distance between two words. \
By default, matches have to be identical.
        :type phondist: int, default=0

        :param phondist_msr: The name of the phonological distance measure, \
which has to be a method of panphon.distance.Distance
        :type phondist_msr: "doglo_prime_distance" | \
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
"levenshtein_distance", default="hamming_feature_edit_distance"

        :param semsim: The minimal semantic similarity between the \
meaning of words. By default, meanings have to be identical.
        :type semsim: int (float between -1 and 1 for gensim), default=1

        :param semsim_msr: The function with which to measure semantic \
similarity.
        :type semsim_msr: function of type func(a: str, b: str) -> int, \
        default=loanpy.helpers.gensim_multiword

        :param scdictlist_ad: list of correspondence dictionaries between \
tentative donor and recipient language generated with \
loanpy.qfysc.get_sound_corresp. Not a dictionary, therefore sequence \
important. \
Will be used in loanpy.loanfinder.Search.likeliestphonmatch to \
calculate likelihood \
(NSE) from predicted adaptation vs source word.
        :type scdictlist_ad: None | list of 6 dicts. Dicts 0, 1, 2 \
capture phonological \
correspondences, dicts 3, 4, 5 phonotactic ones. dict0/dict3: the actual \
correspondences, dict1/dict4: How often they occur in the data, \
dict2/dict5: list of \
cognates in which they occur. default=None

        :param scdictlist_rc: list of correspondence dictionaries between \
present-day language and past stage of that language generated with \
loanpy.qfysc.get_sound_corresp. Not a dictionary, therefore sequence \
important. \
Will be used in loanpy.loanfinder.Search.likeliestphonmatch to \
calculate likelihood \
(NSE) from predicted reconstruction vs source word.
        :type scdictlist_rc: None | list of 6 dicts. Dicts 0, 1, 2 \
capture phonological \
correspondences, dicts 3, 4, 5 phonotactic ones. dict0/dict3: the actual \
correspondences, dict1/dict4: How often they occur in the data, \
dict2/dict5: list of \
cognates in which they occur. default=None

        :Example:

        >>> from pathlib import Path
        >>> from loanpy.loanfinder import Search, __file__
        >>> path2rec = Path(__file__).parent / "tests" \
/ "input_files"/ "hun.csv"
        >>> path2don = Path(__file__).parent / "tests" \
/ "input_files"/ "got.csv"
        >>> path2sc_ad = Path(__file__).parent / "tests" / "input_files" / \
"sc_ad_3cogs.txt"
        >>> path2sc_rc = Path(__file__).parent / "tests" / "input_files" / \
"sc_rc_3cogs.txt"
        >>> search_obj = Search(\
path2donordf=path2don, \
path2recipdf=path2rec, \
scdictlist_ad=path2sc_ad, \
scdictlist_rc=path2sc_rc)

        How to plug in different semantic similarity measurement function, \
e.g. BERT:

        >>> from loanpy import loanfinder
        >>> from loanpy.helpers import plug_in_model
        >>> # pip install transformers==4.19.2
        >>> from sentence_transformers import SentenceTransformer
        >>> from sklearn.metrics.pairwise import cosine_similarity
        >>> plug_in_model(SentenceTransformer("bert-base-nli-mean-tokens"))
        >>> def bert_similarity(sentence1, sentence2):
        >>>     return float(\
cosine_similarity(helpers.model.encode([sentence1]), \
helpers.model.encode([sentence2])))
        >>> path2rec = Path(__file__).parent / "tests" \
        / "input_files"/ "hun.csv"
        >>> path2don = Path(__file__).parent / "tests" \
        / "input_files"/ "got.csv"
        >>> path2sc_ad = Path(__file__).parent / "tests" / "input_files" / \
        "sc_ad_3cogs.txt"
        >>> path2sc_rc = Path(__file__).parent / "tests" / "input_files" / \
        "sc_rc_3cogs.txt"
        >>> # plug in bert_similarity here into param <semsim_msr>
        >>> search_obj = Search(path2donordf=path2don, path2recipdf=path2rec, \
scdictlist_ad=path2sc_ad, scdictlist_rc=path2sc_rc, \
semsim_msr=bert_similarity)

    """
    def __init__(self, path2donordf=None, path2recipdf=None, donorcol="ad",
                 recipcol="rc",
                 phondist=0, phondist_msr="hamming_feature_edit_distance",
                 semsim=1, semsim_msr=gensim_multiword,
                 scdictlist_ad=None, scdictlist_rc=None):

        # pandas Series of predicted adapted donor words in which to search
        self.search_in = read_data(path2donordf, donorcol)
        # pd Series of reg-exes of reconstructed recipient words to search for
        self.search_for = read_data(path2recipdf, recipcol)
        # path to donor and recipient forms.csv to read extra infos later
        self.donpath, self.recpath = path2donordf, path2recipdf
        # names of the columns containing adapted and reconstructed words
        self.doncol, self.reccol = donorcol, recipcol  # used in postprocessing

        self.phondist = phondist  # maximal phonological distance of a mtach
        self.phondist_msr = getattr(Distance(), phondist_msr)  # distnc measure
        self.semsim = semsim  # minimal semantic similarity of a match
        self.semsim_msr = semsim_msr  # semantic similarity measuring function

        # normalised sum of examples for adaptions and reconstructions
        self.get_nse_ad = Adrc(scdictlist=scdictlist_ad, mode="adapt").get_nse
        self.get_nse_rc = Adrc(scdictlist=scdictlist_rc,
                               mode="reconstruct").get_nse

    def phonmatch(self, search_for, index, dropduplicates=True):
        """
        Check if a regular expression is contained \
in a wordlist and replace it with a number. \
The wordlist is a pandas Series object that gets initiated in \
loanpy.loanfinder.Search. To pass a wordlist in through the parameter \
of this function, use loanpy.loanfinder.Search.phonmatch_small

        :param search_for: The regular expression for which to search in the \
donor language.
        :type search_for: str

        :param index: The number with which to replace a match. \
        (This number will be \
        used to merge the rest of the recipient language's \
data frame, so it should represent \
        its index there.)
        :type index: idx

        :param dropduplicates: If set to True, this will drop matches \
that have the same \
        index in the wordlist \
        (There's one adapted donor-word per row, but its index \
        is the same as the original donor word's from which it was adapted. \
        Therefore, one recipient word can match with the same donor \
word through multiple \
        adaptations. Since the semantics are the same for all of \
those matches, the first match can be picked and duplicates \
dropped safely. This saves a lot of time and energy. \
Later, loanpy.loanfinder.Search.likeliestphonmatch calculates \
the likeliest phonological matches, \
but only for those phonological matches, whose semantics already matched.)
        :type dropduplicates: bool, default=True

        :return: a pandas data frame containing \
phonological matches. The index \
        indicates the position (row) of the word in the data frame assigned \
        to loanpy.loanfinder.Search.search_in. \
        The column "recipdf_idx" is intended to indicate \
        the position of the word in the word list of the recipient language. \
        It is the same value as the one passed to param <index>.
        :rtype: pandas.core.series.Series

        :Example:

        >>> from pathlib import Path
        >>> from loanpy.loanfinder import Search, __file__
        >>> path2read_data = Path(__file__).parent / "tests" / \
"input_files" / "ad_read_data.csv"
        >>> search_obj = Search(path2donordf=path2read_data, donorcol="col1")
        >>> search_obj.phonmatch(search_for="(b|c)?lub", index=99,
        >>> dropduplicates=False)
              match  recipdf_idx
        1  blub           99
        1  club           99


        """
        # maximal phonetic distance == 0 means only identical words are matches
        if self.phondist == 0:  # will drop all non-identical elements
            matched = self.search_in[self.search_in.str.match(search_for)]
        else:  # will otherwise drop everything above the max distance
            self.phondist_msr = partial(self.phondist_msr, target=search_for)
            matched = self.search_in[
                self.search_in.apply(self.phondist_msr) <= self.phondist]

        # creates new col "recipdf_idx" - keys to the input df
        dfphonmatch = DataFrame({"match": matched, "recipdf_idx": index})

        # this makes things more economical. dropping redundancies
        if dropduplicates is True:
            dfphonmatch = dfphonmatch[~dfphonmatch.index.duplicated(
                keep='first')]

        # returns a pandas data frame
        return dfphonmatch

    def loans(self, write_to=False, postprocess=False, merge_with_rest=False):
        """
        Searches for phonological matches \
and calculates their semantic similarity. Returns candidate list of loans.

        :param write_to: indicate if results should be written to file. \
If yes, provide path.
        :type write_to: pathlib.PosixPath | str | None | False, \
default=False

        :param postprocess: Indicate if results should be post-processed. See \
loanpy.loanfinder.Search.postprocess for more details
        :type postprocess: bool, default=False

        :param merge_with_rest: Indicate if additional info from input \
data frame columns should be copied into the output data frame. \
Helps with quick debugging sometimes. See \
loanpy.loanfinder.Search.merge_with_rest for more details
        :type merge_with_rest: bool, default=False

        :returns: data frame with potential loanwords
        :rtype: pandas.core.series.Series

        :Example:

        >>> from pathlib import Path
        >>> from loanpy.loanfinder import Search, __file__
        >>> from loanpy.helpers import plug_in_model
        >>> from gensim.models import word2vec
        >>> from gensim.test.utils import common_texts
        >>> in_got = path2donordf=Path(__file__).parent / "tests" / \
"input_files" / "loans_got.csv"
        >>> in_hun = path2donordf=Path(__file__).parent / "tests" / \
"input_files" / "loans_hun.csv"
        >>> search_obj = Search(in_got, in_hun, semsim=0.1)
        >>> # plug in dummy vectors, api (default) would need \
internet + a minute to load
        >>> plug_in_model(word2vec.Word2Vec(common_texts, min_count=1).wv)
        >>> search_obj.loans()
            match recipdf_idx        Meaning_x     Meaning_y gensim_multiword
         0  blub       0       computer, interface   human      0.109408


        """

        # find phonological matches
        dfmatches = concat(gen(self.search_for, self.search_for.index,
                               self.phonmatch,
                               "searching for phonological matches: "))
        # raise exception if no matches found
        if len(dfmatches) == 0:
            raise NoPhonMatch("no phonological matches found")

        # add translations for semantic comparison
        dfmatches = dfmatches.merge(read_csv(self.recpath, encoding="utf-8",
                                             usecols=["Meaning"]).fillna(""),
                                    left_on="recipdf_idx", right_index=True)
        dfmatches = dfmatches.merge(read_csv(self.donpath, encoding="utf-8",
                                             usecols=["Meaning"]).fillna(""),
                                    left_index=True, right_index=True)

        # calculate semantic similarity of phonological matches
        dfmatches[self.semsim_msr.__name__] = list(gen(dfmatches["Meaning_x"],
                                                       dfmatches["Meaning_y"],
                                                       self.semsim_msr,
                                                       "calculating semantic \
similarity of phonological matches: "))

        # sorting and cutting off words with too low semantic similarity
        logger.warning("cutting off by semsim=" +
                       str(self.semsim) +
                       "and ranking by semantic similarity")
        dfmatches = dfmatches[dfmatches[
            self.semsim_msr.__name__] >= self.semsim]
        dfmatches = dfmatches.sort_values(by=self.semsim_msr.__name__,
                                          ascending=False)

        # 3 optional extra steps indicated in params, skipped by default
        if postprocess:
            dfmatches = self.postprocess(dfmatches)
        if merge_with_rest:
            dfmatches = self.merge_with_rest(dfmatches)
        if write_to:
            dfmatches.to_csv(write_to, encoding="utf-8", index=False)
            logger.warning(f"file written to {write_to}")

        logger.warning(f"done. Insert date and time later here.")
        return dfmatches

    def postprocess(self, dfmatches):
        """
        Will replace every phonological match \
in the output data frame with its most likely version.

        :param dfmatches: The entire data frame with potential loanwords
        :type dfmatches: pandas.core.series.Series

        :returns: the same data frame but with likelier adaptations of donor \
words
        :rtype: pandas.core.series.Series

        :Example:

        >>> from pathlib import Path
        >>> from pandas import DataFrame
        >>> from loanpy.loanfinder import Search, __file__
        >>> PATH2SC_AD = Path(__file__).parent / "tests" \
/ "input_files" / "sc_ad_likeliest.txt"
        >>> PATH2SC_RC = Path(__file__).parent / "tests" \
/ "input_files" / "sc_rc_likeliest.txt"
        >>> search_obj = Search(
        >>> path2donordf=Path(__file__).parent / "tests" \
/ "input_files" / "loans_got.csv",
        >>> path2recipdf=Path(__file__).parent / "tests" / \
"input_files" / "loans_hun.csv",
        >>> scdictlist_ad=PATH2SC_AD, scdictlist_rc=PATH2SC_RC,
        >>> semsim=0.2)
        >>> dfin = DataFrame({"match": ["blub"], "recipdf_idx": [0],
        >>> "Meaning_x": ["computer, interface"],
        >>> "Meaning_y": ["human"], "semsim_msr": [0.10940766]})
        >>> search_obj.postprocess(dfin)
        postprocessing...
           recipdf_idx            Meaning_x  ...                      align_ad\
  nse_combined
        0            0  computer, interface  ...  ['b<b', 'l<l', 'u<u', 'b<b']\
        15.0


        """
        logger.warning(f"postprocessing...")
        # read in data for likeliestphonmatch, i.e. col Segments in both,
        # donor and recipient data frames
        dfmatches = dfmatches.merge(read_csv(self.recpath, encoding="utf-8",
                                             usecols=["Segments",
                                                      self.reccol]).fillna(""),
                                    left_on="recipdf_idx", right_index=True)
        dfmatches = dfmatches.merge(read_csv(self.donpath, encoding="utf-8",
                                             usecols=["Segments",
                                                      self.doncol]).fillna(""),
                                    left_index=True, right_index=True)
        dfmatches["Segments_x"] = [i.replace(" ", "")
                                   for i in dfmatches["Segments_x"]]
        dfmatches["Segments_y"] = [i.replace(" ", "")
                                   for i in dfmatches["Segments_y"]]
        # calculate likeliest phonological matches
        newcols = concat([self.likeliestphonmatch(ad, rc, segd, segr)
                          for ad, rc, segd, segr
                          in zip(dfmatches[self.doncol],
                                 dfmatches[self.reccol],
                                 dfmatches["Segments_y"],
                                 dfmatches["Segments_x"])])
        del dfmatches["match"]  # delete non-likeliest matches
        newcols.index = dfmatches.index  # otherwise concat wont work

        dfmatches = concat([dfmatches, newcols], axis=1)  # add new cols
        # delete redundant data
        del (dfmatches["Segments_x"], dfmatches[self.reccol],
             dfmatches["Segments_y"], dfmatches[self.doncol])

        return dfmatches  # same structure as input df

    def likeliestphonmatch(self, donor_ad, recip_rc, donor_segment,
                           recip_segment):
        """
        Called by loanpy.loanfinder.postprocess. \
Calculates the nse of recip_rc-recip_segment \
and donor_ad-donor_segment, adds them together \
and picks the word pair with the highest sum. \
Adds 2*4 columns from loanpy.adrc.Adrc.get_nse.

        :param donor_ad: adapted words in the donor data frame
        :type donor_ad: str (not a regular expression, words separated by ", ")

        :param recip_rc: a reconstructed word
        :type recip_rc: str (regular expression)

        :param donor_segment: the original (non-adapted) donor word
        :type donor_segment: str

        :param recip_segment: the original (non-reconstructed) recipient word
        :type recip_segment: str

        :returns: The likeliest phonological match
        :rtype: pandas.core.series.Series

        :Example:

        >>> from pathlib import Path
        >>> from pandas import DataFrame
        >>> from loanpy.loanfinder import Search, __file__
        >>> PATH2SC_AD = Path(__file__).parent / "tests" \
/ "input_files" / "sc_ad_likeliest.txt"
        >>> PATH2SC_RC = Path(__file__).parent / "tests" \
/ "input_files" / "sc_rc_likeliest.txt"
        >>> PATH2READ_DATA = Path(__file__).parent / "tests" \
/ "input_files" / "ad_read_data.csv"
        >>> search_obj = Search(
        >>> PATH2READ_DATA, donorcol="col1",
        >>> scdictlist_ad=PATH2SC_AD, scdictlist_rc=PATH2SC_RC)
        >>> search_obj.likeliestphonmatch(donor_ad="a, blub, \
club", recip_rc="(b|c)?lub",
        >>> donor_segment="elub", recip_segment="dlub")
            match  nse_rc  se_rc  ...           distr_ad\
            align_ad  nse_combined
            0  blub    10.0     50  ...  [0, 0, 10, 10, 0]  \
            ['e<V', 'C<b', 'l<l', 'u<u', 'b<b']          14.0
            [1 rows x 10 columns]


        """
        # step 1: serach for phonological matches between
        # reconstructed reg-ex and list of predicted adaptations
        dfph = self.phonmatch_small(Series(donor_ad.split(", "), name="match"),
                                    recip_rc, dropduplicates=False)
        # get the nse score between original and predictions
        # and write to new columns
        # cols se_rc, lst_rc, se_ad, lst_ad are just extra info for the user
        dfph = DataFrame([(wrd,) + self.get_nse_rc(recip_segment, wrd) +
                          self.get_nse_ad(donor_segment, wrd)
                          for wrd in dfph["match"]],
                         columns=["match", "nse_rc", "se_rc", "distr_rc",
                                  "align_rc", "nse_ad", "se_ad", "distr_ad",
                                  "align_ad"])
        # add combined nse
        dfph["nse_combined"] = dfph["nse_rc"] + dfph["nse_ad"]
        # get idx of max combined, keep only that idx (=likeliest match)
        dfph = dfph[dfph.index == dfph["nse_combined"].idxmax()]

        return dfph

    def phonmatch_small(self, search_in, search_for, index=None,
                        dropduplicates=True):

        """
        Same as loanpy.loanfinder.Search.phonmatch but search_in \
has to be added as a parameter. Found this \
to be the most elegant solution b/c \
loanpy.loanfinder.Search.likeliestphonmatch() inputs lots of \
small and very different search_in-dfs, while loans() inputs one big df.

        :param search_in: The iterable to search within
        :type search_in: pandas.core.series.Series

        :param search_for: See loanpy.loanfinder.Search.phonmatch
        :type search_for: str

        :param index: See loanpy.loanfinder.Search.phonmatch
        :type index: str | None, default=None

        :param dropduplicates: See loanpy.loanfinder.Search.phonmatch
        :type dropduplicates: bool, default=True

        :returns: See loanpy.loanfinder.Search.phonmatch
        :rtype: pandas.core.series.Series

        """
        # for inline comments see loanpy.loanfinder.Search.phonmatch
        if self.phondist == 0:
            matched = search_in[search_in.str.match(search_for)]
        else:
            self.phondist_msr = partial(self.phondist_msr, target=search_for)
            matched = search_in[
                search_in.apply(self.phondist_msr) <= self.phondist]

        dfphonmatch = DataFrame({"match": matched, "recipdf_idx": index})

        if dropduplicates is True:
            dfphonmatch = dfphonmatch[
                ~dfphonmatch.index.duplicated(keep='first')]
        return dfphonmatch

    def merge_with_rest(self, dfmatches):
        """
        Merges the output data frame with the remaining columns \
from both input data frames. This helps to inspect results quickly manually.

        :param dfmatches: The output data frame
        :type dfmatches: pandas.core.frame.DataFrame

        :returns: same data frame with extra cols added from both \
input forms.csv
        :rtype: pandas.core.frame.DataFrame
        """
        logger.warning("Merging with remaining columns from input data frames")
        # avoid duplicates
        dfmatches = dfmatches.drop(["Meaning_x", "Meaning_y"], axis=1)
        dfmatches = dfmatches.merge(read_csv(self.donpath,
                                             encoding="utf-8").fillna(""),
                                    left_index=True, right_index=True)
        dfmatches = dfmatches.merge(read_csv(self.recpath,
                                             encoding="utf-8").fillna(""),
                                    left_on="recipdf_idx", right_index=True)
        dfmatches = dfmatches.sort_values(by=self.semsim_msr.__name__,
                                          ascending=False)  # unsorted by merge
        return dfmatches
