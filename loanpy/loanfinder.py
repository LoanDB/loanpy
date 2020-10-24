"""
takes two wordlists as input, one in the recipient language (L1) \
and one in the tentative donor language (L2) and returns an excel-sheet with \
a list of candidates for potential loanwords.
"""

import os

import pandas as pd

from loanpy.adapter import adapt, launch as launch_adapter
from loanpy.helpers import (filterdf, gensim_similarity, loadvectors,
                            progressbar)
from loanpy.reconstructor import (getnse, launch as launch_reconstructor,
                                  reconstruct)

os.chdir(os.path.join(os.path.dirname(__file__), "data"))


def adapt_or_reconstruct_col(inputcsv, inputcol, funcname,
                             howmany, struc, vowelharmony,
                             only_documented_clusters=False,
                             substicsv="substi.csv",
                             dfetymology="dfuralonet.csv",
                             timelayer="", soundchangedict="scdict.txt",
                             se_or_edict="sedict.txt",
                             write=True, outputcsv="", outputcol=""):
    """
    Adapt or reconstruct IPA-strings in a csv-column.

    :param inputcsv:
        The name of the file that contains the input column.
    :type inputcsv: str, e.g. "dfhun.csv" or "dfgot.csv".

    :param inputcol:
        The name of the column that contains the input words.
    :type inputcol: str, e.g. "L1_ipa" or "L2_ipa".

    :param funcname:
        The name of the function that should be applied.
    :type funcname: {'adapt', 'reconstruct'}

    :param howmany:
        The desired number of reconstructions or adaptions per word.
    :type howmany: int

    :param struc:
        Accepts only reconstructions that don't violate phonotactic \
constraints and adapts words phonotactically if True.
    :type struc: bool

    :param vowelharmony:
        Accepts only reconstructions that do not violate constraints \
related to vowel harmony and adapts words according to vowel harmony if True.
    :type vowelharmony: bool

    :param only_documented_clusters:
        Reconstructions and adaptions will contain only consonant and vowel \
clusters that are documented in the etymological dictionary of L1 if True.
    :type only_documented_clusters: bool, default=False

    :param substicsv:
        The name of the csv-file in which the sound substitutions for \
adaptation are stored.
    :type substicsv:  str, default="substi.csv"

    :param dfetymology:
        The name of the csv-file containing the etymological dictionary of L1 \
from which a list of allowed phonotactic structures will be extracted.
    :type dfetymology: str, default="dfuralonet.csv"

    :param timelayer:
        The historical layer from which to extract a list of allowed \
phonotactic structures. <U> means 'Uralic', <FU> 'Finno-Ugric' and \
<Ug> 'Ugric'. By default, all three layers \
are taken together into consideration.
    :type timelayer: {"", "U", "FU", "Ug"}, default=""

    :param soundchangedict: str
        A dictionary containing sound changes from modern into proto-L1. \
Created with loanpy.reconstructor.dfetymology2dict. \
This is the basis for reconstructions.
    :type soundchangedict: default="scdict.txt"

    :param se_or_edict:
        A dictionary containing information on how often \
or in which exact words each \
sound change occurs in the etymological dictionary of L1 \
(sum of examples, examples). Created with \
loanpy.reconstructor.dfetymology2dict. \
This is the basis for calculating the normalised \
sum of examples (nse), the sum of examples (se), \
examples (e) and list of number of examples (lne).
    :type se_or_edict: str, default="sedict.txt"

    :param write:
        If True, output will be written to a column in a csv \
and None will be returned. If False, output will be returned.
    :type write: bool, default=True

    :param outputcsv:
        The name of the file to which the output column should be written. \
By default, the output file is identical with the input file.
    :type outputcsv: str, default=""

    :param outputcol:
        The name of the column to which \
the output word list should be written. By default, \
the column name is identical to parameter <funcname>, \
that is "adapt" or "reconstruct".
    :type outputcol: str, default=""

    :return: List of strings representing reconstructed or adapted words.
    :rtype: {None if write=True, list if write=False}

    :raises Exception: If parameter funcname is something else than \
"adapt" or "reconstruct"

    :Example:

    >>> from loanpy import loanfinder as lf
    >>> lf.adapt_or_reconstruct_col(
            inputcsv="dfhun_zaicz_before1600.csv", inputcol="L1_ipa",
            funcname="reconstruct", howmany=float("inf"), struc=False,
            vowelharmony=False, only_documented_clusters=False,
            write=True, outputcsv="example_dfhun_before1600.csv",
            outputcol="example_rc")
    View results in data/example_dfhun_before1600.csv

    >>> from loanpy import loanfinder as lf
    >>> lf.adapt_or_reconstruct_col(
            inputcsv="dfgot_wikiling_backup.csv", inputcol="L2_ipa",
            funcname="adapt", struc=True,howmany=1,
            only_documented_clusters=False, vowelharmony=True,
            write=True,
            outputcsv="example_dfgot.csv", outputcol="example_ad")
    View results in data/example_dfgot.csv

    """

    df = pd.read_csv(inputcsv, encoding="utf-8")
    newcol = []

    if funcname == "adapt":
        launch_adapter(substicsv=substicsv, dfetymology=dfetymology,
                       timelayer=timelayer, soundchangedict=soundchangedict)

        newcol = [(adapt(ipastring=i, howmany=howmany, struc=struc,
                         vowelharmony=vowelharmony,
                         only_documented_clusters=only_documented_clusters))
                  for i in progressbar(df[inputcol], prefix="adapting: ")]

    elif funcname == "reconstruct":
        launch_reconstructor(soundchangedict=soundchangedict,
                             dfetymology=dfetymology,
                             timelayer=timelayer,
                             se_or_edict=se_or_edict)

        newcol = [reconstruct(ipastring=i, howmany=howmany, struc=struc,
                              vowelharmony=vowelharmony,
                              only_documented_clusters=only_documented_clusters)
                  for i
                  in progressbar(df[inputcol], prefix="reconstructing: ")]
    else:
        raise Exception("parameter <funcname> must be \
either 'adapt' or 'reconstruct'")

    if write is True:
        if outputcol == "":
            outputcol = funcname
        if outputcsv == "":
            outputcsv = inputcsv
        df[outputcol] = newcol
        df.to_csv(outputcsv, encoding="utf-8", index=False)

    else:
        return newcol


def findphoneticmatches(root, index, dropduplicates=True):
    """
     Find phonetic matches in a list of L2 words.

    :param root:
        An L1 word to search for in a list of L2 words. \
Can be a regular expression.
    :type root: str

    :param index: The number with which to replace a match. \
This number will be \
used to merge the rest of the L1 data frame, so it should represent \
the index in the L1 data frame. Subtract 2 from the index that Excel shows \
because Python starts counting at zero and does not count headers.
    :type index: str

    :param dropduplicates: Every adapted L2-word is in one row, but its index \
is the same as the original L2 word from which it was adapted. \
Therefore one L1 word can match with the same L2 word through multiple \
adaptations. Since the semantics stay the same it is legitimate to \
drop duplicates. For a more precise search, e.g. to find out \
which of all possible matches has the highest nse, set dropduplicates=False
    :type dropduplicates: bool, default=True

    .. note:: The L2 words to search in are stored in a pandas series as a \
global variable called <dfL2> that gets defined in \
loanpy.loanfinder.findloans. \
It is possible to define it manually as well, as shown in the example.

    :return: a pandas data frame containing phonetic matches. The index \
indicates the position of the word in the L2 word list. \
The column "L1_idx" indicates \
the position of the word in the L1 word list.
    :rtype: pandas.core.frame.DataFrame

    :Example:

    >>> import pandas as pd
    >>> from loanpy import loanfinder as lf
    >>> lf.dfL2 = pd.read_csv("dfgot_wiktionary_backup.csv",
                               encoding="utf-8")["L2_latin"]
    >>> lf.findphoneticmatches("^anna$", 5)
    +---+----------------+-----------+
    |   |L2_latin_match  |  L1_idx   |
    +---+----------------+-----------+
    |288| anna           |     5     |
    +---+----------------+-----------+

    >>> import pandas as pd
    >>> from loanpy import loanfinder as lf
    >>> lf.dfL2 = pd.read_csv("dfgot_wiktionary_backup.csv",
                               encoding="utf-8")["L2_latin"]
    >>> lf.findphoneticmatches("^abraham$|^anna$", 123)
    +---+----------------+-----------+
    |   |L2_latin_match  |  L1_idx   |
    +---+----------------+-----------+
    |6  |      abraham   |     123   |
    +---+----------------+-----------+
    |288|      anna      |     123   |
    +---+----------------+-----------+

    >>> import pandas as pd
    >>> from loanpy import loanfinder as lf
    >>> lf.dfL2 = pd.read_csv("dfgot_wiktionary_backup.csv",
                               encoding="utf-8")["L2_latin"]
    >>> lf.findphoneticmatches("^a(nn|br)a(ham)?$", 5)
    +---+----------------+-----------+
    |   |L2_latin_match  |  L1_idx   |
    +---+----------------+-----------+
    |6  |      abraham   |     5     |
    +---+----------------+-----------+
    |288|      anna      |     5     |
    +---+----------------+-----------+

    """

    dfphonmatch = pd.DataFrame()
    dfphonmatch[dfL2.name+"_match"] = dfL2[dfL2.str.match(root)]
    dfphonmatch["L1_idx"] = index
    if dropduplicates is True:
        dfphonmatch = dfphonmatch[~dfphonmatch.index.duplicated(keep='first')]
    return dfphonmatch


def gen(iterable1, iterable2, function, prefix, *args):
    for ele1, ele2 in zip(progressbar(iterable1, prefix), iterable2):
        yield function(ele1, ele2, *args)


def findloans(sheetname="new", L1="dfhun_zaicz_before1600.csv",
              L2="dfgot_wikiling.csv",
              L1col="hmInf_strF_vhF_onlyF", L2col="hm1_strT_vhT_onlyF",
              semantic_similarity=gensim_similarity,
              wordvectors="glove-wiki-gigaword-50",
              cutoff=100, write=True,
              sedictname="sedict.txt", edictname="edict.txt"):
    """
    Search for phonetic matches and rank them according to semantic similarity.

    :param sheetname:
        The name of the Excel worksheet in results.xlsx to which to write the \
results. Maximum 31 characters long and \\\/\*\?\:\[\] must be excluded.
    :type sheetname: str, default="new"

    :param L1:
        The name of the input-csv that contains the input column for L1.
    :type L1: str, default="dfhun_zaicz_before1600.csv"

    :param L2:
        The name of the input-csv that contains the input column for L2.
    :type L2: str, default="dfgot_wikiling.csv"

    :param L1col:
        The name of the input column that contains the input words for L1.
    :type L1col: str, default="hmInf_strF_vhF_onlyF"

    :param L2col:
        The name of the input column that contains the input words for L2.
    :type L2col: str, default="hm1_onlyF_strF"

    :param semantic_similarity:
        Indicate which function should be used to \
calculate semantic similarities.
    :type semantic_similarity: function, default=gensim_similarity

    :param wordvectors:
        Indicate which word vectors *gensim* should use for calculating \
semantic similarity. Find a table of pretrained models at \
https://github.com/RaRe-Technologies/gensim-data (last access 09.apr 2021) \
or in data/wordvectornames.xlsx
    :type wordvectors: str, default="glove-wiki-gigaword-50"

    :param cutoff:
        Indicate how many of the semantically most similar phonetic matches \
should be returned in the final result.
    :type cutoff: int, default=100

    :param write:
        If True, output will be written to a results.xlsx \
and None will be returned. If False, output will be returned.
    :type write: bool, default=True

    :param sedictname:
        Indicate the name of the file containing the sum of examples \
for each sound change in the etymological dictionary of L1. This file is \
generated by loanpy.reconstructor.dfetymology2dict
    :type sedictname: str, default="sedict.txt"

    :param edictname:
        Indicate the name of the file containing the examples for each sound \
change in the etymological dictionary of L1. This file is generated by \
loanpy.reconstructor.dfetymology2dict
    :type edictname: str, default="edict.txt"

    :return: Phonetic matches ranked according to \
semantic similarity
    :rtype: {None if write=True, pandas.core.frame.DataFrame if write=False}

    :raises Exception: If sheetname contains \\\/\*\?\:\[\] because \
Excel can not handle those
    :raises Exception: If sheetname is more than 31 characters long
    :raises Exception: If no phonetic matches were found

    :Example:

    >>> from loanpy import loanfinder as lf
    >>> lf.findloans("example", "example_dfhun_before1600.csv",
                     "example_dfgot.csv", "example_rc", "example_ad")
    view results in data/results.xlsx on sheet "example"

    """

    print("remember to close results.xlsx")
    if "\\\/\*\?\:\[\]" in sheetname and write is True:
        raise Exception("sheetname must exclude \\\/\*\?\:\[\]")
    if len(sheetname) > 31 and write is True:
        raise Exception("sheetname must be less than 32 characters")

    global dfL2  # input for findphoneticmatches(), must be global
    dfL2 = pd.read_csv(L2, encoding="utf-8", usecols=[L2col])  # save RAM
    dfL2 = dfL2[~dfL2[L2col].str.contains(" not in substi.csv|\
every substituted word contains \
at least one cluster undocumented in proto-L1")]
    dfL2[L2col] = dfL2[L2col].str.split(", ")
    dfL2 = dfL2.explode(L2col)  # one word per row
    dfL2 = dfL2[L2col]  # pd series

    dfL1 = pd.read_csv(L1, encoding="utf-8")
    dfL1 = dfL1[~dfL1[L1col].str.contains(" not old|wrong phonotactics|\
wrong vowel harmony")]  # this way the original index is preserved
    dfL1 = dfL1[L1col]

    dfmatches = pd.DataFrame()  # finding phonetic matches
    for i in gen(dfL1, dfL1.index, findphoneticmatches,
                 "searching for phonetic matches: "):
        dfmatches = dfmatches.append(i)

    if len(dfmatches) == 0:
        raise Exception("no phonetic matches found")

    dfmatches = dfmatches.merge(pd.read_csv(L1, encoding="utf-8",
                                            usecols=["L1_en"]),
                                left_on="L1_idx", right_index=True)
    dfmatches = dfmatches.merge(pd.read_csv(L2, encoding="utf-8",
                                            usecols=["L2_en"]),
                                left_index=True, right_index=True)

    loadvectors(wordvectors)
    dfmatches[semantic_similarity.__name__] = list(gen(dfmatches["L1_en"],
                                                       dfmatches["L2_en"],
                                                       semantic_similarity,
                                                       "calculating \
semantic similarity of phonetic matches: "))

    dfmatches = dfmatches.sort_values(by=semantic_similarity.__name__,
                                      ascending=False).head(cutoff)
    dfmatches = dfmatches.drop(["L1_en", "L2_en"],
                               axis=1)  # to avoid duplicates after merging
    dfmatches = dfmatches.merge(pd.read_csv(L2, encoding="utf-8"),
                                left_index=True, right_index=True)
    dfmatches = dfmatches.merge(pd.read_csv(L1, encoding="utf-8"),
                                left_on="L1_idx", right_index=True)
    dfmatches = dfmatches.sort_values(by=semantic_similarity.__name__,
                                      ascending=False)

    launch_reconstructor(se_or_edict=sedictname)
    for colname, ex, norm in zip(["nse", "se", "lne", "e"],
                                 [False, False, True, True],
                                 [True, False, False, True]):
        if colname == "e":
            launch_reconstructor(se_or_edict=edictname)
        dfmatches[colname] = list(gen(dfmatches["L1_ipa"],
                                      dfmatches[L2col+"_match"],
                                      getnse,
                                      "calculating "+colname+" of \
semantically most similar phonetic matches: ",
                                      ex, norm))

    if write is True:
        dfdict = pd.read_excel("results.xlsx", sheet_name=None,
                               engine="openpyxl")
        dfdict[sheetname] = dfmatches  # add dfmatches to the dictionary
        writer = pd.ExcelWriter("results.xlsx", engine="openpyxl")
        for sheet_name in dfdict.keys():  # write dictionary to results.xlsx
            dfdict[sheet_name].to_excel(writer, sheet_name=sheet_name,
                                        index=False, engine="openpyxl")
        writer.save()

    else:
        return dfmatches
