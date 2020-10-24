"""
Reconstruct words from L1 into proto-L1.
"""

import ast
from collections import Counter
from itertools import product
import math
import os
import re

from ipatok import tokenise
import pandas as pd

from loanpy.helpers import harmony, ipa2clusters, list2regex, word2struc

cns = 'jwʘǀǃǂǁk͡pɡ͡bcɡkqɖɟɠɢʄʈʛbb͡ddd̪pp͡ttt̪ɓɗb͡βk͡xp͡ɸq͡χɡ͡ɣɢ͡ʁc͡çd͡ʒt͡ʃɖ͡ʐɟ͡ʝʈ͡ʂb͡vd̪͡z̪d̪͡ðd̪͡ɮ̪d͡z\
d͡ɮd͡ʑp͡ft̪͡s̪t̪͡ɬ̪t̪͡θt͡st͡ɕt͡ɬxçħɣʁʂʃʐʒʕʝχfss̪vzz̪ðɸβθɧɕɬɬ̪ɮʑɱŋɳɴmnn̪ɲʀʙʟɭɽʎrr̪ɫɺɾhll̪ɦðʲt͡ʃʲnʲ\
ʃʲlʲCl̥m̥n̥r̥jːwːʘːǀːǃːǂːǁːk͡pːɡ͡bːcːɡːkːqːɖːɟːɠːɢːʄːʈːʛːbːb͡dːdːd̪ːpːp͡tːtːt̪ːɓːɗːb͡βːk͡xː\
p͡ɸːq͡χːɡ͡ɣːɢ͡ʁːc͡çːd͡ʒːt͡ʃːɖ͡ʐːɟ͡ʝːʈ͡ʂːb͡vːd̪͡z̪ːd̪͡ðːd̪͡ɮ̪ːd͡zːd͡ɮːd͡ʑːp͡fːt̪͡s̪ːt̪͡ɬ̪ːt̪͡θːt͡sːt͡ɕːt͡ɬːxːçːħː\
ɣːʁːʂːʃːʐːʒːʕːʝːχːfːsːs̪ːvːzːz̪ːðːɸːβːθːɧːɕːɬːɬ̪ːɮːʑːɱːŋːɳːɴːmːnːn̪ːɲːʀːʙːʟːɭːɽː\
ʎːrːr̪ːɫːɺːɾːhːlːl̪ːɦːðʲːt͡ʃʲːnʲːʃʲːlʲːCːl̥ːm̥ːn̥ːr̥ː'
vow = 'ɑɘɞɤɵʉaeiouyæøœɒɔəɘɵɞɜɛɨɪɯɶʊɐʌʏʔɥɰʋʍɹɻ¨ȣȣ̈ɑːɘːɞːɤːɵːʉːaːeːiːoːuːyːæːøːœː\
ɒːɔːəːɘːɵːɞːɜːɛːɨːɪːɯːɶːʊːɐːʌːʏːʔːɥːɰːʋːʍːɹːɻːȣ̈ːFBVͽ'
front = 'jcɖɟʄʈbb͡ddd̪pp͡ttt̪ɓɗc͡çd͡ʒt͡ʃɟ͡ʝb͡vd͡zd͡ʑp͡ft͡st͡ɕçʂʃʐʒʝfss̪vzz̪ðɸβθɕʑɳmnn̪ɲʎhll̪ɦɘɞ\
ɵʉaeiyæøœɛɪɶʏʔɥɻ¨ȣ̈jːcːɖːɟːʄːʈːbːb͡dːdːd̪ːpːp͡tːtːt̪ːɓːɗːc͡çːd͡ʒːt͡ʃːɟ͡ʝːb͡vːd͡zːd͡ʑːp͡fːt͡sː\
t͡ɕːçːʂːʃːʐːʒːʝːfːsːs̪ːvːzːz̪ːðːɸːβːθːɕːʑːɳːmːnːn̪ːɲːʎːhːlːl̪ːɦːɘːɞːɵːʉːaːeːiːyːæːøː\
œːɛːɪːɶːʏːʔːɥːɻːȣ̈ːF'
back = 'wɡkqɠɢʛq͡χɢ͡ʁxħɣʁʕχŋɴʀɫɑɤouɒɔəɘɵɞɜɨɯʊɐʌʍɹȣwːɡːkːqːɠːɢːʛːq͡χːɢ͡ʁːxːħːɣːʁːʕːχː\
ŋːɴːʀːɫːɑːɤːoːuːɒːɔːəːɘːɵːɞːɜːɨːɯːʊːɐːʌːʍːɹːB'
nsedict = {}

os.chdir(os.path.join(os.path.dirname(__file__), "data"))


def launch(soundchangedict="scdict.txt", dfetymology="dfuralonet.csv",
           timelayer="", se_or_edict="sedict.txt"):
    """
    Define global variables necessary for loanpy.loanfinder.reconstruct

    :param soundchangedict:
        The dictionary of sound changes of L1, extracted from the \
etymological dictionary by loanpy.reconstructor.dfetymology2dict. Needed for \
loanpy.reconstructor.reconstruct. If file doesn't exist yet, set \
soundchangedict=""
    :type soundchangedict:  str, default="scdict.txt"

    :param dfetymology:
         The name of the etymological dictionary of L1 that is \
stored in a csv-file with a column named "Old", containing the reconstructed \
forms and a column named "New", containing the modern forms.
    :type dfetymology: str, default="dfuralonet.csv"

    :param timelayer:
        Indicate from which historical layer sound changes \
should be extracted: \
Uralic ('U'), Finno-Ugric ('FU'), or Ugric ('Ug'). \
By default all three are taken into consideration simultaneously. \
Information about these layers must be stored in a column named "Lan" \
in the etymological dictionary.
    :type timelayer:  {"", "U", "FU", "Ug"}, default=""

    :param se_or_edict: name the dictionary necessary for \
loanpy.reconstructor.getnse. If file doesn't exist yet, set \
se_or_edict=""
    :type se_or_edict:
        str, default="sedict.txt", e.g. "edict.txt",

    :return: None, but defines global variables, explained below
    :rtype: NoneType

    :var scdict: dictionary containing information about sound changes \
from modern-L1 into proto-L1. Keys are modern-day L1 phonemes and phoneme \
clusters, values are a list of their possible reconstruction, ordered \
according to their likelihood.
    :vartype scdict: dictionary. Keys: str, values: list of str
    :var allowedphonotactics: set of allowed phonotactic structures. Will \
only be defined if not soundchangedict="".
    :vartype allowedphonotactics: global, set
    :var nsedict: dictionary with sound changes as keys and their sum of \
examples or examples as values. Will only be defined if not se_or_edict="".
    :vartype nsedict: global, dictionary, Keys: str, Values: int or list of str

    :Example:

    >>> from loanpy import reconstructor as rc
    >>> rc.launch()
    >>> rc.scdict
    {'#0': ['0', 's', 'j', 'ʃ', 'w', 'θ', 'm'],
     '#aː': ['ɑ', 'o', 'ȣ'],
     '#b': ['p'],
     ...}

    >>> from loanpy import reconstructor as rc
    >>> rc.launch()
    >>> rc.allowedphonotactics
    {'VCCVCV',
     'CVCCVCV',
     'CVCVCV',
     'CVCCV',
     'CVCCVCCV',
     'CVCVCC',
     'VCCVCVCV',
     'VCVCV',
     'CV',
     'VCCV',
     'CVCV',
     'V',
     'CVCCCV',
     'VCV',
     'VCCCV',
     'CVCVCCV'}

    >>> from loanpy import reconstructor as rc
    >>> rc.launch()
    >>> rc.nsedict
    {'#0<*0': 429,
     '#0<*j': 6,
     '#0<*m': 1,
     ...}

    >>> from loanpy import reconstructor as rc
    >>> rc.launch(se_or_edict="edict.txt")
    >>> rc.nsedict
    {'#0<*0': ['ɛ<*ȣ̈',
     'eːɡ<*æŋͽ',
     'ɒz<*o',
    ...}
    """

    dfety = pd.read_csv(dfetymology, encoding="utf-8")
    if timelayer != "":
        dfety = dfety[dfety.Lan == timelayer].reset_index(drop=True)
    global allowedphonotactics
    allowedphonotactics = set([word2struc(i) for i in dfety["Old"]])

    with open(soundchangedict, "r", encoding="utf-8") as f:
        global scdict
        scdict = ast.literal_eval(f.read())

    if se_or_edict != "":
        with open(se_or_edict, "r", encoding="utf-8") as f:
            global nsedict
            nsedict = ast.literal_eval(f.read())


def getsoundchanges(reflex, root):  # requires two ipastrings as input
    """
    Takes a modern-day L1 word and its reconstructed form and returns \
    a table of sound changes.

    :param reflex: a modern-day L1-word
    :type reflex: str

    :param root: a reconstructed proto-L1 word
    :type root: str

    :return: table of sound changes
    :rtype: pandas.core.frame.DataFrame

    :Example:

    >>> from loanpy import reconstructor as rc
    >>> rc.getsoundchanges("ɟɒloɡ", "jɑlkɑ")
    +---+--------+------+
    | # | reflex | root |
    +---+--------+------+
    | 0 | #0     | 0    |
    +---+--------+------+
    | 1 | #ɟ     | j    |
    +---+--------+------+
    | 2 | ɒ      | ɑ    |
    +---+--------+------+
    | 3 | l      | lk   |
    +---+--------+------+
    | 4 | o      | ɑ    |
    +---+--------+------+
    | 5 | ɡ#     | 0    |
    +---+--------+------+
    """

    reflex = ipa2clusters(reflex)
    root = ipa2clusters(root)

    reflex[0], reflex[-1] = "#" + reflex[0], reflex[-1] + "#"
    reflex, root = ["#0"] + reflex, ["0"] + root

    if reflex[1][1:] in vow and root[1] in cns:
        root = root[1:]
    elif reflex[1][1:] in cns and root[1] in vow:
        reflex = reflex[1:]

    diff = abs(len(root) - len(reflex))  # "a,b","c,d,e,f,g->"a,b,000","c,d,efg
    if len(reflex) < len(root):
        reflex += ["0#"]
        root = root[:-diff] + ["".join(root[-diff:])]
    elif len(reflex) > len(root):
        root += ["0"]
        reflex = reflex[:-diff] + ["".join(reflex[-diff:])]
    else:
        reflex, root = reflex + ["0#"], root + ["0"]

    return pd.DataFrame({"reflex": reflex, "root": root})


def dfetymology2dict(dfetymology="dfuralonet.csv",
                     timelayer="",
                     name_soundchangedict="scdict",
                     name_sumofexamplesdict="sedict",
                     name_listofexamplesdict="edict"):
    """
    Convert an etymological dictionary to a dictionary of sound changes, \
sum of examples, and examples.

    :param dfetymology:
        The name of the etymological dictionary of L1 that is \
stored in a csv-file with a column named "Old", containing the reconstructed \
forms and a column named "New", containing the modern forms.
    :type dfetymology: str, default="dfuralonet.csv"

    :param timelayer:
        Indicate from which historical layer sound changes \
should be extracted, Uralic ('U'), Finno-Ugric ('FU'), or Ugric ('Ug'). \
By default all three are taken into consideration simultaneously.
    :type timelayer:  {"", "U", "FU", "Ug"}, default=""

    :param name_soundchangedict:
        Name of the first output file, without \
the file-extension.
    :type name_soundchangedict: str, default="scdict"

    :param name_sumofexamplesdict:
        Name of the second output file, without \
the file-extension.
    :type name_sumofexamplesdict: str, default="sedict"

    :param name_listofexamplesdict:
        Name of the third output file, without \
the file-extension.
    :type name_listofexamplesdict: str, default="edict"

    :return:
        Three dictionaries containing information about sound changes, \
sum of examples, and examples.
    :rtype: tuple of three dictionaries, keys: str, values: int or list of str

    :Example:

    >>> from loanpy import reconstructor as rc
    >>> rc.dfetymology2dict()
    ({'#0': ['0', 's', 'j', 'ʃ', 'w', 'θ', 'm'],
     '#aː': ['ɑ', 'o', 'ȣ'],
     '#b': ['p'],
     ...)
     view results in data\\scdict.txt, sedict.txt and edict.txt

    """
    dfety = pd.read_csv(dfetymology, encoding="utf-8")
    if timelayer != "":
        dfetym = dfetym[dfetym.Lan == timelayer].reset_index(drop=True)

    dfsoundchange = pd.DataFrame()
    for new, old in zip(dfety["New"], dfety["Old"]):
        dfsoundchange = dfsoundchange.append(getsoundchanges(new, old).
                                             assign(wordchange=new+"<*"+old))
    dfsoundchange["e"] = 1
    dfsoundchange["soundchange"] = dfsoundchange["reflex"] + "<*" +\
        dfsoundchange["root"]

    dfsc = dfsoundchange.groupby("reflex")["root"].\
        apply(lambda x:
              list(dict.fromkeys([n for n, count in
                                  Counter(x).most_common() for i in
                                  range(count)]))).reset_index()
    dfse = dfsoundchange.groupby("soundchange")["e"].sum().reset_index()
    dfe = dfsoundchange.groupby("soundchange")["wordchange"].\
        apply(list).reset_index()

    scdict = dict(zip(dfsc["reflex"], dfsc["root"]))
    sedict = dict(zip(dfse["soundchange"], dfse["e"]))
    edict = dict(zip(dfe["soundchange"], dfe["wordchange"]))

    for dictname, dictionary in\
        zip([name_soundchangedict, name_sumofexamplesdict,
             name_listofexamplesdict], [scdict, sedict, edict]):
        with open(dictname + ".txt", "w", encoding="utf-8") as data:
            data.write(str(dictionary))

    return scdict, sedict, edict


def getnse(reflex, root, examples=False, normalise=True):
    """
    Takes a modern-day L1 word and a reconstruction into proto-L1 \
and assesses the likelihood of the etymology.

    :param reflex:
        A modern-day L1 word
    :type reflex: str

    :param root:
        The reconstructed proto-L1 form. This information \
is stored in etymological dictionaries and is either documented through \
old corpora or reconstructed with the historical-comparative method.
    :type root: str

    :param examples:
        Indicate whether all the examples in which the \
given sound change occurs should be listed up.
    :type examples: bool, default=False

    :param normalise:
        Indicate whether the sum of examples should be \
divided through the number of sound changes in the word.
    :type normalise: bool, default=True

    :return: Normalised sum of examples (nse) with default settings. \
Sum of examples if normalise=False, examples or list of number \
of examples if examples=True.
    :rtype: int or list of str or list of int

    :Example:

    >>> from loanpy import reconstructor as rc
    >>> rc.launch()
    >>> rc.getnse("ɟɒloɡ","jɑlkɑ")
    83.66666666666667

    >>> from loanpy import reconstructor as rc
    >>> rc.launch(se_or_edict="edict.txt")
    >>> rc.getnse("ɟɒloɡ", "jɑlkɑ", examples=True)
    [['ɛ<*ȣ̈',
    'eːɡ<*æŋͽ',
    'ɒz<*o',
    ...]

    >>> from loanpy import reconstructor as rc
    >>> rc.launch(se_or_edict="sedict.txt")
    >>> rc.getnse("ɟɒloɡ", "jɑlkɑ", normalise=False)
    502

    >>> from loanpy import reconstructor as rc
    >>> rc.launch(se_or_edict="sedict.txt")
    >>> rc.getnse("ɟɒloɡ", "jɑlkɑ", examples=True)
    [429, 5, 39, 4, 11, 14]

    .. note:: If both, reflex and root start with a consonant \
or a vowel, the first sound change is #0<\*0

    """
    dfsc = getsoundchanges(reflex, root)
    outlist = [nsedict[i] for i in dfsc["reflex"] +
               "<*" + dfsc["root"] if i in nsedict]
    return outlist if examples is True else\
        (sum(outlist) / len(dfsc) if normalise is True else sum(outlist))


def reconstruct(ipastring, howmany=float("inf"), struc=False,
                vowelharmony=False, only_documented_clusters=True,
                sort_by_nse=False):
    """
    Takes a modern-day L1 word as input and creates a list of possible \
reconstructions into proto-L1.

    :param ipastring: A modern-day L1-word
    :type ipastring: str

    :param howmany: The number of reconstructions to generate. They will be \
ordered according to likelihood.
    :type howmany: int, default=float("inf")

    :param struc: Indicate whether phonotactic constraints should be \
considered.
    :type struc: bool, default=False

    :param vowelharmony: Indicate whether vowelharmonic constraints \
should be considered.
    :type vowelharmony: bool, default=False

    :param only_documented_clusters: Indicate whether only those modern-day L1\
 clusters should be accepted that are documented in the etymological \
dictionary.
    :type only_documented_clusters: bool, default=True

    :param sort_by_nse: Indicate whether reconstructions should be sorted by \
their likelihood.
    :type sort_by_nse: bool, default=False

    :return: a regular expression after which to search in L2
    :rtype: str

    :Example:

    >>> from loanpy import reconstructor as rc
    >>> rc.launch()
    >>> rc.reconstruct('mɒɟɒr', howmany=100)
    '^(m)(ɑ|ͽ|u|o|ȣ|e|æ)(nʲt͡ʃʲ|j|ðʲ|lj|l|n|t͡ʃʲ)(ɑ|ͽ|u)(r)(ͽ)$'

    >>> from loanpy import reconstructor as rc
    >>> rc.launch()
    >>> rc.reconstruct("mɒɟɒr", howmany=5, struc=True,
                        vowelharmony=True, sort_by_nse=True)
    '^mɑnʲt͡ʃʲɑrͽ$|^mɑjɑrͽ$|^mɑðʲɑrͽ$|^mɑlɑrͽ$|^mɑljɑrͽ$'

    """

    ipaword = tokenise(ipastring)
    if not all(phon in cns+vow for phon in ipaword):
        return ", ".join([i for i in ipaword if i not in scdict]) + " not IPA"

    if only_documented_clusters is True:
        ipaword = ipa2clusters(ipastring)

    ipaword[0], ipaword[-1] = "#" + ipaword[0], ipaword[-1] + "#"
    ipaword = ["#0"] + ipaword + ["0#"]

    if not all(phon in scdict for phon in ipaword):
        return ", ".join([i for i in ipaword if i not in scdict]) + " not old"

    proto = [scdict[i] for i in ipaword]

    if howmany >= math.prod([len(i) for i in proto]):
        out = proto
    else:
        out = [[i[0]] for i in proto]
        while math.prod([len(i) for i in out]) < howmany:
            if all([len(i) == 1 for i in proto]):
                break

            difflist = []
            for i, j in enumerate(proto):
                if len(j) > 1:
                    difflist.append((nsedict[ipaword[i] + "<*" + j[0]] -
                                     nsedict[ipaword[i] + "<*" + j[1]]))
                else:
                    difflist.append(float("inf"))
            whichsound = difflist.index(min(difflist))
            proto[whichsound] = proto[whichsound][1:]
            out[whichsound].append(proto[whichsound][0])

    if struc is False and vowelharmony is False:
        return "^" + "".join([list2regex(i) for i in out]) + "$"

    out = [re.sub("0", "", "".join(i)) for i in product(*out)]

    if struc is True:
        out = [i for i in out if word2struc(i) in allowedphonotactics]
        if out == []:
            return "wrong phonotactics"

    if vowelharmony is True:
        out = [i for i in out if harmony(i)]
        if out == []:
            return "wrong vowel harmony"

    if sort_by_nse is True:
        nse = [getnse(ipastring, i) for i in out]
        out = [i[1] for i in sorted(zip(nse, out), reverse=True)]

    return "^"+"$|^".join(out)+"$"
