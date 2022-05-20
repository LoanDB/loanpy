"""
Quantify sound correspondences from etymological data, \
and give etymologies a likelihood-score based on extracted data.
"""

from ast import literal_eval
from collections import Counter
from functools import partial

from ipatok import clusterise
from lingpy.align.pairwise import Pairwise
from lingpy.data.model import Model as LingpyModel
from pandas import DataFrame, concat
from tqdm import tqdm

from loanpy.helpers import Etym, clusterise

class WrongModeError(Exception):
    pass

def read_mode(mode):
    if mode and mode not in ["adapt", "reconstruct"]:
        raise WrongModeError("parameter <mode> must be 'adapt' or 'reconstruct'")
    return mode if mode else "adapt"

def read_connector(connector, mode):
    if connector is None: connector = ("<", "<*")
    return connector[1] if mode=="reconstruct" else connector[0]

def read_scdictbase(scdictbase):
    if scdictbase is None: return {} #needed by get_sound_corresp for substitutions
    if isinstance(scdictbase, dict): return scdictbase
    with open(scdictbase, "r", encoding="utf-8") as f:
        return literal_eval(f.read())

class Qfy(Etym):
    """
    Read etymological data and customise the way in which it \
shall be quantified later.

    Inherit from loanpy.helpers.Etym:

    :param formscsv: The path to cldf's forms.csv. For more details see \
loanpy.helpers.read_forms
    :type formscsv: pathlib.PosixPath | str | None, default=None

    :param srclg: The computational source language (can differ from linguistic \
source). This is the data FROM which we make predictions. \
For more details see loanpy.helpers.Etym.
    :type srclg: str (options are listed in column "ID" in \
cldf / etc / languages.tsv), default=None

    :param tgtlg: The computational target language (can again differ from \
linguistic one). This is the language INTO which we make predictions. \
For more details see loanpy.helpers.Etym.
    :type tgtlg: str (options are listed in column "ID" in \
cldf / etc / languages.tsv), default=None

    :param struc_most_frequent: The n most frequent structures \
that we want to accept into the phonotactic inventory of the target language. \
Sometimes a good idea \
to omit rare ones.
    :type struc_most_frequent: int, default=9999999

    :param struc_inv: All possible phonotactic structures in the target \
language. Will be extracted from target language if set to None. \
See loanpy.helpers.Etym.read_strucinv for more details.
    :type struc_inv: None | set | list, default=None

    Define own attributes:

    :param mode: The mode in which sound correspondences will be \
extracted. Differences between the two modes: Different connectors, \
source and target gets flipped, different alignment. No phonotactic \
correspondences extracted if mode == "reconstruct". \
Flows into self.mode. For more details see loanpy.qfysc.read_mode.
    :type mode: "adapt" | "reconstruct", default="adapt"

    :param connector: The strings that connect the linguistic (!) source word \
with the target adapted or reconstructed one. \
Flows into self.connector. Set to None if \
default settings ("<\*" and "<") should be used. For more details see \
loanpy.qfysc.read_connector.
    :type connector: tuple, default=None

    :param scdictbase: The sound correspondence dictionary base, \
a heuristic approach \
for predicting sound substitutions. loanpy.qfysc.Qfy.get_sound_corresp \
will combine sound substitutions gathered from etymological data with \
this dictionary. If combination should be skipped, set to None or {}. \
Flows into self.scdictbase. \
For more details see loanpy.helpers.Etym.get_scdictbase.
    :type scdictbase: None | dict | pathlib.PosixPath, default=None

    :param vfb: Placeholders for "any vowel", "any front vowel", "any \
back vowel", as these can occur in etymological dictionaries. \
Tokeniser can only handle IPA-characters. Therefore placeholders \
have to be IPA-characters as well. Best is to choose IPA-characters that \
don't occur in the phoneme inventory. By default, no placeholders \
are used. For a list \
of available ipa characters see ipa_all.csv's column "ipa"
    :type vfb: None | iterable, default=None, example: "əœʌ"

    """
    def __init__(self,
# to inherit from loanpy.helpers.Etym
formscsv=None,
srclg=None,
tgtlg=None,
struc_most_frequent=9999999,
struc_inv=None,

mode="adapt",  # or: "reconstruct"
connector=None,  # will automatically get defined
scdictbase=None,  # big file, therefore not generated every time
vfb=None):  # etymological data sometimes has placeholders for "any vowel",
# "any front vowel", or "any back vowel". Those have to be designated by
# ipa characters that are not used in the language. Because the tokeniser
# accepts only ipa-characters.


        super().__init__(
                         formscsv=formscsv,
                         srclg=srclg,
                         tgtlg=tgtlg,
                         struc_most_frequent=struc_most_frequent,
                         struc_inv=struc_inv)

        self.mode = read_mode(mode)
        self.connector = read_connector(connector, mode)
        self.scdictbase = read_scdictbase(scdictbase)
        self.vfb = vfb

    def align(self, left, right, merge_vowels=False, gop=-1, scale=0.5, mode="global",
    factor=0.3, restricted_chars="T_", distance=False, model=LingpyModel, pprint=False):

        if self.mode == "reconstruct": return self.align_clusterwise(left, right)
        elif self.mode == "adapt": return self.align_lingpy(left, right, merge_vowels, gop, scale, mode,
        factor, restricted_chars, distance, model, pprint)

    def align_lingpy(self, left, right, merge_vowels=False, gop=-1, scale=0.5, mode="global",
    factor=0.3, restricted_chars="T_", distance=False, model=None, pprint=False):
        pw = Pairwise(seqs=left, seqB=right, merge_vowels=merge_vowels)
        pw.align(gop=gop, scale=scale, mode=mode, factor=factor, #cant pass on param model for some reason
        restricted_chars=restricted_chars, distance=distance, pprint=pprint)
        leftright = [i.split("\t") for i in str(pw).split("\n")[:-1]]
        leftright[0] = ["C" if new=="-" and self.phon2cv.get(old,"") == "C" else
                        "V" if new=="-" and self.phon2cv.get(old,"") == "V" else
                        new for new,old in zip(leftright[0], leftright[1])]
        leftright[1] = ["C" if old=="-" and self.phon2cv.get(new,"") == "C" else
                        "V" if old=="-" and self.phon2cv.get(new,"") == "V" else
                        old for new,old in zip(leftright[0], leftright[1])]

        return DataFrame({"keys": leftright[1], "vals": leftright[0]})

    def align_clusterwise(self, left, right):
        keys = clusterise(left)
        vals = clusterise(right)

        keys[0], keys[-1] = "#" + keys[0], keys[-1] + "#" #only keys get this!
        keys, vals = ["#-"] + keys, ["-"] + vals

        if self.phon2cv.get(keys[1][1:],"") == "V" and self.phon2cv.get(vals[1][1:],"") == "C":
            vals = vals[1:]
        elif self.phon2cv.get(keys[1][1:],"") == "C" and self.phon2cv.get(vals[1][1:],"") == "V":
            keys = keys[1:]

        diff = abs(len(vals) - len(keys))  # "a,b","c,d,e,f,g->"a,b,000","c,d,efg
        if len(keys) < len(vals):
            keys += ["-#"]
            vals = vals[:-diff] + ["".join(vals[-diff:])]
        elif len(keys) > len(vals):
            vals += ["-"]
            keys = keys[:-diff] + ["".join(keys[-diff:])]
        else:
            keys, vals = keys + ["-#"], vals + ["-"]

        return DataFrame({"keys": keys, "vals": vals})

    def get_sound_corresp(self, write_to=None):

        """
        Convert an etymological dictionary to a dictionary of sound \
and phonotactic correspondences, \
their number of occurences, and IDs of cognate sets in which they occur.

        :param write_to: Indicate if results should be written to a \
text file. If yes, provide the path. None means that no file will be written.
        :type write_to: None | pathlib.PosixPath | str, default=None

        :returns: list of 6 dicts. Dicts 0, 1, 2 capture phonological \
correspondences, dicts 3, 4, 5 phonotactical ones. dict0/dict3: the actual \
correspondences, dict1/dict4: How often each correspondence occurs in the data, \
dict2/dict5: list of cognates in which each correspondence occurs.
        :rtype: [dict, dict, dict, dict, dict, dict]

        """

        soundchange = []
        wordchange = []
        for left, right, cog in zip(self.dfety["Target_Form"], self.dfety["Source_Form"],
                                        self.dfety["Cognacy"]):
            dfalign = self.align(left, right)
            soundchange.append(dfalign)
            wordchange += [cog]*len(dfalign)

        dfsoundchange = concat(soundchange)
        dfsoundchange["wordchange"] = wordchange
        dfsoundchange["e"] = 1

        if self.mode == "adapt": #important! since for rc we're predicting backwards
            dfsoundchange["soundchange"] = dfsoundchange["vals"] + self.connector +\
                dfsoundchange["keys"]
        else:
            dfsoundchange["soundchange"] = dfsoundchange["keys"] + self.connector +\
                dfsoundchange["vals"]

        dfsc = dfsoundchange.groupby("keys")["vals"].apply(
        lambda x: [n[0] for n in Counter(x).most_common()]).reset_index()
        dfse = dfsoundchange.groupby("soundchange")["e"].sum().reset_index()
        dfe = dfsoundchange.groupby("soundchange")["wordchange"].apply(
        lambda x: sorted(set(x))).reset_index()

        scdict = dict(zip(dfsc["keys"], dfsc["vals"]))
        sedict = dict(zip(dfse["soundchange"], dfse["e"]))
        edict = dict(zip(dfe["soundchange"], dfe["wordchange"]))

        if self.vfb: #"əœʌ"
            for i in scdict:
                if any(self.phon2cv.get(j,"") == "V" for j in scdict[i]) and self.vfb[0] not in scdict[i]:
                    scdict[i].append(self.vfb[0])
                    if any(self.vow2fb.get(j,"") == "F" for j in scdict[i]) and self.vfb[1] not in scdict[i]:
                        scdict[i].append(self.vfb[1])
                    if any(self.vow2fb.get(j,"") == "B" for j in scdict[i]) and self.vfb[2] not in scdict[i]:
                        scdict[i].append(self.vfb[2])

        if self.mode == "adapt":
            for k in scdict:
                scdict[k] = ["" if j=="C" or j=="V" else j for j in scdict[k]]
            for i in self.scdictbase | scdict: #loop through keys of both dicts combined
                if i in self.scdictbase and i in scdict: #combine keys that are in both
                    scdict[i] = list(dict.fromkeys(scdict[i]+self.scdictbase[i]))
                elif i in self.scdictbase: #if key is missing from scdict but is in scdictbase
                    scdict[i] = self.scdictbase[i] #then add it to scdict
                #like this we make sure that no non-ipa chars are taken

        out = [scdict, sedict, edict]
        out += self.get_struc_corresp() if self.mode=="adapt" else [{}, {}, {}]

        if write_to:
            with open(write_to, "w", encoding="utf-8") as data:
                data.write(str(out))

        return out

    def get_struc_corresp(self, write_to=None):

        """
        Convert an etymological dictionary to a dictionary of sound correspondences, \
        their number of occurences, and word pairs in which they occur.

        :param dfetymology: The name of the etymological dictionary. Has to be \
        stored in a csv-file.
        :type dfetymology: str, default="dfuralonet.csv"

        :param left: The column containing the reflexes or donorwords
        :type left: str, default="New"

        :param right: The column containing the roots or loanwords
        :type right: str, default="Old"

        :param layer: Indicate from which historical layer sound changes \
        should be extracted, Uralic ('U'), Finno-Ugric ('FU'), or Ugric ('Ug'). \
        By default all three are taken into consideration simultaneously.
        :type layer: str, e.g. {"", "U", "FU", "Ug"}, default=""

        :param method: the type of methodment
        :type method: {'uralonet', 'lingpy'}, default="uralonet"

        :param defaultsc: Name of the file containing default sound-correspondences, \
        to fill out gaps that are missing in the etymological data.
        :type defaultsc: str, default=""

        :param substitute: Set to True if wordpairs are in a donor-loanword relationship
        :type substitute: bool, default=False

        :param write: If set to True, three dictionaries will be written to .txt-files and \
        None will be returned. Else a tuple of three dictionaries will be returned and no \
        files written.

        :param name_soundchangedict: Name of sound correspondence dictionary, without \
        file-extension.
        :type name_soundchangedict: str, default="scdict"

        :param name_sumofexampledict: Name of dictionary counting occurences of \
        sound correspondences, without file-extension.
        :type name_soundchangedict: str, default="sedict"

        :param name_examplesdict: Name of dictionary counting all examples of \
        wordpairs exhibiting a sound correspondence, without file-extension.
        :type name_soundchangedict: str, default="edict"

        :return: See parameter <write>
        :rtype: {None, tuple of 3 dictionaries}

        :Example:

        >>> from loanpy import qfysc
        >>> qfy = qfysc.Qfy("")
        >>> qfy.dfetymology2dict(dfetymology="dfpii_Holopainen2019.csv",
                                 left="L2_ipa", right="L1_ipa",
                                 layer="", method="lingpy",
                                 defaultsc="substi_Dellert2017.csv",
                                 substitute=True, write=True,
                                 name_soundchangedict="substi",
                                 name_sumofexamplesdict="sedict_ad",
                                 name_examplesdict="edict_ad",)
        view results in data\\substi.txt, sedict_ad.txt and edict_ad.txt

        >>> from loanpy import qfysc
        >>> qfy = qfysc.Qfy("")
        >>> qfy.dfetymology2dict(dfetymology="dfuralonet.csv",
                                 left="New", right="Old",
                                 method="uralonet",
                                  write=True,
                                 name_soundchangedict="scdict",
                                 name_sumofexamplesdict="sedict",
                                 name_examplesdict="edict")
        view results in data\\scdict.txt, sedict.txt and edict.txt
        """

        keys = [self.word2struc(i) for i in self.dfety["Source_Form"]]
        vals = [self.word2struc(i) for i in self.dfety["Target_Form"]]
        wordchange = self.dfety["Cognacy"]

        dfstrucchange = DataFrame(zip(keys, vals, wordchange),
                                     columns=["keys", "vals", "wordchange"])
        dfstrucchange["e"] = 1

        if self.mode == "adapt": #important! #search explanation tweet
            dfstrucchange["strucchange"] = dfstrucchange["vals"] + self.connector +\
                dfstrucchange["keys"]
        else:
            dfstrucchange["soundchange"] = dfstrucchange["keys"] + self.connector +\
                dfstrucchange["vals"]

        dfsc = dfstrucchange.groupby("keys")["vals"].apply(
        lambda x: [n[0] for n in Counter(x).most_common()]).reset_index()

        dfse = dfstrucchange.groupby("strucchange")["e"].sum().reset_index()

        dfe = dfstrucchange.groupby("strucchange")["wordchange"].\
            apply(list).reset_index()

        scdict = dict(zip(dfsc["keys"], dfsc["vals"]))
        sedict = dict(zip(dfse["strucchange"], dfse["e"]))
        edict = dict(zip(dfe["strucchange"], dfe["wordchange"]))

        for i in scdict:
            scdict[i] = list(dict.fromkeys(scdict[i] + self.rank_closest_struc(i).split(", ")))

        if write_to:
            with open(write_to, "w", encoding="utf-8") as data:
                data.write(str([scdict, sedict, edict]))

        return [scdict, sedict, edict]
