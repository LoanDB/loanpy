        # without this break it will prefer the unlikeliest scs bc 0-0=0
        # 1. Are there multiple sounds with zero difference?
    #    if Counter(difflist)[0] > 1:
    #        #2. does any of the 0 diff candidates DO have examples?
    #        if any(fsc!=0 for fsc, diff in zip(firstsclist, difflist) if diff==0):
    #            # 3. freeze all the ones with 0 examples
    #            difflist = [float("inf") if fsc==0 else diff for fsc, diff in zip(firstsclist, difflist)]

        #print("difflist before exc2:", difflist, "firstsclist:", firstsclist)
    #    if 0 in firstsclist and not all(i==0 for i in firstsclist):
    #        difflist = [float("inf") if fsc==0 else diff for fsc, diff in zip(firstsclist, difflist)]
        #print("difflist:", difflist, "sclistlist:", sclistlist, "ipa:", ipa)


#change path to "data", then change back after function executed or failed
def setpath(method):
    @wraps(method)
    def wrapped(*args, **kwargs):
        owd = os.getcwd() #orginal working directory. Change back to this in the end
        os.chdir(os.path.join(os.path.dirname(__file__), "data"))
        try:
            result = method(*args, **kwargs)
        finally:
            os.chdir(owd)
        return result
    return wrapped

phon2cv = {phoneme : "C" if cons == "+" else
                          "V" if cons == "-" else ""
                for phoneme,cons in zip(self.ipa_all["ipa"],
                                        self.ipa_all["cons"])}

vow2fb = {phoneme : "F" if back == "-" and cons == "-" else
                         "B" if back == "+" and cons == "-" else ""
                for phoneme,back,cons in zip(self.ipa_all["ipa"],
                                             self.ipa_all["back"],
                                             self.ipa_all["cons"])}
vow2fb["V"] = "V"

self.phon2cv = {}
for i in phon2cv:
    if phon2cv[i]:
        self.phon2cv[i] = phon2cv[i]

self.vow2fb = {}
for i in vow2fb:
    if vow2fb[i]:
        self.vow2fb[i] = vow2fb[i]



        #if dfetymology:
        #    dfforms = pd.read_csv(dfetymology[0], usecols=["Segments", "Cognacy", "Language_ID"])
        #    dfforms["Segments"] = [i.replace(" ", "") for i in dfforms.Segments] #delete spaces
        #    self.dfety = self.cldf2pd(dfforms, dfetymology[1], dfetymology[2]) #needed by dfetymology2dict
        #    forms = list(dfforms[dfforms["Language_ID"]==dfetymology[2]]["Segments"])

        #    if inventory is None:
        #        self.phoneme_inventory = set(self.tokenise("".join(forms)))
        #    if clusters is None:
        #        self.clusters = set(self.ipa2clusters("".join(forms)))
        #    if struc_inv is None:
        #        self.struc_inv = []
        #        forms_struc = [self.word2struc(i) for i in forms]
        #        for i in set(forms_struc):
        #            cnt = forms_struc.count(i)
        #            if cnt > ptct_thresh:
        #                self.struc_inv.append((i, cnt))
        #        self.struc_inv = [i[0] for i in sorted(self.struc_inv, key=lambda x: x[1], reverse=True)]
        #if distance_measure:
        #    dst = panphon.distance.Distance()
        #    self.distance_measure = getattr(dst, distance_measure)

        #print(type(self.struc_inv))

    #gensim_similarit's docstring
    """\
    load vectors if global model==None and calculate word similiarites. \
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

    :param L1_en: English translations of L1-word
    :type L1_en: str, words are separated by ", "

    :param L2_en: English translations of L2-word
    :type L2_en: str, words are separated by ", "

    :param return_wordpair: Indicate whether the word pair itself should be \
     returned too.
    :type return_wordpair: bool, default=False

    :return: The most similar word pair of the two lists
    :rtype: int or tuple of (int and tuple of two str)

    :Example:

    >>> from loanpy import helpers
    >>> hp = helpers.Help_sem()
    >>> hp.gensim_similarity("hovercraft, full, eels", "nipples, explode, delight",
                           return_wordpair=False)
    0.21005636

    >>> from loanpy import helpers
    >>> hp = helpers.Help_sem()
    >>> hp.gensim_similarity("drop, panties, William", "cannot, wait, lunchtime",
                           return_wordpair=True)
    (0.54949486, ('house', 'cottage'))

    """


    #filterdf's docstring
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

def align_old(self, left, right):  # requires two ipastrings as input
    """
    Takes two words as input that stand in a reflex-root or donorword-loanword relationship to each other \
    and returns a table of sound correspondences

    :param left: a reflex or a loanword
    :type left: str

    :param right: a root or a donorword
    :type right: str

    :param method: the type of methodment
    :type method: {'uralonet', 'lingpy'}

    :return: table of sound correspondences
    :rtype: pandas.core.frame.DataFrame

    :Example:

    >>> from loanpy import qfysc
    >>> qfy = qfysc.Qfy("")
    >>> qfy.align("ɟɒloɡ", "jɑlkɑ")
    +---+----------+----------+
    | # | left   | right   |
    +---+----------+----------+
    | 0 | #-       | -        |
    +---+----------+----------+
    | 1 | #ɟ       | j        |
    +---+----------+----------+
    | 2 | ɒ        | ɑ        |
    +---+----------+----------+
    | 3 | l        | lk       |
    +---+----------+----------+
    | 4 | o        | ɑ        |
    +---+----------+----------+
    | 5 | ɡ#       | -        |
    +---+----------+----------+

    >>> from loanpy import qfysc
    >>> qfy = qfysc.Qfy("")
    >>> qfy.align("ɟɒloɡ", "jɑlkɑ", method="lingpy")
    +---+----------+----------+
    | # | left   |   right |
    +---+----------+----------+
    | 0 | ɟ        | j        |
    +---+----------+----------+
    | 1 | ɒ        | ɑ        |
    +---+----------+----------+
    | 2 | l        | l        |
    +---+----------+----------+
    | 3 | o        | -        |
    +---+----------+----------+
    | 4 | g        | k        |
    +---+----------+----------+
    | 5 |          | ɑ        |
    +---+----------+----------+

    """
    if self.mode == "adapt":
        pw = Pairwise(left, right, merge_vowels=False)
        pw.align()
        leftright = [i.split("\t") for i in str(pw).split("\n")[:-1]]
        leftright[0] = ["C" if new=="-" and self.phon2cv.get(old,"") == "C" else
                        "V" if new=="-" and self.phon2cv.get(old,"") == "V" else
                        new for new,old in zip(leftright[0], leftright[1])]
        leftright[1] = ["C" if old=="-" and self.phon2cv.get(new,"") == "C" else
                        "V" if old=="-" and self.phon2cv.get(new,"") == "V" else
                        old for new,old in zip(leftright[0], leftright[1])]

        return pd.DataFrame({"keys": leftright[1], "vals": leftright[0]})

    else:
        keys = self.ipa2clusters(left)
        vals = self.ipa2clusters(right)

        keys[0], keys[-1] = "#" + keys[0], keys[-1] + "#"
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

        return pd.DataFrame({"keys": keys, "vals": vals})

#docstrings of getnse()... "invalid escape sequence "\*"
        """
        Takes a reflex/donorword and a root/loanword and \
        assesses the likelihood of the etymology.

        :param left: A reflex or a donorword
        :type left: str

        :param right: A root or a loanword. This information \
        is generally stored in etymological dictionaries.
        :type right: str

        :param examples: Indicate whether other wordpairs exhibiting a sound-correspondence \
        should be listed up.
        :type examples: bool, default=False

        :param normalise: Indicate whether the sum of examples should be \
        normalised by dividing it through the number of sound changes between two words.
        :type normalise: bool, default=True

        :return: Normalised sum of examples (nse) with default settings. \
        Sum of examples if normalise==False, examples or list of number \
        of examples if examples==True.
        :rtype: int or list of str or list of int

        :Example:

        >>> from loanpy import reconstructor as rc
        >>> rc.launch()
        >>> rc.getnse("ɟɒloɡ","jɑlkɑ")
        83.66666666666667

        >>> from loanpy import reconstructor as rc
        >>> rc.launch(se_or_edict="edict.txt")
        >>> rc.getnse("ɟɒloɡ", "jɑlkɑ", examples=True)
        [['ɛ<ȣ̈',
        'eːɡ<æŋͽ',
        'ɒz<o',
        ...]

        >>> from loanpy import reconstructor as rc
        >>> rc.launch(se_or_edict="sedict.txt")
        >>> rc.getnse("ɟɒloɡ", "jɑlkɑ", normalise=False)
        502

        >>> from loanpy import reconstructor as rc
        >>> rc.launch(se_or_edict="sedict.txt")
        >>> rc.getnse("ɟɒloɡ", "jɑlkɑ", examples=True)
        [429, 5, 39, 4, 11, 14]

        .. note:: If both left and right start with a consonant \
          or a vowel, the first sound change is #0<\*0

        """

#adrc.py:
#import logging
#rootdir = os.path.join(os.path.dirname(__file__), "data")
#logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)
#formatter = logging.Formatter('%(asctime)s:%(message)s')
#file_handler = logging.FileHandler(f'{rootdir}\\ipa.log', encoding="utf-8")
#file_handler.setFormatter(formatter)
#logger.addHandler(file_handler)

#Adrc() __init__
        #else:
        #    self.scdict = scdict

        #self.sedict = self.scdict[1]
        #self.edict = self.scdict[2]
        #self.scdict = self.scdict[0]

        #self.struc_inv=None #WHY TF WOULD I WRITE A LINE LIKE THIS
        #if isinstance(scdict_struc, str):
        #    with open(scdict_struc, "r", encoding="utf-8") as f:
        #        self.scdict_struc = literal_eval(f.read())
        #        self.struc_inv = set(sum(self.scdict_struc.values(), []))
        #elif isinstance(scdict_struc, dict):
        #    self.scdict_struc = scdict_struc
        #    self.struc_inv = set(sum(self.scdict_struc.values(), []))

        #if isinstance(sedict_struc, str):
        #    with open(sedict_struc, "r", encoding="utf-8") as f:
        #        self.sedict_struc = literal_eval(f.read())
        #elif isinstance(sedict_struc, dict):
        #    self.sedict_struc = sedict_struc

        #if isinstance(scdict, str):
            #print(os.getcwd())

            #exception 2:
            #if firstsc == 0: #if a soundchange has zero examples
            #    for i, sl in enumerate(sclistlist):  # look at the other sounds in the word
            #        if len(sl) > 1 and self.nsedict.get(ipa[i]+self.connector+sl[0],0) > 0:
                    #if there is at least one that hasnt reached the end AND has more than 0 examples
            #            difflist.append(float("inf"))  # "dont move (the sound with 0 examples) in readsc()"
                        # instead give (the one that hasnt reached the end and has more than zero examples) a chance later
            #            break

#documentation of findloans
        """
        Search for phonetic matches and rank them according to semantic similarity.

        :param outputname: The name of the Excel worksheet in results.xlsx to which to write the \
         results. Maximum 31 characters long and \\\/\*\?\:\[\] must be excluded.
        :type outputname: str, default="new"

        :param cutoff: Indicate how many of the semantically most similar phonetic matches \
         should be returned in the final result.
        :type cutoff: int, default=100

        :param write: If True, output will be written to a sheet in results.xlsx \
         and None will be returned. If False, a Pandas DataFrame will be returned.
        :type write: bool, default=True

        :param semantic_similarity: Indicate which function should be used to \
         calculate semantic similarities. See example two for an example of how to plug in \
         an external function for similarity measurement.
        :type semantic_similarity: function, default=loanpy.helpers.gensim_similarity

        :return: Phonetic matches ranked according to \
         semantic similarity
        :rtype: {None if write=True, pandas.core.frame.DataFrame if write=False}

        :raises Exception: If no phonetic matches were found

        :Example:

        >>> from loanpy import loanfinder as lf
        >>> finder = lf.Semantix(L1='hun_re.csv', L2='ad_pii.csv', L1col='re_577TTT', L2col='ad_test')
        >>> finder.findloans("outputname='example', cutoff=250, write=True)
        view results in data/results.xlsx on sheet "example"

        >>> from loanpy import loanfinder
        >>> from loanpy import helpers
        >>> from sentence_transformers import SentenceTransformer
        >>> from sklearn.metrics.pairwise import cosine_similarity
        >>> helpers.model = SentenceTransformer("bert-base-nli-mean-tokens")
        >>> def bert_similarity(sentence1, sentence2):
        >>>     return float(cosine_similarity(helpers.model.encode([sentence1]), helpers.model.encode([sentence2])))
        >>> finder = loanfinder.Semantix(L1='hun_re.csv', L2='ad_pii.csv', L1col='re_577TTT', L2col='ad_1TFT', sedict='sedict.txt', edict='edict.txt')
        >>> finder.findloans(outputname='577TTT1TFT_bert', cutoff=250, write=True, semantic_similarity=bert_similarity)
        view results in data/results.xlsx on sheet "577TTT1TFT_bert"
        """


def filterdf(df, col, occurs_or_bigger, term, write=False,
             name="dffiltered.csv"):
    """
    docstring threw deprecation warning. rewrite.
    """
    out = None
    if isinstance(term, str):
        df = df.fillna('')
        if occurs_or_bigger:
            out = df[df[col].str.contains(term, na=False)]
        else:
            out = df[~df[col].str.contains(term, na=False)]

    elif isinstance(term, (float, int)):
        df = df.fillna(0)
        if occurs_or_bigger:
            out = df[df[col] > term]
        else:
            out = df[df[col] <= term]

    if write:
        out.to_csv(name, encoding="utf_8_sig", index=False)
    else:
        return out

def test_filterdfin():
    dfwinemenu = DataFrame({"wine": ["Egri Bikavér", "Tokaji Aszú"],
                               "colour": ["red", "white"],
                               "price_in_forint": [10000, 30000]})
    dfred = DataFrame({"wine": ["Egri Bikavér"],
                          "colour": ["red"],
                          "price_in_forint": [10000]})
    dfwhite = DataFrame({"wine": ["Tokaji Aszú"],
                            "colour": ["white"],
                            "price_in_forint": [30000]})
    dfin1 = filterdf(df=dfwinemenu, col="colour",
                        occurs_or_bigger=True,
                        term="red").reset_index(drop=True)
    dfin2 = filterdf(df=dfwinemenu, col="colour",
                        occurs_or_bigger=False,
                        term="red").reset_index(drop=True)
    dfin3 = filterdf(df=dfwinemenu, col="price_in_forint",
                        occurs_or_bigger=True,
                        term=10000).reset_index(drop=True)
    dfin4 = filterdf(df=dfwinemenu, col="price_in_forint",
                        occurs_or_bigger=False,
                        term=10000).reset_index(drop=True)
    assert_frame_equal(dfin1, dfred.reset_index(drop=True))
    assert_frame_equal(dfin2, dfwhite.reset_index(drop=True))
    assert_frame_equal(dfin3, dfwhite.reset_index(drop=True))
    assert_frame_equal(dfin4, dfred.reset_index(drop=True))

done = False
def animate():
    starttime = datetime.now().strftime('%H:%M')
    for c in cycle(["➼   🎯 ", " ➼  🎯 ", "  ➼ 🎯 ", "   ➼🎯 ", "     🎯"]):
        if done:
            break
        stdout.write(f"\r{starttime} Loading vectors" + c)
        stdout.flush()
        sleep(0.1)
    stdout.flush()

#tried to mock pandas.DataFrame.groupby but it's too difficult
#groupby is going to return these 6.
    #create a mock version of pandas dataframe with a mock-groupby method
    class DataFrameMock:
        def __init__(self, groupby_returns: list):
            self.groupby_returns = iter(groupby_returns)
            self.groupby_called_with = []
        def groupby(self, arg):
            self.groupby_called_with.append(arg)
            return next(groupby_returns)

dfg1 = DataFrame({"keys": ["k", "i", "b", "u", "a"], "vals": ["h", "e", "p", "u", "a"]}).groupby("keys")
dfg2 = DataFrame({"soundchange": ["k<h", "i<e", "b<p", "u<u", "a<a"], "e": [2, 2, 2, 1, 1]}).groupby("soundchange")
dfg3 = DataFrame({"soundchange": ["k<h", "i<e", "b<p", "u<u", "a<a"],
                  "wordchange": [[12], [12], [13], [13], [13]]}).groupby("soundchange")
dfg4 = DataFrame({"keys": ["#-","#k", "i", "k", "i#", "#b", "u", "b", "a#"],
"vals": ["#-", "#h", "e", "h", "e#", "#p", "u", "p", "a#"]}).groupby("keys")
dfg5 = DataFrame({"soundchange": ["#-<*#-","#h<*#k","e<*i","h<*k","e#<*i#",
"#b<*#p","u<*u","b<*p","a#<*a#"], "e": [2]+[1]*8}).groupby("soundchange")
dfg6 = DataFrame({"soundchange": ["#-<*#-","#h<*#k","e<*i","h<*k","e#<*i#",
"#b<*#p","u<*u","b<*p","a#<*a#"], "wordchange": [[12, 13], [12], [12], [12], [12],
[13], [13], [13], [13]]}).groupby("soundchange")


mockdf = DataFrameMock([dfg1, dfg2, dfg3])
mockdf2 = DataFrameMock([dfg4, dfg5, dfg6])


#for some reason "groupby" just doesn't get overwritten whatever I do.
#create a mock version of pandas dataframe with a mock-groupby method
#groupby_returns = iter([dfg1, dfg2, dfg3, dfg4, dfg5, dfg6])
#groupby_called_with = []
#class DataFrameMock(DataFrame):
#    def __init__(self):
#        super().__init__()
#    def groupby(self, arg):
#        groupby_called_with.append(arg)
#        return next(groupby_returns)


#set up: groupby is going to return these 6.
dfg1 = DataFrame({"keys": ["k", "i", "b", "u", "a"], "vals": ["h", "e", "p", "u", "a"]}).groupby("keys")
#groupby would sort the sound changes below alphabetically, so I did it manually beforehand, for transparency
#so groupby here is really doing nothing except creating a groupby-object
#if mode=="adapt", sc are stored as vals<keys from dfconcat
dfg2 = DataFrame({"soundchange": ['a<a', 'e<i', 'h<k', 'p<b', 'u<u'], "e": [1, 2, 2, 2, 1]}).groupby("soundchange")
dfg3 = DataFrame({"soundchange": ['a<a', 'e<i', 'h<k', 'p<b', 'u<u'],"wordchange": [[13], [13], [12], [12], [13]]}).groupby("soundchange")
dfg4 = DataFrame({"keys": ["#-","#k", "i", "k", "i#", "#b", "u", "b", "a#"],
"vals": ["#-", "#h", "e", "h", "e#", "#p", "u", "p", "a#"]}).groupby("keys")
dfg5 = DataFrame({"soundchange": ["#-<*#-","#h<*#k","e<*i","h<*k","e#<*i#",
"#b<*#p","u<*u","b<*p","a#<*a#"], "e": [2]+[1]*8}).groupby("soundchange")
dfg6 = DataFrame({"soundchange": ["#-<*#-","#h<*#k","e<*i","h<*k","e#<*i#",
"#b<*#p","u<*u","b<*p","a#<*a#"], "wordchange": [[12, 13], [12], [12], [12], [12],
[13], [13], [13], [13]]}).groupby("soundchange")

groupby_returns = iter([dfg1, dfg2, dfg3, dfg4, dfg5, dfg6])

#get-struc_corresp
    dfg1 = DataFrame({"keys": ["CVCV"]*2, "vals": ["CVCV"]*2}).groupby("keys")
    dfg2 = DataFrame({"strucchange": ["CVCV<CVCV"]*2, "e": [1]*2}).groupby("strucchange")
    dfg3 = DataFrame({"strucchange": ["CVCV<CVCV"], "wordchange": [[12, 13]]}).groupby("strucchange")

with patch("loanpy.qfysc.DataFrame.groupby", side_effect=[dfg1, dfg2, dfg3]):

#write factsheet (sanity.py)
tp4 = len([i for i in workflow[f"step4_tp"] if i])
tp7 = len([i for i in step7_fp if i])
DataFrame({
"step4_tp": [f"{tp4}/{len(workflow)}", f"{round(100*tp4/len(workflow))}%"],
"step7_tp": [f"{tp7}/{len(workflow)}", f"{round(100*tp7/len(workflow))}%"],
"diff_step4_vs7": [f"{round(tp4-tp7)}/{len(workflow)}",
f"{round(100*abs(tp4-tp7)/len(workflow))}%"]
}).to_excel(f"{outname}_factsheet.csv", encoding="utf_8_sig", index=False)

#no idea what this is supposed to do
text(0, tpr_fpr_opt[0][-1], f"tpr: 100%={len_df}")

#def test_read_nsedict():
#    """test if nse-dict is read correctly from different sources"""
    #just one patch less than unittest, otherwise same.

    # setup
#    soundchanges = [{"dict1": "dict1"}, {"d2": "d2"}]
#    path = Path(__file__).parent / "test_read_nsedict.txt"
#    with open(path, "w") as f:
#        f.write(str(soundchanges))

    # assert
#    assert read_nsedict(soundchanges) == {"d2": "d2"}
#    assert read_nsedict(path) == {"d2": "d2"}

    # tear down
#    remove(path)
#    del soundchanges

#@setpath
#class Roc_opt(Roc):

 #       def __init__(self, heur=True, dfetymology=None):
  #          if not os.path.exists("cache.csv"):
   #             raise Missing_dfargscsv("cache.csv has to be created by initiating ROC_ad with different arguments. The best combo will be on top.")
    #
     #       args = "mode,scdictbase,dfetymology,left,right,ptct_thresh,scdict,scdict_struc,crossval,guesslist,hm_struc_ceiling,hm_paths_ceiling,writesc,writesc_struc,vowelharmony,only_documented_clusters,sort_by_nse,\
#struc_filter,show_workflow,write,outname,plot".split(",")

 #           for idx, row in pd.read_csv("cache.csv", usecols=args).fillna("None").head(1).iterrows():
  #              args = list(map(str, list(row)))
   #
    #        args[2] = tuple(args[2][2:-2].split("', '"))
     #       if dfetymology:
      #          args[2] = dfetymology
#
 #           args[5] = int(args[5])
  #          args[9] = list(map(int, args[9][2:-2].split(", ")))
   #         args[10], args[11] = int(args[10]), int(args[11])
    #        if heur:
     #           args[6] = args[1] #scdict = scdictbase
      #          args[7] = {} #scdict_struc = {}
       #         args[8] = False #crossval = False
        #        args[20] = f"{args[20]}_heuristic"

         #   super().__init__(*args)
