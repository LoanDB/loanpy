"""unit tests for loanpy.helpers.py (2.0 BETA) with pytest 7.1.1"""

from unittest.mock import call, patch
from pathlib import Path
from os import remove
from inspect import ismethod
from ast import literal_eval

from pandas import DataFrame, Series, read_csv
from pandas.testing import assert_frame_equal, assert_series_equal
from pytest import raises
from gensim.models import word2vec
from gensim.test.utils import common_texts, get_tmpfile
from numpy import array, float32
from numpy.testing import assert_array_equal
from networkx import DiGraph

from loanpy import helpers as hp  # needed for hp.model, to plug in
from loanpy.helpers import (
    Etym,
    InventoryMissingError,
    apply_edit,
    combine_ipalists,
    cldf2pd,
    edit_distance_with2ops,
    editops,
    flatten,
    forms2list,
    gensim_multiword,
    get_howmany,
    get_mtx,
    list2regex,
    model,
    mtx2graph,
    pick_minmax,
    plug_in_model,
    read_cvfb,
    read_dst,
    read_forms,
    tuples2editops)


class EtymMonkey:
    """used throughout the module"""
    pass


def test_plug_in_model():
    """test if model gets plugged into global variable correctly"""
    plug_in_model("xyz")
    assert hp.model == "xyz"
    plug_in_model(None)
    assert hp.model is None


def test_read_cvfb():
    """check if cvfb.txt, a tuple of 2 long dicts is correctly read in"""

    # set up
    with patch("loanpy.helpers.literal_eval") as literal_eval_mock:
        literal_eval_mock.return_value = [
            {"a": "V", "b": "C"}, {"e": "F", "o": "B"}]

        # assert
        assert read_cvfb() == ({"a": "V", "b": "C"}, {"e": "F", "o": "B"})

        # assert calls
        literal_eval_mock.assert_called()
        assert isinstance(literal_eval_mock.call_args_list[0][0][0], str)
        assert len(literal_eval_mock.call_args_list[0][0][0]) == 92187
        assert literal_eval_mock.call_args_list[0][0][0][:
                                                         20] == "[\
{'q': 'C', 'q̟': 'C"


def test_read_forms():
    """Check if CLDF's forms.csv is read in correctly"""

    # set up
    dfin = DataFrame({"Segments": ["a b c", "d e f"],  # pull together later
                      "Cognacy": ["ghi", "jkl"],
                      "Language_ID": [123, 456],
                      "randcol": ["mno", "pqr"]})
    dfexp = DataFrame({"Segments": ["abc", "def"],  # pulled together segments
                       "Cognacy": ["ghi", "jkl"],
                       "Language_ID": [123, 456]})
    path = Path(__file__).parent / "test_read_forms.csv"
    with patch("loanpy.helpers.read_csv") as read_csv_mock:
        read_csv_mock.return_value = dfexp

        # assert
        assert read_forms(None) is None
        assert_frame_equal(read_forms(path), dfexp)

    # assert calls
    read_csv_mock.assert_called_with(
        path, usecols=['Segments', 'Cognacy', 'Language_ID'])

    # tear down
    del path, dfin, dfexp


def test_cldf2pd():
    """test if the CLDF format is correctly tranformed to a pandas dataframe"""

    # set up
    dfin = DataFrame({"Segments": ["a", "b", "c", "d", "e", "f", "g"],
                      "Cognacy": [1, 1, 2, 2, 3, 3, 3],
                      "Language_ID": ["lg1", "lg2", "lg1", "lg3",
                                      "lg1", "lg2", "lg3"]})
    dfexp = DataFrame({"Target_Form": ["b", "f"],
                       "Source_Form": ["a", "e"],
                       "Cognacy": [1, 3]})
    # only cognates are taken, where source and target language occur
    dfout = cldf2pd(dfin, source_language="lg1", target_language="lg2")

    # assert
    assert cldf2pd(None, source_language="whatever",
                   target_language="wtvr2") is None
    assert_frame_equal(dfout, dfexp)

    # tear down
    del dfout, dfexp, dfin


def test_read_dst():
    """check if getattr gets euclidean distance of panphon feature vectors"""

    # assert where no setup needed
    assert read_dst("") is None

    # set up monkey class
    class MonkeyDist:  # mock panphon's Distance() class
        def __init__(self): pass

        def weighted_feature_edit_distance(self): pass

    # initiate monkey class
    mockdist = MonkeyDist()

    # mock panphon.distance.Distance
    with patch("loanpy.helpers.Distance") as Distance_mock:
        Distance_mock.return_value = mockdist
        out = read_dst("weighted_feature_edit_distance")

        # assert
        assert ismethod(out)  # check if it is a method of the mocked class

    # assert calls
    assert out == mockdist.weighted_feature_edit_distance
    Distance_mock.assert_called_with()  # the class was called without args

    # tear down
    del mockdist, out, MonkeyDist


def test_flatten():
    """check if nested lists are flattened and "" thrown out"""
    assert flatten([["a", "b"], ["c"]]) == ["a", "b", "c"]
    assert flatten([["wrd1", "wrd2", ""], ["wrd3", "", ""]]) == [
        "wrd1", "wrd2", "wrd3"]


def test_combine_ipalists():
    """test if old sounds are combined correctly"""

    # set up
    with patch("loanpy.helpers.flatten") as flatten_mock:
        flatten_mock.return_value = ["ki", "ke", "gi", "bu", "bo", "pu"]
        with patch("loanpy.helpers.product", side_effect=[
            [('k', 'i'), ('k', 'e'), ('g', 'i'), ('g', 'e')],
                [('b', 'u'), ('b', 'o'),
                 ('p', 'u'), ('p', 'o')]]) as product_mock:
            inlist = [[["k", "g"], ["i", "e"]], [["b", "p"], ["u", "o"]]]
            out = ["ki", "ke", "gi", "bu", "bo", "pu"]

            # assert
            assert combine_ipalists(inlist) == out

    # assert calls
    flatten_mock.assert_called_with(
        [["ki", "ke", "gi", "ge"], ["bu", "bo", "pu", "po"]])
    product_mock.assert_has_calls([call(["k", "g"], ["i", "e"]),
                                   call(["b", "p"], ["u", "o"])])

    # tear down
    del inlist, out


def test_forms2list():
    "test if dff is converted to a list correctly"
    # test first break: return None if dff is None
    assert forms2list(None, "sth") is None

    # set up fake input df
    mock_df_in = DataFrame({"Segments": ["abc", "def", "pou"],
                            "Cognacy": [1, 1, 2],
                            "Language_ID": ["lg2", "lg1", "lg2"]})
    assert forms2list(mock_df_in, "lg2") == ["abc", "pou"]


def test_init():
    """test if class Etym is initiated correctly without args"""

    # set up (spaghetti alarm)
    with patch("loanpy.helpers.read_forms") as read_forms_mock:
        read_forms_mock.return_value = None
        with patch("loanpy.helpers.forms2list") as forms2list_mock:
            forms2list_mock.return_value = None
            with patch("loanpy.helpers.read_cvfb") as read_cvfb_mock:
                read_cvfb_mock.return_value = ({"d1": 123}, {"d2": 456})
                with patch("loanpy.helpers.cldf2pd") as cldf2pd_mock:
                    cldf2pd_mock.return_value = None
                    with patch("loanpy.helpers.Etym.get_inventories"
                               ) as get_inventories_mock:
                        get_inventories_mock.return_value = (None, None, None)
                        with patch("loanpy.helpers.read_dst"
                                   ) as read_dst_mock:
                            read_dst_mock.return_value = "distfunc"

                            # initiate without args
                            mocketym = Etym()

                            # assert if initiation went correctly
                            assert mocketym.phon2cv == {"d1": 123}
                            assert mocketym.vow2fb == {"d2": 456}
                            assert mocketym.dfety is None
                            assert mocketym.phoneme_inventory is None
                            assert mocketym.cluster_inventory is None
                            assert mocketym.phonotactic_inventory is None
                            assert mocketym.distance_measure == "distfunc"

                            # double check with __dict__
                            assert len(mocketym.__dict__) == 8
                            assert mocketym.__dict__ == {
                                'cluster_inventory': None,
                                'phoneme_inventory': None,
                                'dfety': None,
                                'distance_measure': 'distfunc',
                                'forms_target_language': None,
                                'phon2cv': {'d1': 123},
                                'phonotactic_inventory': None,
                                'vow2fb': {'d2': 456}}

    # assert calls
    read_forms_mock.assert_called_with(None)
    forms2list_mock.assert_called_with(None, None)
    read_cvfb_mock.assert_called_with()
    cldf2pd_mock.assert_called_with(
        None, None, None)
    get_inventories_mock.assert_called_with(None, None, None, 9999999)
    read_dst_mock.assert_called_with(
        "weighted_feature_edit_distance")

    # tear down
    del mocketym

    # set up2
    dfmk = DataFrame({"Segments": ["abc", "def", "pou"],
                      "Cognacy": [1, 1, 2],
                      "Language_ID": ["lg2", "lg1", "lg2"]})
    with patch("loanpy.helpers.read_forms") as read_forms_mock:
        read_forms_mock.return_value = dfmk
        with patch("loanpy.helpers.forms2list") as forms2list_mock:
            forms2list_mock.return_value = ["abc", "pou"]
            with patch("loanpy.helpers.read_cvfb") as read_cvfb_mock:
                read_cvfb_mock.return_value = ("sth1", "sth2")
                with patch("loanpy.helpers.cldf2pd") as cldf2pd_mock:
                    cldf2pd_mock.return_value = "sth3"
                    with patch("loanpy.helpers.Etym.get_inventories"
                               ) as get_inventories_mock:
                        get_inventories_mock.return_value = (
                            "sth4", "sth5", "sth6")
                        with patch("loanpy.helpers.read_dst") as read_dst_mock:
                            read_dst_mock.return_value = "sth7"

                            # initiate with pseudo arguments
                            mocketym = Etym(
                                forms_csv="path", source_language="lg1",
                                target_language="lg2")

                            # assert if initiation went right
                            assert mocketym.phon2cv == "sth1"
                            assert mocketym.vow2fb == "sth2"
                            assert mocketym.dfety == "sth3"
                            assert mocketym.phoneme_inventory == "sth4"
                            assert mocketym.cluster_inventory == "sth5"
                            assert mocketym.phonotactic_inventory == "sth6"
                            assert mocketym.distance_measure == "sth7"

                            # double check with __dict__
                            assert len(mocketym.__dict__) == 8
                            assert mocketym.__dict__ == {
                                'cluster_inventory': "sth5",
                                'phoneme_inventory': "sth4",
                                'dfety': "sth3",
                                'distance_measure': 'sth7',
                                'forms_target_language': ['abc', 'pou'],
                                'phon2cv': "sth1",
                                'phonotactic_inventory': "sth6",
                                'vow2fb': "sth2"}

    # assert calls
    read_forms_mock.assert_called_with("path")
    forms2list_mock.assert_called_with(dfmk, "lg2")
    read_cvfb_mock.assert_called_with()
    cldf2pd_mock.assert_called_with(dfmk, "lg1", "lg2")
    get_inventories_mock.assert_called_with(None, None, None, 9999999)
    read_dst_mock.assert_called_with(
        "weighted_feature_edit_distance")

    # tear down
    del mocketym, dfmk


def test_read_inventory():
    """check if phoneme inventory is extracted correctly"""

    class EtymMonkey:
        pass
    etym_monkey = EtymMonkey()
    # assert where no setup needed
    etym_monkey.forms_target_language = "whatever"
    assert Etym.read_inventory(etym_monkey, "whatever2") == "whatever2"
    etym_monkey.forms_target_language = None
    assert Etym.read_inventory(etym_monkey, None) is None

    # set up
    # this is the vocabulary of the language
    etym_monkey.forms_target_language = ["a", "aab", "bc"]
    with patch("loanpy.helpers.tokenise") as tokenise_mock:
        # these are all letters of the language
        tokenise_mock.return_value = ['a', 'a', 'a', 'b', 'b', 'c']

        # assert
        assert Etym.read_inventory(etym_monkey,
                                   None, tokenise_mock) == set(['a', 'b', 'c'])

    # assert calls
    tokenise_mock.assert_called_with("aaabbc")

    # set up2: for clusterise
    etym_monkey.forms_target_language = ["a", "ab", "baac"]
    with patch("loanpy.helpers.clusterise") as clusterise_mock:
        clusterise_mock.return_value = [
            'aa', 'bb', 'aa', 'c']  # clusterised vocab

        # assert
        assert Etym.read_inventory(
            etym_monkey, None, clusterise_mock) == set(['aa', 'bb', 'c'])

    # assert calls
    # all words are pulled together to one string
    clusterise_mock.assert_called_with("aabbaac")

    # tear down
    del etym_monkey, EtymMonkey


def test_get_inventories():
    """test if phoneme/cluster/phonotactic inventories are read in well"""
    # set up
    class EtymMonkey():
        def __init__(self):
            self.read_inventory_called_with = []
            self.read_phonotactic_inv_called_with = []

        def read_inventory(self, *args):
            self.read_inventory_called_with.append([*args])
            return "read_inventory_returned_this"

        def read_phonotactic_inv(self, *args):
            self.read_phonotactic_inv_called_with.append([*args])
            return "read_phonotactic_inv_returned_this"

    # create instancce
    etym_monkey = EtymMonkey()
    # run func
    assert Etym.get_inventories(self=etym_monkey) == (
        "read_inventory_returned_this",
        "read_inventory_returned_this",
        "read_phonotactic_inv_returned_this"
    )

    # assert calls
    assert etym_monkey.read_inventory_called_with == [
        [None], [None, hp.clusterise]]
    assert etym_monkey.read_phonotactic_inv_called_with == [[None, 9999999]]

    # run func without default parameters

    # create instancce
    etym_monkey = EtymMonkey()
    # assert assigned attributes
    assert Etym.get_inventories(etym_monkey, "param1", "param2", "param3", 4
                                ) == ("read_inventory_returned_this",
                                      "read_inventory_returned_this",
                                      "read_phonotactic_inv_returned_this")
    # assert calls
    assert etym_monkey.read_inventory_called_with == [["param1"], [
        "param2", hp.clusterise]]
    assert etym_monkey.read_phonotactic_inv_called_with == [["param3", 4]]

    # tear down
    del etym_monkey, EtymMonkey


def test_read_phonotactic_inv():
    """test if inventory of phonotactic structures is extracted correctly"""

    # set up custom class
    class EtymMonkeyReadstrucinv:
        def __init__(self):
            self.forms_target_language = ["ab", "ab", "aa", "bb", "bb", "bb"]
            self.phonotactics_readstrucinv = iter(
                ["VV", "VC", "VC", "CC", "CC", "CC"])
            self.called_with = []

        def word2phonotactics(self, word):
            self.called_with.append(word)
            return next(self.phonotactics_readstrucinv)

    # set up rest
    mocketym = EtymMonkeyReadstrucinv()

    # assert with different parameter combinations
    assert Etym.read_phonotactic_inv(self=mocketym, phonotactic_inventory=[
        "a", "b", "c"]) == ["a", "b", "c"]
    mocketym.forms_target_language = None
    assert Etym.read_phonotactic_inv(self=mocketym, phonotactic_inventory=None,
                                     ) is None
    mocketym.forms_target_language = ["ab", "ab", "aa", "bb", "bb", "bb"]
    # now just read the most frquent 2 structures. VV is the 3rd frquent. so
    # not in the output.
    assert Etym.read_phonotactic_inv(self=mocketym, phonotactic_inventory=None,
                                     howmany=2) == {"CC", "VC"}

    # assert calls
    assert mocketym.called_with == mocketym.forms_target_language

    # tear down
    del mocketym, EtymMonkeyReadstrucinv


def test_word2phonotactics():
    """test is the phonotactic structure of a word is returned correctly"""

    # set up1
    mocketym = EtymMonkey()  # empty global mockclass
    mocketym.phon2cv = {"a": "V", "b": "C"}  # lets say "c" is missing

    # assert without mocking tokenise (input is tokenised already)
    # sounds missing from dict are ignored
    assert Etym.word2phonotactics(self=mocketym,
                                  ipa_in=["a", "b", "c"]) == "VC"

    # set up2
    with patch("loanpy.helpers.tokenise") as tokenise_mock:
        tokenise_mock.return_value = ['a', 'b', 'c']

        # assert if it works with tokenisation as well
        assert Etym.word2phonotactics(self=mocketym, ipa_in="abc") == "VC"

        # assert call
        tokenise_mock.assert_called_with("abc")

    # tear down
    del mocketym


def test_word2phonotactics_keepcv():
    """test if phonotactic structure is returned while C and V stay the same"""

    # set up
    mocketym = EtymMonkey()
    mocketym.phon2cv = {"a": "V", "b": "C"}

    # assert
    assert Etym.word2phonotactics_keepcv(
        self=mocketym, ipa_in=[
            'a', 'b', 'c', 'C', 'V']) == "VCCV"
    # the "c" is ignored bc it's not in phon2cv

    # tear down
    del mocketym


def test_harmony():
    """test if a words front-back vowel harmony is inferred correctly"""

    # set up1
    mocketym = EtymMonkey()
    mocketym.vow2fb = {"o": "B", "i": "F"}

    # assert without tokenisation
    assert Etym.has_harmony(
        self=mocketym,
        ipalist=['b', 'o', 't͡s', 'i', 'b', 'o', 't͡s', 'i']) is False

    # set up2: add mock tokeniser
    with patch("loanpy.helpers.tokenise") as tokenise_mock:
        tokenise_mock.return_value = ['b', 'o', 't͡s', 'i',
                                      'b', 'o', 't͡s', 'i']

        # assert with tokenisation
        assert Etym.has_harmony(self=mocketym, ipalist="bot͡sibot͡si") is False

    # assert calls
    tokenise_mock.assert_called_with("bot͡sibot͡si")

    # set up 3: overwrite vow2fb
    mocketym.vow2fb = {"ɒ": "B"}
    # assert
    assert Etym.has_harmony(self=mocketym, ipalist=['t', 'ɒ', 'r', 'k', 'ɒ'])
    # set up 4: add mock tokeniser
    with patch("loanpy.helpers.tokenise") as tokenise_mock:
        tokenise_mock.return_value = ['t', 'ɒ', 'r', 'k', 'ɒ']

        # assert
        assert Etym.has_harmony(self=mocketym, ipalist="tɒrkɒ") is True

    # assert call
    tokenise_mock.assert_called_with("tɒrkɒ")

    # set up 5: overwrite vow2fb
    mocketym.vow2fb = {"ɛ": "F", "y": "F"}
    # assert
    assert Etym.has_harmony(
        self=mocketym,
        ipalist=['ʃ', 'ɛ', 'f', 'y', 'l', 'ɛ', 'ʃ', 'ɛ'])

    # set up 6: add mock tokeniser
    with patch("loanpy.helpers.tokenise") as tokenise_mock:
        tokenise_mock.return_value = ['ʃ', 'ɛ', 'f', 'y', 'l', 'ɛ', 'ʃ', 'ɛ']

        # assert
        assert Etym.has_harmony(self=mocketym, ipalist="ʃɛfylɛʃɛ") is True

    # assert call
    tokenise_mock.assert_called_with("ʃɛfylɛʃɛ")

    # tear down
    del mocketym


def test_repair_harmony():
    """test if a words vowelharmony is repaired correctly"""

    # set up1: custom class, create an instance of it, mock tokeniser
    class EtymMonkeyHarmonyTrue:
        def has_harmony(self, ipalist):
            self.called_with = ipalist
            return True
    mocketym = EtymMonkeyHarmonyTrue()
    with patch("loanpy.helpers.tokenise") as tokenise_mock:
        tokenise_mock.return_value = ['k', 'ɛ', 's', 't', 'h', 'ɛ', 'j']

        # assert that nothing happens if input word has vowel harmony
        assert Etym.repair_harmony(self=mocketym, ipalist='kɛsthɛj') == [
            ['k', 'ɛ', 's', 't', 'h', 'ɛ', 'j']]

    # assert calls
    assert mocketym.called_with == ['k', 'ɛ', 's', 't', 'h', 'ɛ', 'j']
    tokenise_mock.assert_called_with('kɛsthɛj')

    # set up2: new custom class where has_harmony returns False,
    # overwrite mocketym with it, add vow2fb-dict to it, mock tokeniser
    class EtymMonkeyHarmonyFalse:
        def __init__(self, *get_fb_returns):
            self.get_fb_returns = iter([*get_fb_returns])
            self.get_fb_called_with = []

        def has_harmony(self, ipalist):
            self.harmony_called_with = ipalist
            return False

        def get_fb(self, ipalist, turnto):
            self.get_fb_called_with.append((ipalist, turnto))
            return next(self.get_fb_returns)

    mocketym = EtymMonkeyHarmonyFalse(['ɒ', 'l', 'ʃ', 'oː', 'B', 'r', 'ʃ'])
    mocketym.vow2fb = {"ɒ": "B", "oː": "B", "ø": "F"}
    with patch("loanpy.helpers.tokenise") as tokenise_mock:
        tokenise_mock.return_value = ['ɒ', 'l', 'ʃ', 'oː', 'ø', 'r', 'ʃ']

        # assert that the wrong front vowel ø is replaced by "B"
        assert Etym.repair_harmony(self=mocketym, ipalist='ɒlʃoːørʃ') == [
            ['ɒ', 'l', 'ʃ', 'oː', 'B', 'r', 'ʃ']]

        # assert calls
        assert mocketym.get_fb_called_with == [
            (['ɒ', 'l', 'ʃ', 'oː', 'ø', 'r', 'ʃ'], "B")]
        assert mocketym.harmony_called_with == [
            'ɒ', 'l', 'ʃ', 'oː', 'ø', 'r', 'ʃ']
        tokenise_mock.assert_called_with("ɒlʃoːørʃ")

    # set up3, overwrite mock class instance with new input, overwrite vow2fb
    mocketym = EtymMonkeyHarmonyFalse(
        ['b', 'eː', 'l', 'F', 't', 'ɛ', 'l', 'ɛ', 'p'])
    mocketym.vow2fb = {"eː": "F", "ɒ": "B", "ɛ": "F"}

    # assert that the wrong front vowel "ɒ" is replace by "F"
    assert Etym.repair_harmony(
        self=mocketym,
        ipalist=['b', 'eː', 'l', 'ɒ', 't', 'ɛ', 'l', 'ɛ', 'p']) == [
        ['b', 'eː', 'l', 'F', 't', 'ɛ', 'l', 'ɛ', 'p']]
    # assert calls
    assert mocketym.get_fb_called_with == [
        (['b', 'eː', 'l', 'ɒ', 't', 'ɛ', 'l', 'ɛ', 'p'], "F")]
    assert mocketym.harmony_called_with == [
        'b', 'eː', 'l', 'ɒ', 't', 'ɛ', 'l', 'ɛ', 'p']

    # set up4: define repetitive variables, create new instance of mockclass,
    # overwrite vow2fb
    bk = ['b', 'ɒ', 'l', 'ɒ', 't', 'o', 'n', 'k', 'ɛ', 'n', 'ɛ', 'ʃ', 'ɛ']
    bkf = ['b', 'F', 'l', 'F', 't', 'F', 'n', 'k', 'ɛ', 'n', 'ɛ', 'ʃ', 'ɛ']
    bkb = ['b', 'ɒ', 'l', 'ɒ', 't', 'o', 'n', 'k', 'B', 'n', 'B', 'ʃ', 'B']
    mocketym = EtymMonkeyHarmonyFalse(bkb, bkf)
    mocketym.vow2fb = {"ɒ": "B", "o": "B", "ɛ": "F"}

    # assert words without vowelharmony with equally many front and back vowels
    # are repaired in both possible ways
    assert Etym.repair_harmony(
        self=mocketym, ipalist=list('bɒlɒtonkɛnɛʃɛ')) == [bkb, bkf]

    # assert calls
    assert mocketym.harmony_called_with == bk
    assert mocketym.get_fb_called_with == [(bk, "F"), (bk, "B")]

    # tear down
    del mocketym, bk, bkf, bkb, EtymMonkeyHarmonyTrue, EtymMonkeyHarmonyFalse


def test_get_fb():
    """test if vowels violating vowelharmony turn into "F" or "B" correctly"""

    # set up1: instance of mock class, plug in vow2fb
    mocketym = EtymMonkey()
    mocketym.vow2fb = {"ɒ": "B", "ɛ": "F"}

    # assert nothing happens if all vowels are frount and turnto='F'
    assert Etym.get_fb(
        self=mocketym,
        ipalist=['k', 'ɛ', 's', 't', 'h', 'ɛ', 'j']) == [
        'k', 'ɛ', 's', 't', 'h', 'ɛ', 'j']

    # set up2: plug in new vow2fb
    mocketym.vow2fb = {"ɒ": "B", "oː": "B", "ø": "F"}

    # assert front vowels are turned to "B"
    assert Etym.get_fb(
        self=mocketym, ipalist=[
            'ɒ', 'l', 'ʃ', 'oː', 'ø', 'r', 'ʃ'], turnto="B") == [
        'ɒ', 'l', 'ʃ', 'oː', 'B', 'r', 'ʃ']

    # set up3: overwrite vow2fb
    mocketym.vow2fb = {"ɛ": "F", "o": "B", "ɒ": "B"}

    # assert vowels violating vowel harmony can be turned to "B" or "F"
    assert Etym.get_fb(
        self=mocketym,
        ipalist=['b', 'ɒ', 'l', 'ɒ', 't', 'o', 'n',
                 'k', 'ɛ', 'n', 'ɛ', 'ʃ', 'ɛ'],
        turnto="B") == ['b', 'ɒ', 'l', 'ɒ', 't', 'o', 'n',
                        'k', 'B', 'n', 'B', 'ʃ', 'B']
    assert Etym.get_fb(
        self=mocketym,
        ipalist=['b', 'ɒ', 'l', 'ɒ', 't', 'o', 'n',
                 'k', 'ɛ', 'n', 'ɛ', 'ʃ', 'ɛ'],
        turnto="F") == ['b', 'F', 'l', 'F', 't', 'F', 'n',
                        'k', 'ɛ', 'n', 'ɛ', 'ʃ', 'ɛ']

    # tear down
    del mocketym


def test_get_scdictbase():
    """test if heuristic prediction of sound substitutions works,
    i.e. if phoneme inventory can be ranked
    according to feature vector distance to any sound"""

    # set up mockclass, mock read_csv, create instance of mockclass,
    # plug phoneme_inventory, phon2cv, vow2fb into it
    # set up 2 path variables to test files
    # define expected output as variable
    # define mock function for tqdm and plug it into lonapy.helpers
    class EtymMonkeyget_scdictbase:
        def __init__(self):
            self.rnkcls = iter(["e, f, d", "d, f, e", "f, d, e", "f, d", "e",
                                "e", "d", "f", "f", "e"])
            self.rank_closest_called_with = []

        def rank_closest(self, *args):
            self.rank_closest_called_with.append([*args])
            return next(self.rnkcls)
    with patch("loanpy.helpers.read_csv") as read_csv_mock:
        read_csv_mock.return_value = DataFrame({"ipa": ["a", "b", "c"]})
        mocketym = EtymMonkeyget_scdictbase()
        mocketym.phoneme_inventory = ["d", "e", "f"]
        mocketym.phon2cv = {"d": "C", "e": "V", "f": "C"}
        mocketym.vow2fb = {"e": "F"}
        path2test_scdictbase = Path(__file__).parent / "test_scdictbase.txt"
        exp = {"a": ["e", "f", "d"],
               "b": ["d", "f", "e"],
               "c": ["f", "d", "e"],
               "C": ["f", "d"],
               "V": ["e"],
               "F": ["e"],
               "B": []}
        exp2 = {"a": ["e"],
                "b": ["d"],
                "c": ["f"],
                "C": ["f"],
                "V": ["e"],
                "F": ["e"],
                "B": []}

        def tqdm_mock(pdseries):
            tqdm_mock.called_with = pdseries
            return pdseries
        tqdm = hp.tqdm
        hp.tqdm = tqdm_mock

        # assert if output is returned, asigned to self, and written correctly
        assert Etym.get_scdictbase(
            self=mocketym, write_to=path2test_scdictbase) == exp
        assert mocketym.scdictbase == exp
        with open(path2test_scdictbase, "r", encoding="utf-8") as f:
            assert literal_eval(f.read()) == exp

        # assert correct output with howmany=1 instead of inf
        assert Etym.get_scdictbase(
            self=mocketym,
            write_to=path2test_scdictbase,
            most_common=1) == exp2
        assert mocketym.scdictbase == exp2
        with open(path2test_scdictbase, "r", encoding="utf-8") as f:
            assert literal_eval(f.read()) == exp2

    # assert calls
    assert_series_equal(hp.tqdm.called_with, Series(
        ["a", "b", "c"], name="ipa"))
    read_csv_mock.assert_called_with(
        Path(__file__).parent.parent / "ipa_all.csv")
    assert mocketym.rank_closest_called_with == [
        ['a', float("inf")], ['b', float("inf")], ['c', float("inf")],
        ['ə', float("inf"), ['d', 'f']], ['ə', float("inf"), ['e']],
        ['a', 1], ['b', 1], ['c', 1], ['ə', 1, ['d', 'f']], ['ə', 1, ['e']]]
    # tear down
    hp.tqdm = tqdm
    remove(path2test_scdictbase)
    del mocketym, path2test_scdictbase, exp, tqdm, EtymMonkeyget_scdictbase


def test_rank_closest():
    """test if phoneme-inventory is ranked correctly
    according to feature vectore distance to a given phoneme"""

    # set up custom class, create instance of it
    class EtymMonkeyrank_closest:
        def __init__(self):
            self.phoneme_inventory, self.dm_called_with = None, []
            self.dm_return = iter([1, 0, 2])

        def distance_measure(self, *args):
            arglist = [*args]
            self.dm_called_with.append(arglist)
            return next(self.dm_return)

    mocketym = EtymMonkeyrank_closest()

    # assert exception and exception message
    with raises(InventoryMissingError) as inventorymissingerror_mock:
        Etym.rank_closest(
            self=mocketym,
            ph="d",
            howmany=float("inf"),
            inv=None)
    assert str(inventorymissingerror_mock.value
               ) == "define phoneme inventory or forms.csv"

    # set up2: mock pick_minmax
    with patch("loanpy.helpers.pick_minmax") as pick_minmax_mock:
        pick_minmax_mock.return_value = ["b", "a", "c"]

        # assert
        assert Etym.rank_closest(
            self=mocketym, ph="d", inv=[
                "a", "b", "c"]) == "b, a, c"

    # assert calls
    assert mocketym.dm_called_with == [['d', 'a'], ['d', 'b'], ['d', 'c']]
    pick_minmax_mock.assert_called_with(
        [('a', 1), ('b', 0), ('c', 2)], float("inf"))

    # set up3: overwrite mock class instance, mock pick_minmax anew
    mocketym = EtymMonkeyrank_closest()
    with patch("loanpy.helpers.pick_minmax") as pick_minmax_mock:
        pick_minmax_mock.return_value = ["b", "a"]

        # assert pick_minmax picks mins correctly again
        assert Etym.rank_closest(
            self=mocketym, ph="d", inv=[
                "a", "b", "c"], howmany=2) == "b, a"

    # assert calls
    assert mocketym.dm_called_with == [['d', 'a'], ['d', 'b'], ['d', 'c']]
    pick_minmax_mock.assert_called_with([('a', 1), ('b', 0), ('c', 2)], 2)

    # set up4: check if phoneme inventory can be accessed through self
    mocketym = EtymMonkeyrank_closest()
    mocketym.phoneme_inventory = ["a", "b", "c"]
    with patch("loanpy.helpers.pick_minmax") as pick_minmax_mock:
        pick_minmax_mock.return_value = "b"

        # assert pick_minmax picks mins correctly again
        assert Etym.rank_closest(
            self=mocketym,
            ph="d",
            inv=None,
            howmany=1) == "b"

    # assert calls
    assert mocketym.dm_called_with == [['d', 'a'], ['d', 'b'], ['d', 'c']]
    pick_minmax_mock.assert_called_with([('a', 1), ('b', 0), ('c', 2)], 1)

    # tear down
    del mocketym, EtymMonkeyrank_closest


def test_rank_closest_phonotactics():
    """test if getting the distance between to phonotactic structures works"""

    # set up
    mocketym = EtymMonkey()
    mocketym.phonotactic_inventory = None
    with raises(InventoryMissingError) as inventorymissingerror_mock:
        # assert error is raised
        Etym.rank_closest_phonotactics(
            self=mocketym,
            struc="CV",
            howmany=float("inf"))
    # assert error message
    assert str(
        inventorymissingerror_mock.value
    ) == "define phonotactic inventory or forms.csv"

    # set up: create instance of empty mock class,
    #  plug in inventory of phonotactic structures,
    # mock edit_distance_with2ops and pick_minmax
    mocketym = EtymMonkey()
    mocketym.phonotactic_inventory = ["CVC", "CVCVCC"]
    with patch("loanpy.helpers.edit_distance_with2ops", side_effect=[
            1, 0.98]) as edit_distance_with2ops_mock:
        with patch("loanpy.helpers.pick_minmax") as pick_minmax_mock:
            pick_minmax_mock.return_value = ["CVCVCC", "CVC"]

            # assert the correct closest string is picked
            assert Etym.rank_closest_phonotactics(
                self=mocketym, struc="CVCV") == "CVCVCC, CVC"

    # assert calls
    edit_distance_with2ops_mock.assert_has_calls(
        [call("CVCV", "CVC"), call("CVCV", "CVCVCC")])
    pick_minmax_mock.assert_called_with(
        [('CVC', 1), ('CVCVCC', 0.98)], 9999999)

    # tear down
    del mocketym


def test_gensim_multiword():
    """use gensim's built-in test suit to check if it works"""

    # test first where no setup needed
    # test first break without returning wordpair
    assert gensim_multiword(
        recip_transl=None,
        donor_transl=0.1,
        return_wordpair=False) == -1
    # test first break with returning wordpair
    assert gensim_multiword(recip_transl=None,
                            donor_transl=0.1,
                            return_wordpair=True) == (-1,
                                                      "!<class 'NoneType'>!",
                                                      "!<class 'float'>!")

    class GensimMonkey:
        def __init__(self): pass

        def similarity(self, word1, word2): return 0

        def has_index_for(self, w): return True
    # set up: mock gensim.api.load:
    with patch("loanpy.helpers.load") as load_mock:
        mockgensim = GensimMonkey()
        load_mock.return_value = mockgensim

        # assert gensim_multiword's result
        assert gensim_multiword("word1", "word2") == 0
        # assert api would have loadad the correct model
        assert hp.model == mockgensim

        # assert call
        load_mock.assert_called_with("word2vec-google-news-300")

        # tear down hp.model
        hp.model = None

        # plug in different wordvectors
        # assert gensim_multiword's result
        assert gensim_multiword("word1", "word2", wordvectors="somemodel") == 0
        # assert api would have loadad the correct model
        assert hp.model == mockgensim

    # assert call
    load_mock.assert_called_with("somemodel")

    # todo: find out how to mock a MemoryError

    # set up2: plug in a mock word2vec model from gensim's test suite
    hp.model = word2vec.Word2Vec(common_texts, min_count=1).wv

    # assert that some distances are calculated correctly
    assert gensim_multiword("human, computer",
                            "interface") == float32(0.10940766334533691)
    assert gensim_multiword(
        "human, computer",
        "interface",
        return_wordpair=True) == (
        float32(0.10940766),
        'human',
        'interface')
    assert gensim_multiword("computer, human",
                            "interface") == float32(0.10940766334533691)
    assert gensim_multiword(
        "computer, human",
        "interface",
        return_wordpair=True) == (
        float32(0.10940766),
        'human',
        'interface')

    # assert KeyError is skipped
    assert gensim_multiword("human, missingword",
                            "interface") == float32(0.10940766334533691)
    assert gensim_multiword(
        "human, missingword",
        "interface",
        return_wordpair=True) == (
        float32(0.10940766),
        'human',
        'interface')

    # assert missing words result in similarity score of -1
    assert gensim_multiword("human, computer", "missingword") == float32(-1)

    # assert missing src word shows right warning if wordpairs are returned
    assert gensim_multiword("human, computer",
                            "missingword",
                            return_wordpair=True) == (float32(-1),
                                                      '',
                                                      'source word \
not in model')
    # assert missing tgt word shows right warning if wordpairs are returned
    assert gensim_multiword("missingword",
                            "human, computer",
                            return_wordpair=True) == (float32(-1),
                                                      'target word \
not in model',
                                                      '')
    # assert right warning if both words are missing
    assert gensim_multiword("missing1", "missing2", return_wordpair=True) == (
        float32(-1), 'target word not in model', 'source word not in model')

    # assert loop is interrupted as soon as similarity score == 1
    # loop is certainly broken because "1" is not a numpy.float32
    assert gensim_multiword("human, computer", "computer") == 1
    assert gensim_multiword(
        "human, computer",
        "computer",
        return_wordpair=True) == (
        1,
        'computer',
        'computer')

    # tear down hp.model by setting it to its previous value (None)
    hp.model = None  # del hp.model would lead to errors
    del GensimMonkey


def test_list2regex():
    """test if list of phonemes is correctly converted to regular expression"""
    assert list2regex(["b", "k", "v"]) == "(b|k|v)"
    assert list2regex(["b", "k", "-", "v"]) == "(b|k|v)?"
    assert list2regex(["b", "k", "-", "v", "mp"]) == "(b|k|v|mp)?"
    assert list2regex(["b", "k", "-", "v", "mp", "mk"]) == "(b|k|v|mp|mk)?"
    assert list2regex(["o"]) == '(o)'
    assert list2regex(["ʃʲk"]) == '(ʃʲk)'
    assert list2regex(["-"]) == ""


def test_edit_distance_with2ops():
    """test if editdistances are calculated correctly"""

    # default weight is 100 per deletion and 49 per insertion
    # in 80 tests around the world
    assert edit_distance_with2ops("ajka", "Rajka") == 49
    assert edit_distance_with2ops("Rajka", "ajka") == 100
    assert edit_distance_with2ops("Debrecen", "Mosonmagyaróvár") == 1386
    assert edit_distance_with2ops("Bécs", "Hegyeshalom") == 790
    assert edit_distance_with2ops("Hegyeshalom", "Mosonmagyaróvár") == 1388
    assert edit_distance_with2ops("Mosonmagyaróvár", "Győr") == 1398
    # 4 del + 4 ins = 4*49+4*100
    assert edit_distance_with2ops("Győr", "Tata") == 596
    assert edit_distance_with2ops("Tata", "Tatabánya") == 245  # 5 ins: 5*49
    assert edit_distance_with2ops("Tatabánya", "Budapest") == 994
    assert edit_distance_with2ops("Budapest", "Komárom") == 1143
    # 4 ins + 1 del: 4*49+100
    assert edit_distance_with2ops("Komárom", "Révkomárom") == 296
    # 4 del + 1 ins: 4*100+49
    assert edit_distance_with2ops("Révkomárom", "Komárom") == 449
    assert edit_distance_with2ops("Komárom", "Budapest") == 1092
    assert edit_distance_with2ops("Budapest", "Debrecen") == 1043
    assert edit_distance_with2ops("Debrecen", "Beregszász") == 843
    assert edit_distance_with2ops("Beregszász", "Kiev") == 1047
    assert edit_distance_with2ops("Kiev", "Moszkva") == 594
    assert edit_distance_with2ops("Moszkva", "Szenpétervár") == 990
    assert edit_distance_with2ops("Szentpétervár", "Vlagyivosztok") == 1639
    assert edit_distance_with2ops("Vlagyivosztok", "Tokió") == 1247
    assert edit_distance_with2ops("Tokió", "New York") == 594
    assert edit_distance_with2ops("New York", "Bécs") == 996

    # check if custom weights for insertion work. deletion always costs 1.
    assert edit_distance_with2ops("ajka", "Rajka", w_ins=90) == 90
    assert edit_distance_with2ops("Rajka", "ajka", w_ins=90) == 100
    assert edit_distance_with2ops(
        "Debrecen", "Mosonmagyaróvár", w_ins=90) == 1960


def test_get_mtx():
    """test if distance matrix between two words is set up correctly"""

    # set up by mocking numpy.zeros and defining the expected output
    with patch("loanpy.helpers.zeros") as zeros_mock:
        zeros_mock.return_value = array([[0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0.]])

        exp = array([[0., 1., 2., 3., 4.],
                     [1., 2., 3., 4., 5.],
                     [2., 3., 2., 3., 4.],
                     [3., 4., 3., 2., 3.],
                     [4., 5., 4., 3., 2.]])

        # assert
        assert_array_equal(get_mtx("Bécs", "Pécs"), exp)

    # assert call
    zeros_mock.assert_called_with((5, 5))

    # tear down
    del exp


def test_mtx2graph():
    """test if numpy matrix is correctly converted to
     networkx DiGraph (directed graph)"""

    # set up by mocking get_mtx and defining actual and expected output
    # output is tuple: element 1 is a Graph, two is the height, 3 the width.
    # expG is expected output graph. It's a directed graph. The tuples show the
    # nodes in the coordinate system between which the edges go,
    # the number next to the tuples is the weight of each edge
    # graph objects store edges in arbitrary order, cant compare them directly
    # instead have to turn their data into lists and compare their sets

    with patch("loanpy.helpers.get_mtx") as get_mtx_mock:
        get_mtx_mock.return_value = array([[0., 1., 2.],
                                           [1., 2., 3.],
                                           [2., 3., 2.]])
        expG = DiGraph()
        expG.add_weighted_edges_from([((2, 2), (2, 1), 100),
                                      ((2, 2), (1, 2), 49),
                                      ((2, 2), (1, 1), 0),
                                      ((2, 1), (2, 0), 100),
                                      ((2, 1), (1, 1), 49),
                                      ((2, 0), (1, 0), 49),
                                      ((1, 2), (1, 1), 100),
                                      ((1, 2), (0, 2), 49),
                                      ((1, 1), (1, 0), 100),
                                      ((1, 1), (0, 1), 49),
                                      ((1, 0), (0, 0), 49),
                                      ((0, 2), (0, 1), 100),
                                      ((0, 1), (0, 0), 100)])
        exp = [(e, datadict["weight"]) for e, datadict in expG.edges.items()]

        outtuple = mtx2graph("ló", "hó")
        out = [(e, datadict["weight"])
               for e, datadict in outtuple[0].edges.items()]

        # assert expected and actual output is the same
        assert len(outtuple) == 3
        assert isinstance(outtuple, tuple)
        assert set(out) == set(exp)
        # the height. always 1 longer than the word bc + "#" (#ló)
        assert outtuple[1] == 3
        assert outtuple[2] == 3  # the width.

    # assert call
    get_mtx_mock.assert_called_with("ló", "hó")

    # set up2: assert weights are passed on correctly
    with patch("loanpy.helpers.get_mtx") as get_mtx_mock:
        get_mtx_mock.return_value = array([[0., 1., 2.],
                                           [1., 2., 3.],
                                           [2., 3., 2.]])
        expG = DiGraph()
        expG.add_weighted_edges_from([((2, 2), (2, 1), 11),
                                      ((2, 2), (1, 2), 7),
                                      ((2, 2), (1, 1), 0),
                                      ((2, 1), (2, 0), 11),
                                      ((2, 1), (1, 1), 7),
                                      ((2, 0), (1, 0), 7),
                                      ((1, 2), (1, 1), 11),
                                      ((1, 2), (0, 2), 7),
                                      ((1, 1), (1, 0), 11),
                                      ((1, 1), (0, 1), 7),
                                      ((1, 0), (0, 0), 7),
                                      ((0, 2), (0, 1), 11),
                                      ((0, 1), (0, 0), 11)])
        exp = [(e, datadict["weight"]) for e, datadict in expG.edges.items()]

        outtuple = mtx2graph("ló", "hó", w_del=11, w_ins=7)
        out = [(e, datadict["weight"])
               for e, datadict in outtuple[0].edges.items()]

        # assert expected and actual output is the same
        assert len(outtuple) == 3
        assert isinstance(outtuple, tuple)
        assert set(out) == set(exp)
        # the height. always 1 longer than the word bc + "#" (#ló)
        assert outtuple[1] == 3
        assert outtuple[2] == 3  # the width.

    # assert call
    get_mtx_mock.assert_called_with("ló", "hó")

    # tear down
    del out, exp, expG, outtuple


def test_tuples2editops():
    """assert that edit operations coded as tuples
    are converted to natural language correctly"""

    # set up by mocking numpy.subtract and numpy.array_equiv
    arrayequiv_expcalls = [
        (array([0, 1]), [1, 1]),  # first step diagonal? No.
        (array([0, 1]), [0, 1]),  # first step horizontal? Yes.
        (array([1, 0]), [1, 1]),  # second step diagonal? No.
        (array([1, 0]), [0, 1]),  # second step horizontal? No.
        (array([1, 0]), [1, 0]),  # second step vertical? Yes.
        (array([0, 1]), [0, 1]),  # step before (first step) horizontal? Yes.
        (array([1, 1]), [1, 1])  # third step diagonal? Yes.
    ]

    with patch("loanpy.helpers.subtract", side_effect=[
        array([0, 1]), array([1, 0]), array([0, 1]), array([1, 1])]
    ) as subtract_mock:
        with patch("loanpy.helpers.array_equiv", side_effect=[
                False, True, False, False,
                True, True, True]) as array_equiv_mock:

            # assert list of tuples is correctly converted to list of strings
            assert tuples2editops([(0, 0), (0, 1), (1, 1), (2, 2)],
                                  "ló", "hó") == ['substitute l by h',
                                                    'keep ó']

    # assert calls
    subtract_mock.assert_has_calls([
        call((0, 1), (0, 0)),  # first step = horizontal
        call((1, 1), (0, 1)),  # second step = vertical
        # recheck first step -> convert to "substitute"
        call((0, 1), (0, 0)),
        call((2, 2), (1, 1))  # last step diagonal
    ])
    # cant use assert_called_with bc nparrays can only be compared
    # through assert_array_equal
    for arraypair0, arraypair1 in zip(
            array_equiv_mock.call_args_list, arrayequiv_expcalls):
        # first [0] turns call object to tuple
        assert_array_equal(arraypair0[0][0], arraypair1[0])
        assert_array_equal(arraypair0[0][1], arraypair1[1])

    # tear down
    del arrayequiv_expcalls


def test_editops():
    """test if all shortest paths of editoperations are extracted correctly"""

    # set up by mocking mtx2graph, networkx.all_shortest_paths, tuples2editops
    G = DiGraph()
    G.add_weighted_edges_from(
        [((2, 2), (2, 1), 100),
         ((2, 2), (1, 2), 49),
         ((2, 2), (1, 1), 0),
         ((2, 1), (2, 0), 100),
         ((2, 1), (1, 1), 49),
         ((2, 0), (1, 0), 49),
         ((1, 2), (1, 1), 100),
         ((1, 2), (0, 2), 49),
         ((1, 1), (1, 0), 100),
         ((1, 1), (0, 1), 49),
         ((1, 0), (0, 0), 49),
         ((0, 2), (0, 1), 100),
         ((0, 1), (0, 0), 100)]
    )
    with patch("loanpy.helpers.mtx2graph") as mtx2graph_mock:
        mtx2graph_mock.return_value = (G, 3, 3)
        with patch("loanpy.helpers.shortest_path") as shortest_path_mock:
            shortest_path_mock.return_value = [(2, 2), (1, 1), (0, 1), (0, 0)]
            with patch("loanpy.helpers.tuples2editops", side_effect=[
                    ['substitute l by h', 'keep ó']]) as tuples2editops_mock:

                # assert that 2 strings are correctly converted to editops
                assert editops("ló", "hó") == [('substitute l by h', 'keep ó')]

    # assert calls
    mtx2graph_mock.assert_called_with("ló", "hó", 100, 49)
    shortest_path_mock.assert_called_with(
        G, (2, 2), (0, 0), weight="weight")
    tuples2editops_mock.assert_called_with(
        [(0, 0), (0, 1), (1, 1), (2, 2)], "ló", "hó")

    # set up2: to return 2 paths
    with patch("loanpy.helpers.mtx2graph") as mtx2graph_mock2:
        mtx2graph_mock2.return_value = (G, 3, 4)
        with patch("loanpy.helpers.all_shortest_paths"
                   ) as all_shortest_paths_mock:
            all_shortest_paths_mock.return_value = [
                [(2, 3), (1, 2), (0, 1), (0, 0)],
                [(2, 3), (1, 2), (1, 1), (0, 0)]]
            with patch("loanpy.helpers.tuples2editops", side_effect=[
                ['delete C', 'keep C', 'keep V'],
                ['keep C', 'delete C', 'keep V']]
            ) as tuples2editops_mock:

                # assert that both paths are extracted
                assert editops("CCV", "CV", howmany_paths=2) == [
                    ('delete C', 'keep C', 'keep V'),
                    ('keep C', 'delete C', 'keep V')]

    # assert calls
    mtx2graph_mock2.assert_called_with("CCV", "CV", 100, 49)
    all_shortest_paths_mock.assert_called_with(
        G, (2, 3), (0, 0), weight="weight")
    tuples2editops_mock.assert_has_calls([
        call([(0, 0), (0, 1), (1, 2), (2, 3)], "CCV", "CV"),
        call([(0, 0), (1, 1), (1, 2), (2, 3)], "CCV", "CV")])

    # set up3: assert weights are passed on correctly
    with patch("loanpy.helpers.mtx2graph") as mtx2graph_mock2:
        mtx2graph_mock2.return_value = (G, 3, 4)
        with patch("loanpy.helpers.all_shortest_paths"
                   ) as all_shortest_paths_mock:
            all_shortest_paths_mock.return_value = [
                [(2, 3), (1, 2), (0, 1), (0, 0)],
                [(2, 3), (1, 2), (1, 1), (0, 0)]]
            with patch("loanpy.helpers.tuples2editops", side_effect=[
                ['delete C', 'keep C', 'keep V'],
                ['keep C', 'delete C', 'keep V']]
            ) as tuples2editops_mock:

                # assert that both paths are extracted
                assert editops("CCV", "CV", howmany_paths=2,
                               w_del=4, w_ins=35) == [
                    ('delete C', 'keep C', 'keep V'),
                    ('keep C', 'delete C', 'keep V')]

    # assert calls
    mtx2graph_mock2.assert_called_with("CCV", "CV", 4, 35)
    all_shortest_paths_mock.assert_called_with(
        G, (2, 3), (0, 0), weight="weight")
    tuples2editops_mock.assert_has_calls([
        call([(0, 0), (0, 1), (1, 2), (2, 3)], "CCV", "CV"),
        call([(0, 0), (1, 1), (1, 2), (2, 3)], "CCV", "CV")])

    # tear down
    del G


def test_apply_edit():
    """test if editoperations are correctly applied to words"""
    assert apply_edit("ló", ('substitute l by h', 'keep ó')) == ['h', 'ó']
    assert apply_edit(["l", "ó"],
                      ('substitute l by h', 'keep ó')) == ['h', 'ó']
    assert apply_edit(['f', 'ɛ', 'r', 'i', 'h', 'ɛ', 'ɟ'],
                      ('insert d',
                       'insert u',
                       'insert n',
                       'insert ɒ',
                       'insert p',
                       'substitute f by ɒ',
                       'delete ɛ',
                       'keep r',
                       'delete i',
                       'delete h',
                       'delete ɛ',
                       'substitute ɟ by t')
                      ) == ['d', 'u', 'n', 'ɒ', 'p', 'ɒ', 'r', 't']
    assert apply_edit(['t͡ʃ',
                       'ø',
                       't͡ʃ'],
                      ("substitute t͡ʃ by f",
                       "insert r",
                       "keep ø",
                       "substitute t͡ʃ by t͡ʃː")) == ['f', 'r', 'ø', 't͡ʃː']


def test_get_howmany():
    """test if gethowmany correctly returns a tuple whose product is \
as close as possible, but not less than the first number of the input tuple, \
while the two last elements of the \
output tuple are not higher than the two last numbers of the input tuple"""
    assert get_howmany(10, 2, 2) == (3, 2, 2)
    assert get_howmany(100, 2, 2) == (25, 2, 2)
    assert get_howmany(100, 9, 2) == (8, 7, 2)
    assert get_howmany(1000, 9, 2) == (56, 9, 2)
    assert get_howmany(1000, 3, 2) == (167, 3, 2)
    assert get_howmany(0, 0, 0) == (0, 0, 0)
    assert get_howmany(1000, 1000, 1000) == (10, 10, 10)
    assert get_howmany(500, 0, 2) == (500, 0, 2)


def test_pick_minmax():
    """test if correct number of mins/maxs is picked"""
    assert pick_minmax([("a", 5), ("b", 7), ("c", 3)], float("inf")
                       ) == ["c", "a", "b"]
    assert pick_minmax([("a", 5), ("b", 7), ("c", 3)], 1) == ["c"]
    assert pick_minmax([("a", 5), ("b", 7), ("c", 3)], 2) == ["c", "a"]
    assert pick_minmax([("a", 5), ("b", 7), ("c", 3)], 3) == ["c", "a", "b"]
    # test with max
    assert pick_minmax([("a", 5), ("b", 7), ("c", 3)], float("inf"),
                       max) == ["b", "a", "c"]
    assert pick_minmax([("a", 5), ("b", 7), ("c", 3)], 1, max) == ["b"]
    assert pick_minmax([("a", 5), ("b", 7), ("c", 3)], 2, max) == ["b", "a"]
    assert pick_minmax([("a", 5), ("b", 7), ("c", 3)], 3,
                       max) == ["b", "a", "c"]

    # test with return_all=True
    assert pick_minmax([("a", 5), ("b", 7), ("c", 3)], float("inf"), True
                       ) == ["c", "a", "b"]
    assert pick_minmax([("a", 5), ("b", 7), ("c", 3)], 1, min, True
                       ) == ["c", "a", "b"]
    assert pick_minmax([("a", 5), ("b", 7), ("c", 3)], 2, min, True
                       ) == ["c", "a", "b"]
    assert pick_minmax([("a", 5), ("b", 7), ("c", 3)], 3, min, True
                       ) == ["c", "a", "b"]
    # test with max
    assert pick_minmax([("a", 5), ("b", 7), ("c", 3)], float("inf"),
                       max, True) == ["b", "a", "c"]
    assert pick_minmax([("a", 5), ("b", 7), ("c", 3)], 1, max, True
                       ) == ["b", "a", "c"]
    assert pick_minmax([("a", 5), ("b", 7), ("c", 3)], 2, max, True
                       ) == ["b", "a", "c"]
    assert pick_minmax([("a", 5), ("b", 7), ("c", 3)], 3,
                       max, True) == ["b", "a", "c"]
