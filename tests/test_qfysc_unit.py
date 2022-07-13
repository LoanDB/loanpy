"""unit test for loanpy.qfysc.py (2.0 BETA) for pytest 7.1.1"""

from ast import literal_eval
from inspect import ismethod
from os import remove
from pathlib import Path
from unittest.mock import patch, call

from loanpy import qfysc as qs
from loanpy import helpers as hp
from loanpy.qfysc import (
    Etym,
    cldf2pd,
    read_dst,
    read_scdictbase)

from pytest import raises
from pandas import DataFrame, Series
from pandas.testing import assert_frame_equal, assert_series_equal

# used throughout the module


class EtymMonkey:
    pass

# used in multiple places


class PairwiseMonkey:
    def __init__(self, *args):
        self.alignments = [("kala", "hal-")]
        self.align_called_with = []

    def align(self, *args):
        self.align_called_with.append([*args])

# used in multiple places


class EtymMonkeyGetSoundCorresp:
    def __init__(
            self,
            adapting,
            connector,
            alignreturns1,
            alignreturns2,
            vfb=None):
        self.df1 = alignreturns1
        self.df2 = alignreturns2
        self.align_returns = iter([self.df1, self.df2])
        self.align_called_with = []
        self.rank_closest_phonotactics_called_with = []
        self.dfety = DataFrame({"Segments_tgt": ["kiki", "buba"],
                                "Segments_src": ["hehe", "pupa"],
                                "CV_Segments_tgt": ["kiki", "buba"],
                                "CV_Segments_src": ["hehe", "pupa"],
                                "ProsodicStructure_tgt": ["CVCV", "CVCV"],
                                "ProsodicStructure_src": ["CVCV", "CVCV"],
                                "Cognacy": [12, 13]})
        self.adapting = adapting
        self.connector = connector
        self.scdictbase = {"k": ["h", "c"], "i": ["i", "e"], "b": ["v"],
                           "u": ["o", "u"], "a": ["y", "ü"]}
        self.vfb = vfb
        self.phon2cv = {
            "k": "C",
            "i": "V",
            "b": "C",
            "u": "V",
            "a": "V",
            "e": "V"}
        self.vow2fb = {"i": "F", "e": "F", "u": "B", "a": "B"}
        self.inventories = {
        "Segments": {"k", "i", "h", "e", "b", "u", "a", "p"},
        "CV_Segments": {"k", "i", "h", "e", "b", "u", "a", "p"},
        "ProdosicStructure": {"CVCV"}
        }

    def align(self, left, right):
        self.align_called_with.append((left, right))
        return next(self.align_returns)

    def rank_closest_phonotactics(self, struc):
        self.rank_closest_phonotactics_called_with.append(struc)
        return "CVC, CVCCC"

    def get_phonotactics_corresp(self, *args):
        return [{"d1": 0}, {"d2": 0}, {"d3": 0}]


def test_get_inventories():
    """test if phoneme/cluster/phonotactic inventories are read in well"""

    #todo: re-write this test.
    pass


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
            self.rnkcls = iter(["e, f, d", "d, f, e", "f, d, e", "e, f, d",
                                "f, d", "e", "e", "d", "f", "f", "e"])
            self.rank_closest_called_with = []

        def rank_closest(self, *args):
            self.rank_closest_called_with.append([*args])
            return next(self.rnkcls)

    with patch("loanpy.qfysc.read_csv") as read_csv_mock:
        read_csv_mock.return_value = DataFrame({"ipa": ["a", "b", "c", "ə"]})
        mocketym = EtymMonkeyget_scdictbase()
        mocketym.inventories = {"Segments": {"d", "e", "f"}}

        path2test_scdictbase = Path(__file__).parent / "test_scdictbase.txt"
        exp = {"a": ["e", "f", "d"],
               "b": ["d", "f", "e"],
               "c": ["f", "d", "e"],
               "ə": ["e", "f", "d"],
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
        tqdm = qs.tqdm
        qs.tqdm = tqdm_mock

        # assert if output is returned, asigned to self, and written correctly
        assert Etym.get_scdictbase(
            self=mocketym, write_to=path2test_scdictbase) == exp
        assert mocketym.scdictbase == exp
        with open(path2test_scdictbase, "r", encoding="utf-8") as f:
            assert literal_eval(f.read()) == exp

    # tear down
    qs.tqdm = tqdm
    remove(path2test_scdictbase)
    del mocketym, path2test_scdictbase, exp, tqdm, EtymMonkeyget_scdictbase

def test_rank_closest_phonotactics():
    """test if getting the distance between to phonotactic structures works"""

    # set up: create instance of empty mock class,
    #  plug in inventory of phonotactic structures,
    # mock edit_distance_with2ops and nsmallest
    mocketym = EtymMonkey()
    mocketym.inventories = {"ProsodicStructure": {"CVC", "CVCVCC"}}
    with patch("loanpy.qfysc.edit_distance_with2ops", side_effect=[
            1, 0.98]) as edit_distance_with2ops_mock:
        with patch("loanpy.qfysc.nsmallest") as nsmallest_mock:
            nsmallest_mock.return_value = [(1, "CVCVCC"), (2, "CVC")]

            # assert the correct closest string is picked
            assert Etym.rank_closest_phonotactics(
                self=mocketym, struc="CVCV") == "CVCVCC, CVC"

    # assert calls
    edit_distance_with2ops_mock.assert_has_calls(
        [call("CVCV", "CVC"), call("CVCV", "CVCVCC")],
        any_order=True)  # since based on a set!
    try:
        nsmallest_mock.assert_called_with(
            9999999, [(1, 'CVC'), (0.98, 'CVCVCC')])
    except AssertionError:  # since based on a set!
        nsmallest_mock.assert_called_with(
            9999999, [(1, 'CVCVCC'), (0.98, 'CVC')])

    # tear down
    del mocketym

def test_rank_closest():
    """test if phoneme-inventory is ranked correctly
    according to feature vectore distance to a given phoneme"""

    # set up custom class, create instance of it
    class EtymMonkeyrank_closest:
        def __init__(self):
            self.inventories = {"Segments": None}
            self.dm_called_with = []
            self.dm_return = iter([1, 0, 2])

        def distance_measure(self, *args):
            arglist = [*args]
            self.dm_called_with.append(arglist)
            return next(self.dm_return)

    mocketym = EtymMonkeyrank_closest()
    mocketym.inventories["Segments"] = ["a", "b", "c"]

    # set up2: mock nsmallest
    with patch("loanpy.qfysc.nsmallest") as nsmallest_mock:
        nsmallest_mock.return_value = [(1, "b"), (2, "a"), (3, "c")]

        # assert
        assert Etym.rank_closest(
            self=mocketym, ph="d") == "b, a, c"

    # assert calls
    assert mocketym.dm_called_with == [['d', 'a'], ['d', 'b'], ['d', 'c']]
    nsmallest_mock.assert_called_with(
        99999, [(1, 'a'), (0, 'b'), (2, 'c')])

    # set up3: overwrite mock class instance, mock nsmallest anew
    mocketym = EtymMonkeyrank_closest()
    mocketym.inventories["Segments"] = ["a", "b", "c"]
    with patch("loanpy.qfysc.nsmallest") as nsmallest_mock:
        nsmallest_mock.return_value =  [(1, "b"), (2, "a")]

        # assert nsmallest picks mins correctly again
        assert Etym.rank_closest(
            self=mocketym, ph="d", howmany=2) == "b, a"

    # assert calls
    assert mocketym.dm_called_with == [['d', 'a'], ['d', 'b'], ['d', 'c']]
    nsmallest_mock.assert_called_with(2, [(1, 'a'), (0, 'b'), (2, 'c')])

    # set up4: check if phoneme inventory can be accessed through self
    mocketym = EtymMonkeyrank_closest()
    mocketym.inventories["Segments"] = ["a", "b", "c"]
    with patch("loanpy.qfysc.nsmallest") as nsmallest_mock:
        nsmallest_mock.return_value = [(1, "b")]

        # assert nsmallest picks mins correctly again
        assert Etym.rank_closest(
            self=mocketym,
            ph="d",
            howmany=1) == "b"

    # assert calls
    assert mocketym.dm_called_with == [['d', 'a'], ['d', 'b'], ['d', 'c']]
    nsmallest_mock.assert_called_with(1, [(1, 'a'), (0, 'b'), (2, 'c')])

    # tear down
    del mocketym, EtymMonkeyrank_closest

def test_read_scdictbase():
    """test if scdictbase is generated correctly from ipa_all.csv"""

    # no setup needed for this assertion
    assert read_scdictbase(None) == {}

    # setup
    base = {"a": ["e", "o"], "b": ["p", "v"]}
    path = Path(__file__).parent / "test_read_scdictbase.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(base))

    with patch("loanpy.qfysc.literal_eval") as literal_eval_mock:
        literal_eval_mock.return_value = base

        # assert
        assert read_scdictbase(base) == base
        assert read_scdictbase(path) == base

    # assert call
    literal_eval_mock.assert_called_with(str(base))

    # tear down
    remove(path)
    del base, path

#rewrite this by merging it with the other test_init below.
def test_init():
    """test if class Etym is initiated correctly"""

    # set up mock class to patch panphon
    class DistMonkey:
        def __init__(self):
            pass
        def weighted_feature_edit_distance():
            pass

    #set up patches
    with patch("loanpy.qfysc.cldf2pd") as cldf2pd_mock:
        cldf2pd_mock.return_value = None, None
        with patch("loanpy.qfysc.Etym.get_inventories"
                   ) as get_inventories_mock:
            get_inventories_mock.return_value = {}
            with patch("loanpy.qfysc.read_dst"
                       ) as read_dst_mock:
                read_dst_mock.return_value = "distfunc"
                # with patch("loanpy.qfysc.read_nsedict") as read_nsedict_mock:
                #    read_nsedict_mock.return_value = {}
                with patch("loanpy.qfysc.read_scdictbase"
                           ) as read_scdictbase_mock:
                    read_scdictbase_mock.return_value = {}
                    with patch("loanpy.qfysc.Distance.weighted_feature_edit_distance") as dist_mock:
                        dist_mock.return_value = DistMonkey().weighted_feature_edit_distance
                        mocketym = Etym()

                        # assert
                        assert mocketym.adapting is True
                        assert mocketym.connector == "<"
                        assert mocketym.scdictbase == {}
                        assert mocketym.dfety is None
                        assert mocketym.inventories == {}
                        assert mocketym.distance_measure == dist_mock

                        # double check with __dict__
                        assert len(mocketym.__dict__) == 7
                        assert mocketym.__dict__ == {
                            'connector': '<',
                            'adapting': True,
                            'scdictbase': {},
                            'dfety': None,
                            'dfrest': None,
                            'distance_measure': dist_mock,
                            'inventories': {}}

                read_scdictbase_mock.assert_called_with(None)
                cldf2pd_mock.assert_called_with(
                    None, None, None)
                get_inventories_mock.assert_called_with()
                read_dst_mock.assert_not_called()

    del mocketym
    # set up 2
    dfmk = DataFrame({"Segments": ["abc", "def", "pou"],
               "Cognacy": [1, 1, 2],
               "Language_ID": ["lg2", "lg1", "lg2"]})
    with patch("loanpy.qfysc.read_csv") as read_csv_mock:
        read_csv_mock.return_value = dfmk
        with patch("loanpy.qfysc.cldf2pd") as cldf2pd_mock:
            cldf2pd_mock.return_value = "sth3", "sth5"
            with patch("loanpy.qfysc.Etym.get_inventories"
                       ) as get_inventories_mock:
                get_inventories_mock.return_value = {"sth4": "xy"}
                with patch("loanpy.qfysc.read_dst"
                           ) as read_dst_mock:
                    read_dst_mock.return_value = "sth7"
                    # with patch("loanpy.qfysc.read_nsedict") as read_nsedict_mock:
                    #    read_nsedict_mock.return_value = {}
                    with patch("loanpy.qfysc.read_scdictbase"
                               ) as read_scdictbase_mock:
                        read_scdictbase_mock.return_value = {}
                        with patch("loanpy.qfysc.Distance.weighted_feature_edit_distance") as dist_mock:
                            dist_mock.return_value = DistMonkey().weighted_feature_edit_distance

                            mocketym = Etym(
                                forms_csv="path", source_language="lg1",
                                target_language="lg2")

                            # assert
                            assert mocketym.adapting is True
                            assert mocketym.connector == "<"
                            assert mocketym.scdictbase == {}
                            assert mocketym.dfety is "sth3"
                            assert mocketym.inventories == {"sth4": "xy"}
                            assert mocketym.distance_measure == dist_mock

                            # double check with __dict__
                            assert len(mocketym.__dict__) == 7
                            assert mocketym.__dict__ == {
                                'connector': '<',
                                'adapting': True,
                                'scdictbase': {},
                                'dfety': "sth3",
                                'dfrest': "sth5",
                                'distance_measure': dist_mock,
                                'inventories': {"sth4": "xy"}}

                read_scdictbase_mock.assert_called_with(None)
                cldf2pd_mock.assert_called_with("path", "lg1", "lg2")
                get_inventories_mock.assert_called_with()
                read_dst_mock.assert_not_called()

    # tear down
    del mocketym

def test_cldf2pd():
    """test if the CLDF format is correctly tranformed to a pandas dataframe"""

    # set up
    dfin = DataFrame({"Segments": ["a", "b", "c", "d", "e", "f", "g", "ə"],
                      "CV_Segments": ["hi", "jk", "lm", "no", "pq", "rs", "tu", "vw"],
                      "Cognacy": [1, 1, 2, 2, 3, 3, 4, 4],
                      "Language_ID": ["lg1", "lg2", "lg1", "lg3",
                                      "lg3", "lg2", "lg3", "lg4"],
                      "ProsodicStructure": ["V", "C", "C", "C", "V", "C", "C", "V"]
                      }).to_csv(
                            "test_cldf2pd.csv",
                            encoding="utf-8", index=False
                      )
    dfexp = DataFrame({"Segments_tgt": ["b"],
                       "Segments_src": ["a"],
                       "CV_Segments_tgt": ["jk"],
                       "CV_Segments_src": ["hi"],
                       "ProsodicStructure_tgt": ["C"],
                       "ProsodicStructure_src": ["V"],
                       "Cognacy": [1],
                       })
    dfexp2 = DataFrame({"Segments_tgt": ["f"],
                        "CV_Segments_tgt": ["rs"],
                        "ProsodicStructure_tgt": ["C"]})
    # only cognates are taken, where source and target language occur
    dfout = cldf2pd(
    "test_cldf2pd.csv", source_language="lg1", target_language="lg2"
    )

    # assert
    assert cldf2pd(None, source_language="whatever",
                   target_language="wtvr2") == (None, None)
    assert_frame_equal(dfout[0], dfexp)
    assert_frame_equal(dfout[1], dfexp2)

    # tear down
    remove("test_cldf2pd.csv")
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
    with patch("loanpy.qfysc.Distance") as Distance_mock:
        Distance_mock.return_value = mockdist
        out = read_dst("weighted_feature_edit_distance")

        # assert
        assert ismethod(out)  # check if it is a method of the mocked class

    # assert calls
    assert out == mockdist.weighted_feature_edit_distance
    Distance_mock.assert_called_with()  # the class was called without args

    # tear down
    del mockdist, out, MonkeyDist


def test_align():
    """test if align assigns the correct alignment-function to 2 strings"""

    # set up mock class
    class EtymMonkeyAlign:
        def __init__(self):
            self.align_lingpy_called_with = []
            self.align_clusterwise_called_with = []

        def align_lingpy(self, *args):
            self.align_lingpy_called_with.append([*args])
            return "lingpyaligned"

        def align_clusterwise(self, *args):
            self.align_clusterwise_called_with.append([*args])
            return "clusterwisealigned"

    # initiate mock class, plug in mode
    mockqfy = EtymMonkeyAlign()
    mockqfy.adapting = True
    # assert if lingpy-alignment is assigned correctly if mode=="adapt"
    assert Etym.align(
        self=mockqfy,
        left="leftstr",
        right="rightstr") == "lingpyaligned"
    # assert call
    assert mockqfy.align_lingpy_called_with == [['leftstr', 'rightstr']]

    # set up mock class, plug in mode
    mockqfy = EtymMonkeyAlign()
    mockqfy.adapting = False
    # assert
    assert Etym.align(
        self=mockqfy,
        left="leftstr",
        right="rightstr") == "clusterwisealigned"
    # assert call
    assert mockqfy.align_clusterwise_called_with == [["leftstr", "rightstr"]]

    # tear down
    del mockqfy


def test_align_lingpy():
    """test if lingpy's pairwise alignment function is called correctly"""

    # set up instance of basic mock class, plug in phon2cv,
    # mock lingpy.Pairwise, mock pandas.DataFrame
    mockqfy = EtymMonkey()
    mockpairwise = PairwiseMonkey()
    mockqfy.phon2cv = {"h": "C", "a": "V", "l": "C"}
    with patch("loanpy.qfysc.Pairwise") as Pairwise_mock:
        Pairwise_mock.return_value = mockpairwise
        with patch("loanpy.qfysc.DataFrame") as DataFrame_Monkey:
            exp = {"keys": ["h", "a", "l", "V"], "vals": ["k", "a", "l", "a"]}
            DataFrame_Monkey.return_value = DataFrame(exp)

            # assert
            assert_frame_equal(
                Etym.align_lingpy(
                    self=mockqfy,
                    left="kala",
                    right="hal"),
                DataFrame(exp))

    # assert calls
    Pairwise_mock.assert_has_calls([call(
        seqs='kala', seqB='hal', merge_vowels=False)])
    # Pairwise always initiated by 3 args
    assert DataFrame_Monkey.call_args_list == [call(exp)]
    assert mockpairwise.align_called_with == [[]]

    # tear down
    del mockqfy, exp


def test_align_clusterwise():
    """test if loanpy's own clusterwise-alignment works"""

    # set up basic mock class, plug in phon2cv, create expected output var
    mockqfy = EtymMonkey()
    mockqfy.phon2cv = {
        "ɟ": "C", "ɒ": "V", "l": "C", "o": "V", "ɡ": "C", "j": "C"}
    exp = DataFrame({"keys": ['#-', '#ɟ', 'ɒ', 'l', 'o', 'ɡ#'],
                    "vals": ['-', 'j', 'ɑ', 'l.k', 'ɑ', '-']})

    # assert
    assert_frame_equal(
        Etym.align_clusterwise(self=mockqfy, left="ɟ ɒ l o ɡ", right="j ɑ l.k ɑ"), exp)

    # tear down
    del mockqfy, exp


def test_get_sound_corresp():
    """test if sound correspondences are extracted correctly"""

    # set up: the expected outcome of assert while mode=="adapt"
    exp = [{"a": ["a", "y", "ü"], "b": ["p", "v"], "i": ["e", "i"],
            "k": ["h", "c"], "u": ["u", "o"]},
           {'a<a': 1, 'e<i': 2, 'h<k': 2, 'p<b': 2, 'u<u': 1},
           {'a<a': [13], 'e<i': [12], 'h<k': [12], 'p<b': [13], 'u<u': [13]},
           {'d1': 0}, {'d2': 0}, {'d3': 0}]

    # set up: the expected outcome of assert while mode=="reconstruct"
    exp2 = [{'#-': ['-'], '#b': ['p'], '#k': ['h'], 'a#': ['a'],
             'b': ['p'], 'i': ['e'], 'i#': ['e'],
             'k': ['h'], 'u': ['u']},
            {'#-<*-': 2, '#b<*p': 1, '#k<*h': 1, 'a#<*a': 1, 'b<*p': 1,
             'i#<*e': 1, 'i<*e': 1, 'k<*h': 1, 'u<*u': 1},
            {'#-<*-': [12, 13], '#b<*p': [13], '#k<*h': [12],
             'a#<*a': [13], 'b<*p': [13], 'i#<*e': [12], 'i<*e': [12],
             'k<*h': [12], 'u<*u': [13]},
            {}, {}, {}]

    # set up: create instance 1 of mock class
    mockqfy = EtymMonkeyGetSoundCorresp(
        adapting=True, connector="<", alignreturns1=DataFrame(
            {
                "keys": [
                    "k", "i", "k", "i"], "vals": [
                    "h", "e", "h", "e"]}), alignreturns2=DataFrame(
                        {
                            "keys": [
                                "b", "u", "b", "a"], "vals": [
                                    "p", "u", "p", "a"]}))

    # set up: create instance 2 of mock class
    mockqfy2 = EtymMonkeyGetSoundCorresp(  # necessary bc of iter()
        adapting=False, connector="<*", vfb="əœʌ",
        alignreturns1=DataFrame(
            {"keys": ["#-", "#k", "i", "k", "i#"],
             "vals": ["-", "h", "e", "h", "e"]}),
        alignreturns2=DataFrame(
            {"keys": ["#-", "#b", "u", "b", "a#"],
             "vals": ["-", "p", "u", "p", "a"]}))

    # set up the side_effects of pandas.concat
    dfconcat = DataFrame({"keys": list("kikibuba"), "vals": list("hehepupa")})
    dfconcat2 = DataFrame(
        {"keys": ["#-", "#k", "i", "k", "i#", "#-", "#b", "u", "b", "a#"],
         "vals": ["-", "h", "e", "h", "e", "-", "p", "u", "p", "a"]})

    # set up path for param write_to
    path2test_get_sound_corresp = Path(
        __file__).parent / "test_get_sound_corresp.txt"

    # mock pandas.concat
    with patch("loanpy.qfysc.concat", side_effect=[
            dfconcat, dfconcat2]) as concat_mock:
        # groupby too difficult to mock
        # assert while mode=="adapt"
        assert Etym.get_sound_corresp(
            self=mockqfy, write_to=None) == exp
        try:  # assert that no file was being written
            remove(Path(__file__).parent / "soundchanges.txt")
            assert 1 == 2  # this asserts that the except part was being run
        except FileNotFoundError:  # i.e. the file was correctly not written
            pass
        # assert while mode=="reconstruct"
        assert Etym.get_sound_corresp(
            self=mockqfy2,
            write_to=path2test_get_sound_corresp) == exp2
        # assert that file was written
        with open(path2test_get_sound_corresp, "r", encoding="utf-8") as f:
            assert literal_eval(f.read()) == exp2

        # assert calls from assert while mode == "adapt"
        assert mockqfy.align_called_with == [
            ("kiki", "hehe"), ("buba", "pupa")]
        assert mockqfy.rank_closest_phonotactics_called_with == []  # no called
        for act, exp in zip(
            concat_mock.call_args_list[0][0][0], [
                mockqfy.df1, mockqfy.df2]):
            # first [0] picks the only call from list of calls
            # second [0] converts call object to tuple
            # third [0] picks the first element of the tuple (2nd is empty),
            # which is a list of two dataframes
            assert_frame_equal(act, exp)

    # assert calls of assert while mode == "reconstruct"
    assert mockqfy2.align_called_with == [
        ("kiki", "hehe"), ("buba", "pupa")]
    assert mockqfy2.rank_closest_phonotactics_called_with == []  # not called
    for act, exp in zip(
        concat_mock.call_args_list[1][0][0], [
            mockqfy2.df1, mockqfy2.df2]):
        # first [0] picks the only call from list of calls
        # second [0] converts call object to tuple
        # third [0] picks the first element of the tuple (2nd is empty),
        # which is a list of two dataframes
        assert_frame_equal(act, exp)

    # tear down
    remove(path2test_get_sound_corresp)
    del mockqfy, dfconcat, exp, path2test_get_sound_corresp


def test_get_phonotactics_corresp():
    """test if phonotactic correspondences are extracted correctly from data"""

    # set up instance of mock class, plug in attributes
    # def expected outcome var, vars for side-effects of mock-functions,
    # vars for expected calls
    # mock pandasDataFrame
    mockqfy = EtymMonkeyGetSoundCorresp(
        adapting=True,
        connector="<",
        alignreturns1=None,
        alignreturns2=None)

    mockqfy.dfety = DataFrame({"Segments_tgt": ["k i k i", "b u b a"],
                               "Segments_src": ["h e h e", "p u p a"],
                               "CV_Segments_tgt": ["k i k i ", "b u b a"],
                               "CV_Segments_src": ["h e h e", "p u p a"],
                               "ProsodicStructure_tgt": ["CVCV", "CVCV"],
                               "ProsodicStructure_src": ["CVCV", "CVCV"],
                               "Cognacy": [12, 13]})
    mockqfy.left = "Target_Form"
    mockqfy.right = "Source_Form"
    mockqfy.connector = "<"
    exp = [{"CVCV": ["CVCV", "CVC", "CVCCC"]},
           {"CVCV<CVCV": 2},
           {"CVCV<CVCV": [12, 13]}]
    exp_call1 = list(zip(["CVCV", "CVCV"], ["CVCV", "CVCV"], [12, 13]))
    exp_call2 = {"columns": ["keys", "vals", "wordchange"]}

    path2test_get_phonotactics_corresp = Path(
        __file__).parent / "phonotctchange.txt"

    with patch("loanpy.qfysc.DataFrame") as DataFrame_mock:
        DataFrame_mock.return_value = DataFrame(
            {"keys": ["CVCV"] * 2, "vals": ["CVCV"] * 2,
             "wordchange": [12, 13]})

        # assert
        assert Etym.get_phonotactics_corresp(
            self=mockqfy, write_to=path2test_get_phonotactics_corresp) == exp
        # assert file was written
        with open(path2test_get_phonotactics_corresp, "r",
                  encoding="utf-8") as f:
            assert literal_eval(f.read()) == exp

    # assert calls
    assert list(DataFrame_mock.call_args_list[0][0][0]) == exp_call1
    assert DataFrame_mock.call_args_list[0][1] == exp_call2
    assert mockqfy.rank_closest_phonotactics_called_with == ["CVCV"]

    # tear down
    remove(path2test_get_phonotactics_corresp)
    del mockqfy, exp, path2test_get_phonotactics_corresp
